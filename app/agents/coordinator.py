"""
Coordinator（协调器）——多 Agent 架构的「确定性大脑」。

采用 hub-and-spoke 模式：所有 Agent 通过 Coordinator 调度，互不直接通信。

每轮流程：
  1. 并行运行分析 Agent（TriageAgent 每轮 + K6ScorerAgent 仅 K6 阶段）
  2. 合并信号 → analysis dict
  3. 确定性逻辑：更新 R(t)、更新 K6、FSM 状态决策
  4. 运行 TherapistAgent 生成回复

Coordinator 原地修改 session 和 fsm，返回回复文本与一份 trace（用于可观测性）。
会话持久化由上层 orchestrator 负责。
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.agents.base import AgentContext
from app.agents.k6_scorer_agent import K6ScorerAgent
from app.agents.therapist import TherapistAgent
from app.agents.triage import TriageAgent
from app.config import settings
from app.orchestrator.fsm import PM_STRATEGY_STATES, SessionFSM, SessionState
from app.safety import k6_scorer, risk_monitor

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorResult:
    response_text: str
    analysis: dict
    trace: dict = field(default_factory=dict)


class Coordinator:
    def __init__(self):
        self.triage = TriageAgent()
        self.k6_scorer_agent = K6ScorerAgent()
        self.therapist = TherapistAgent()

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(
        self,
        session: dict,
        fsm: SessionFSM,
        user_message: str,
        history: list[dict],
    ) -> CoordinatorResult:
        trace: dict = {}
        ctx = AgentContext(
            user_message=user_message,
            history=history,
            session=session,
            fsm_state=fsm.state.value,
        )

        # ── 1. 并行运行分析 Agent ───────────────────────────────────
        analysis = self._run_analysis_agents(ctx, fsm, trace)
        ctx.analysis = analysis

        # ── 2. 更新 R(t) 风险评分（始终运行，安全监控） ─────────────
        risk = risk_monitor.update(session, analysis)

        # ── 3. 更新 K6 分数（跨轮 max 平滑） ────────────────────────
        k6_dim_scores = analysis.get("k6_dim_scores") or {}
        k6_scorer.update_scores(session, k6_dim_scores)

        # ── 4. FSM 状态决策（确定性） ───────────────────────────────
        if risk_monitor.should_force_crisis(risk) or analysis.get("crisis_detected"):
            fsm.force_crisis()
        elif fsm.state == SessionState.CLOSURE:
            fsm.closure_done = True
        else:
            self._advance_fsm(fsm, session, analysis)

        # ── 5. 准备回复上下文 ───────────────────────────────────────
        ctx.fsm_state = fsm.state.value
        ctx.alert_level = risk.level.value
        ctx.stabilize = risk_monitor.should_stabilize(session, risk)
        ctx.language = _map_language(analysis.get("language", "cantonese"))
        ctx.hotline_already_given = _hotline_mentioned_before(session)
        ctx.k6_progress = _build_k6_progress(session, fsm)
        ctx.pm_strategies_used = fsm.pm_strategies_used
        ctx.remaining_strategies = _remaining_pm_strategies(fsm.pm_strategies_used)

        # ── 6. 运行 TherapistAgent ──────────────────────────────────
        t0 = time.monotonic()
        response_text = self.therapist.respond(ctx)
        trace["therapist_ms"] = round((time.monotonic() - t0) * 1000)

        return CoordinatorResult(
            response_text=response_text,
            analysis=analysis,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # 并行分析
    # ------------------------------------------------------------------

    def _run_analysis_agents(
        self, ctx: AgentContext, fsm: SessionFSM, trace: dict
    ) -> dict:
        """并行运行分析 Agent，合并结果。"""
        # 决定本轮要跑哪些分析 agent
        agents = [self.triage]
        if fsm.state == SessionState.K6_ASSESSMENT:
            agents.append(self.k6_scorer_agent)

        results: dict[str, dict] = {}

        def _run(agent):
            t0 = time.monotonic()
            out = agent.analyze(ctx)
            trace[f"{agent.name}_ms"] = round((time.monotonic() - t0) * 1000)
            return agent.name, out

        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            for name, out in pool.map(_run, agents):
                results[name] = out

        # 合并：triage 提供主信号，k6_scorer 提供 k6_dim_scores
        merged: dict = {}
        merged.update(results.get("triage", self.triage.safe_default()))
        if "k6_scorer" in results:
            merged.update(results["k6_scorer"])
        else:
            # 非 K6 阶段：补全零分（经 max 平滑不影响已有分数）
            merged.update(self.k6_scorer_agent.safe_default())
        return merged

    # ------------------------------------------------------------------
    # FSM 推进逻辑（确定性，从 orchestrator 迁移而来）
    # ------------------------------------------------------------------

    def _advance_fsm(self, fsm: SessionFSM, session: dict, analysis: dict) -> None:
        current = fsm.state

        # WELCOME → K6_ASSESSMENT
        if current == SessionState.WELCOME:
            fsm.transition(SessionState.K6_ASSESSMENT)
            return

        # K6_ASSESSMENT：检查是否完成
        if current == SessionState.K6_ASSESSMENT:
            result = k6_scorer.evaluate(session, fsm.state_turn_count + 1)
            if result.complete:
                k6_scorer.mark_complete(session, result)
                next_strategy = k6_scorer.select_next_pm_strategy(
                    result.scores, fsm.pm_strategies_used
                )
                if next_strategy:
                    target = SessionState(next_strategy)
                    if fsm.can_transition(target):
                        fsm.transition(target)
                        return
                if fsm.can_transition(SessionState.CLOSURE):
                    fsm.transition(SessionState.CLOSURE)
                return
            fsm.stay()
            return

        # PM+ 策略状态：聊够最少轮数后进 PM_DECISION
        if current in PM_STRATEGY_STATES:
            if fsm.state_turn_count + 1 < settings.min_turns_per_screening:
                fsm.stay()
                return
            if fsm.can_transition(SessionState.PM_DECISION):
                fsm.transition(SessionState.PM_DECISION)
            return

        # PM_DECISION：根据用户意愿决定
        if current == SessionState.PM_DECISION:
            wants_continue = analysis.get("wants_to_continue")
            if wants_continue is False:
                if fsm.can_transition(SessionState.CLOSURE):
                    fsm.transition(SessionState.CLOSURE)
                return
            if wants_continue is True:
                k6_scores = session.get("k6_scores", {})
                next_strategy = k6_scorer.select_next_pm_strategy(
                    k6_scores, fsm.pm_strategies_used
                )
                if next_strategy:
                    target = SessionState(next_strategy)
                    if fsm.can_transition(target):
                        fsm.transition(target)
                        return
                if fsm.can_transition(SessionState.CLOSURE):
                    fsm.transition(SessionState.CLOSURE)
                return
            # 意愿不明 → 停留等待
            fsm.stay()
            return


# ────────────────────────────────────────────────────────────────
# 辅助函数（从 orchestrator 迁移）
# ────────────────────────────────────────────────────────────────


def _build_k6_progress(session: dict, fsm: SessionFSM) -> dict:
    scores = session.get("k6_scores", {})
    nonzero = [
        k6_scorer.K6_LABELS_ZH[d]
        for d in k6_scorer.K6_DIMENSIONS
        if int(scores.get(d, 0)) > 0
    ]
    missing = [
        k6_scorer.K6_LABELS_ZH[d]
        for d in k6_scorer.K6_DIMENSIONS
        if int(scores.get(d, 0)) == 0
    ]
    return {
        "nonzero_dims": nonzero,
        "missing_dims": missing,
        "turns": fsm.state_turn_count,
    }


def _remaining_pm_strategies(used: list[str]) -> list[str]:
    labels = {
        "pm_stress_mgmt": "管理壓力",
        "pm_problem_solving": "解決問題",
        "pm_behavioral_activation": "行為激活（重拾愉悅）",
        "pm_social_support": "強化社交支持",
    }
    return [labels[s] for s in labels if s not in used]


_HOTLINE_MARKERS = (
    "2389 2222", "23892222",
    "2382 0000", "23820000",
    "2466 7350", "24667350",
    "2777 8899", "27778899",
    "2711 6622", "27116622",
    "2377 8511", "23778511",
    "18288",
    "撒瑪利亞", "撒玛利亚",
    "生命熱線", "生命热线",
    "關心一線", "关心一线",
    "向晴熱線", "向晴热线",
    "精神健康專線", "精神健康专线",
    "突破輔導", "突破辅导",
    "青年協會", "青年协会",
    "明愛", "明爱",
)


def _hotline_mentioned_before(session: dict) -> bool:
    for msg in session.get("history", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if any(marker in content for marker in _HOTLINE_MARKERS):
            return True
    return False


def _map_language(lang_code: str) -> str:
    mapping = {
        "cantonese": "粵語",
        "mandarin": "普通話",
        "english": "English",
        "mixed": "粵語",
    }
    return mapping.get(lang_code, "粵語")
