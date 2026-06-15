"""
主编排器：将所有模块串联成一条完整的处理流水线。

处理流程：
1. 加载/创建会话 + 初始化 FSM
2. Analysis LLM 调用 → 结构化 JSON（R(t) 信号 + K6 各维度分数 + 危机标志 + 用户意愿）
3. 更新 R(t) 风险评分（实时安全监控）
4. 更新 K6 分数（跨轮平滑，max 累积）
5. FSM 状态决策：
   - 危机 → force_crisis（最高优先级）
   - WELCOME → K6_ASSESSMENT
   - K6_ASSESSMENT → 若完成则选 PM+ 策略，否则 stay
   - PM+ 策略状态 → 聊够 5 轮则进 PM_DECISION
   - PM_DECISION → 根据用户意愿选下个策略或 CLOSURE
6. Response LLM 调用 → 自然语言回复
7. 持久化会话
"""
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.intelligence import llm, prompt_builder
from app.orchestrator.fsm import (
    PM_STRATEGY_STATES,
    SessionFSM,
    SessionState,
)
from app.safety import k6_scorer, risk_monitor
from app.storage import session_store


@dataclass
class ProcessResult:
    response_text: str
    state: str
    alert_level: str
    risk_score: float
    k6_total: int
    k6_severity: str
    k6_complete: bool


def process(phone: str, user_message: str) -> ProcessResult:
    """
    处理一条来自用户的消息，返回机器人回复。
    phone: 原始手机号（仅用于哈希，不持久化）
    """
    # ── 1. 加载会话 ────────────────────────────────────────────────
    user_hash, session = session_store.get_or_create(phone)

    # 用户发「开始」→ 重置会话
    clean_msg = user_message.strip().strip("「」『』\"'")
    if clean_msg in ("開始", "开始"):
        session = session_store.reset(user_hash)

    fsm = SessionFSM(session)

    # 会话已结束，不再处理
    if fsm.is_terminal():
        return _make_result(
            session,
            "我哋嘅對話已經結束喇。如果你想重新傾計，可以發送「開始」。",
        )

    # 超过轮次上限
    if fsm.turn_count >= settings.max_turns:
        fsm.state = SessionState.CLOSURE
        fsm.apply_to(session)
        session_store.save(user_hash, session)
        return _make_result(
            session,
            "我哋已經傾咗好耐喇，今日嘅對話就到呢度先。有需要隨時可以返嚟！",
        )

    history = session_store.recent_history(session, n_turns=8)

    # ── 2. Analysis LLM 调用 ────────────────────────────────────────
    try:
        analysis_system, analysis_messages = prompt_builder.build_analysis_prompt(
            user_message=user_message,
            history=history,
        )
        analysis = llm.complete_json(analysis_messages, system=analysis_system)
    except Exception:
        # 分析失败时用安全默认值，不中断主流程
        analysis = {
            "s_emotion": 0.0,
            "s_keyword": 0.0,
            "s_behavior": 0.0,
            "crisis_detected": False,
            "k6_dim_scores": {dim: 0 for dim in k6_scorer.K6_DIMENSIONS},
            "language": "cantonese",
            "emotion_labels": [],
            "wants_to_continue": None,
        }

    # ── 3. 更新 R(t) 风险评分（始终运行，作为安全监控） ────────────
    risk = risk_monitor.update(session, analysis)

    # ── 4. 更新 K6 分数（跨轮平滑） ────────────────────────────────
    k6_dim_scores = analysis.get("k6_dim_scores") or {}
    k6_scorer.update_scores(session, k6_dim_scores)

    # ── 5. FSM 状态决策 ─────────────────────────────────────────────
    if risk_monitor.should_force_crisis(risk) or analysis.get("crisis_detected"):
        fsm.force_crisis()
    elif fsm.state == SessionState.CLOSURE:
        # 已在告别轮，标记完成
        fsm.closure_done = True
    else:
        _advance_fsm(fsm, session, analysis)

    stabilize = risk_monitor.should_stabilize(session, risk)

    # ── 6. 生成回复 ─────────────────────────────────────────────────
    language = _map_language(analysis.get("language", "cantonese"))
    hotline_already_given = _hotline_mentioned_before(session)
    k6_progress = _build_k6_progress(session, fsm)
    remaining_strategies = _remaining_pm_strategies(fsm.pm_strategies_used)

    response_system, response_messages = prompt_builder.build_response_prompt(
        state=fsm.state,
        history=history,
        user_message=user_message,
        analysis=analysis,
        stabilize=stabilize,
        alert_level=risk.level.value,
        language=language,
        hotline_already_given=hotline_already_given,
        k6_progress=k6_progress,
        pm_strategies_used=fsm.pm_strategies_used,
        remaining_strategies=remaining_strategies,
    )
    response_text = llm.complete(response_messages, system=response_system)

    # ── 7. 持久化 ───────────────────────────────────────────────────
    session_store.append_message(session, "user", user_message)
    session_store.append_message(session, "assistant", response_text)
    fsm.apply_to(session)
    session_store.save(user_hash, session)

    return _make_result(session, response_text)


# ────────────────────────────────────────────────────────────────
# FSM 推进逻辑
# ────────────────────────────────────────────────────────────────


def _advance_fsm(fsm: SessionFSM, session: dict, analysis: dict) -> None:
    """根据当前状态、K6 进度、用户意愿推进 FSM。"""
    current = fsm.state

    # WELCOME → K6_ASSESSMENT（自动推进）
    if current == SessionState.WELCOME:
        fsm.transition(SessionState.K6_ASSESSMENT)
        return

    # K6_ASSESSMENT：检查是否完成
    if current == SessionState.K6_ASSESSMENT:
        # 用本轮的 state_turn_count + 1（因为本轮还没结算）
        result = k6_scorer.evaluate(session, fsm.state_turn_count + 1)
        if result.complete:
            # 标记 K6 完成 + 选第一个 PM+ 策略
            k6_scorer.mark_complete(session, result)
            next_strategy = k6_scorer.select_next_pm_strategy(
                result.scores, fsm.pm_strategies_used
            )
            if next_strategy:
                target = SessionState(next_strategy)
                if fsm.can_transition(target):
                    fsm.transition(target)
                    return
            # 选不出策略（极少见，例如所有维度都 0）→ 直接 closure
            if fsm.can_transition(SessionState.CLOSURE):
                fsm.transition(SessionState.CLOSURE)
            return
        # 未完成，继续聊
        fsm.stay()
        return

    # PM+ 策略状态：聊够最少轮数后进 PM_DECISION
    if current in PM_STRATEGY_STATES:
        if fsm.state_turn_count + 1 < settings.min_turns_per_screening:
            fsm.stay()
            return
        # 聊够了
        if fsm.can_transition(SessionState.PM_DECISION):
            fsm.transition(SessionState.PM_DECISION)
        return

    # PM_DECISION：根据用户意愿决定
    if current == SessionState.PM_DECISION:
        wants_continue = analysis.get("wants_to_continue")
        if wants_continue is False:
            # 用户明确想结束
            if fsm.can_transition(SessionState.CLOSURE):
                fsm.transition(SessionState.CLOSURE)
            return
        if wants_continue is True:
            # 用户明确想继续 → 选下一个策略
            k6_scores = session.get("k6_scores", {})
            next_strategy = k6_scorer.select_next_pm_strategy(
                k6_scores, fsm.pm_strategies_used
            )
            if next_strategy:
                target = SessionState(next_strategy)
                if fsm.can_transition(target):
                    fsm.transition(target)
                    return
            # 无更多策略 → closure
            if fsm.can_transition(SessionState.CLOSURE):
                fsm.transition(SessionState.CLOSURE)
            return
        # 用户意愿不明（null），继续停留等待回复
        fsm.stay()
        return


# ────────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────────


def _build_k6_progress(session: dict, fsm: SessionFSM) -> dict:
    """构造给 k6_assessment 模板的进度信息。"""
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
    """返回还未使用的 PM+ 策略中文标签。"""
    labels = {
        "pm_stress_mgmt": "管理壓力",
        "pm_problem_solving": "解決問題",
        "pm_behavioral_activation": "行為激活（重拾愉悅）",
        "pm_social_support": "強化社交支持",
    }
    return [labels[s] for s in labels if s not in used]


# 危机热线号码（用于检测历史中是否已提及）
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
    """检查历史 assistant 消息中是否已经提过危机热线。"""
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


def _make_result(session: dict, response_text: str) -> ProcessResult:
    return ProcessResult(
        response_text=response_text,
        state=session["state"],
        alert_level=session.get("alert_level", "green"),
        risk_score=session.get("risk_score", 0.0),
        k6_total=session.get("k6_total", 0),
        k6_severity=session.get("k6_severity", "mild"),
        k6_complete=session.get("k6_complete", False),
    )
