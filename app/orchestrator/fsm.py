"""
会话有限状态机（FSM）。
LLM 只负责生成文字，状态流转完全由规则决定，不由 LLM 控制。

流程：
  WELCOME → K6_ASSESSMENT → (PM_STRESS_MGMT|PM_PROBLEM_SOLVING|
                             PM_BEHAVIORAL_ACTIVATION|PM_SOCIAL_SUPPORT)
          → PM_DECISION → (下一个 PM+ 策略 | CLOSURE)
  任何状态 → CRISIS_INTERVENTION（危机触发）
"""
from enum import Enum
from typing import Optional


class SessionState(str, Enum):
    WELCOME = "welcome"
    K6_ASSESSMENT = "k6_assessment"                      # K6 评估（自由对话中推断分数）
    PM_STRESS_MGMT = "pm_stress_mgmt"                    # PM+ 压力管理（呼吸训练）
    PM_PROBLEM_SOLVING = "pm_problem_solving"            # PM+ 解决问题
    PM_BEHAVIORAL_ACTIVATION = "pm_behavioral_activation"  # PM+ 行为激活
    PM_SOCIAL_SUPPORT = "pm_social_support"              # PM+ 强化社交支持
    PM_DECISION = "pm_decision"                          # 询问用户是否继续下一个策略
    CRISIS_INTERVENTION = "crisis_intervention"          # 危机干预
    CLOSURE = "closure"                                  # 结束


# 所有 PM+ 策略状态
PM_STRATEGY_STATES = [
    SessionState.PM_STRESS_MGMT,
    SessionState.PM_PROBLEM_SOLVING,
    SessionState.PM_BEHAVIORAL_ACTIVATION,
    SessionState.PM_SOCIAL_SUPPORT,
]


# 允许的状态转换表
TRANSITIONS: dict[SessionState, list[SessionState]] = {
    SessionState.WELCOME: [
        SessionState.K6_ASSESSMENT,
        SessionState.CRISIS_INTERVENTION,
    ],
    SessionState.K6_ASSESSMENT: PM_STRATEGY_STATES + [
        SessionState.CRISIS_INTERVENTION,
        SessionState.CLOSURE,
    ],
    SessionState.PM_STRESS_MGMT: [
        SessionState.PM_DECISION,
        SessionState.CRISIS_INTERVENTION,
    ],
    SessionState.PM_PROBLEM_SOLVING: [
        SessionState.PM_DECISION,
        SessionState.CRISIS_INTERVENTION,
    ],
    SessionState.PM_BEHAVIORAL_ACTIVATION: [
        SessionState.PM_DECISION,
        SessionState.CRISIS_INTERVENTION,
    ],
    SessionState.PM_SOCIAL_SUPPORT: [
        SessionState.PM_DECISION,
        SessionState.CRISIS_INTERVENTION,
    ],
    SessionState.PM_DECISION: PM_STRATEGY_STATES + [
        SessionState.CLOSURE,
        SessionState.CRISIS_INTERVENTION,
    ],
    SessionState.CRISIS_INTERVENTION: [
        SessionState.CLOSURE,
    ],
    SessionState.CLOSURE: [],
}


class SessionFSM:
    """
    管理单个会话的状态流转。
    从 session dict 初始化，操作完成后调用 apply_to() 写回。
    """

    def __init__(self, session: dict):
        self.state = SessionState(session.get("state", SessionState.WELCOME))
        self.pm_strategies_used: list[str] = session.get("pm_strategies_used", [])
        self.turn_count: int = session.get("turn_count", 0)
        self.state_turn_count: int = session.get("state_turn_count", 0)
        self.closure_done: bool = session.get("closure_done", False)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.state == SessionState.CLOSURE and self.closure_done

    def can_transition(self, target: SessionState) -> bool:
        return target in TRANSITIONS.get(self.state, [])

    def is_pm_strategy(self) -> bool:
        return self.state in PM_STRATEGY_STATES

    # ------------------------------------------------------------------
    # 状态转换
    # ------------------------------------------------------------------

    def stay(self) -> None:
        """留在当前状态，增加轮次计数。"""
        self.turn_count += 1
        self.state_turn_count += 1

    def transition(self, target: SessionState) -> None:
        if not self.can_transition(target):
            raise ValueError(f"不允许的状态转换: {self.state} → {target}")
        # 如果离开某个 PM+ 策略状态，记录已用过
        if self.state in PM_STRATEGY_STATES:
            name = self.state.value
            if name not in self.pm_strategies_used:
                self.pm_strategies_used.append(name)
        self.state = target
        self.turn_count += 1
        self.state_turn_count = 0  # 进入新状态，重置状态内轮次

    def force_crisis(self) -> None:
        """绕过正常转换直接进入危机干预（最高优先级）。"""
        if self.state in PM_STRATEGY_STATES:
            name = self.state.value
            if name not in self.pm_strategies_used:
                self.pm_strategies_used.append(name)
        self.state = SessionState.CRISIS_INTERVENTION
        self.turn_count += 1
        self.state_turn_count = 0

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def apply_to(self, session: dict) -> None:
        """将 FSM 状态写回 session dict。"""
        session["state"] = self.state.value
        session["pm_strategies_used"] = self.pm_strategies_used
        session["turn_count"] = self.turn_count
        session["state_turn_count"] = self.state_turn_count
        session["closure_done"] = self.closure_done
