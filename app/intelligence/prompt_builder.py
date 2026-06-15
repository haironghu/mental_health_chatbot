"""
多层 Prompt 组装器。
按照 FSM 当前状态选择 Task Prompt 模板，拼接 System + Task + Safety。
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from app.orchestrator.fsm import SessionState

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)

# FSM 状态 → task 模板文件名
_TASK_TEMPLATE: dict[SessionState, str] = {
    SessionState.WELCOME: "tasks/welcome.jinja2",
    SessionState.K6_ASSESSMENT: "tasks/k6_assessment.jinja2",
    SessionState.PM_STRESS_MGMT: "tasks/pm_stress_mgmt.jinja2",
    SessionState.PM_PROBLEM_SOLVING: "tasks/pm_problem_solving.jinja2",
    SessionState.PM_BEHAVIORAL_ACTIVATION: "tasks/pm_behavioral_activation.jinja2",
    SessionState.PM_SOCIAL_SUPPORT: "tasks/pm_social_support.jinja2",
    SessionState.PM_DECISION: "tasks/pm_decision.jinja2",
    SessionState.CRISIS_INTERVENTION: "tasks/crisis_intervention.jinja2",
    SessionState.CLOSURE: "tasks/closure.jinja2",
}


def _render(template_name: str, **kwargs) -> str:
    return _env.get_template(template_name).render(**kwargs)


def build_triage_prompt(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    构造分诊 Prompt（TriageAgent 用）：R(t) 信号 + 危机 + 语言 + 用户意愿。
    返回 (system_text, messages)。
    """
    system = _render("agents/triage.jinja2", user_message=user_message, history=history)
    messages = [{"role": "user", "content": user_message}]
    return system, messages


def build_k6_prompt(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    构造 K6 评分 Prompt（K6ScorerAgent 用）：六维度分数。
    返回 (system_text, messages)。
    """
    system = _render("agents/k6_scoring.jinja2", user_message=user_message, history=history)
    messages = [{"role": "user", "content": user_message}]
    return system, messages


def build_safety_prompt(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    构造安全监测 Prompt（SafetyMonitorAgent 用）：危机检测。
    返回 (system_text, messages)。
    """
    system = _render("agents/safety_monitor.jinja2", user_message=user_message, history=history)
    messages = [{"role": "user", "content": user_message}]
    return system, messages


def build_response_prompt(
    state: SessionState,
    history: list[dict],
    user_message: str,
    analysis: dict,
    stabilize: bool,
    alert_level: str,
    language: str = "粵語",
    hotline_already_given: bool = False,
    k6_progress: dict | None = None,
    pm_strategies_used: list[str] | None = None,
    remaining_strategies: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """
    构造回复 Prompt（System + Task + Safety 三层）。
    返回 (system_text, messages)。

    新增参数：
    - k6_progress: K6 评估进度（用于 k6_assessment 模板），含 nonzero_dims / missing_dims / turns
    - pm_strategies_used: 已用 PM+ 策略列表（用于 pm_decision 模板）
    - remaining_strategies: 仍可推荐的策略列表（用于 pm_decision 模板）
    """
    task_template = _TASK_TEMPLATE.get(state, "tasks/welcome.jinja2")

    # 默认空值，模板会处理 None 情况
    k6_progress = k6_progress or {"nonzero_dims": [], "missing_dims": [], "turns": 0}
    pm_strategies_used = pm_strategies_used or []
    remaining_strategies = remaining_strategies or []

    system_parts = [
        _render("system.jinja2", language=language),
        _render(
            task_template,
            analysis=analysis,
            alert_level=alert_level,
            hotline_already_given=hotline_already_given,
            k6_progress=k6_progress,
            pm_strategies_used=pm_strategies_used,
            remaining_strategies=remaining_strategies,
        ),
        _render(
            "safety.jinja2",
            stabilize=stabilize,
            alert_level=alert_level,
            hotline_already_given=hotline_already_given,
        ),
    ]
    system = "\n\n---\n\n".join(system_parts)

    # 保留最近对话历史 + 当前用户消息
    messages = list(history) + [{"role": "user", "content": user_message}]
    return system, messages
