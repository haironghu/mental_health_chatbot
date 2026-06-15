"""
TherapistAgent（治疗师 Agent）。

职责：每轮运行，根据 FSM 状态 + 分析信号生成自然语言回复。
特点：质量优先，使用较好的模型。

Phase 1 为单一治疗师；Phase 3 会按状态拆成多个专家
（k6_interview / stress / problem / activation / social / crisis / closure）。
"""
import logging

from app.agents.base import AgentContext, ResponseAgent
from app.config import settings
from app.intelligence import llm, prompt_builder
from app.orchestrator.fsm import SessionState

logger = logging.getLogger(__name__)


class TherapistAgent(ResponseAgent):
    name = "therapist"

    def __init__(self):
        self.model = settings.model_therapist

    def respond(self, ctx: AgentContext) -> str:
        system, messages = prompt_builder.build_response_prompt(
            state=SessionState(ctx.fsm_state),
            history=ctx.history,
            user_message=ctx.user_message,
            analysis=ctx.analysis,
            stabilize=ctx.stabilize,
            alert_level=ctx.alert_level,
            language=ctx.language,
            hotline_already_given=ctx.hotline_already_given,
            k6_progress=ctx.k6_progress,
            pm_strategies_used=ctx.pm_strategies_used,
            remaining_strategies=ctx.remaining_strategies,
        )
        return llm.complete(messages, system=system, model=self.model)
