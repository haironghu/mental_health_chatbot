"""
TriageAgent（分诊 Agent）。

职责：每轮运行，从用户消息中提取情绪/行为风险信号（R(t) 用）、语言、用户意愿。
危机/自杀意念检测已分离到 SafetyMonitorAgent（Phase 2）。
特点：高频，使用便宜模型。失败时返回安全默认值（全 0）。
"""
import logging

from app.agents.base import AgentContext, AnalysisAgent
from app.config import settings
from app.intelligence import llm, prompt_builder

logger = logging.getLogger(__name__)


class TriageAgent(AnalysisAgent):
    name = "triage"

    def __init__(self):
        self.model = settings.model_triage  # 空字符串则 llm 内部回退默认模型

    def safe_default(self) -> dict:
        return {
            "s_emotion": 0.0,
            "s_behavior": 0.0,
            "language": "cantonese",
            "emotion_labels": [],
            "wants_to_continue": None,
        }

    def analyze(self, ctx: AgentContext) -> dict:
        try:
            system, messages = prompt_builder.build_triage_prompt(
                user_message=ctx.user_message,
                history=ctx.history,
                memory_summary=ctx.memory_summary,
            )
            result = llm.complete_json(messages, system=system, model=self.model)
            # 合并到安全默认值上，保证字段齐全
            merged = self.safe_default()
            merged.update(result)
            return merged
        except Exception:
            logger.exception("[triage] 分析失败，使用安全默认值")
            return self.safe_default()
