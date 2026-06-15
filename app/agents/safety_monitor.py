"""
SafetyMonitorAgent（安全监测 Agent）。

职责：每轮并行运行，专注检测自杀/自伤/危机意念。
这是从 TriageAgent 分离出来的安全关键功能——专注的 prompt 比混在分诊里更可靠。

输出：crisis_detected（权威危机信号）+ s_keyword（危机关键词强度，供 R(t)）+ crisis_reason。
失败时返回 crisis_detected=False，但 Coordinator 还有确定性关键词兜底（crisis_keywords）。
"""
import logging

from app.agents.base import AgentContext, AnalysisAgent
from app.config import settings
from app.intelligence import llm, prompt_builder

logger = logging.getLogger(__name__)


class SafetyMonitorAgent(AnalysisAgent):
    name = "safety_monitor"

    def __init__(self):
        self.model = settings.model_safety

    def safe_default(self) -> dict:
        return {
            "crisis_detected": False,
            "s_keyword": 0.0,
            "crisis_reason": "",
        }

    def analyze(self, ctx: AgentContext) -> dict:
        try:
            system, messages = prompt_builder.build_safety_prompt(
                user_message=ctx.user_message,
                history=ctx.history,
                memory_summary=ctx.memory_summary,
            )
            result = llm.complete_json(messages, system=system, model=self.model)
            merged = self.safe_default()
            merged.update(result)
            return merged
        except Exception:
            logger.exception("[safety_monitor] 检测失败，回退安全默认（依赖关键词兜底）")
            return self.safe_default()
