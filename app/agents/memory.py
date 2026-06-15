"""
MemoryAgent（记忆 Agent）。

职责：把较早嘅对话压缩成滚动摘要，使后续轮次只需发送「摘要 + 最近几轮」，
而唔系成段历史，降低长会话嘅 token 成本。

按节奏（memory_summary_every）触发，使用便宜模型。
失败时返回旧摘要不变（唔会丢失已有脉络）。
"""
import logging

from app.agents.base import Agent
from app.config import settings
from app.intelligence import llm, prompt_builder

logger = logging.getLogger(__name__)


class MemoryAgent(Agent):
    name = "memory"

    def __init__(self):
        self.model = settings.model_memory

    def summarize(self, previous_summary: str, older_messages: list[dict]) -> str:
        """
        将 previous_summary + older_messages 压缩成新摘要。
        失败时返回 previous_summary 不变。
        """
        if not older_messages:
            return previous_summary
        try:
            system, messages = prompt_builder.build_memory_prompt(
                previous_summary=previous_summary,
                messages=older_messages,
            )
            summary = llm.complete(messages, system=system, model=self.model)
            return (summary or previous_summary).strip()
        except Exception:
            logger.exception("[memory] 摘要失败，保留旧摘要")
            return previous_summary
