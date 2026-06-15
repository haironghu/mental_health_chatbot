"""
K6ScorerAgent（K6 评分 Agent）。

职责：仅在 K6_ASSESSMENT 状态运行，从对话推断 K6 六维度分数（0-4）。
特点：需要一定推理质量，使用中档模型。失败时返回全 0（经 max 平滑不影响已有分数）。

注意：本 Agent 只输出「本轮推断」的维度分数，跨轮平滑（max 累积）由 k6_scorer.update_scores 负责。
"""
import logging

from app.agents.base import AgentContext, AnalysisAgent
from app.config import settings
from app.intelligence import llm, prompt_builder
from app.safety.k6_scorer import K6_DIMENSIONS

logger = logging.getLogger(__name__)


class K6ScorerAgent(AnalysisAgent):
    name = "k6_scorer"

    def __init__(self):
        self.model = settings.model_k6

    def safe_default(self) -> dict:
        # 返回 {"k6_dim_scores": {全 0}}，与 Coordinator 合并约定一致
        return {"k6_dim_scores": {dim: 0 for dim in K6_DIMENSIONS}}

    def analyze(self, ctx: AgentContext) -> dict:
        try:
            system, messages = prompt_builder.build_k6_prompt(
                user_message=ctx.user_message,
                history=ctx.history,
            )
            result = llm.complete_json(messages, system=system, model=self.model)
            # 只取六个维度，缺失补 0
            dim_scores = {dim: int(result.get(dim, 0)) for dim in K6_DIMENSIONS}
            return {"k6_dim_scores": dim_scores}
        except Exception:
            logger.exception("[k6_scorer] 评分失败，使用安全默认值")
            return self.safe_default()
