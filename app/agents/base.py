"""
Agent 框架基类。

两类 Agent：
- AnalysisAgent：输入对话上下文，输出结构化信号 dict（如 R(t) 信号、K6 分数）
- ResponseAgent：输入对话上下文，输出自然语言回复 str

所有 Agent 都应在内部捕获异常并返回安全默认值，避免单个 Agent 失败拖垮整条流水线。
Coordinator 负责调度，Agent 之间不互相调用。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """传给 Agent 的对话上下文（只读快照）。"""
    user_message: str
    history: list[dict]
    session: dict                      # 当前会话（含 k6_scores、risk_score 等）
    fsm_state: str                     # 当前 FSM 状态值
    # 以下字段在分析完成后由 Coordinator 填充，供 ResponseAgent 使用
    analysis: dict = field(default_factory=dict)
    alert_level: str = "green"
    stabilize: bool = False
    language: str = "粵語"
    hotline_already_given: bool = False
    k6_progress: dict = field(default_factory=dict)
    pm_strategies_used: list[str] = field(default_factory=list)
    remaining_strategies: list[str] = field(default_factory=list)


class Agent(ABC):
    """所有 Agent 的共同基类。"""

    #: Agent 名称（用于日志 / trace）
    name: str = "agent"
    #: 该 Agent 使用的模型档位（None / "" 则回退默认模型）
    model: str | None = None


class AnalysisAgent(Agent):
    """分析型 Agent：输出结构化信号 dict。"""

    @abstractmethod
    def analyze(self, ctx: AgentContext) -> dict[str, Any]:
        """分析上下文，返回信号 dict。失败时应返回安全默认值。"""
        ...

    @abstractmethod
    def safe_default(self) -> dict[str, Any]:
        """该 Agent 失败时的安全默认输出。"""
        ...


class ResponseAgent(Agent):
    """响应型 Agent：输出自然语言回复。"""

    @abstractmethod
    def respond(self, ctx: AgentContext) -> str:
        """根据上下文生成回复文本。"""
        ...
