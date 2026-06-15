"""
多 Agent 模块。

设计原则：FSM 是确定性的「大脑」，Agents 是专家，由 Coordinator 统一调度（hub-and-spoke）。
Agents 之间不互相决策，各自只负责擅长的事。

详见 docs/MULTI_AGENT_DESIGN.md
"""
