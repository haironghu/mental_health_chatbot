"""
主编排器：负责会话生命周期，将分析/决策/回复委托给 Coordinator（多 Agent 架构）。

职责分工：
- orchestrator：会话加载/重置/终止判断/轮次上限/持久化（基础设施）
- Coordinator：分析 Agent 调度 + 确定性决策 + Therapist 回复（业务大脑）

详见 docs/MULTI_AGENT_DESIGN.md
"""
import logging
from dataclasses import dataclass

from app.agents.coordinator import Coordinator
from app.config import settings
from app.orchestrator.fsm import SessionFSM, SessionState
from app.storage import session_store

logger = logging.getLogger(__name__)

# 全局协调器（无状态，可复用）
_coordinator = Coordinator()


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

    # ── 2. 委托 Coordinator 处理（分析 + 决策 + 回复） ──────────────
    result = _coordinator.run(
        session=session,
        fsm=fsm,
        user_message=user_message,
        history=history,
    )

    # ── 3. 持久化 ───────────────────────────────────────────────────
    session_store.append_message(session, "user", user_message)
    session_store.append_message(session, "assistant", result.response_text)
    fsm.apply_to(session)
    session_store.save(user_hash, session)

    if result.trace:
        logger.info("[trace] %s", result.trace)

    return _make_result(session, result.response_text)


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
