"""
多轮风险累积监控。
R(t) = α×R(t-1) + β×S_emotion(t) + γ×S_keyword(t) + δ×S_behavior(t)

各分量由 LLM Analysis Prompt 的结构化输出提供（0~100 范围）。
"""
from dataclasses import dataclass
from enum import Enum

from app.config import settings


class AlertLevel(str, Enum):
    GREEN = "green"    # R < 30  正常对话
    YELLOW = "yellow"  # 30 ≤ R < 60  增加关怀，资源推送
    ORANGE = "orange"  # 60 ≤ R < 80  危机支持模式，引导热线
    RED = "red"        # R ≥ 80  立即危机协议，通知社工


@dataclass
class RiskResult:
    score: float        # 0~100
    level: AlertLevel
    s_emotion: float
    s_keyword: float
    s_behavior: float
    previous_score: float


def _determine_level(score: float) -> AlertLevel:
    if score < 30:
        return AlertLevel.GREEN
    if score < 60:
        return AlertLevel.YELLOW
    if score < 80:
        return AlertLevel.ORANGE
    return AlertLevel.RED


def update(session: dict, analysis: dict) -> RiskResult:
    """
    根据 LLM 分析结果更新会话风险分数，原地修改 session 并返回结果。

    analysis 需包含：
        s_emotion  float 0~100  情绪信号强度
        s_keyword  float 0~100  危机关键词信号
        s_behavior float 0~100  行为信号（回避、语言模式等）
    """
    r_prev = float(session.get("risk_score", 0.0))
    s_emotion = float(analysis.get("s_emotion", 0.0))
    s_keyword = float(analysis.get("s_keyword", 0.0))
    s_behavior = float(analysis.get("s_behavior", 0.0))

    r_new = (
        settings.risk_alpha * r_prev
        + settings.risk_beta * s_emotion
        + settings.risk_gamma * s_keyword
        + settings.risk_delta * s_behavior
    )
    r_new = max(0.0, min(100.0, r_new))
    level = _determine_level(r_new)

    # 更新连续负面轮次计数
    if s_emotion > 40 or s_keyword > 20:
        session["consecutive_negative_turns"] = session.get("consecutive_negative_turns", 0) + 1
    else:
        session["consecutive_negative_turns"] = 0

    session["risk_score"] = r_new
    session["alert_level"] = level.value

    return RiskResult(
        score=r_new,
        level=level,
        s_emotion=s_emotion,
        s_keyword=s_keyword,
        s_behavior=s_behavior,
        previous_score=r_prev,
    )


def should_force_crisis(risk: RiskResult) -> bool:
    return risk.level == AlertLevel.RED


def should_stabilize(session: dict, risk: RiskResult) -> bool:
    """检测脆弱放大循环：连续多轮负面情绪升级。"""
    return session.get("consecutive_negative_turns", 0) >= 3 or risk.level == AlertLevel.ORANGE
