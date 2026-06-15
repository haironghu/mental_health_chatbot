"""
本地JSON文件会话存储。
每个用户一个文件：data/sessions/{sha256(phone)}.json
"""
import hashlib
import json
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path(__file__).parent.parent.parent / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _user_hash(phone: str) -> str:
    """将手机号哈希化，不在磁盘上存储明文手机号。"""
    return hashlib.sha256(phone.encode()).hexdigest()[:32]


def _session_path(user_hash: str) -> Path:
    return SESSIONS_DIR / f"{user_hash}.json"


def _default_session() -> dict:
    return {
        "state": "welcome",
        "turn_count": 0,
        "state_turn_count": 0,
        # 实时风险评分（R(t)），用于危机检测和稳定介入
        "risk_score": 0.0,
        "alert_level": "green",
        "consecutive_negative_turns": 0,
        # K6 心理困扰量表评分（六维度 0-4，跨轮累积取最大值）
        "k6_scores": {
            "tense": 0,        # 緊張
            "helpless": 0,     # 無助
            "restless": 0,     # 焦躁/坐立不安
            "depressed": 0,    # 抑鬱/乜都唔開心
            "effortful": 0,    # 所有嘢都好費力
            "worthless": 0,    # 覺得自己冇價值
        },
        "k6_total": 0,                    # 0-24 总分
        "k6_severity": "mild",            # mild / moderate / severe
        "k6_complete": False,             # K6 评估是否完成
        "k6_completed_at": None,          # 完成时间（ISO 8601）
        # PM+ 已使用过的策略状态名（避免重复）
        "pm_strategies_used": [],
        "closure_done": False,
        # 较早对话的滚动摘要（Memory Agent 维护，降低长会话 token 成本）
        "memory_summary": "",
        # 决策审计日志（PM+ 策略选择等关键确定性决策的依据记录）
        "decision_log": [],
        "history": [],  # [{role: "user"|"assistant", content: "..."}]
    }


def load(user_hash: str) -> Optional[dict]:
    """加载会话，不存在返回 None。"""
    path = _session_path(user_hash)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save(user_hash: str, session: dict) -> None:
    """保存会话到JSON文件。"""
    path = _session_path(user_hash)
    with path.open("w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def get_or_create(phone: str) -> tuple[str, dict]:
    """
    根据手机号获取或创建会话。
    返回 (user_hash, session_dict)。
    """
    user_hash = _user_hash(phone)
    session = load(user_hash)
    if session is None:
        session = _default_session()
        save(user_hash, session)
    return user_hash, session


def reset(user_hash: str) -> dict:
    """重置用户会话，返回新的默认会话。"""
    session = _default_session()
    save(user_hash, session)
    return session


def append_message(session: dict, role: str, content: str) -> None:
    """向会话历史追加一条消息（原地修改）。"""
    session["history"].append({"role": role, "content": content})


def recent_history(session: dict, n_turns: int = 10) -> list[dict]:
    """
    返回最近 n_turns 轮的对话历史（每轮含用户+助手共2条）。
    用于构造 LLM messages 列表。
    """
    history = session["history"]
    return history[-(n_turns * 2):]
