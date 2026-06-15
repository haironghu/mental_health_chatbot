"""
测试公共 fixtures。
设置必要的环境变量，使 app.config.Settings 可以在无 .env 文件时初始化。
"""
import os
import shutil
import tempfile

import pytest

# 在任何 app 模块被导入之前设置环境变量
os.environ.setdefault("OPENROUTER_API_KEY", "fake-test-key")


@pytest.fixture()
def tmp_sessions_dir(monkeypatch, tmp_path):
    """
    将 session_store 的存储目录指向临时目录，避免污染真实数据。
    """
    import app.storage.session_store as store
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(store, "SESSIONS_DIR", sessions_dir)
    return sessions_dir


@pytest.fixture()
def default_session():
    """返回一个默认的会话 dict。"""
    return {
        "state": "welcome",
        "turn_count": 0,
        "state_turn_count": 0,
        "risk_score": 0.0,
        "alert_level": "green",
        "consecutive_negative_turns": 0,
        "k6_scores": {
            "tense": 0, "helpless": 0, "restless": 0,
            "depressed": 0, "effortful": 0, "worthless": 0,
        },
        "k6_total": 0,
        "k6_severity": "mild",
        "k6_complete": False,
        "k6_completed_at": None,
        "pm_strategies_used": [],
        "closure_done": False,
        "memory_summary": "",
        "decision_log": [],
        "history": [],
    }
