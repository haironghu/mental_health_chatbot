"""会话存储测试（使用临时目录，不污染真实数据）。"""
import json

from app.storage import session_store


class TestUserHash:
    def test_hash_deterministic(self):
        h1 = session_store._user_hash("+85298765432")
        h2 = session_store._user_hash("+85298765432")
        assert h1 == h2

    def test_hash_different_for_different_phones(self):
        h1 = session_store._user_hash("+85298765432")
        h2 = session_store._user_hash("+85211111111")
        assert h1 != h2

    def test_hash_length(self):
        h = session_store._user_hash("+85298765432")
        assert len(h) == 32


class TestGetOrCreate:
    def test_creates_new_session(self, tmp_sessions_dir):
        user_hash, session = session_store.get_or_create("+85299990000")
        assert session["state"] == "welcome"
        assert session["turn_count"] == 0
        assert session["history"] == []
        # 文件应已创建
        path = tmp_sessions_dir / f"{user_hash}.json"
        assert path.exists()

    def test_loads_existing_session(self, tmp_sessions_dir):
        # 第一次创建
        user_hash, session = session_store.get_or_create("+85299990000")
        session["state"] = "screening_emotion"
        session["turn_count"] = 2
        session_store.save(user_hash, session)

        # 第二次加载
        user_hash2, session2 = session_store.get_or_create("+85299990000")
        assert user_hash2 == user_hash
        assert session2["state"] == "screening_emotion"
        assert session2["turn_count"] == 2


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_sessions_dir):
        data = {
            "state": "deep_inquiry",
            "turn_count": 5,
            "risk_score": 42.5,
            "alert_level": "yellow",
            "completed_dimensions": ["screening_emotion"],
            "consecutive_negative_turns": 1,
            "history": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "嗨！"},
            ],
        }
        session_store.save("testhash123", data)
        loaded = session_store.load("testhash123")
        assert loaded == data

    def test_load_nonexistent_returns_none(self, tmp_sessions_dir):
        result = session_store.load("nonexistent_hash_abc")
        assert result is None

    def test_chinese_content_preserved(self, tmp_sessions_dir):
        """确保中文（粤语）内容存储不乱码。"""
        data = {
            "state": "welcome",
            "turn_count": 0,
            "risk_score": 0,
            "alert_level": "green",
            "completed_dimensions": [],
            "consecutive_negative_turns": 0,
            "history": [
                {"role": "user", "content": "我最近好唔開心"},
                {"role": "assistant", "content": "我聽到你講最近唔開心，可以同我傾多啲嗎？"},
            ],
        }
        session_store.save("chinese_test", data)
        loaded = session_store.load("chinese_test")
        assert loaded["history"][0]["content"] == "我最近好唔開心"


class TestAppendMessage:
    def test_appends_to_history(self, default_session):
        session_store.append_message(default_session, "user", "hello")
        session_store.append_message(default_session, "assistant", "hi")
        assert len(default_session["history"]) == 2
        assert default_session["history"][0] == {"role": "user", "content": "hello"}
        assert default_session["history"][1] == {"role": "assistant", "content": "hi"}


class TestRecentHistory:
    def test_returns_last_n_turns(self, default_session):
        # 添加 6 条消息（3 轮）
        for i in range(3):
            default_session["history"].append({"role": "user", "content": f"msg {i}"})
            default_session["history"].append({"role": "assistant", "content": f"reply {i}"})

        recent = session_store.recent_history(default_session, n_turns=2)
        assert len(recent) == 4  # 2 轮 × 2 条
        assert recent[0]["content"] == "msg 1"

    def test_returns_all_if_less_than_n(self, default_session):
        default_session["history"].append({"role": "user", "content": "only one"})
        recent = session_store.recent_history(default_session, n_turns=10)
        assert len(recent) == 1
