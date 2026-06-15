"""编排器集成测试（mock LLM 调用）。

注意：多 Agent 架构下，LLM 由各 Agent 调用，因此 mock 点是
共享的 app.intelligence.llm 模块（TriageAgent/K6ScorerAgent 用 complete_json，
TherapistAgent 用 complete）。
"""
from unittest.mock import patch

from app.orchestrator.orchestrator import process


# Triage agent 的 JSON 输出（中性、低风险）
_TRIAGE_OUT = {
    "s_emotion": 20,
    "s_keyword": 0,
    "s_behavior": 5,
    "language": "cantonese",
    "emotion_labels": ["neutral"],
    "crisis_detected": False,
    "wants_to_continue": None,
}

# K6 agent 的 JSON 输出（全 0）
_K6_OUT = {
    "tense": 0, "helpless": 0, "restless": 0,
    "depressed": 0, "effortful": 0, "worthless": 0,
}


def _make_complete_json(triage_out=None, k6_out=None):
    """
    返回 fake complete_json：根据 system prompt 区分 triage / k6 调用。
    K6 评分 prompt 的 system 含「K6 凱斯勒」，分诊 prompt 不含。
    """
    triage_out = triage_out if triage_out is not None else _TRIAGE_OUT
    k6_out = k6_out if k6_out is not None else _K6_OUT

    def fake(messages, *, system="", model=None):
        if "K6 凱斯勒" in system:
            return dict(k6_out)
        return dict(triage_out)
    return fake


class TestProcess:
    """测试完整的 process() 流水线，mock 掉 LLM 调用。"""

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_first_message_welcome_to_k6(self, mock_json, mock_complete, tmp_sessions_dir):
        mock_json.side_effect = _make_complete_json()
        mock_complete.return_value = "嗨！我係你嘅情緒支援小幫手"

        result = process("+85299990001", "你好")
        assert result.response_text == "嗨！我係你嘅情緒支援小幫手"
        assert result.state == "k6_assessment"
        assert result.alert_level == "green"
        assert result.k6_complete is False

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_crisis_detected_forces_crisis_state(self, mock_json, mock_complete, tmp_sessions_dir):
        crisis_triage = {
            **_TRIAGE_OUT,
            "s_emotion": 90, "s_keyword": 95, "s_behavior": 70,
            "crisis_detected": True,
        }
        mock_json.side_effect = _make_complete_json(triage_out=crisis_triage)
        mock_complete.return_value = "我好擔心你，你而家安全嗎？"

        result = process("+85299990002", "我唔想活了")
        assert result.state == "crisis_intervention"
        assert result.alert_level in ("orange", "red")

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_k6_scores_accumulate_across_turns(self, mock_json, mock_complete, tmp_sessions_dir):
        # K6 评分仅在 K6_ASSESSMENT 状态运行，故预置会话于该状态
        from app.storage import session_store
        user_hash, session = session_store.get_or_create("+85299990010")
        session["state"] = "k6_assessment"
        session_store.save(user_hash, session)

        mock_complete.return_value = "..."

        # 第一轮：tense=2, depressed=1
        mock_json.side_effect = _make_complete_json(
            k6_out={"tense": 2, "helpless": 0, "restless": 0,
                    "depressed": 1, "effortful": 0, "worthless": 0}
        )
        process("+85299990010", "msg1")

        # 第二轮：tense=1（应被 max 保留 2）, helpless=3
        mock_json.side_effect = _make_complete_json(
            k6_out={"tense": 1, "helpless": 3, "restless": 0,
                    "depressed": 0, "effortful": 0, "worthless": 0}
        )
        process("+85299990010", "msg2")

        session = session_store.load(user_hash)
        scores = session["k6_scores"]
        assert scores["tense"] == 2       # max(2, 1)
        assert scores["helpless"] == 3    # max(0, 3)
        assert scores["depressed"] == 1   # max(1, 0)

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_terminal_session_returns_fixed_message(self, mock_json, mock_complete, tmp_sessions_dir):
        from app.storage import session_store
        user_hash, session = session_store.get_or_create("+85299990003")
        session["state"] = "closure"
        session["closure_done"] = True
        session_store.save(user_hash, session)

        result = process("+85299990003", "hello again")
        assert "已經結束" in result.response_text
        mock_complete.assert_not_called()

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_session_persisted_after_process(self, mock_json, mock_complete, tmp_sessions_dir):
        mock_json.side_effect = _make_complete_json()
        mock_complete.return_value = "test reply"

        process("+85299990004", "test message")

        from app.storage import session_store
        user_hash = session_store._user_hash("+85299990004")
        session = session_store.load(user_hash)
        assert session is not None
        assert len(session["history"]) == 2
        assert session["history"][0]["content"] == "test message"
        assert session["history"][1]["content"] == "test reply"

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_triage_failure_uses_safe_defaults(self, mock_json, mock_complete, tmp_sessions_dir):
        """Triage LLM 失败时不崩溃，agent 内部回退安全默认值。"""
        mock_json.side_effect = Exception("API Error")
        mock_complete.return_value = "我聽到你啦"

        result = process("+85299990005", "test")
        assert result.response_text == "我聽到你啦"
        assert result.alert_level == "green"  # 安全默认 → 绿色

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_start_keyword_resets_session(self, mock_json, mock_complete, tmp_sessions_dir):
        from app.storage import session_store
        user_hash, session = session_store.get_or_create("+85299990006")
        session["state"] = "pm_decision"
        session["k6_complete"] = True
        session["k6_total"] = 12
        session_store.save(user_hash, session)

        mock_json.side_effect = _make_complete_json()
        mock_complete.return_value = "歡迎"

        result = process("+85299990006", "開始")
        assert result.state == "k6_assessment"
        assert result.k6_complete is False
        assert result.k6_total == 0
