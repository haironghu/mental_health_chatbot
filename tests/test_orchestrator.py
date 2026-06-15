"""编排器集成测试（mock LLM 调用）。"""
from unittest.mock import patch

from app.config import settings
from app.orchestrator.orchestrator import process, _advance_fsm, _map_language
from app.orchestrator.fsm import SessionFSM, SessionState


# 默认 LLM 分析结果（中性、低风险、无 K6 信号）
_BASE_ANALYSIS = {
    "s_emotion": 20,
    "s_keyword": 0,
    "s_behavior": 5,
    "language": "cantonese",
    "emotion_labels": ["neutral"],
    "crisis_detected": False,
    "k6_dim_scores": {
        "tense": 0, "helpless": 0, "restless": 0,
        "depressed": 0, "effortful": 0, "worthless": 0,
    },
    "wants_to_continue": None,
}


# ── _advance_fsm 单元测试 ────────────────────────────────────────


class TestAdvanceFSM:
    def test_welcome_advances_to_k6_assessment(self, default_session):
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, _BASE_ANALYSIS)
        assert fsm.state == SessionState.K6_ASSESSMENT

    def test_k6_assessment_stays_when_incomplete(self, default_session):
        default_session["state"] = "k6_assessment"
        default_session["state_turn_count"] = 1  # 才聊 1 轮，肯定不够
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, _BASE_ANALYSIS)
        assert fsm.state == SessionState.K6_ASSESSMENT

    def test_k6_assessment_completes_and_picks_strategy(self, default_session):
        # 4 个维度有非零信号 + state_turn_count + 1 >= 5
        default_session["state"] = "k6_assessment"
        default_session["state_turn_count"] = 4
        default_session["k6_scores"] = {
            "tense": 3, "helpless": 2, "restless": 0,
            "depressed": 3, "effortful": 2, "worthless": 0,
        }
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, _BASE_ANALYSIS)
        # 应该进入某个 PM+ 策略状态
        assert fsm.state in (
            SessionState.PM_STRESS_MGMT,
            SessionState.PM_BEHAVIORAL_ACTIVATION,
            SessionState.PM_SOCIAL_SUPPORT,
            SessionState.PM_PROBLEM_SOLVING,
        )

    def test_k6_completion_marks_session(self, default_session):
        default_session["state"] = "k6_assessment"
        default_session["state_turn_count"] = 4
        default_session["k6_scores"] = {
            "tense": 3, "helpless": 2, "restless": 2,
            "depressed": 3, "effortful": 0, "worthless": 0,
        }
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, _BASE_ANALYSIS)
        assert default_session["k6_complete"] is True
        assert default_session["k6_total"] == 10
        assert default_session["k6_severity"] == "moderate"
        assert default_session["k6_completed_at"] is not None

    def test_pm_strategy_stays_until_min_turns(self, default_session):
        default_session["state"] = "pm_stress_mgmt"
        default_session["state_turn_count"] = 2  # 还没到 5
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, _BASE_ANALYSIS)
        assert fsm.state == SessionState.PM_STRESS_MGMT

    def test_pm_strategy_to_decision_after_enough_turns(self, default_session):
        default_session["state"] = "pm_stress_mgmt"
        # min_turns_per_screening = 5; +1 之后 == 5 OK
        default_session["state_turn_count"] = settings.min_turns_per_screening - 1
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, _BASE_ANALYSIS)
        assert fsm.state == SessionState.PM_DECISION

    def test_pm_decision_continue_picks_next_strategy(self, default_session):
        default_session["state"] = "pm_decision"
        default_session["pm_strategies_used"] = ["pm_stress_mgmt"]
        default_session["k6_scores"] = {
            "tense": 1, "helpless": 3, "restless": 0,
            "depressed": 3, "effortful": 2, "worthless": 0,
        }
        analysis = {**_BASE_ANALYSIS, "wants_to_continue": True}
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, analysis)
        # depressed/helpless 高 → behavioral_activation
        assert fsm.state == SessionState.PM_BEHAVIORAL_ACTIVATION

    def test_pm_decision_end_goes_to_closure(self, default_session):
        default_session["state"] = "pm_decision"
        default_session["pm_strategies_used"] = ["pm_stress_mgmt"]
        analysis = {**_BASE_ANALYSIS, "wants_to_continue": False}
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, analysis)
        assert fsm.state == SessionState.CLOSURE

    def test_pm_decision_unclear_stays(self, default_session):
        default_session["state"] = "pm_decision"
        default_session["pm_strategies_used"] = ["pm_stress_mgmt"]
        analysis = {**_BASE_ANALYSIS, "wants_to_continue": None}
        fsm = SessionFSM(default_session)
        _advance_fsm(fsm, default_session, analysis)
        assert fsm.state == SessionState.PM_DECISION


class TestMapLanguage:
    def test_cantonese(self):
        assert _map_language("cantonese") == "粵語"

    def test_english(self):
        assert _map_language("english") == "English"

    def test_mandarin(self):
        assert _map_language("mandarin") == "普通話"

    def test_mixed_defaults_to_cantonese(self):
        assert _map_language("mixed") == "粵語"

    def test_unknown_defaults_to_cantonese(self):
        assert _map_language("unknown") == "粵語"


# ── process() 集成测试（mock LLM） ──────────────────────────────


class TestProcess:
    """测试完整的 process() 流水线，mock 掉 LLM 调用。"""

    @patch("app.orchestrator.orchestrator.llm")
    def test_first_message_welcome_to_k6(self, mock_llm, tmp_sessions_dir):
        mock_llm.complete_json.return_value = _BASE_ANALYSIS
        mock_llm.complete.return_value = "嗨！我係你嘅情緒支援小幫手"

        result = process("+85299990001", "你好")
        assert result.response_text == "嗨！我係你嘅情緒支援小幫手"
        assert result.state == "k6_assessment"
        assert result.alert_level == "green"
        assert result.k6_total == 0
        assert result.k6_complete is False

    @patch("app.orchestrator.orchestrator.llm")
    def test_crisis_detected_forces_crisis_state(self, mock_llm, tmp_sessions_dir):
        crisis_analysis = {
            **_BASE_ANALYSIS,
            "s_emotion": 90,
            "s_keyword": 95,
            "s_behavior": 70,
            "crisis_detected": True,
        }
        mock_llm.complete_json.return_value = crisis_analysis
        mock_llm.complete.return_value = "我好擔心你，你而家安全嗎？"

        result = process("+85299990002", "我唔想活了")
        assert result.state == "crisis_intervention"
        assert result.alert_level in ("orange", "red")

    @patch("app.orchestrator.orchestrator.llm")
    def test_k6_scores_accumulate_across_turns(self, mock_llm, tmp_sessions_dir):
        # 第一轮：报告了 tense=2, depressed=1
        first_analysis = {
            **_BASE_ANALYSIS,
            "k6_dim_scores": {
                "tense": 2, "helpless": 0, "restless": 0,
                "depressed": 1, "effortful": 0, "worthless": 0,
            },
        }
        mock_llm.complete_json.return_value = first_analysis
        mock_llm.complete.return_value = "..."
        process("+85299990010", "msg1")

        # 第二轮：本轮只检测到 tense=1（应被 max 平滑保留 2）
        second_analysis = {
            **_BASE_ANALYSIS,
            "k6_dim_scores": {
                "tense": 1, "helpless": 3, "restless": 0,
                "depressed": 0, "effortful": 0, "worthless": 0,
            },
        }
        mock_llm.complete_json.return_value = second_analysis
        process("+85299990010", "msg2")

        # 验证 session 里的累积值
        from app.storage import session_store
        user_hash = session_store._user_hash("+85299990010")
        session = session_store.load(user_hash)
        scores = session["k6_scores"]
        assert scores["tense"] == 2       # max(2, 1) = 2
        assert scores["helpless"] == 3    # max(0, 3) = 3
        assert scores["depressed"] == 1   # max(1, 0) = 1

    @patch("app.orchestrator.orchestrator.llm")
    def test_terminal_session_returns_fixed_message(self, mock_llm, tmp_sessions_dir):
        from app.storage import session_store
        user_hash, session = session_store.get_or_create("+85299990003")
        session["state"] = "closure"
        session["closure_done"] = True
        session_store.save(user_hash, session)

        result = process("+85299990003", "hello again")
        assert "已經結束" in result.response_text
        mock_llm.complete.assert_not_called()

    @patch("app.orchestrator.orchestrator.llm")
    def test_session_persisted_after_process(self, mock_llm, tmp_sessions_dir):
        mock_llm.complete_json.return_value = _BASE_ANALYSIS
        mock_llm.complete.return_value = "test reply"

        process("+85299990004", "test message")

        from app.storage import session_store
        user_hash = session_store._user_hash("+85299990004")
        session = session_store.load(user_hash)
        assert session is not None
        assert len(session["history"]) == 2
        assert session["history"][0]["content"] == "test message"
        assert session["history"][1]["content"] == "test reply"

    @patch("app.orchestrator.orchestrator.llm")
    def test_llm_failure_uses_safe_defaults(self, mock_llm, tmp_sessions_dir):
        mock_llm.complete_json.side_effect = Exception("API Error")
        mock_llm.complete.return_value = "我聽到你啦"

        result = process("+85299990005", "test")
        assert result.response_text == "我聽到你啦"
        assert result.alert_level == "green"

    @patch("app.orchestrator.orchestrator.llm")
    def test_start_keyword_resets_session(self, mock_llm, tmp_sessions_dir):
        from app.storage import session_store
        user_hash, session = session_store.get_or_create("+85299990006")
        session["state"] = "pm_decision"
        session["k6_complete"] = True
        session["k6_total"] = 12
        session_store.save(user_hash, session)

        mock_llm.complete_json.return_value = _BASE_ANALYSIS
        mock_llm.complete.return_value = "歡迎"

        result = process("+85299990006", "開始")
        # 重置后 welcome → k6_assessment
        assert result.state == "k6_assessment"
        assert result.k6_complete is False
        assert result.k6_total == 0
