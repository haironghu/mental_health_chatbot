"""多 Agent 测试：各 Agent 单测 + Coordinator 调度。"""
from unittest.mock import patch

from app.agents.base import AgentContext
from app.agents.coordinator import Coordinator, _map_language
from app.agents.k6_scorer_agent import K6ScorerAgent
from app.agents.therapist import TherapistAgent
from app.agents.triage import TriageAgent
from app.orchestrator.fsm import SessionFSM, SessionState


def _ctx(session, state="k6_assessment", user_message="test", history=None):
    return AgentContext(
        user_message=user_message,
        history=history or [],
        session=session,
        fsm_state=state,
    )


# ── TriageAgent ──────────────────────────────────────────────────


class TestTriageAgent:
    @patch("app.intelligence.llm.complete_json")
    def test_returns_merged_signals(self, mock_json, default_session):
        mock_json.return_value = {
            "s_emotion": 40, "crisis_detected": False, "language": "cantonese",
        }
        agent = TriageAgent()
        out = agent.analyze(_ctx(default_session))
        assert out["s_emotion"] == 40
        # 缺失字段由安全默认补全
        assert "s_keyword" in out
        assert "wants_to_continue" in out

    @patch("app.intelligence.llm.complete_json")
    def test_failure_returns_safe_default(self, mock_json, default_session):
        mock_json.side_effect = Exception("boom")
        agent = TriageAgent()
        out = agent.analyze(_ctx(default_session))
        assert out["crisis_detected"] is False
        assert out["s_emotion"] == 0.0


# ── K6ScorerAgent ────────────────────────────────────────────────


class TestK6ScorerAgent:
    @patch("app.intelligence.llm.complete_json")
    def test_returns_dim_scores(self, mock_json, default_session):
        mock_json.return_value = {
            "tense": 3, "helpless": 2, "restless": 0,
            "depressed": 2, "effortful": 1, "worthless": 0,
        }
        agent = K6ScorerAgent()
        out = agent.analyze(_ctx(default_session))
        assert out["k6_dim_scores"]["tense"] == 3
        assert out["k6_dim_scores"]["depressed"] == 2

    @patch("app.intelligence.llm.complete_json")
    def test_failure_returns_zeros(self, mock_json, default_session):
        mock_json.side_effect = Exception("boom")
        agent = K6ScorerAgent()
        out = agent.analyze(_ctx(default_session))
        assert all(v == 0 for v in out["k6_dim_scores"].values())


# ── TherapistAgent ───────────────────────────────────────────────


class TestTherapistAgent:
    @patch("app.intelligence.llm.complete")
    def test_generates_response(self, mock_complete, default_session):
        mock_complete.return_value = "我喺度聽緊你"
        agent = TherapistAgent()
        ctx = _ctx(default_session, state="welcome")
        ctx.analysis = {"s_emotion": 0, "emotion_labels": []}
        out = agent.respond(ctx)
        assert out == "我喺度聽緊你"


# ── _map_language ────────────────────────────────────────────────


class TestMapLanguage:
    def test_cantonese(self):
        assert _map_language("cantonese") == "粵語"

    def test_english(self):
        assert _map_language("english") == "English"

    def test_mandarin(self):
        assert _map_language("mandarin") == "普通話"

    def test_unknown_defaults_cantonese(self):
        assert _map_language("xyz") == "粵語"


# ── Coordinator._advance_fsm（确定性逻辑） ───────────────────────

_BASE = {
    "s_emotion": 20, "s_keyword": 0, "s_behavior": 5,
    "language": "cantonese", "emotion_labels": [], "crisis_detected": False,
    "k6_dim_scores": {"tense": 0, "helpless": 0, "restless": 0,
                      "depressed": 0, "effortful": 0, "worthless": 0},
    "wants_to_continue": None,
}


class TestCoordinatorAdvanceFSM:
    def setup_method(self):
        self.coord = Coordinator()

    def test_welcome_to_k6(self, default_session):
        fsm = SessionFSM(default_session)
        self.coord._advance_fsm(fsm, default_session, _BASE)
        assert fsm.state == SessionState.K6_ASSESSMENT

    def test_k6_stays_when_incomplete(self, default_session):
        default_session["state"] = "k6_assessment"
        default_session["state_turn_count"] = 1
        fsm = SessionFSM(default_session)
        self.coord._advance_fsm(fsm, default_session, _BASE)
        assert fsm.state == SessionState.K6_ASSESSMENT

    def test_k6_completes_picks_strategy(self, default_session):
        default_session["state"] = "k6_assessment"
        default_session["state_turn_count"] = 4
        default_session["k6_scores"] = {
            "tense": 3, "helpless": 2, "restless": 0,
            "depressed": 3, "effortful": 2, "worthless": 0,
        }
        fsm = SessionFSM(default_session)
        self.coord._advance_fsm(fsm, default_session, _BASE)
        assert fsm.is_pm_strategy()
        assert default_session["k6_complete"] is True
        assert default_session["k6_total"] == 10

    def test_pm_strategy_to_decision(self, default_session):
        from app.config import settings
        default_session["state"] = "pm_stress_mgmt"
        default_session["state_turn_count"] = settings.min_turns_per_screening - 1
        fsm = SessionFSM(default_session)
        self.coord._advance_fsm(fsm, default_session, _BASE)
        assert fsm.state == SessionState.PM_DECISION

    def test_pm_decision_continue(self, default_session):
        default_session["state"] = "pm_decision"
        default_session["pm_strategies_used"] = ["pm_stress_mgmt"]
        default_session["k6_scores"] = {
            "tense": 1, "helpless": 3, "restless": 0,
            "depressed": 3, "effortful": 2, "worthless": 0,
        }
        fsm = SessionFSM(default_session)
        analysis = {**_BASE, "wants_to_continue": True}
        self.coord._advance_fsm(fsm, default_session, analysis)
        assert fsm.state == SessionState.PM_BEHAVIORAL_ACTIVATION

    def test_pm_decision_end(self, default_session):
        default_session["state"] = "pm_decision"
        default_session["pm_strategies_used"] = ["pm_stress_mgmt"]
        fsm = SessionFSM(default_session)
        analysis = {**_BASE, "wants_to_continue": False}
        self.coord._advance_fsm(fsm, default_session, analysis)
        assert fsm.state == SessionState.CLOSURE


# ── Coordinator.run（集成，mock LLM） ───────────────────────────


class TestCoordinatorRun:
    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_run_returns_response_and_trace(self, mock_json, mock_complete, default_session):
        def fake_json(messages, *, system="", model=None):
            if "K6 凱斯勒" in system:
                return {"tense": 0, "helpless": 0, "restless": 0,
                        "depressed": 0, "effortful": 0, "worthless": 0}
            return dict(_BASE)
        mock_json.side_effect = fake_json
        mock_complete.return_value = "你好呀 😊"

        coord = Coordinator()
        fsm = SessionFSM(default_session)
        result = coord.run(default_session, fsm, "你好", history=[])

        assert result.response_text == "你好呀 😊"
        # trace 应记录 triage 和 therapist 的耗时
        assert "triage_ms" in result.trace
        assert "therapist_ms" in result.trace

    @patch("app.intelligence.llm.complete")
    @patch("app.intelligence.llm.complete_json")
    def test_k6_agent_only_runs_in_k6_state(self, mock_json, mock_complete, default_session):
        """非 K6 状态不应调用 K6 评分 prompt。"""
        seen_systems = []

        def fake_json(messages, *, system="", model=None):
            seen_systems.append(system)
            return dict(_BASE)
        mock_json.side_effect = fake_json
        mock_complete.return_value = "..."

        # 处于 PM 状态（非 K6）
        default_session["state"] = "pm_stress_mgmt"
        coord = Coordinator()
        fsm = SessionFSM(default_session)
        coord.run(default_session, fsm, "test", history=[])

        # 不应有 K6 评分调用
        assert not any("K6 凱斯勒" in s for s in seen_systems)
