"""Prompt 组装器测试。"""
from app.intelligence.prompt_builder import (
    build_k6_prompt,
    build_response_prompt,
    build_triage_prompt,
)
from app.orchestrator.fsm import SessionState


_DEFAULT_ANALYSIS = {
    "s_emotion": 0,
    "s_keyword": 0,
    "s_behavior": 0,
    "emotion_labels": [],
    "crisis_detected": False,
    "k6_dim_scores": {dim: 0 for dim in
                      ["tense", "helpless", "restless", "depressed", "effortful", "worthless"]},
    "wants_to_continue": None,
}


class TestBuildTriagePrompt:
    def test_returns_system_and_messages(self):
        system, messages = build_triage_prompt(
            user_message="我最近好唔開心",
            history=[],
        )
        assert isinstance(system, str)
        assert len(system) > 0
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "我最近好唔開心"

    def test_system_contains_triage_signals(self):
        system, _ = build_triage_prompt("test", history=[])
        assert "s_emotion" in system
        assert "s_behavior" in system
        assert "wants_to_continue" in system

    def test_triage_does_not_contain_k6_or_crisis(self):
        # 分诊 prompt 不应包含 K6 评分（在 k6_scoring）或危机检测（在 safety_monitor）
        system, _ = build_triage_prompt("test", history=[])
        assert "k6_dim_scores" not in system
        assert "s_keyword" not in system
        assert "crisis_detected" not in system

    def test_history_included(self):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨！"},
        ]
        system, _ = build_triage_prompt("第二句", history=history)
        assert "你好" in system


class TestBuildK6Prompt:
    def test_returns_system_and_messages(self):
        system, messages = build_k6_prompt("我覺得好攰", history=[])
        assert isinstance(system, str)
        assert len(messages) == 1
        assert messages[0]["content"] == "我覺得好攰"

    def test_system_contains_k6_dimensions(self):
        system, _ = build_k6_prompt("test", history=[])
        assert "tense" in system
        assert "helpless" in system
        assert "worthless" in system
        assert "K6" in system

    def test_history_included(self):
        history = [{"role": "user", "content": "之前嘅話"}]
        system, _ = build_k6_prompt("新話", history=history)
        assert "之前嘅話" in system


class TestMemorySummaryInjection:
    def test_summary_injected_into_triage(self):
        system, _ = build_triage_prompt("test", history=[], memory_summary="用戶有學業壓力")
        assert "對話摘要" in system
        assert "用戶有學業壓力" in system

    def test_no_summary_when_empty(self):
        system, _ = build_triage_prompt("test", history=[], memory_summary="")
        assert "對話摘要" not in system

    def test_summary_injected_into_response(self):
        system, _ = build_response_prompt(
            state=SessionState.K6_ASSESSMENT,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="green",
            memory_summary="用戶提過孤獨感",
        )
        assert "用戶提過孤獨感" in system


class TestBuildResponsePrompt:
    def test_returns_system_and_messages(self):
        system, messages = build_response_prompt(
            state=SessionState.WELCOME,
            history=[],
            user_message="hi",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="green",
        )
        assert isinstance(system, str)
        assert len(messages) == 1
        assert messages[0]["content"] == "hi"

    def test_system_contains_three_layers(self):
        system, _ = build_response_prompt(
            state=SessionState.K6_ASSESSMENT,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="green",
        )
        assert "心理健康" in system  # system 层
        assert "K6" in system  # task 层
        assert "安全指令" in system  # safety 层

    def test_each_state_has_template(self):
        """每个 FSM 状态都应能正常渲染，不报错。"""
        for state in SessionState:
            system, _ = build_response_prompt(
                state=state,
                history=[],
                user_message="test",
                analysis=_DEFAULT_ANALYSIS,
                stabilize=False,
                alert_level="green",
            )
            assert len(system) > 0

    def test_stabilize_flag_included(self):
        system, _ = build_response_prompt(
            state=SessionState.K6_ASSESSMENT,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=True,
            alert_level="green",
        )
        assert "情緒穩定介入" in system

    def test_orange_alert_includes_hotline(self):
        system, _ = build_response_prompt(
            state=SessionState.K6_ASSESSMENT,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="orange",
            hotline_already_given=False,
        )
        # 至少有一个青少年热线（关心一线 2777 8899 或撒玛利亚 2389 2222）
        assert "2777 8899" in system or "2389 2222" in system

    def test_orange_alert_hotline_already_given(self):
        system, _ = build_response_prompt(
            state=SessionState.K6_ASSESSMENT,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="orange",
            hotline_already_given=True,
        )
        # 已经给过热线时不应再贴号码
        assert "唔好再重複" in system or "簡短提示" in system

    def test_red_alert_crisis_mode(self):
        system, _ = build_response_prompt(
            state=SessionState.CRISIS_INTERVENTION,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="red",
        )
        assert "危機" in system

    def test_history_carried_into_messages(self):
        history = [
            {"role": "user", "content": "之前的消息"},
            {"role": "assistant", "content": "之前的回复"},
        ]
        _, messages = build_response_prompt(
            state=SessionState.WELCOME,
            history=history,
            user_message="新消息",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="green",
        )
        assert len(messages) == 3
        assert messages[-1]["content"] == "新消息"

    def test_language_parameter(self):
        system, _ = build_response_prompt(
            state=SessionState.WELCOME,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="green",
            language="English",
        )
        assert "English" in system

    def test_k6_progress_passed_to_template(self):
        progress = {
            "nonzero_dims": ["緊張", "抑鬱"],
            "missing_dims": ["無助", "焦躁", "費力", "無價值"],
            "turns": 2,
        }
        system, _ = build_response_prompt(
            state=SessionState.K6_ASSESSMENT,
            history=[],
            user_message="test",
            analysis=_DEFAULT_ANALYSIS,
            stabilize=False,
            alert_level="green",
            k6_progress=progress,
        )
        # k6 模板应展开 progress 信息
        assert "緊張" in system
        assert "無助" in system
