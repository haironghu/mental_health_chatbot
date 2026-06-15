"""风险评分 R(t) 公式和预警级别测试。"""
import pytest

from app.safety.risk_monitor import (
    AlertLevel,
    RiskResult,
    _determine_level,
    update,
    should_force_crisis,
    should_stabilize,
)


class TestDetermineLevel:
    def test_green(self):
        assert _determine_level(0) == AlertLevel.GREEN
        assert _determine_level(29.9) == AlertLevel.GREEN

    def test_yellow(self):
        assert _determine_level(30) == AlertLevel.YELLOW
        assert _determine_level(59.9) == AlertLevel.YELLOW

    def test_orange(self):
        assert _determine_level(60) == AlertLevel.ORANGE
        assert _determine_level(79.9) == AlertLevel.ORANGE

    def test_red(self):
        assert _determine_level(80) == AlertLevel.RED
        assert _determine_level(100) == AlertLevel.RED


class TestUpdate:
    def test_first_turn_zero_history(self, default_session):
        """首轮对话，所有信号为 0 → R=0，绿色。"""
        analysis = {"s_emotion": 0, "s_keyword": 0, "s_behavior": 0}
        result = update(default_session, analysis)
        assert result.score == 0.0
        assert result.level == AlertLevel.GREEN
        assert result.previous_score == 0.0
        assert default_session["risk_score"] == 0.0
        assert default_session["alert_level"] == "green"

    def test_formula_correctness(self, default_session):
        """验证 R(t) = α×R(t-1) + β×s_e + γ×s_k + δ×s_b。"""
        # 默认系数：α=0.6, β=0.25, γ=0.3, δ=0.15
        default_session["risk_score"] = 20.0
        analysis = {"s_emotion": 50, "s_keyword": 30, "s_behavior": 20}
        result = update(default_session, analysis)
        # R = 0.6×20 + 0.25×50 + 0.3×30 + 0.15×20
        # R = 12 + 12.5 + 9 + 3 = 36.5
        assert abs(result.score - 36.5) < 0.01
        assert result.level == AlertLevel.YELLOW

    def test_score_capped_at_100(self, default_session):
        """R 值不应超过 100。"""
        default_session["risk_score"] = 90.0
        analysis = {"s_emotion": 100, "s_keyword": 100, "s_behavior": 100}
        result = update(default_session, analysis)
        assert result.score <= 100.0

    def test_score_floor_at_zero(self, default_session):
        """R 值不应低于 0。"""
        analysis = {"s_emotion": 0, "s_keyword": 0, "s_behavior": 0}
        result = update(default_session, analysis)
        assert result.score >= 0.0

    def test_cumulative_risk_buildup(self, default_session):
        """多轮累积：连续负面输入使 R 值逐轮增长。"""
        analysis = {"s_emotion": 60, "s_keyword": 40, "s_behavior": 30}
        scores = []
        for _ in range(5):
            result = update(default_session, analysis)
            scores.append(result.score)
        # 每轮应该递增（因为衰减系数 < 1 且有持续负面输入）
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]

    def test_risk_decay_on_positive_input(self, default_session):
        """负面之后转正面，R 值应逐步衰减。"""
        # 先推高分数
        default_session["risk_score"] = 60.0
        analysis_positive = {"s_emotion": 5, "s_keyword": 0, "s_behavior": 0}
        result = update(default_session, analysis_positive)
        # R = 0.8×60 + 0.4×5 + 0 + 0 = 48 + 2 = 50
        assert result.score < 60.0

    def test_consecutive_negative_turns_increment(self, default_session):
        """高情绪信号应增加连续负面轮次计数。"""
        analysis = {"s_emotion": 50, "s_keyword": 0, "s_behavior": 0}
        update(default_session, analysis)
        assert default_session["consecutive_negative_turns"] == 1
        update(default_session, analysis)
        assert default_session["consecutive_negative_turns"] == 2

    def test_consecutive_negative_turns_reset(self, default_session):
        """低信号应重置连续负面轮次。"""
        default_session["consecutive_negative_turns"] = 3
        analysis = {"s_emotion": 10, "s_keyword": 5, "s_behavior": 0}
        update(default_session, analysis)
        assert default_session["consecutive_negative_turns"] == 0

    def test_missing_fields_default_to_zero(self, default_session):
        """analysis 缺少字段时默认 0。"""
        result = update(default_session, {})
        assert result.score == 0.0


class TestShouldForceCrisis:
    def test_red_triggers_crisis(self):
        risk = RiskResult(score=85, level=AlertLevel.RED,
                          s_emotion=90, s_keyword=80, s_behavior=70,
                          previous_score=70)
        assert should_force_crisis(risk) is True

    def test_non_red_does_not_trigger(self):
        risk = RiskResult(score=50, level=AlertLevel.YELLOW,
                          s_emotion=50, s_keyword=30, s_behavior=20,
                          previous_score=30)
        assert should_force_crisis(risk) is False


class TestShouldStabilize:
    def test_three_consecutive_negatives(self):
        session = {"consecutive_negative_turns": 3}
        risk = RiskResult(score=40, level=AlertLevel.YELLOW,
                          s_emotion=50, s_keyword=20, s_behavior=10,
                          previous_score=30)
        assert should_stabilize(session, risk) is True

    def test_orange_level_triggers(self):
        session = {"consecutive_negative_turns": 1}
        risk = RiskResult(score=65, level=AlertLevel.ORANGE,
                          s_emotion=60, s_keyword=50, s_behavior=40,
                          previous_score=50)
        assert should_stabilize(session, risk) is True

    def test_green_no_consecutive_does_not_trigger(self):
        session = {"consecutive_negative_turns": 0}
        risk = RiskResult(score=10, level=AlertLevel.GREEN,
                          s_emotion=10, s_keyword=0, s_behavior=5,
                          previous_score=5)
        assert should_stabilize(session, risk) is False
