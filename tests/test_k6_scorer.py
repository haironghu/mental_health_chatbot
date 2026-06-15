"""K6 评分器测试。"""
import pytest

from app.safety.k6_scorer import (
    K6_DIMENSIONS,
    evaluate,
    mark_complete,
    select_next_pm_strategy,
    severity_of,
    update_scores,
)


class TestSeverityOf:
    @pytest.mark.parametrize("total,expected", [
        (0, "mild"),
        (4, "mild"),
        (5, "moderate"),
        (12, "moderate"),
        (13, "severe"),
        (24, "severe"),
    ])
    def test_boundaries(self, total, expected):
        assert severity_of(total) == expected


class TestUpdateScores:
    def test_first_update_sets_scores(self):
        session = {}
        result = update_scores(session, {
            "tense": 2, "helpless": 1, "restless": 0,
            "depressed": 3, "effortful": 0, "worthless": 0,
        })
        assert result["tense"] == 2
        assert result["depressed"] == 3
        assert session["k6_scores"]["tense"] == 2

    def test_max_smoothing_keeps_higher(self):
        session = {"k6_scores": {dim: 0 for dim in K6_DIMENSIONS}}
        session["k6_scores"]["tense"] = 3
        # 新一轮只看到 tense=1，应保留 3
        result = update_scores(session, {
            "tense": 1, "helpless": 0, "restless": 0,
            "depressed": 0, "effortful": 0, "worthless": 0,
        })
        assert result["tense"] == 3

    def test_clamps_to_0_4(self):
        session = {}
        result = update_scores(session, {"tense": 99, "helpless": -5})
        assert result["tense"] == 4
        assert result["helpless"] == 0

    def test_missing_dim_in_input_keeps_old(self):
        session = {"k6_scores": {"tense": 2, "helpless": 1}}
        result = update_scores(session, {"depressed": 4})
        assert result["tense"] == 2
        assert result["helpless"] == 1
        assert result["depressed"] == 4


class TestEvaluate:
    def test_not_complete_when_few_nonzero_dims(self):
        session = {"k6_scores": {
            "tense": 3, "helpless": 0, "restless": 0,
            "depressed": 2, "effortful": 0, "worthless": 0,
        }}
        # 只有 2 个非零，<4
        result = evaluate(session, state_turn_count=10)
        assert not result.complete

    def test_not_complete_when_few_turns(self):
        session = {"k6_scores": {
            "tense": 1, "helpless": 1, "restless": 1,
            "depressed": 1, "effortful": 1, "worthless": 1,
        }}
        # 6 个非零但只聊 3 轮
        result = evaluate(session, state_turn_count=3)
        assert not result.complete

    def test_complete_when_enough_dims_and_turns(self):
        session = {"k6_scores": {
            "tense": 2, "helpless": 0, "restless": 1,
            "depressed": 3, "effortful": 2, "worthless": 0,
        }}
        result = evaluate(session, state_turn_count=5)
        assert result.complete
        assert result.total == 8
        assert result.severity == "moderate"

    def test_total_and_severity_computed(self):
        session = {"k6_scores": {
            "tense": 4, "helpless": 4, "restless": 4,
            "depressed": 4, "effortful": 4, "worthless": 4,
        }}
        result = evaluate(session, state_turn_count=5)
        assert result.total == 24
        assert result.severity == "severe"


class TestMarkComplete:
    def test_writes_completion_fields(self):
        session = {}
        from app.safety.k6_scorer import K6Result
        result = K6Result(
            scores={"tense": 2, "helpless": 3, "restless": 1,
                    "depressed": 2, "effortful": 0, "worthless": 0},
            total=8,
            severity="moderate",
            complete=True,
        )
        mark_complete(session, result)
        assert session["k6_complete"] is True
        assert session["k6_total"] == 8
        assert session["k6_severity"] == "moderate"
        assert session["k6_completed_at"] is not None


class TestSelectNextPMStrategy:
    def test_high_tense_picks_stress_mgmt(self):
        scores = {"tense": 4, "helpless": 1, "restless": 0,
                  "depressed": 1, "effortful": 0, "worthless": 0}
        assert select_next_pm_strategy(scores, used=[]) == "pm_stress_mgmt"

    def test_high_restless_also_picks_stress_mgmt(self):
        scores = {"tense": 0, "helpless": 1, "restless": 4,
                  "depressed": 1, "effortful": 0, "worthless": 0}
        assert select_next_pm_strategy(scores, used=[]) == "pm_stress_mgmt"

    def test_high_depressed_picks_behavioral_activation(self):
        scores = {"tense": 1, "helpless": 0, "restless": 0,
                  "depressed": 4, "effortful": 2, "worthless": 0}
        assert select_next_pm_strategy(scores, used=[]) == "pm_behavioral_activation"

    def test_high_worthless_picks_social_support(self):
        scores = {"tense": 1, "helpless": 1, "restless": 0,
                  "depressed": 1, "effortful": 1, "worthless": 4}
        assert select_next_pm_strategy(scores, used=[]) == "pm_social_support"

    def test_skips_already_used_strategy(self):
        scores = {"tense": 4, "helpless": 3, "restless": 0,
                  "depressed": 2, "effortful": 0, "worthless": 0}
        # stress_mgmt 用过了，应该选下一个对应高分维度（helpless → behavioral_activation）
        result = select_next_pm_strategy(scores, used=["pm_stress_mgmt"])
        assert result == "pm_behavioral_activation"

    def test_falls_back_to_problem_solving(self):
        # 所有专属维度对应策略都用过，但还有困扰 → 兜底解决问题
        scores = {"tense": 2, "helpless": 0, "restless": 0,
                  "depressed": 0, "effortful": 0, "worthless": 0}
        result = select_next_pm_strategy(
            scores,
            used=["pm_stress_mgmt", "pm_behavioral_activation", "pm_social_support"],
        )
        assert result == "pm_problem_solving"

    def test_returns_none_when_all_used(self):
        scores = {"tense": 2, "helpless": 2, "restless": 0,
                  "depressed": 0, "effortful": 0, "worthless": 0}
        result = select_next_pm_strategy(
            scores,
            used=["pm_stress_mgmt", "pm_behavioral_activation",
                  "pm_social_support", "pm_problem_solving"],
        )
        assert result is None

    def test_returns_none_when_all_zero_scores(self):
        scores = {dim: 0 for dim in K6_DIMENSIONS}
        result = select_next_pm_strategy(scores, used=[])
        assert result is None


class TestSelectStrategyWithReason:
    def test_returns_strategy_and_reason(self):
        from app.safety.k6_scorer import select_next_pm_strategy_with_reason
        scores = {"tense": 4, "helpless": 1, "restless": 0,
                  "depressed": 1, "effortful": 0, "worthless": 0}
        strategy, reason = select_next_pm_strategy_with_reason(scores, used=[])
        assert strategy == "pm_stress_mgmt"
        assert "緊張" in reason  # 依据提到最高维度

    def test_none_has_reason(self):
        from app.safety.k6_scorer import select_next_pm_strategy_with_reason
        scores = {dim: 0 for dim in K6_DIMENSIONS}
        strategy, reason = select_next_pm_strategy_with_reason(scores, used=[])
        assert strategy is None
        assert reason
