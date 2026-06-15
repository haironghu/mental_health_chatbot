"""FSM 状态机测试。"""
import pytest

from app.orchestrator.fsm import (
    PM_STRATEGY_STATES,
    SessionFSM,
    SessionState,
)


class TestSessionFSMInit:
    def test_default_state(self, default_session):
        fsm = SessionFSM(default_session)
        assert fsm.state == SessionState.WELCOME
        assert fsm.turn_count == 0
        assert fsm.state_turn_count == 0
        assert fsm.pm_strategies_used == []

    def test_restore_from_existing_session(self):
        session = {
            "state": "pm_stress_mgmt",
            "turn_count": 6,
            "state_turn_count": 2,
            "pm_strategies_used": [],
        }
        fsm = SessionFSM(session)
        assert fsm.state == SessionState.PM_STRESS_MGMT
        assert fsm.turn_count == 6
        assert fsm.state_turn_count == 2


class TestTransition:
    def test_welcome_to_k6_assessment(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        assert fsm.state == SessionState.K6_ASSESSMENT
        assert fsm.turn_count == 1
        assert fsm.state_turn_count == 0  # 进入新状态重置

    def test_invalid_transition_raises(self, default_session):
        fsm = SessionFSM(default_session)
        # WELCOME 不能直接进 CLOSURE
        with pytest.raises(ValueError, match="不允许的状态转换"):
            fsm.transition(SessionState.CLOSURE)

    def test_k6_to_pm_strategy(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.PM_STRESS_MGMT)
        assert fsm.state == SessionState.PM_STRESS_MGMT

    def test_pm_strategy_records_when_leaving(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.PM_STRESS_MGMT)
        # 离开 PM 策略时应记录
        fsm.transition(SessionState.PM_DECISION)
        assert "pm_stress_mgmt" in fsm.pm_strategies_used

    def test_pm_decision_to_next_strategy(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.PM_STRESS_MGMT)
        fsm.transition(SessionState.PM_DECISION)
        fsm.transition(SessionState.PM_BEHAVIORAL_ACTIVATION)
        assert fsm.state == SessionState.PM_BEHAVIORAL_ACTIVATION

    def test_pm_decision_to_closure(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.PM_STRESS_MGMT)
        fsm.transition(SessionState.PM_DECISION)
        fsm.transition(SessionState.CLOSURE)
        assert fsm.state == SessionState.CLOSURE

    def test_closure_is_terminal(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.CLOSURE)
        # 未标记 closure_done 时不算终态
        assert not fsm.is_terminal()
        fsm.closure_done = True
        assert fsm.is_terminal()

    def test_state_turn_count_resets_on_transition(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.stay()
        fsm.stay()
        assert fsm.state_turn_count == 2
        fsm.transition(SessionState.PM_STRESS_MGMT)
        assert fsm.state_turn_count == 0


class TestStay:
    def test_stay_increments_counts(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        before_turn = fsm.turn_count
        before_state_turn = fsm.state_turn_count
        fsm.stay()
        assert fsm.turn_count == before_turn + 1
        assert fsm.state_turn_count == before_state_turn + 1


class TestForceCrisis:
    def test_force_crisis_from_any_state(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.PM_STRESS_MGMT)
        fsm.force_crisis()
        assert fsm.state == SessionState.CRISIS_INTERVENTION
        # 离开 PM 策略时记录
        assert "pm_stress_mgmt" in fsm.pm_strategies_used

    def test_force_crisis_resets_state_turn(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.stay()
        fsm.force_crisis()
        assert fsm.state_turn_count == 0

    def test_crisis_can_transition_to_closure(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.force_crisis()
        assert fsm.can_transition(SessionState.CLOSURE)
        fsm.transition(SessionState.CLOSURE)
        assert not fsm.is_terminal()
        fsm.closure_done = True
        assert fsm.is_terminal()


class TestIsPMStrategy:
    @pytest.mark.parametrize("state", PM_STRATEGY_STATES)
    def test_pm_strategies_recognized(self, state, default_session):
        session = {**default_session, "state": state.value}
        fsm = SessionFSM(session)
        assert fsm.is_pm_strategy()

    def test_non_pm_states_not_recognized(self, default_session):
        fsm = SessionFSM(default_session)
        assert not fsm.is_pm_strategy()


class TestApplyTo:
    def test_apply_writes_state_back(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.apply_to(default_session)
        assert default_session["state"] == "k6_assessment"
        assert default_session["turn_count"] == 1

    def test_apply_writes_pm_strategies(self, default_session):
        fsm = SessionFSM(default_session)
        fsm.transition(SessionState.K6_ASSESSMENT)
        fsm.transition(SessionState.PM_STRESS_MGMT)
        fsm.transition(SessionState.PM_DECISION)
        fsm.apply_to(default_session)
        assert "pm_stress_mgmt" in default_session["pm_strategies_used"]
