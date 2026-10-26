"""Tests of the negotiation configurations (``control/conditions.py``)."""

import pytest

from emergency_sim.control.conditions import NEGOTIATION, NegotiationConfig
from emergency_sim.domain.enums import GREEDY_CONDITIONS, Condition, CounterCondition


def test_every_agent_condition_has_a_config_and_baselines_have_none() -> None:
    assert set(NEGOTIATION) == set(Condition) - GREEDY_CONDITIONS


def test_no_escalation_keeps_float_counters_but_not_reserve_release() -> None:
    config = NEGOTIATION[Condition.FULL_NO_ESCALATION]
    assert config.allowed_counter_conditions() == [
        CounterCondition.FLOAT_DOCTOR, CounterCondition.FLOAT_NURSE,
    ]


def test_full_allows_every_counter_and_b1_none() -> None:
    assert set(NEGOTIATION[Condition.FULL].allowed_counter_conditions()) == set(CounterCondition)
    assert NEGOTIATION[Condition.B1].allowed_counter_conditions() == []
    assert NEGOTIATION[Condition.B1].allowed_responses() == ["ACCEPT", "REJECT"]


def test_diversion_without_escalation_is_rejected() -> None:
    with pytest.raises(ValueError, match="escalation"):
        NegotiationConfig(True, True, allow_escalation=False, allow_diversion=True)

