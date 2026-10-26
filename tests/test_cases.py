"""Tests of the escalation-case utilities in ``control/cases.py``."""

import pytest

from emergency_sim.control.cases import contention_case, incident_utility
from emergency_sim.domain.agent_io import ContentionFlag
from emergency_sim.domain.enums import ResponderCapability
from emergency_sim.policies.params import PolicyParams
from tests.control_fakes import incident_view, public_view, unit_view
from tests.test_policies import make_need

PARAMS = PolicyParams()


def test_travel_and_access_wait_overlap() -> None:
    # The ambulance drives while the responders work: loading starts after
    # max(eta, access), not eta + access (the old, additive model).
    need = make_need("INC-01", 5.0, 5.0, 95, suspected_red=True, deadline_in_min=40.0, access_wait_min=20.0)
    _, slack_near, _ = incident_utility(need, 3.0, PARAMS)
    _, slack_far, _ = incident_utility(need, 15.0, PARAMS)
    assert slack_near == slack_far == pytest.approx(40.0 - 20.0 - PARAMS.scene_overhead_min)
    _, slack_late, _ = incident_utility(need, 25.0, PARAMS)
    assert slack_late == pytest.approx(40.0 - 25.0 - PARAMS.scene_overhead_min)


def test_unknown_access_uses_no_unit_wait() -> None:
    need = make_need("INC-01", 5.0, 5.0, 95, deadline_in_min=40.0, access_wait_min=None,
                     responder_caps_missing=[ResponderCapability.SCENE_CONTROL])
    _, slack, _ = incident_utility(need, 3.0, PARAMS)
    assert slack == pytest.approx(40.0 - PARAMS.no_unit_wait_min - PARAMS.scene_overhead_min)


def test_contested_als_goes_to_the_waiting_red_not_the_blocked_suspected_scene() -> None:
    # Regression (surge seed 2, t=56.7, CASE-0001): the additive access model
    # scored "ALS to a suspected-RED scene with no responders yet" above "ALS
    # to a triaged, accessible scene with REDs waiting", so waiting REDs starved
    # while the ALS sat staged at the blocked scene.
    waiting = make_need("INC-04", 4.0, 5.0, 100, suspected_red=True, triaged=True, red_waiting=2,
                        als_needed=2, est_patients_remaining=8, deadline_in_min=22.3)
    blocked = make_need("INC-08", 6.0, 5.0, 95, suspected_red=True, als_needed=1, est_patients_remaining=6,
                        deadline_in_min=30.3, access_wait_min=None,
                        responder_caps_missing=[ResponderCapability.HEAVY_RESCUE, ResponderCapability.SCENE_CONTROL])
    view = public_view(
        incidents=[incident_view("INC-04", x=4.0), incident_view("INC-08", x=6.0)],
        units=[unit_view("AMB-03", x=5.0)],
    )
    flag = ContentionFlag(unit_id="AMB-03", incident_ids=["INC-04", "INC-08"], rationale="r")
    plan = contention_case("CASE-0001", view, flag, {"INC-04": waiting, "INC-08": blocked}, set(), PARAMS)
    utilities = {o.option_id: o.utility for o in plan.case.options}
    assert utilities["ASSIGN:AMB-03->INC-04"] > utilities["ASSIGN:AMB-03->INC-08"]
