"""Pure, deterministic, SDK-free agent policies and the estimates they share.

Modules:
    params: ``PolicyParams``, the a-priori decision constants.
    estimates: ETAs, rankings, capacity checks, deadlines and utility.
    assessment: ``assess(inp, params) -> AssessmentOutput``.
    dispatch: ``dispatch(inp, params) -> DispatchOutput``.
    hospital: ``hospital(inp, params) -> HospitalOutput``.
    allocation: ``allocate(inp, params) -> AllocationOutput``.

Only ``PolicyParams`` is re-exported here; import the policy functions from
their modules (re-exporting ``dispatch``/``hospital`` would shadow the
submodules of the same name).
"""

from emergency_sim.policies.params import PolicyParams

__all__ = ["PolicyParams"]
