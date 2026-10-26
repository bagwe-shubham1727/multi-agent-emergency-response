"""Contract tests for ``emergency_sim.domain`` and ``emergency_sim.eventlog``."""

import dataclasses
import subprocess
import sys
import types
import typing
from pathlib import Path

import pydantic
import pytest

from emergency_sim import geo
from emergency_sim.domain import agent_io, views
from emergency_sim.domain.actions import (
    Action,
    ActionCheck,
    AssignTransport,
    DispatchUnit,
    DivertUnit,
    LendStaff,
)
from emergency_sim.domain.agent_io import AgentDecision, AllocationOutput, CaseDecision
from emergency_sim.domain.base import AgentIOModel, FrozenModel
from emergency_sim.domain.enums import TRIAGE_RANK, SimEventType, Triage
from emergency_sim.domain.log_records import LogRecord, RunStartRecord
from emergency_sim.domain.messages import (
    ALLOCATION_AGENT,
    ASSESSMENT_AGENT,
    BROADCAST,
    DISPATCH_AGENT,
    CommitAction,
    DeclineAction,
    Message,
    hospital_agent_name,
)
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.ports import EventSink
from emergency_sim.domain.scenario import Scenario
from emergency_sim.eventlog import MemoryEventSink, read_log, write_jsonl
from tests import factories

ACTION_ADAPTER = pydantic.TypeAdapter(Action)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HIDDEN_KEYS = frozenset(
    {"golden_window_min", "deteriorates_to_red_at_min", "true_victims", "worst_triage", "time_sensitivity_min"}
)
STRICT_OUTPUT_TYPES = (
    agent_io.AssessmentOutput,
    agent_io.DispatchOutput,
    agent_io.HospitalOutput,
    agent_io.AllocationOutput,
)


# --- Log records and event log ---------------------------------------------


def test_sample_log_covers_every_record_kind() -> None:
    kinds = [r.kind for r in factories.sample_log_records()]
    union_members = typing.get_args(typing.get_args(LogRecord)[0])
    expected = [m.model_fields["kind"].default for m in union_members]
    assert kinds == expected


def test_every_log_record_round_trips_through_jsonl(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    for record in factories.sample_log_records():
        sink.emit(record)
    path = tmp_path / "nested" / "run" / "events.jsonl"

    write_jsonl(sink.records, path)
    restored = read_log(path)

    assert restored == list(sink.records)
    assert [type(r) for r in restored] == [type(r) for r in sink.records]
    assert len(path.read_text().splitlines()) == len(sink.records)


def test_message_with_nested_engine_action_round_trips(tmp_path: Path) -> None:
    record = factories.sample_log_records()[3].model_copy(
        update={"message": factories.commit_message()}
    )
    path = tmp_path / "events.jsonl"
    write_jsonl([record], path)

    [restored] = read_log(path)

    assert restored == record
    assert isinstance(restored.message.proposed_action, CommitAction)
    assert isinstance(restored.message.proposed_action.actions[0], DispatchUnit)


def test_write_jsonl_is_byte_deterministic(tmp_path: Path) -> None:
    records = factories.sample_log_records()
    write_jsonl(records, tmp_path / "a.jsonl")
    write_jsonl(records, tmp_path / "b.jsonl")

    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_read_log_rejects_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "telemetry", "seq": 0, "t_min": 0.0}\n')

    with pytest.raises(pydantic.ValidationError):
        read_log(path)


def test_memory_sink_assigns_sequential_seq_without_mutating_input() -> None:
    sink = MemoryEventSink()
    records = factories.sample_log_records()
    for record in records:
        sink.emit(record.model_copy(update={"seq": 42}))

    assert [r.seq for r in sink.records] == list(range(len(records)))
    assert isinstance(sink.records, tuple)
    assert isinstance(sink, EventSink)


def test_run_start_record_embeds_full_scenario(tiny_scenario: Scenario) -> None:
    start = factories.sample_log_records(tiny_scenario)[0]

    assert isinstance(start, RunStartRecord)
    assert start.scenario_data == tiny_scenario


# --- Messages ---------------------------------------------------------------


@pytest.mark.parametrize("field_name", sorted(Message.model_fields))
def test_message_requires_every_field(field_name: str) -> None:
    data = factories.message().model_dump()
    del data[field_name]

    with pytest.raises(pydantic.ValidationError):
        Message.model_validate(data)


def test_message_is_frozen_and_rejects_extra_fields() -> None:
    msg = factories.message()
    with pytest.raises(pydantic.ValidationError):
        msg.round = 2  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        Message.model_validate({**msg.model_dump(), "priority": 1})


def test_proposed_action_is_discriminated_by_action() -> None:
    data = factories.message().model_dump()
    data["proposed_action"] = {"action": "decline", "reason": "NO_BEDS"}

    assert isinstance(Message.model_validate(data).proposed_action, DeclineAction)


def test_agent_names() -> None:
    assert hospital_agent_name("H2") == "hospital_H2"
    assert (ASSESSMENT_AGENT, DISPATCH_AGENT, ALLOCATION_AGENT, BROADCAST) == (
        "incident_assessment",
        "dispatch",
        "resource_allocation",
        "all",
    )


# --- Actions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        DispatchUnit(unit_id="AMB-01", incident_id="INC-01"),
        DivertUnit(unit_id="AMB-01", incident_id="INC-02"),
        AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True),
        LendStaff(staff_id="DR-FLOAT-1", hospital_id="H1"),
    ],
)
def test_actions_round_trip_through_discriminated_union(action: FrozenModel) -> None:
    assert ACTION_ADAPTER.validate_json(action.model_dump_json()) == action


def test_action_kind_cannot_be_mislabelled() -> None:
    with pytest.raises(pydantic.ValidationError):
        ACTION_ADAPTER.validate_python({"kind": "lend_staff", "unit_id": "AMB-01", "incident_id": "INC-01"})


def test_action_check_defaults() -> None:
    assert ActionCheck(ok=True) == ActionCheck(ok=True, reason=None, detail="")


# --- Enums and params -------------------------------------------------------


def test_sim_event_values_are_lowercase_names() -> None:
    assert all(member.value == member.name.lower() for member in SimEventType)


def test_triage_rank_orders_red_first_and_is_read_only() -> None:
    assert sorted(Triage, key=TRIAGE_RANK.__getitem__) == [
        Triage.RED,
        Triage.YELLOW,
        Triage.GREEN,
        Triage.BLACK,
    ]
    with pytest.raises(TypeError):
        TRIAGE_RANK[Triage.RED] = 5  # type: ignore[index]


def test_sim_params_defaults_and_treatment_durations() -> None:
    params = SimParams()

    assert params.ambulance_turnout_min == geo.TURNOUT_MIN["ambulance"] == 1.0
    assert params.responder_turnout_min == geo.TURNOUT_MIN["responder"] == 2.0
    assert [params.treatment_min(t) for t in (Triage.RED, Triage.YELLOW, Triage.GREEN)] == [60.0, 40.0, 20.0]
    with pytest.raises(ValueError):
        params.treatment_min(Triage.BLACK)
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.handover_min = 1.0  # type: ignore[misc]


# --- Information boundary ---------------------------------------------------


def _field_names(model: type[pydantic.BaseModel], seen: set[type] | None = None) -> set[str]:
    """Collects field names of ``model`` and every model nested in it."""
    seen = seen if seen is not None else set()
    if model in seen:
        return set()
    seen.add(model)
    names = set(model.model_fields)
    for info in model.model_fields.values():
        for nested in _nested_models(info.annotation):
            names |= _field_names(nested, seen)
    return names


def _nested_models(annotation: object) -> list[type[pydantic.BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return [annotation]
    return [m for arg in typing.get_args(annotation) for m in _nested_models(arg)]


def test_public_view_has_no_hidden_truth_fields() -> None:
    assert not _field_names(views.PublicView) & HIDDEN_KEYS
    assert not _field_names(views.DecisionPoint) & HIDDEN_KEYS


@pytest.mark.parametrize("model", [agent_io.AssessmentInput, agent_io.DispatchInput, agent_io.HospitalInput, agent_io.AllocationInput])
def test_agent_inputs_have_no_hidden_truth_fields(model: type[AgentIOModel]) -> None:
    assert not _field_names(model) & HIDDEN_KEYS


# --- LLM-facing models ------------------------------------------------------


def _models_in(model: type[pydantic.BaseModel]) -> set[type[pydantic.BaseModel]]:
    found = {model}
    for info in model.model_fields.values():
        for nested in _nested_models(info.annotation):
            if nested not in found:
                found |= _models_in(nested)
    return found


def _mentions_dict(annotation: object) -> bool:
    origin = typing.get_origin(annotation)
    if annotation is dict or origin is dict:
        return True
    return any(_mentions_dict(arg) for arg in typing.get_args(annotation))


@pytest.mark.parametrize("output_type", STRICT_OUTPUT_TYPES)
def test_agent_outputs_are_strict_schema_friendly(output_type: type[AgentIOModel]) -> None:
    for model in _models_in(output_type):
        assert issubclass(model, AgentIOModel), model
        assert model.model_config.get("extra") == "forbid", model
        for name, info in model.model_fields.items():
            assert info.is_required(), f"{model.__name__}.{name} has a default"
            assert not info.metadata, f"{model.__name__}.{name} has constraints"
            assert not _mentions_dict(info.annotation), f"{model.__name__}.{name} is a dict"
            assert not isinstance(info.annotation, types.UnionType) or type(None) in typing.get_args(info.annotation)


def test_agent_io_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        CaseDecision.model_validate({"case_id": "C", "chosen_option_id": "O", "rationale": "", "score": 1})


def test_agent_decision_is_frozen() -> None:
    decision = AgentDecision(
        output=AllocationOutput(decisions=[]),
        source="model",
        error=None,
        tool_calls=(),
        usage=None,
        latency_ms=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.source = "policy_fallback"  # type: ignore[misc]


def test_domain_import_does_not_load_sdk_or_ui_libraries() -> None:
    code = (
        "import sys\n"
        "import emergency_sim.domain.ports, emergency_sim.domain.params, emergency_sim.eventlog, emergency_sim.scenarios\n"
        "bad = sorted(m for m in ('agents', 'openai', 'streamlit', 'plotly', 'pandas') if m in sys.modules)\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
