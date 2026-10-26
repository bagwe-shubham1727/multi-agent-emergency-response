"""The four agent output types and all tools build OpenAI strict JSON schemas."""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from agents import AgentOutputSchema

from emergency_sim.domain.agent_io import AllocationOutput, AssessmentOutput, DispatchOutput, HospitalOutput
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.mas import tools
from emergency_sim.mas.specs import agent_specs
from tests import factories

OUTPUT_TYPES = (AssessmentOutput, DispatchOutput, HospitalOutput, AllocationOutput)
FORBIDDEN_KEYWORDS = frozenset({"default", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"})


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


@pytest.mark.parametrize("output_type", OUTPUT_TYPES, ids=lambda t: t.__name__)
def test_strict_output_schema_builds(output_type: type[AgentIOModel]) -> None:
    schema = AgentOutputSchema(output_type, strict_json_schema=True)
    assert schema.is_strict_json_schema()
    assert not schema.is_plain_text()
    json_schema = schema.json_schema()
    assert "anyOf" not in json_schema and "oneOf" not in json_schema  # root is not a union
    for node in _walk(json_schema):
        assert not FORBIDDEN_KEYWORDS & node.keys(), node
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node.get("properties", {}))


@pytest.mark.parametrize("output_type", OUTPUT_TYPES, ids=lambda t: t.__name__)
def test_output_schema_validates_its_own_json(output_type: type[AgentIOModel]) -> None:
    empty = {name: [] for name in output_type.model_fields}
    parsed = AgentOutputSchema(output_type).validate_json(json.dumps(empty))
    assert isinstance(parsed, output_type)


def test_all_tools_are_strict() -> None:
    for tool in tools.make_tools(*tools.TOOL_FUNCTIONS):
        assert tool.strict_json_schema, tool.name
        assert tool.params_json_schema["additionalProperties"] is False


def test_spec_output_types_are_the_four_outputs() -> None:
    specs = agent_specs([factories.hospital("H1"), factories.hospital("H2")])
    assert {s.output_type for s in specs} == set(OUTPUT_TYPES)
