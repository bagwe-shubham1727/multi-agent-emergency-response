"""Tests of the narrow engine port given to controllers (``engine/port.py``)."""

from emergency_sim.domain.actions import DispatchUnit
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.ports import EnginePort
from emergency_sim.engine.engine import Engine
from emergency_sim.engine.port import EnginePortAdapter
from emergency_sim.eventlog import MemoryEventSink
from tests import factories


def _port() -> tuple[Engine, EnginePortAdapter]:
    engine = Engine(factories.tiny_scenario(), SimParams(), MemoryEventSink())
    return engine, EnginePortAdapter(engine)


def test_adapter_is_an_engine_port_exposing_nothing_else() -> None:
    _, port = _port()
    assert isinstance(port, EnginePort)
    public = {name for name in dir(port) if not name.startswith("_")}
    assert public == {"now", "view", "check", "apply"}
    assert not hasattr(port, "__dict__")


def test_adapter_delegates_to_the_engine() -> None:
    engine, port = _port()
    engine.advance()
    assert port.now == engine.now
    assert port.view() == engine.view()
    action = DispatchUnit(unit_id="AMB-01", incident_id="INC-01")
    assert port.check(action) == engine.check(action)
    assert port.apply(action, "test").ok
    assert engine.view() == port.view()
