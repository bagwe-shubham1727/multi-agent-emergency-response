"""Shared pytest fixtures.

Every test is hermetic: ``OPENAI_API_KEY`` is removed for each test (a test
that needs a fake key sets it with ``monkeypatch.setenv``), and files go to
``tmp_path``. Tiny hand-made worlds come from ``tests.factories``.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from emergency_sim.domain.scenario import Scenario
from emergency_sim.scenarios import DEFAULT_DATA_ROOT, load_scenario
from tests import factories

SCENARIO_NAMES = ("normal", "surge", "constrained")


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Removes ``OPENAI_API_KEY`` so no test can reach the network by accident."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


@pytest.fixture(scope="session")
def data_root() -> Path:
    """The committed ``data/`` folder (read-only in tests)."""
    return DEFAULT_DATA_ROOT


@pytest.fixture(scope="session")
def real_scenarios(data_root: Path) -> Mapping[str, Scenario]:
    """The three committed scenarios (seed 7380), keyed by name."""
    return MappingProxyType({name: load_scenario(data_root / name) for name in SCENARIO_NAMES})


@pytest.fixture
def tiny_scenario() -> Scenario:
    """A one-hospital, two-ambulance, one-incident world (see ``factories.tiny_scenario``)."""
    return factories.tiny_scenario()
