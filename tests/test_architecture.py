"""Enforces the layer import rules of the implementation plan (section 2.1) with ``ast``.

Each production module is assigned a *layer* (``domain``, ``engine``,
``mas``...). A layer may import the standard library, its own layer, the
internal layers listed in ``RULES`` and, where ``third_party`` is given, only
those third-party packages. Independently of the table, the agents SDK and
``openai`` may only be imported by ``mas``, and ``streamlit`` only by
``dashboard``. Packages that do not exist yet are simply not scanned.

The module also checks two code standards of section 2.3 that are cheap to
verify statically: no ``from __future__ import annotations`` and no ``print``
outside ``cli.py``. The pre-existing phase-1 modules (``geo``, ``datagen``,
``validate``) have no rule and are not scanned.
"""

import ast
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "emergency_sim"
SOURCE_ROOTS = ("emergency_sim", "dashboard")

# Top-level names inside emergency_sim that form a layer.
EMS_LAYERS = frozenset(
    {
        "domain", "engine", "policies", "mas", "control", "evaluation",
        "geo", "datagen", "validate", "scenarios", "eventlog", "cli",
    }
)
SDK_PACKAGES = frozenset({"agents", "openai"})
UI_PACKAGES = frozenset({"streamlit"})
TEST_PACKAGES = frozenset({"tests", "conftest", "pytest"})


@dataclass(frozen=True)
class LayerRule:
    """What one layer may import.

    Attributes:
        layers: Internal layers allowed besides its own; None means any.
        third_party: Allowed third-party top-level packages; None means any
            (still subject to the SDK / UI isolation rules).
    """

    layers: frozenset[str] | None
    third_party: frozenset[str] | None = None


RULES: Mapping[str, LayerRule] = MappingProxyType({
    "domain": LayerRule(frozenset({"geo"}), frozenset({"pydantic"})),
    "scenarios": LayerRule(frozenset({"domain", "datagen"}), frozenset({"pydantic"})),
    "eventlog": LayerRule(frozenset({"domain"}), frozenset({"pydantic"})),
    "engine": LayerRule(frozenset({"domain", "geo"})),
    "policies": LayerRule(frozenset({"domain", "geo"})),
    "mas": LayerRule(frozenset({"domain", "policies"})),
    "control": LayerRule(frozenset({"domain", "policies", "geo"})),
    "evaluation": LayerRule(None),
    "cli": LayerRule(None),
    # geo is allowed for the map's street/avenue ticks (plan, WP5 figures.py).
    "dashboard": LayerRule(frozenset({"domain", "eventlog", "evaluation", "geo"})),
})


@dataclass(frozen=True)
class SourceModule:
    """A production module to check."""

    path: Path
    name: str
    layer: str
    is_package: bool


def _layer_of_module(dotted: str) -> str | None:
    """Returns the layer of an internal module name, or None if not internal."""
    parts = dotted.split(".")
    if parts[0] == "dashboard":
        return "dashboard"
    if parts[0] != PACKAGE:
        return None
    if len(parts) == 1:
        return PACKAGE  # the root package itself (docstring only)
    return parts[1] if parts[1] in EMS_LAYERS else PACKAGE


def _iter_source_modules() -> Iterator[SourceModule]:
    for root_name in SOURCE_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts
            is_package = parts[-1] == "__init__"
            dotted = ".".join(parts[:-1] if is_package else parts)
            layer = _layer_of_module(dotted)
            if layer in RULES:
                yield SourceModule(path, dotted, layer, is_package)


def _resolve_from(node: ast.ImportFrom, module: str, is_package: bool) -> list[str]:
    """Returns the absolute module names an ``ImportFrom`` refers to."""
    if node.level:
        package = module.split(".") if is_package else module.split(".")[:-1]
        package = package[: len(package) - (node.level - 1)]
        base = ".".join(package + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    if base == PACKAGE:
        # ``from emergency_sim import geo`` imports the submodule.
        return [f"{base}.{alias.name}" for alias in node.names]
    return [base]


def imported_modules(source: str, module: str, is_package: bool = False) -> list[tuple[int, str]]:
    """Lists ``(line, absolute module name)`` for every import in ``source``.

    Args:
        source: Python source code.
        module: Dotted name of the module the source belongs to.
        is_package: Whether the source is a package ``__init__``.

    Returns:
        Imports in source order (including ones under ``if TYPE_CHECKING``).
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.extend((node.lineno, name) for name in _resolve_from(node, module, is_package))
    return sorted(found)


def _check_import(layer: str, target: str) -> str | None:
    """Returns a violation message if ``layer`` may not import ``target``."""
    top = target.split(".")[0]
    rule = RULES[layer]
    target_layer = _layer_of_module(target)
    if target_layer is not None:
        if target_layer in (layer, PACKAGE) or rule.layers is None or target_layer in rule.layers:
            return None
        return f"layer {layer!r} may not import layer {target_layer!r}"
    if top in sys.stdlib_module_names or top == "__future__":
        return None
    if top in TEST_PACKAGES:
        return f"production code may not import test package {top!r}"
    if top in SDK_PACKAGES and layer != "mas":
        return f"only 'mas' may import the SDK package {top!r}"
    if top in UI_PACKAGES and layer != "dashboard":
        return f"only 'dashboard' may import {top!r}"
    if rule.third_party is not None and top not in rule.third_party:
        return f"layer {layer!r} may not import third-party package {top!r}"
    return None


def violations_for_source(source: str, module: str, is_package: bool = False) -> list[str]:
    """Checks one module's imports against the layer rules.

    Args:
        source: Python source code.
        module: Dotted module name (decides the layer).
        is_package: Whether the source is a package ``__init__``.

    Returns:
        Human-readable violations; empty if the module complies.
    """
    layer = _layer_of_module(module)
    if layer not in RULES:
        return []
    problems = []
    for line, target in imported_modules(source, module, is_package):
        message = _check_import(layer, target)
        if message:
            problems.append(f"{module}:{line} imports {target}: {message}")
    return problems


# --- The real code base -----------------------------------------------------


def test_scanner_finds_the_foundation_modules() -> None:
    names = {m.name for m in _iter_source_modules()}
    assert {"emergency_sim.domain.ports", "emergency_sim.scenarios", "emergency_sim.eventlog"} <= names
    assert "emergency_sim.geo" not in names  # phase-1 module, no rule


def test_layer_import_rules() -> None:
    problems = [
        problem
        for module in _iter_source_modules()
        for problem in violations_for_source(module.path.read_text(), module.name, module.is_package)
    ]
    assert problems == []


def test_no_future_annotations_and_no_print() -> None:
    problems = []
    for module in _iter_source_modules():
        tree = ast.parse(module.path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                problems.append(f"{module.name}:{node.lineno} uses from __future__")
            if (
                module.layer != "cli"
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                problems.append(f"{module.name}:{node.lineno} calls print (use logging)")
    assert problems == []


def test_domain_package_init_reexports_nothing() -> None:
    tree = ast.parse((PROJECT_ROOT / PACKAGE / "domain" / "__init__.py").read_text())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


# --- The checker itself (so the real-code test cannot pass vacuously) -------


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("emergency_sim.engine.engine", "import agents"),
        ("emergency_sim.engine.engine", "from emergency_sim.control.greedy import GreedyController"),
        ("emergency_sim.engine.engine", "from emergency_sim import policies"),
        ("emergency_sim.engine.rules", "from ..policies import estimates"),
        ("emergency_sim.control.coordinator", "from emergency_sim.engine.engine import Engine"),
        ("emergency_sim.control.coordinator", "from agents import Runner"),
        ("emergency_sim.policies.dispatch", "from openai.types.responses import ResponseOutputText"),
        ("emergency_sim.mas.backend", "from emergency_sim.engine import Engine"),
        ("emergency_sim.domain.views", "import numpy"),
        ("emergency_sim.domain.views", "from emergency_sim.scenarios import load_scenario"),
        ("emergency_sim.eventlog", "from emergency_sim import scenarios"),
        ("emergency_sim.evaluation.runner", "import agents"),
        ("emergency_sim.evaluation.report", "import streamlit as st"),
        ("dashboard.app", "from emergency_sim.engine.engine import Engine"),
        ("dashboard.figures", "from emergency_sim.control import greedy"),
        ("emergency_sim.control.greedy", "from tests.factories import hospital"),
    ],
)
def test_checker_flags_forbidden_imports(module: str, source: str) -> None:
    assert violations_for_source(source, module, is_package=False)


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("emergency_sim.domain.params", "from emergency_sim import geo"),
        ("emergency_sim.domain.views", "from .enums import Triage\nfrom pydantic import Field"),
        ("emergency_sim.scenarios", "from . import datagen\nfrom emergency_sim.domain.scenario import Scenario"),
        ("emergency_sim.engine.engine", "import heapq\nfrom .state import WorldState\nfrom emergency_sim import geo"),
        ("emergency_sim.mas.backend", "from agents import Runner\nimport openai\nfrom emergency_sim.policies import dispatch"),
        ("emergency_sim.control.greedy", "from emergency_sim.policies.estimates import eta_min\nfrom emergency_sim.domain.ports import EnginePort"),
        ("emergency_sim.evaluation.runner", "from emergency_sim.engine.engine import Engine\nfrom emergency_sim.mas.backend import make_offline_backend\nimport pandas"),
        ("emergency_sim.cli", "from emergency_sim.evaluation import runner"),
        ("dashboard.app", "import streamlit as st\nimport plotly.express as px\nfrom dashboard import replay\nfrom emergency_sim.evaluation.metrics import kpis_at"),
    ],
)
def test_checker_allows_permitted_imports(module: str, source: str) -> None:
    assert violations_for_source(source, module, is_package=False) == []


def test_relative_import_resolution_from_package_init() -> None:
    imports = imported_modules("from . import state\nfrom ..domain import enums", "emergency_sim.engine", is_package=True)
    assert [name for _, name in imports] == ["emergency_sim.engine", "emergency_sim.domain"]
