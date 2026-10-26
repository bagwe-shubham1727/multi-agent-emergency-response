"""Loads scenarios from disk, or generates them for other seeds.

    scenario = scenario_for("surge", 7380)     # committed data/surge
    scenario = scenario_for("surge", 1001)     # regenerated in a temp dir

The committed data (seed 7380) is read from ``data/<name>/``. Any other seed is
generated with ``datagen.generate_scenario`` into a temporary directory, loaded
and discarded, so evaluation seeds never touch ``data/``.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Final

from emergency_sim import datagen
from emergency_sim.domain.scenario import Scenario

DEFAULT_DATA_ROOT: Final[Path] = Path(__file__).resolve().parent.parent / "data"

_MANIFEST: Final = "manifest.json"
_ENTITY_FILES: Final = (
    "hospitals",
    "stations",
    "ambulances",
    "staff",
    "responders",
    "incidents",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenario(directory: Path) -> Scenario:
    """Reads the six JSON files + manifest.json (seed, horizon_min).

    Args:
        directory: A scenario folder such as ``data/normal``.

    Returns:
        The validated, immutable scenario.

    Raises:
        FileNotFoundError: If a required file is missing.
        pydantic.ValidationError: If a file does not match the domain schema.
    """
    manifest = _read_json(directory / _MANIFEST)
    entities = {name: _read_json(directory / f"{name}.json") for name in _ENTITY_FILES}
    return Scenario.model_validate(
        {
            "name": manifest["scenario"],
            "seed": manifest["seed"],
            "horizon_min": manifest["horizon_min"],
            **entities,
        }
    )


def _committed_seed(directory: Path) -> int | None:
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        return None
    return int(_read_json(manifest_path)["seed"])


def scenario_for(name: str, seed: int, data_root: Path = DEFAULT_DATA_ROOT) -> Scenario:
    """Returns scenario ``name`` generated with ``seed``.

    Uses the committed data if ``data_root/name/manifest.json`` has this seed;
    otherwise runs ``datagen.generate_scenario(name, seed, tmp)`` in a
    ``TemporaryDirectory``, loads the result and discards the files.

    Args:
        name: Scenario name (``normal``, ``surge`` or ``constrained``).
        seed: Generator seed.
        data_root: Folder holding the committed scenario folders.

    Returns:
        The scenario.

    Raises:
        KeyError: If ``name`` is not a known scenario.
    """
    if name not in datagen.SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; expected one of {sorted(datagen.SCENARIOS)}")
    committed = data_root / name
    if _committed_seed(committed) == seed:
        return load_scenario(committed)
    with tempfile.TemporaryDirectory(prefix="ems-scenario-") as tmp:
        tmp_root = Path(tmp)
        datagen.generate_scenario(name, seed, tmp_root)
        return load_scenario(tmp_root / name)
