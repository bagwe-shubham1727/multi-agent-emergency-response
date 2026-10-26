"""In-memory event sink and JSONL persistence of run logs.

A run's records are collected by ``MemoryEventSink`` (which assigns ``seq``),
written with ``write_jsonl`` and read back with ``read_log``. Serialisation is
deterministic: the same records always give byte-identical files.
"""

from collections.abc import Iterable
from pathlib import Path

from emergency_sim.domain.log_records import LOG_RECORD_ADAPTER, LogRecord


class MemoryEventSink:
    """Collects log records in memory; implements ``domain.ports.EventSink``.

    Each emitted record gets ``seq`` equal to its position in the log.
    """

    def __init__(self) -> None:
        """Creates an empty sink."""
        self._records: list[LogRecord] = []

    @property
    def records(self) -> tuple[LogRecord, ...]:
        """All records emitted so far, in order."""
        return tuple(self._records)

    def emit(self, record: LogRecord) -> None:
        """Appends a copy of ``record`` with ``seq`` set to its position.

        Args:
            record: The record to store; its own ``seq`` is ignored.
        """
        self._records.append(record.model_copy(update={"seq": len(self._records)}))


def write_jsonl(records: Iterable[LogRecord], path: Path) -> None:
    """Writes records as JSON Lines, one ``model_dump_json()`` per line.

    Args:
        records: Records in log order.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_log(path: Path) -> list[LogRecord]:
    """Reads a JSONL run log written by ``write_jsonl``.

    Args:
        path: The ``events.jsonl`` file.

    Returns:
        The records in file order. Blank lines are skipped.

    Raises:
        pydantic.ValidationError: If a line is not a valid ``LogRecord``.
    """
    with path.open(encoding="utf-8") as handle:
        return [LOG_RECORD_ADAPTER.validate_json(line) for line in handle if line.strip()]
