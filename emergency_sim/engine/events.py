"""Internal event queue of the discrete-event engine (plan section 4.1).

Events are ordered by ``(t, priority, seq)``: time first, then the fixed
priority of the event kind (resources are freed before units arrive, units
arrive before new incidents are reported, deterioration comes last), then
insertion order. ``seq`` is a global counter, so ties are deterministic.
"""

import heapq
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from emergency_sim.engine.state import EngineError, round_time


class EventKind(StrEnum):
    """Kind of internal (scheduled) engine event."""

    TREATMENT_END = "treatment_end"
    T0_BED_RELEASE = "t0_bed_release"
    T0_STAFF_RELEASE = "t0_staff_release"
    SHIFT_END = "shift_end"
    BUSY_T0_END = "busy_t0_end"
    ARRIVE_HOSPITAL = "arrive_hospital"
    HANDOVER_DONE = "handover_done"
    ARRIVE_SCENE = "arrive_scene"
    ACCESS_GRANTED = "access_granted"
    LOAD_DONE = "load_done"
    INCIDENT_REPORTED = "incident_reported"
    DETERIORATION = "deterioration"


EVENT_PRIORITY: Final = MappingProxyType(
    {
        EventKind.TREATMENT_END: 0,
        EventKind.T0_BED_RELEASE: 0,
        EventKind.T0_STAFF_RELEASE: 0,
        EventKind.SHIFT_END: 0,
        EventKind.BUSY_T0_END: 1,
        EventKind.ARRIVE_HOSPITAL: 2,
        EventKind.HANDOVER_DONE: 2,
        EventKind.ARRIVE_SCENE: 3,
        EventKind.ACCESS_GRANTED: 3,
        EventKind.LOAD_DONE: 3,
        EventKind.INCIDENT_REPORTED: 4,
        EventKind.DETERIORATION: 5,
    }
)
"""Priority at equal time, lowest first."""


@dataclass(frozen=True)
class Event:
    """A scheduled engine event.

    Attributes:
        kind: What happens.
        subject_id: The unit, victim, hospital, staff member or incident concerned.
        token: Unit trip token for unit events; a stale token means the event
            was superseded (for example by a divert) and is ignored.
    """

    kind: EventKind
    subject_id: str
    token: int = 0


class EventQueue:
    """A deterministic min-heap of ``(t, priority, seq, event)``."""

    def __init__(self) -> None:
        """Creates an empty queue."""
        self._heap: list[tuple[float, int, int, Event]] = []
        self._seq = 0

    def __len__(self) -> int:
        """Number of pending events."""
        return len(self._heap)

    def push(self, t: float, event: Event) -> float:
        """Schedules an event.

        Args:
            t: Event time in minutes (rounded to 0.1 here).
            event: The event.

        Returns:
            The rounded event time.
        """
        when = round_time(t)
        heapq.heappush(self._heap, (when, EVENT_PRIORITY[event.kind], self._seq, event))
        self._seq += 1
        return when

    def peek_time(self) -> float | None:
        """Returns the time of the next event, or None if the queue is empty."""
        return self._heap[0][0] if self._heap else None

    def pop(self) -> tuple[float, Event]:
        """Removes and returns the next event.

        Returns:
            ``(time, event)``.

        Raises:
            EngineError: If the queue is empty.
        """
        if not self._heap:
            raise EngineError("pop from an empty event queue")
        t, _, _, event = heapq.heappop(self._heap)
        return t, event
