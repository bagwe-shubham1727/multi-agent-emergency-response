"""Message bus: builds, numbers and logs every inter-agent message.

Identifiers are sequential per run and never use the wall clock:
messages ``M-0001``, conversations ``NEG-0001``, cases ``CASE-0001``.
"""

import logging
from typing import Final

from emergency_sim.control import ProtocolError
from emergency_sim.domain.enums import MessageType
from emergency_sim.domain.log_records import MessageRecord
from emergency_sim.domain.messages import Message, ProposedAction
from emergency_sim.domain.ports import EventSink

LOGGER = logging.getLogger(__name__)

MAX_RATIONALE_CHARS: Final = 200
"""Rationales are truncated to this length (same limit the agents are given)."""

MAX_ROUND: Final = 99
"""Sanity bound on ``round``; anything above is a programming error."""


class MessageBus:
    """Creates ``Message``s with sequential ids and emits ``MessageRecord``s.

    The bus also keeps the per-incident conversation used by round-0
    (non-negotiation) messages such as PRIORITY_REPORT and dispatch INFORM.
    """

    def __init__(self, sink: EventSink) -> None:
        """Creates an empty bus.

        Args:
            sink: Destination of the message records.
        """
        self._sink = sink
        self._messages: list[Message] = []
        self._conversations = 0
        self._cases = 0
        self._incident_threads: dict[str, str] = {}

    @property
    def messages(self) -> tuple[Message, ...]:
        """All messages sent so far, in order."""
        return tuple(self._messages)

    def new_conversation(self) -> str:
        """Allocates a new conversation id.

        Returns:
            ``"NEG-0001"``, ``"NEG-0002"``, ...
        """
        self._conversations += 1
        return f"NEG-{self._conversations:04d}"

    def new_case_id(self) -> str:
        """Allocates a new case id.

        Returns:
            ``"CASE-0001"``, ``"CASE-0002"``, ...
        """
        self._cases += 1
        return f"CASE-{self._cases:04d}"

    def incident_conversation(self, incident_id: str) -> str:
        """Returns the incident's round-0 conversation, creating it on first use.

        Args:
            incident_id: The incident.

        Returns:
            Its conversation id.
        """
        if incident_id not in self._incident_threads:
            self._incident_threads[incident_id] = self.new_conversation()
        return self._incident_threads[incident_id]

    def send(
        self,
        *,
        t_min: float,
        sender: str,
        recipient: str,
        type: MessageType,
        incident_id: str,
        action: ProposedAction,
        rationale: str,
        conversation_id: str,
        in_reply_to: str | None = None,
        round: int = 0,
    ) -> Message:
        """Builds, records and emits one message.

        Args:
            t_min: Decision-point time.
            sender: Sender agent name.
            recipient: Recipient agent name or ``BROADCAST``.
            type: Performative.
            incident_id: Triggering incident.
            action: Structured content.
            rationale: Human-readable reason (truncated to 200 chars).
            conversation_id: Thread id.
            in_reply_to: Answered message id, if a reply.
            round: Negotiation round (0 outside transport negotiation).

        Returns:
            The message sent.

        Raises:
            ProtocolError: If ``round`` is out of range or ``in_reply_to``
                names a message that was never sent.
        """
        if not 0 <= round <= MAX_ROUND:
            raise ProtocolError(f"invalid round {round}")
        if in_reply_to is not None and not self._known(in_reply_to):
            raise ProtocolError(f"in_reply_to {in_reply_to} was never sent")
        message = Message(
            message_id=f"M-{len(self._messages) + 1:04d}",
            t_min=t_min,
            sender=sender,
            recipient=recipient,
            type=type,
            incident_id=incident_id,
            proposed_action=action,
            rationale=(rationale or type.value.lower())[:MAX_RATIONALE_CHARS],
            in_reply_to=in_reply_to,
            conversation_id=conversation_id,
            round=round,
        )
        self._messages.append(message)
        self._sink.emit(MessageRecord(t_min=t_min, message=message))
        LOGGER.debug("%s %s -> %s %s", message.message_id, sender, recipient, type)
        return message

    def _known(self, message_id: str) -> bool:
        """Tells whether a message id has been issued.

        Args:
            message_id: Candidate id.

        Returns:
            True if sent earlier in this run.
        """
        prefix, _, number = message_id.partition("-")
        return prefix == "M" and number.isdigit() and 1 <= int(number) <= len(self._messages)
