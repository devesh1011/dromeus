"""Deterministic random perfect matchings for fixed membership."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from dromeus.manifests.models import (
    MAX_PARTICIPANT_COUNT,
    Participant,
    ParticipantCountError,
    PublicKey,
    RoundId,
)


@dataclass(frozen=True, slots=True)
class Pairing:
    """One round's disjoint peer pairs."""

    round_id: RoundId
    pairs: tuple[tuple[PublicKey, PublicKey], ...]

    @property
    def peers(self) -> tuple[PublicKey, ...]:
        return tuple(peer for pair in self.pairs for peer in pair)

    def peer_for(self, local_public_key: PublicKey) -> PublicKey:
        for left, right in self.pairs:
            if left == local_public_key:
                return right
            if right == local_public_key:
                return left
        raise KeyError(f"unknown peer: {local_public_key}")


class PeerScheduler:
    """Derive identical pairings from membership, seed, and round number."""

    def __init__(
        self,
        participants: Sequence[Participant | PublicKey],
        *,
        seed: int,
    ) -> None:
        self._members = self._normalise_members(participants)
        self._seed = seed

    def schedule(self, round_id: RoundId) -> Pairing:
        """Return the deterministic perfect matching for one round."""
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        ordered = sorted(
            self._members,
            key=lambda key: (
                hashlib.sha256(
                    f"{self._seed}\x00{round_id}\x00{key}".encode()
                ).digest(),
                key,
            ),
        )
        pairs: list[tuple[PublicKey, PublicKey]] = []
        for index in range(0, len(ordered), 2):
            left, right = sorted((ordered[index], ordered[index + 1]))
            pairs.append((left, right))
        return Pairing(round_id=round_id, pairs=tuple(pairs))

    @staticmethod
    def _normalise_members(
        participants: Sequence[Participant | PublicKey],
    ) -> tuple[PublicKey, ...]:
        if not participants:
            raise ParticipantCountError(
                "scheduler participant count must be between 2 and 16 inclusive"
            )
        first = participants[0]
        if isinstance(first, Participant):
            if not all(
                isinstance(participant, Participant) for participant in participants
            ):
                raise ValueError(
                    "participants must all be Participant models or public keys"
                )
            models = cast(Sequence[Participant], participants)
            if len({participant.node_index for participant in models}) != len(models):
                raise ValueError("participant node indices must be unique")
            members: tuple[PublicKey, ...] = tuple(
                participant.public_key
                for participant in sorted(models, key=lambda item: item.node_index)
            )
        elif any(isinstance(participant, Participant) for participant in participants):
            raise ValueError(
                "participants must all be Participant models or public keys"
            )
        else:
            members = tuple(
                cast(PublicKey, participant) for participant in participants
            )
        if any(not member for member in members):
            raise ValueError("participant public keys must not be empty")
        if len(set(members)) != len(members):
            raise ValueError("participant public keys must be unique")
        count = len(members)
        # The pair scheduler also serves two-peer engine seams; manifests enforce
        # the production run-group minimum of four participants.
        if count < 2 or count > MAX_PARTICIPANT_COUNT:
            raise ParticipantCountError(
                "scheduler participant count must be between 2 and 16 inclusive"
            )
        if count % 2:
            raise ParticipantCountError("participant count must be even")
        return members


__all__ = ["Pairing", "ParticipantCountError", "PeerScheduler"]
