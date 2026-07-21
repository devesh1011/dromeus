"""Deterministic random perfect matchings for fixed membership."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from dromeus.manifests.models import Participant, PublicKey, RoundId


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
        if len(participants) == 0 or len(participants) % 2:
            raise ValueError("participant count must be positive and even")
        self._members = self._normalise_members(participants)
        self._seed = seed
        self._cache: dict[RoundId, Pairing] = {}

    def schedule(self, round_id: RoundId) -> Pairing:
        """Return the deterministic perfect matching for one round."""
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        cached = self._cache.get(round_id)
        if cached is not None:
            return cached
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
        pairing = Pairing(round_id=round_id, pairs=tuple(pairs))
        self._cache[round_id] = pairing
        return pairing

    def history(self) -> tuple[Pairing, ...]:
        """Return schedules requested so far in round order."""
        return tuple(self._cache[round_id] for round_id in sorted(self._cache))

    def cumulative_edges(self) -> dict[tuple[PublicKey, PublicKey], int]:
        """Count how often each undirected edge appeared in requested rounds."""
        edges: dict[tuple[PublicKey, PublicKey], int] = {}
        for pairing in self.history():
            for edge in pairing.pairs:
                edges[edge] = edges.get(edge, 0) + 1
        return edges

    @staticmethod
    def _normalise_members(
        participants: Sequence[Participant | PublicKey],
    ) -> tuple[PublicKey, ...]:
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
        return members


__all__ = ["Pairing", "PeerScheduler"]
