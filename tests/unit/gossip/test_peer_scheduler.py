from __future__ import annotations

import pytest

from dromeus.gossip.peer_scheduler import (
    Pairing,
    ParticipantCountError,
    PeerScheduler,
)
from dromeus.manifests.models import Participant


def test_scheduler_is_independent_and_forms_a_perfect_matching() -> None:
    participants = tuple(
        Participant(public_key=f"peer-{index}", node_index=index) for index in range(4)
    )
    first = PeerScheduler(participants, seed=8)
    second = PeerScheduler(tuple(reversed(participants)), seed=8)

    pairing = first.schedule(3)

    assert pairing == second.schedule(3)
    assert len(pairing.pairs) == 2
    assert set(pairing.peers) == {f"peer-{index}" for index in range(4)}
    assert pairing.peer_for("peer-0") != "peer-0"


def test_scheduler_records_history_and_cumulative_edges() -> None:
    scheduler = PeerScheduler([f"peer-{index}" for index in range(4)], seed=2)

    scheduler.schedule(0)
    scheduler.schedule(1)

    assert [item.round_id for item in scheduler.history()] == [0, 1]
    assert sum(scheduler.cumulative_edges().values()) == 4


def test_scheduler_rejects_invalid_membership() -> None:
    with pytest.raises(ParticipantCountError, match="even"):
        PeerScheduler(
            ["peer-0", "peer-1", "peer-2", "peer-3", "peer-4"],
            seed=1,
        )
    with pytest.raises(ValueError, match="unique"):
        PeerScheduler(["peer-0", "peer-0"], seed=1)


@pytest.mark.parametrize("participant_count", [4, 8, 16])
@pytest.mark.parametrize("seed", [17, 29, 41])
def test_scheduler_builds_disjoint_perfect_matchings_at_m2_sizes(
    participant_count: int,
    seed: int,
) -> None:
    members = [f"peer-{index}" for index in range(participant_count)]
    scheduler = PeerScheduler(members, seed=seed)

    for round_id in range(32):
        pairing = scheduler.schedule(round_id)
        assert len(pairing.pairs) == participant_count // 2
        assert len(pairing.peers) == participant_count
        assert len(set(pairing.peers)) == participant_count
        assert set(pairing.peers) == set(members)


def test_pairing_rejects_unknown_peer() -> None:
    pairing = Pairing(round_id=0, pairs=(("peer-0", "peer-1"),))

    with pytest.raises(KeyError, match="unknown peer"):
        pairing.peer_for("peer-2")
