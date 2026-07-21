from __future__ import annotations

import pytest

from dromeus.gossip.scheduler import Pairing, PeerScheduler
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
    with pytest.raises(ValueError, match="even"):
        PeerScheduler(["peer-0", "peer-1", "peer-2"], seed=1)
    with pytest.raises(ValueError, match="unique"):
        PeerScheduler(["peer-0", "peer-0"], seed=1)


def test_pairing_rejects_unknown_peer() -> None:
    pairing = Pairing(round_id=0, pairs=(("peer-0", "peer-1"),))

    with pytest.raises(KeyError, match="unknown peer"):
        pairing.peer_for("peer-2")
