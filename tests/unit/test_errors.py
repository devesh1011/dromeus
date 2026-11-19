from __future__ import annotations

import pytest

from dromeus.errors import (
    ConfigError,
    DraftError,
    DromeusError,
    FormationError,
    PrivacyError,
    RunError,
    TaskError,
)
from dromeus.gossip.interfaces import PairCommitError
from dromeus.manifests.models import ParticipantCountError
from dromeus.membership.formation import ReadyValidationError
from dromeus.persistence.archive import RunArchiveError
from dromeus.persistence.run_store import RunStoreError
from dromeus.protocol.codec import EnvelopeError, ProtocolDecodeError
from dromeus.runtime import NodeRuntimeError, PeerRunFailureError
from dromeus.telemetry.consensus import ConsensusSketchError
from dromeus.telemetry.evidence import EvidenceError
from dromeus.transport.interface import TransportError
from dromeus.transport.receiver import ReceiverError
from dromeus.transport.transfer import TransferError


@pytest.mark.parametrize(
    ("error", "builtin"),
    [
        (ConfigError, ValueError),
        (TaskError, ValueError),
        (DraftError, ValueError),
        (PrivacyError, ValueError),
        (FormationError, RuntimeError),
        (RunError, RuntimeError),
    ],
)
def test_public_errors_follow_builtin_convention(
    error: type[DromeusError], builtin: type[Exception]
) -> None:
    assert issubclass(error, DromeusError)
    assert issubclass(error, builtin)


@pytest.mark.parametrize(
    ("internal", "public"),
    [
        (NodeRuntimeError, RunError),
        (PeerRunFailureError, RunError),
        (TransportError, RunError),
        (ReceiverError, RunError),
        (TransferError, RunError),
        (PairCommitError, RunError),
        (RunStoreError, RunError),
        (RunArchiveError, RunError),
        (ProtocolDecodeError, RunError),
        (EnvelopeError, RunError),
        (ConsensusSketchError, RunError),
        (EvidenceError, RunError),
        (ReadyValidationError, FormationError),
        (ParticipantCountError, DraftError),
    ],
)
def test_internal_errors_are_caught_as_public_errors(
    internal: type[Exception], public: type[DromeusError]
) -> None:
    assert issubclass(internal, public)
