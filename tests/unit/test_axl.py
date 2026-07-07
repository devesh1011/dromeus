from dromeus.transport.axl import matches_yggdrasil_sender


def test_yggdrasil_sender_identity_uses_known_bits_not_string_prefix() -> None:
    bridge_sender = "d5bcbd9608ae04b5a88dbafe46287" + "f" * 35
    actual_sender = "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0a"
    different_sender = (
        "d5bcbd9608ae04b5a88dbafe4628e72ed7d776c8fdef2857c3f09093d9ecbe0a"
    )

    assert matches_yggdrasil_sender(actual_sender, bridge_sender)
    assert not matches_yggdrasil_sender(different_sender, bridge_sender)
