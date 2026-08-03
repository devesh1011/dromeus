from typing import cast

from demo.formation.server import DashboardState


def test_dashboard_marks_completed_nodes_as_complete() -> None:
    state = DashboardState(containerized=True)

    state.complete(
        "manifest-hash",
        detail="Four nodes completed 10 D-PSGD rounds over AXL",
    )

    snapshot = state.snapshot()
    nodes = cast(list[dict[str, object]], snapshot["nodes"])
    assert snapshot["status"] == "complete"
    assert snapshot["status_detail"] == "Four nodes completed 10 D-PSGD rounds over AXL"
    assert [node["state"] for node in nodes] == [
        "complete",
        "complete",
        "complete",
        "complete",
    ]
