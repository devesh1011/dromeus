"""Check direct TCP reachability across the task's private worker mesh."""

from __future__ import annotations

import argparse
import json
import socket
import time
from collections.abc import Sequence


def server(port: int, expected: int, timeout: float) -> dict[str, object]:
    peers: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", port))
        listener.listen(expected)
        listener.settimeout(timeout)
        deadline = time.monotonic() + timeout
        while len(peers) < expected:
            remaining = max(0.1, deadline - time.monotonic())
            listener.settimeout(remaining)
            connection, address = listener.accept()
            with connection:
                connection.settimeout(remaining)
                if connection.recv(16) != b"dromeus-route\n":
                    raise RuntimeError("private-route probe payload mismatch")
                connection.sendall(b"route-ok\n")
            peers.append(address[0])
    return {"mode": "server", "port": port, "expected": expected, "peers": peers}


def client(local_rank: int, targets: Sequence[tuple[int, str, int]], timeout: float) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for rank, host, port in targets:
        if rank == local_rank:
            continue
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.sendall(b"dromeus-route\n")
            response = connection.recv(32)
        if response != b"route-ok\n":
            raise RuntimeError(f"private-route response mismatch for rank {rank}")
        results.append({"rank": rank, "host": host, "port": port, "status": "ok"})
    return {"mode": "client", "local_rank": local_rank, "targets": results}


def parse_targets(values: Sequence[str]) -> tuple[tuple[int, str, int], ...]:
    parsed = []
    for value in values:
        rank, host, port = value.split(",", 2)
        parsed.append((int(rank), host, int(port)))
    return tuple(parsed)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--port", type=int, required=True)
    server_parser.add_argument("--expected", type=int, required=True)
    server_parser.add_argument("--timeout", type=float, default=120.0)
    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("--local-rank", type=int, required=True)
    client_parser.add_argument("--target", action="append", required=True)
    client_parser.add_argument("--timeout", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.mode == "server":
        result = server(arguments.port, arguments.expected, arguments.timeout)
    else:
        result = client(
            arguments.local_rank,
            parse_targets(arguments.target),
            arguments.timeout,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
