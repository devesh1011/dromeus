"""Control the four-node Docker demo from a terminal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        method=method,
        data=body,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=10.0) as response:
            value = json.load(response)
    except HTTPError as error:
        try:
            value = json.load(error)
            detail = value.get("error") if isinstance(value, dict) else None
        except (ValueError, TypeError):
            detail = None
        raise RuntimeError(detail or f"dashboard returned HTTP {error.code}") from error
    if not isinstance(value, dict):
        raise RuntimeError("dashboard returned an invalid response")
    return cast(dict[str, object], value)


def _follow(base_url: str, target: str) -> int:
    seen_logs: set[tuple[str, str, str]] = set()
    while True:
        snapshot = request_json(base_url, "/api/state")
        training = snapshot.get("training")
        if isinstance(training, dict):
            logs = training.get("logs")
            if isinstance(logs, list):
                for raw in logs:
                    if not isinstance(raw, dict):
                        continue
                    timestamp = str(raw.get("timestamp", ""))
                    node = str(raw.get("node", ""))
                    message = str(raw.get("message", ""))
                    key = (timestamp, node, message)
                    if key in seen_logs:
                        continue
                    seen_logs.add(key)
                    print(f"{timestamp[11:23]} {node}  {message}", flush=True)
        status = str(snapshot.get("status", "unknown"))
        if status == target:
            print(str(snapshot.get("status_detail", status)))
            return 0
        if status == "failed":
            print(str(snapshot.get("error") or "run failed"), file=sys.stderr)
            return 1
        time.sleep(0.25)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="dashboard base URL",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    form = commands.add_parser("form", help="form the four-node group")
    form.add_argument("--rounds", type=int, required=True)
    form.add_argument("--wait", action="store_true")
    train = commands.add_parser("train", help="start training explicitly")
    train.add_argument("--follow", action="store_true")
    commands.add_parser("state", help="print dashboard state")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "state":
            print(json.dumps(request_json(args.url, "/api/state"), indent=2))
            return 0
        if args.command == "form":
            if not 1 <= args.rounds <= 100:
                raise RuntimeError("--rounds must be between 1 and 100")
            result = request_json(
                args.url,
                "/api/start",
                method="POST",
                payload={"round_count": args.rounds},
            )
            print(json.dumps(result))
            return _follow(args.url, "formed") if args.wait else 0
        result = request_json(args.url, "/api/train", method="POST", payload={})
        print(json.dumps(result))
        return _follow(args.url, "complete") if args.follow else 0
    except (RuntimeError, URLError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
