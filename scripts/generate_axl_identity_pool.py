"""Generate the persistent 16-key AXL identity pool for frozen M2 runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.request import urlopen

AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"
_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")


def generate_identity_pool(*, binary: Path, output_root: Path) -> Path:
    """Start isolated pinned AXL nodes and retain their owner-only private keys."""
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("AXL binary must be an executable file")
    output_root.mkdir(parents=True, exist_ok=False)
    output_root.chmod(0o700)
    previous_umask = os.umask(0o077)
    try:
        generated = tuple(
            (
                index,
                _generate_identity(
                    binary=binary,
                    output_root=output_root,
                    index=index,
                ),
            )
            for index in range(16)
        )
        public_keys = tuple(public_key for _, public_key in generated)
        if len(set(public_keys)) != 16:
            raise RuntimeError("AXL generated duplicate public keys")
        registry = output_root / "identities.json"
        _write_json_atomic(
            registry,
            {
                "axl_commit": AXL_COMMIT,
                "identities": [
                    {
                        "private_key": f"private-{index:02d}.pem",
                        "public_key": public_key,
                    }
                    for index, public_key in generated
                ],
                "identity_count": 16,
                "public_keys": sorted(public_keys),
                "schema_version": 1,
            },
        )
        registry.chmod(0o600)
        return registry
    finally:
        os.umask(previous_umask)


def _generate_identity(*, binary: Path, output_root: Path, index: int) -> str:
    private_key = output_root / f"private-{index:02d}.pem"
    config = output_root / f"node-{index:02d}.json"
    log = output_root / f"node-{index:02d}.log"
    api_port = 9500 + index
    subprocess.run(
        [
            _find_openssl(),
            "genpkey",
            "-algorithm",
            "ed25519",
            "-out",
            str(private_key),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    private_key.chmod(0o600)
    _write_json_atomic(
        config,
        {
            "PrivateKeyPath": str(private_key),
            "Peers": [],
            "Listen": [],
            "api_port": api_port,
            "bridge_addr": "127.0.0.1",
            "max_message_size": 16 * 1024 * 1024,
        },
    )
    with log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(binary), "-config", str(config)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            public_key = _wait_for_public_key(api_port=api_port, process=process)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    if not private_key.is_file():
        raise RuntimeError("AXL did not persist a private key")
    private_key.chmod(0o600)
    config.chmod(0o600)
    log.chmod(0o600)
    return public_key


def _find_openssl() -> str:
    for candidate in (
        "/opt/homebrew/opt/openssl/bin/openssl",
        "/usr/local/opt/openssl/bin/openssl",
        shutil.which("openssl"),
    ):
        if candidate:
            return candidate
    raise RuntimeError("OpenSSL is required to create AXL identities")


def _wait_for_public_key(*, api_port: int, process: subprocess.Popen[str]) -> str:
    deadline = time.monotonic() + 15.0
    url = f"http://127.0.0.1:{api_port}/topology"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("AXL exited before publishing its identity")
        try:
            with urlopen(url, timeout=1.0) as response:  # noqa: S310
                payload = cast(
                    dict[str, object],
                    json.loads(response.read().decode()),
                )
            public_key = payload.get("our_public_key")
            if isinstance(public_key, str) and _PUBLIC_KEY.fullmatch(public_key):
                return public_key
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            time.sleep(0.1)
    raise TimeoutError("AXL did not publish its identity before the deadline")


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(
        generate_identity_pool(
            binary=arguments.binary.resolve(),
            output_root=arguments.output_root.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
