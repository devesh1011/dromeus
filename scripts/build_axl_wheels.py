"""Build the dromeus-axl platform wheels from the pinned AXL commit.

Usage: python scripts/build_axl_wheels.py [--target NAME ...] [--out DIR]
                                          [--source DIR] [--smoke]
       python scripts/build_axl_wheels.py --smoke-wheel WHEEL

For each target: cross-compile AXL with CGO disabled, check the binary's
embedded VCS metadata, collect the licenses of every linked Go module, and
build a wheel tagged for that platform. `--smoke` installs the wheel for this
machine into a fresh virtual environment and starts the bundled node until its
HTTP API reports a public key. Needs Go, Git and uv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "dromeus-axl"
sys.path.insert(0, str(PACKAGE / "src"))
from dromeus_axl import AXL_COMMIT, AXL_REPOSITORY  # noqa: E402

BUILD_FLAGS = ("-mod=readonly", "-trimpath", "-buildvcs=true")
LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "patents")


@dataclass(frozen=True)
class Target:
    name: str
    goos: str
    goarch: str
    platform_tag: str

    def is_this_machine(self) -> bool:
        machine = platform.machine().lower()
        arch = {
            "x86_64": "amd64",
            "amd64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(machine)
        return sys.platform == self.goos and arch == self.goarch


# PyTorch publishes no macOS x86_64 wheels, so Dromeus cannot run there.
TARGETS = {
    target.name: target
    for target in (
        Target("linux-amd64", "linux", "amd64", "manylinux_2_17_x86_64"),
        Target("linux-arm64", "linux", "arm64", "manylinux_2_17_aarch64"),
        Target("darwin-arm64", "darwin", "arm64", "macosx_12_0_arm64"),
    )
}


def _run(
    *command: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_source(directory: Path) -> Path:
    directory.mkdir(parents=True)
    _run("git", "init", "-q", cwd=directory)
    _run("git", "remote", "add", "origin", AXL_REPOSITORY + ".git", cwd=directory)
    _run("git", "fetch", "-q", "--depth", "1", "origin", AXL_COMMIT, cwd=directory)
    _run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=directory)
    return directory


def check_source(source: Path) -> None:
    head = _run("git", "rev-parse", "HEAD", cwd=source)
    if head != AXL_COMMIT:
        raise SystemExit(f"AXL source is at {head}, expected {AXL_COMMIT}")
    if _run("git", "status", "--porcelain", cwd=source):
        raise SystemExit(f"AXL source at {source} has local changes")


def _go_env(target: Target) -> dict[str, str]:
    return {
        **os.environ,
        "CGO_ENABLED": "0",
        "GOOS": target.goos,
        "GOARCH": target.goarch,
        "GOTOOLCHAIN": "local",
    }


def compile_binary(source: Path, target: Target, output: Path) -> str:
    go = shutil.which("go")
    if go is None:
        raise SystemExit("Go is required to build AXL")
    _run(
        go,
        "build",
        *BUILD_FLAGS,
        "-o",
        str(output),
        "./cmd/node",
        cwd=source,
        env=_go_env(target),
    )
    metadata = _run(go, "version", "-m", str(output))
    for required in (
        f"vcs.revision={AXL_COMMIT}",
        "vcs.modified=false",
        "-trimpath=true",
        "CGO_ENABLED=0",
        f"GOOS={target.goos}",
        f"GOARCH={target.goarch}",
    ):
        if required not in metadata:
            raise SystemExit(f"{target.name}: binary lacks build setting {required}")
    return _run(go, "env", "GOVERSION")


def _go_license(go: str) -> Path:
    """The Go license; distribution packages (e.g. Homebrew) drop GOROOT/LICENSE,
    so fall back to the official toolchain module of the same version."""
    license_file = Path(_run(go, "env", "GOROOT")) / "LICENSE"
    if license_file.is_file():
        return license_file
    goos, goarch, version = (
        _run(go, "env", name) for name in ("GOHOSTOS", "GOHOSTARCH", "GOVERSION")
    )
    module = f"golang.org/toolchain@v0.0.1-{version}.{goos}-{goarch}"
    download = json.loads(_run(go, "mod", "download", "-json", module))
    return Path(download["Dir"]) / "LICENSE"


def third_party_licenses(source: Path, target: Target) -> str:
    """License texts of the Go standard library and every linked module."""
    go = shutil.which("go") or "go"
    listing = _run(
        go,
        "list",
        "-mod=readonly",
        "-deps",
        "-f",
        "{{with .Module}}{{.Path}}\t{{.Version}}\t{{.Dir}}{{end}}",
        "./cmd/node",
        cwd=source,
        env=_go_env(target),
    )
    modules = sorted(
        {
            tuple(line.split("\t"))
            for line in listing.splitlines()
            if line and not line.startswith("github.com/gensyn-ai/axl\t")
        }
    )
    sections = [("Go standard library", _go_license(go))]
    for path, version, directory in modules:
        found = sorted(
            entry
            for entry in Path(directory).iterdir()
            if entry.is_file() and entry.name.lower().startswith(LICENSE_PREFIXES)
        )
        if not found:
            raise SystemExit(f"no license file in Go module {path} {version}")
        sections.extend((f"{path} {version} ({entry.name})", entry) for entry in found)
    rule = "=" * 78
    return "".join(
        f"{rule}\n{title}\n{rule}\n\n{file.read_text().strip()}\n\n"
        for title, file in sections
    )


def build_wheel(source: Path, target: Target, out: Path) -> Path:
    stage = PACKAGE / "build" / target.name
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    binary = stage / "axl"
    go_version = compile_binary(source, target, binary)
    binary.chmod(0o755)
    (stage / "THIRD_PARTY_LICENSES.txt").write_text(
        third_party_licenses(source, target)
    )
    record = {
        "axl_commit": AXL_COMMIT,
        "axl_repository": AXL_REPOSITORY,
        "go_version": go_version,
        "build_flags": " ".join(("CGO_ENABLED=0", *BUILD_FLAGS)),
        "target": target.name,
        "platform_tag": target.platform_tag,
        "binary_sha256": _sha256(binary),
    }
    (stage / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(stage / "dist"), str(PACKAGE)],
        check=True,
        env={**os.environ, "DROMEUS_AXL_STAGE": str(stage)},
    )
    (built,) = (stage / "dist").glob("*.whl")
    if not built.name.endswith(f"-py3-none-{target.platform_tag}.whl"):
        raise SystemExit(f"{built.name}: expected platform {target.platform_tag}")
    check_wheel(built, record)
    wheel = out / built.name
    shutil.move(built, wheel)
    return wheel


def check_wheel(wheel: Path, record: dict[str, str]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        info = archive.getinfo("dromeus_axl/bin/axl")
        if (info.external_attr >> 16) & 0o111 != 0o111:
            raise SystemExit(f"{wheel.name}: AXL binary is not executable")
        digest = hashlib.sha256(archive.read(info)).hexdigest()
        if digest != record["binary_sha256"]:
            raise SystemExit(f"{wheel.name}: AXL binary differs from the build")
        names = set(archive.namelist())
        for required in (
            "dromeus_axl/__init__.py",
            "dromeus_axl/build.json",
            "dromeus_axl/THIRD_PARTY_LICENSES.txt",
        ):
            if required not in names:
                raise SystemExit(f"{wheel.name}: missing {required}")
        wheel_meta = next(n for n in names if n.endswith(".dist-info/WHEEL"))
        text = archive.read(wheel_meta).decode()
        if "Root-Is-Purelib: false" not in text:
            raise SystemExit(f"{wheel.name}: wheel is marked pure Python")


def smoke(wheel: Path) -> None:
    """Install the wheel in a fresh venv and start AXL until its API answers."""
    with tempfile.TemporaryDirectory(prefix="dromeus-axl-smoke-") as directory:
        root = Path(directory)
        venv = root / "venv"
        _run("uv", "venv", "-q", "--python", "3.12", str(venv))
        python = venv / "bin" / "python"
        _run("uv", "pip", "install", "-q", "--python", str(python), str(wheel))
        info = json.loads(
            _run(
                str(python),
                "-c",
                "import json, dromeus_axl; print(json.dumps(dromeus_axl.build_info()))",
            )
        )
        if info["axl_commit"] != AXL_COMMIT:
            raise SystemExit("installed dromeus-axl reports another AXL commit")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        config = root / "axl.json"
        config.write_text(
            json.dumps(
                {
                    "Peers": [],
                    "Listen": [],
                    "api_port": port,
                    "bridge_addr": "127.0.0.1",
                }
            )
        )
        log = root / "axl.log"
        with log.open("wb") as output:
            process = subprocess.Popen(
                [str(venv / "bin" / "dromeus-axl"), "-config", str(config)],
                cwd=root,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        try:
            key = _wait_for_key(f"http://127.0.0.1:{port}/topology", process)
        except RuntimeError as error:
            raise SystemExit(f"{error}\n{log.read_text()}") from error
        finally:
            process.terminate()
            process.wait(timeout=10)
        print(f"smoke: {wheel.name} installed and started; AXL public key {key}")


def _wait_for_key(url: str, process: subprocess.Popen[bytes]) -> str:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"bundled AXL exited with {process.returncode}")
        try:
            with urlopen(url, timeout=1) as response:
                key = json.loads(response.read()).get("our_public_key")
            if isinstance(key, str) and len(key) == 64:
                return key
        except (URLError, OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("bundled AXL API did not answer within 30 s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dromeus-axl platform wheels.")
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(TARGETS),
        help="build only this target (repeatable); default all",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "axl")
    parser.add_argument(
        "--source", type=Path, help="existing AXL checkout at the pinned commit"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="install and start the wheel built for this machine",
    )
    parser.add_argument(
        "--smoke-wheel",
        type=Path,
        help="only install and start this already built wheel",
    )
    args = parser.parse_args()
    if args.smoke_wheel:
        smoke(args.smoke_wheel.resolve())
        return
    targets = [TARGETS[name] for name in (args.target or sorted(TARGETS))]
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="axl-source-") as directory:
        source = (
            args.source.resolve()
            if args.source
            else fetch_source(Path(directory) / "axl")
        )
        check_source(source)
        wheels = {target.name: build_wheel(source, target, out) for target in targets}
    for name, wheel in wheels.items():
        print(f"built {wheel.name} ({TARGETS[name].goos}/{TARGETS[name].goarch})")
    if args.smoke:
        native = [wheels[t.name] for t in targets if t.is_this_machine()]
        if not native:
            raise SystemExit("--smoke: no wheel was built for this machine")
        smoke(native[0])


if __name__ == "__main__":
    main()
