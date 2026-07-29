"""Check architectural invariants without importing project code."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "dromeus"

ALLOWED: dict[str, frozenset[str]] = {
    "algorithms": frozenset({"algorithms", "manifests", "training"}),
    "gossip": frozenset(
        {
            "algorithms",
            "gossip",
            "manifests",
            "persistence",
            "protocol",
            "telemetry",
            "training",
            "transport",
        }
    ),
    "manifests": frozenset({"manifests", "protocol"}),
    "membership": frozenset(
        {"manifests", "membership", "protocol", "telemetry", "transport"}
    ),
    "node": frozenset(
        {"manifests", "membership", "protocol", "runtime", "telemetry", "transport"}
    ),
    "persistence": frozenset({"manifests", "persistence"}),
    "protocol": frozenset({"protocol"}),
    "runtime": frozenset(
        {
            "algorithms",
            "gossip",
            "manifests",
            "membership",
            "persistence",
            "protocol",
            "runtime",
            "telemetry",
            "training",
            "transport",
        }
    ),
    "telemetry": frozenset({"manifests", "persistence", "telemetry"}),
    "training": frozenset({"manifests", "training"}),
    "transport": frozenset({"manifests", "protocol", "telemetry", "transport"}),
}


def owner(path: Path) -> str:
    relative = path.relative_to(SOURCE)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def parsed(path: Path) -> ast.AST:
    return ast.parse(path.read_text(), filename=str(path))


def imported_names(tree: ast.AST) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> list[str] | None:
        if module in visiting:
            start = visiting.index(module)
            return [*visiting[start:], module]
        if module in visited:
            return None
        visiting.append(module)
        for dependency in graph[module]:
            found = visit(dependency)
            if found:
                return found
        visiting.pop()
        visited.add(module)
        return None

    for module in graph:
        found = visit(module)
        if found:
            return found
    return None


def main() -> int:
    errors: list[str] = []
    graph: dict[str, set[str]] = defaultdict(set)
    files = sorted(SOURCE.rglob("*.py"))

    for path in files:
        source_owner = owner(path)
        allowed = ALLOWED.get(source_owner, frozenset())
        tree = parsed(path)
        for imported in imported_names(tree):
            if imported == "support" or imported.startswith(
                ("support.", "tests.", "benchmarks.")
            ):
                errors.append(
                    f"production isolation: {path.relative_to(ROOT)} imports {imported}"
                )
            if not imported.startswith("dromeus."):
                if imported == "msgpack" and source_owner != "protocol":
                    errors.append(
                        f"protocol locality: {path.relative_to(ROOT)} imports "
                        "msgpack; only protocol may encode Dromeus wire values"
                    )
                if source_owner == "transport" and (
                    imported == "safetensors" or imported.startswith("safetensors.")
                ):
                    errors.append(
                        f"transfer opacity: {path.relative_to(ROOT)} imports "
                        "safetensors; transport may only verify bytes and hashes"
                    )
                continue
            target = imported.split(".", 2)[1]
            graph.setdefault(source_owner, set())
            graph.setdefault(target, set())
            if target != source_owner:
                graph[source_owner].add(target)
            if target not in allowed:
                choices = ", ".join(sorted(allowed)) or "no Dromeus modules"
                errors.append(
                    f"dependency direction: {path.relative_to(ROOT)} imports "
                    f"{imported}; {source_owner} may import {choices}"
                )

        if path != SOURCE / "transport" / "receiver.py":
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if isinstance(call.func, ast.Attribute) and call.func.attr == "recv":
                    errors.append(
                        f"single receiver: {path.relative_to(ROOT)}:{call.lineno} "
                        "calls recv(); only transport/receiver.py may drain transport"
                    )

    found_cycle = find_cycle(graph)
    if found_cycle:
        errors.append(f"cycle freedom: {' -> '.join(found_cycle)}")

    if errors:
        print("Structural checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Structural checks passed: dependency direction, production isolation, "
        f"single receiver, transfer opacity, cycle freedom ({len(files)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
