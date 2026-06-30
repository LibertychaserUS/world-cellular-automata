#!/usr/bin/env python3
"""Project-specific AST guardrails for WCA research invariants.

This is intentionally separate from generic lint. Ruff catches style and common
Python mistakes; this file catches architecture violations that would invalidate
WCA experiments, such as model code importing oracle/data layers or reading
supervised labels directly.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (ROOT / "src" / "wca", ROOT / "scripts", ROOT / "tests")
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "runs",
}

MODEL_FORBIDDEN_IMPORT_PREFIXES = (
    "wca.data",
    "wca.training",
    "wca.experiments",
    "scripts",
)
DATA_FORBIDDEN_IMPORT_PREFIXES = (
    "wca.models",
    "wca.training",
    "wca.experiments",
    "scripts",
)
MODEL_FORBIDDEN_LABEL_KEYS = {
    "distance_field",
    "distance_mask",
    "raw_distance",
    "goal_idx",
    "start_idx",
    "label",
    "target",
    "target_idx",
    "field_target",
}
ARGMAX_GUARDED_FILES = {
    Path("src/wca/data/maze/metrics.py"),
    Path("src/wca/data/maze/forensics.py"),
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    code: str
    message: str


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        if not path.exists() or _is_excluded(path):
            continue
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.py")):
                if not _is_excluded(candidate):
                    yield candidate


def _rel(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _rel_posix(path: Path, root: Path) -> str:
    return _rel(path, root).as_posix()


def _starts_with_any(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _import_names(node: ast.AST) -> Iterable[tuple[str, int]]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name, node.lineno
    elif isinstance(node, ast.ImportFrom) and node.module:
        yield node.module, node.lineno


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _subscript_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _string_literal(node.slice)


def _line_has_allow_comment(lines: list[str], lineno: int) -> bool:
    if lineno <= 0 or lineno > len(lines):
        return False
    return "ast-guard: allow" in lines[lineno - 1]


def _collect_import_violations(path: Path, root: Path, tree: ast.AST) -> list[Violation]:
    rel = _rel(path, root)
    rel_text = rel.as_posix()
    violations: list[Violation] = []

    forbidden_prefixes: Sequence[str] = ()
    if rel.parts[:3] == ("src", "wca", "models"):
        forbidden_prefixes = MODEL_FORBIDDEN_IMPORT_PREFIXES
    elif rel.parts[:3] == ("src", "wca", "data"):
        forbidden_prefixes = DATA_FORBIDDEN_IMPORT_PREFIXES

    if not forbidden_prefixes:
        return violations

    for node in ast.walk(tree):
        for name, line in _import_names(node):
            if _starts_with_any(name, forbidden_prefixes):
                violations.append(
                    Violation(
                        path=rel_text,
                        line=line,
                        code="WCA001",
                        message=f"forbidden architectural import `{name}`",
                    )
                )
    return violations


def _collect_model_label_key_violations(path: Path, root: Path, tree: ast.AST) -> list[Violation]:
    rel = _rel(path, root)
    if rel.parts[:3] != ("src", "wca", "models"):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        key = _subscript_key(node)
        if key in MODEL_FORBIDDEN_LABEL_KEYS:
            violations.append(
                Violation(
                    path=rel.as_posix(),
                    line=getattr(node, "lineno", 0),
                    code="WCA002",
                    message=f"model code must not read supervised batch key `{key}`",
                )
            )
    return violations


def _collect_maze_argmax_violations(path: Path, root: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    rel = _rel(path, root)
    if rel not in ARGMAX_GUARDED_FILES:
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        is_argmax = (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "argmax")
                or (isinstance(node.func, ast.Name) and node.func.id == "argmax")
            )
        )
        if is_argmax and not _line_has_allow_comment(lines, getattr(node, "lineno", 0)):
            violations.append(
                Violation(
                    path=rel.as_posix(),
                    line=getattr(node, "lineno", 0),
                    code="WCA003",
                    message="maze metrics/forensics must not infer start/goal via argmax; use explicit batch indices",
                )
            )
    return violations


def collect_violations(paths: Sequence[Path] = DEFAULT_TARGETS, root: Path = ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=path.as_posix())
        except SyntaxError as exc:
            violations.append(
                Violation(
                    path=_rel_posix(path, root),
                    line=exc.lineno or 0,
                    code="WCA000",
                    message=f"syntax error: {exc.msg}",
                )
            )
            continue

        lines = source.splitlines()
        violations.extend(_collect_import_violations(path, root, tree))
        violations.extend(_collect_model_label_key_violations(path, root, tree))
        violations.extend(_collect_maze_argmax_violations(path, root, tree, lines))
    return sorted(violations, key=lambda item: (item.path, item.line, item.code))


def write_markdown(path: Path, violations: Sequence[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# WCA AST Guard Report", ""]
    if not violations:
        lines.append("No AST guard violations found.")
    else:
        lines.extend(
            [
                "| file | line | code | message |",
                "|---|---:|---|---|",
            ]
        )
        for violation in violations:
            lines.append(
                f"| `{violation.path}` | {violation.line} | `{violation.code}` | {violation.message} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WCA-specific AST guardrails.")
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_TARGETS))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    target_paths = tuple(path if path.is_absolute() else ROOT / path for path in args.paths)
    violations = collect_violations(target_paths, ROOT)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"violations": [asdict(violation) for violation in violations]}, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown:
        write_markdown(args.markdown, violations)

    if violations:
        for violation in violations:
            print(f"{violation.path}:{violation.line}: {violation.code} {violation.message}", file=sys.stderr)
        raise SystemExit(1)

    print("WCA AST guard passed")


if __name__ == "__main__":
    main()
