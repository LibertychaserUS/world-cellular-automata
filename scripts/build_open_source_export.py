#!/usr/bin/env python3
"""Build a curated public export of the WCA source tree.

The working repo contains raw artifacts, remote queue state, local caches, and
unfinished experiment branches. Public release must be allowlist-based.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INCLUDE_DIRS = [
    "src",
    "native/wca_kernels",
]

DEFAULT_INCLUDE_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-real-data.txt",
    "requirements-viz.txt",
    ".gitignore",
    "docs/00_project_overview.md",
    "docs/02_architecture.md",
    "docs/34_strict_experiment_route_and_dense_to_sparse_plan.md",
    "docs/60_wca_experiment_constitution_and_macro_plan.md",
    "docs/77_wca_macro_architecture_and_algorithm_validity_criteria.md",
    "docs/80_loop_engineering_operating_system.md",
    "docs/90_evidence_tree_operating_memo.md",
    "docs/94_eval_attribution_and_anti_cheat_protocol.md",
    "docs/98_strict_experiment_manifest_template.md",
    "docs/100_v28_interface_attribution_plan.md",
    "docs/101_unified_physical_latent_design.md",
    "docs/102_open_source_release_and_versioning.md",
    "configs/control/README.md",
    "scripts/analyze_maze_paths.py",
    "scripts/ast_guard.py",
    "scripts/audit_model_ladder.py",
    "scripts/audit_pdebench_dataset.py",
    "scripts/build_open_source_export.py",
    "scripts/build_paper_figures.py",
    "scripts/build_paper_statistical_tables.py",
    "scripts/check_dense_native_mock_equivalence.py",
    "scripts/eval_field_h8_rollout_twice.py",
    "scripts/eval_field_horizon_stratified.py",
    "scripts/eval_field_sentinels.py",
    "scripts/eval_maze.py",
    "scripts/eval_weatherbench_baseline_horizons.py",
    "scripts/eval_weatherbench_horizons.py",
    "scripts/field_eval_plan.py",
    "scripts/gate_experiment_results.py",
    "scripts/generate_maze_pool.py",
    "scripts/generate_pdebench_cache.py",
    "scripts/profile_model.py",
    "scripts/report_v20_pdebench.py",
    "scripts/train_field.py",
    "scripts/train_field_baseline.py",
    "scripts/train_field_token_baseline.py",
    "scripts/train_maze.py",
    "scripts/wca_exp_control.py",
    "scripts/wca_plan_slots.py",
    "tests/test_model_shapes.py",
    "tests/test_maze_batch_schema.py",
    "tests/test_no_label_leakage.py",
    "tests/test_field_model_smoke.py",
    "tests/test_field_tokenizers.py",
    "tests/test_field_metrics.py",
    "tests/test_field_horizon_stratified_eval.py",
    "tests/test_eval_field_sentinels.py",
    "tests/test_dense_native_mock_equivalence.py",
    "tools/wca-exp-control-ts/README.md",
    "tools/wca-exp-control-ts/package.json",
    "tools/wca-exp-control-ts/tsconfig.json",
]

DEFAULT_INCLUDE_GLOBS = [
    "tools/wca-exp-control-ts/src/**/*.ts",
    "tools/wca-exp-control-ts/fixtures/*.json",
    "configs/experiments/field_synthetic_heat_smoke.yaml",
    "configs/experiments/smoke_cpu.yaml",
    "configs/experiments/westb_pdebench_v25*_*.yaml",
    "configs/experiments/westb_pdebench_v26c_*.yaml",
    "configs/experiments/westb_pdebench_v27_*.yaml",
    "configs/experiments/westb_pdebench_v28_*.yaml",
]

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "runs",
    "artifacts",
    "paper",
    "node_modules",
    "dist",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".DS_Store",
}

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN " + r"OPENSSH PRIVATE KEY-----"),
    re.compile(r"\b" + r"gho_" + r"[A-Za-z0-9_]{20,}"),
    re.compile(r"\b" + r"hf_" + r"[A-Za-z0-9_]{20,}"),
    re.compile(r"github" + r"_pat_" + r"[A-Za-z0-9_]{20,}"),
    re.compile(r"ssh\s+-p\s+\d+\s+root@connect\.west[a-z]\.seetacloud\.com"),
    re.compile(r"(\.ssh/|/private/tmp/)[A-Za-z0-9_.-]*(" + r"id_" + r"ed25519" + r"|" + r"known" + r"_hosts" + r")"),
    re.compile(r"\b(password|passwd|api[_-]?key|secret)\s*[:=]", re.IGNORECASE),
]


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    sha256: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def has_excluded_part(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts)


def is_excluded_file(path: Path) -> bool:
    if has_excluded_part(path):
        return True
    return path.name in EXCLUDE_SUFFIXES or any(path.name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def copy_file(src_root: Path, dst_root: Path, rel: Path) -> None:
    if is_excluded_file(rel):
        return
    src = src_root / rel
    if not src.exists() or not src.is_file():
        return
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src_root: Path, dst_root: Path, rel_dir: Path) -> None:
    src_dir = src_root / rel_dir
    if not src_dir.exists():
        return
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        if is_excluded_file(rel):
            continue
        copy_file(src_root, dst_root, rel)


def collect_manifest(dst_root: Path) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for path in sorted(p for p in dst_root.rglob("*") if p.is_file()):
        rel = path.relative_to(dst_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(FileEntry(path=rel, size=path.stat().st_size, sha256=digest))
    return entries


def scan_for_secrets(dst_root: Path) -> list[str]:
    findings: list[str] = []
    text_suffixes = {".py", ".ts", ".tsx", ".js", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg"}
    for path in sorted(p for p in dst_root.rglob("*") if p.is_file()):
        if path.suffix not in text_suffixes and path.name not in {"README", "LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scan_text = "\n".join(line for line in text.splitlines() if "re.compile(" not in line)
        for pattern in SECRET_PATTERNS:
            if pattern.search(scan_text):
                findings.append(f"{path.relative_to(dst_root)} matches {pattern.pattern}")
    return findings


def write_release_files(dst_root: Path, entries: list[FileEntry], source_root: Path) -> None:
    readme = dst_root / "README.md"
    readme.write_text(
        """# World Cellular Automata

World Cellular Automata (WCA) is a research codebase for recursive
world-state prediction. The protected baseline represents a world state as
node vectors, expands it into node-indexed local worlds, evolves those local
worlds through pair interactions, and recomposes evolved centers into the next
world state.

```text
H_t [B,N,D] -> L_t [B,N,N,D] -> local-world pair evolution -> H_{t+1} [B,N,D]
```

## What This Repository Contains

- WCA model code under `src/wca/`.
- Training and evaluation entrypoints for public smoke and field tasks.
- Strict local evaluation and audit utilities.
- Public experiment configs for reproducible follow-up.
- Tests for model shapes, schema boundaries, no-label-leakage, evaluation, and
  sentinel gates.

## Related Public Repositories

- Technical report, PDFs, LaTeX, figures, tables, and release package:
  <https://github.com/LibertychaserUS/world-cellular-automata-technical-report>
- Hugging Face report/artifact mirror:
  <https://huggingface.co/datasets/Chaser111/world-cellular-automata-technical-report>
- Zenodo archive / DOI:
  pending. The report repository contains `wca_technical_report_v0.1_release.zip`
  and `zenodo_metadata.json`; use those files to mint the archival DOI, then
  replace this line with the final Zenodo record URL.

## What This Repository Does Not Contain

Raw artifacts, private VPS logs, remote machine configs, checkpoints, and large
prediction dumps are intentionally excluded. Report PDFs and figures are
released separately in `world-cellular-automata-technical-report`. Heavy result
tables or prediction panels should use a dataset/artifact repository.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,real-data,viz]
```

Use `.[dev]` for code/test work only. Add `real-data` for PDEBench or
WeatherBench-style data preparation, and `viz` for figure/table generation.

## Quick Verification

Run the small local checks before trusting any experiment output:

```bash
python -m pytest tests/test_model_shapes.py tests/test_field_model_smoke.py tests/test_no_label_leakage.py
python -m pytest tests/test_field_horizon_stratified_eval.py tests/test_eval_field_sentinels.py
```

The exported test suite is intentionally lightweight; it checks tensor shape
contracts, schema separation, horizon-stratified evaluation, sentinel behavior,
and dense/native mock equivalence without requiring private artifacts.

## Smoke Training And Evaluation

The synthetic heat smoke config exercises the public field-prediction training
path without downloading large datasets:

```bash
python scripts/train_field.py --config configs/experiments/field_synthetic_heat_smoke.yaml
```

If a run produces an output directory, evaluate it with the horizon-stratified
evaluator using the corresponding config/checkpoint paths:

```bash
python scripts/eval_field_horizon_stratified.py --help
python scripts/eval_field_sentinels.py --help
```

For maze smoke work:

```bash
python scripts/generate_maze_pool.py --help
python scripts/train_maze.py --help
python scripts/eval_maze.py --help
```

## Experiment Control Plane

The public release includes the Python control-plane entrypoint and a
TypeScript shadow control-plane prototype. The control plane is designed around
manifested experiments rather than ad hoc shell commands.

Typical lifecycle:

```bash
python scripts/wca_exp_control.py plan path/to/manifest.json
python scripts/wca_exp_control.py validate path/to/manifest.json
python scripts/wca_exp_control.py submit path/to/manifest.json
python scripts/wca_exp_control.py status path/to/manifest.json
python scripts/wca_exp_control.py fetch path/to/manifest.json
python scripts/wca_exp_control.py inventory path/to/manifest.json
python scripts/wca_exp_control.py report path/to/manifest.json
```

Remote execution requires machine-specific SSH configuration that is not part
of the public repository. Before submitting to paid hardware, local static,
compile, preflight, and contract gates should pass.

For the TypeScript shadow control plane:

```bash
cd tools/wca-exp-control-ts
npm install
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
```

It is not yet the canonical launcher; it is a typed replacement path for future
control-plane hardening.

## Reproducing The Main Evidence Path

The public config names encode the staged evidence tree:

- V25: recursion-depth ladder and h8x2 rollout evidence.
- V25c/V25e: attribution and guardrail controls.
- V26/V27/V28: N=256 capacity, token-equivalent baselines, and interface
  attribution diagnostics.

To reproduce a formal line, use the exact config/manifest, keep the declared
split/seed/eval-index plan fixed, and report both numerical metrics and gate
status. Historical raw anchors and token-level fair comparisons should not be
mixed into one formal table without stating the interface contract.

## Evidence Boundary

Current public evidence should be read as a staged research record:

- strongest formal row: token-level PDEBench reaction-diffusion h8x2 rollout;
- attribution work: learnable-interface and no-recursion controls;
- secondary evidence: maze potential fields and WeatherBench-style transfer;
- diagnostic boundary: larger-token PatchMean interfaces under current capacity.

Do not interpret patch-repeated qualitative panels as native full-field
superiority claims.

## Reporting

The report repository contains the human-readable technical report and figure
package. This source repository contains the executable code path. When citing
or reproducing results, cite both the code commit and the report package
manifest.

## Archive And DOI

For archival citation, use the technical-report release package and
`zenodo_metadata.json` in the report repository to create the Zenodo record.
After Zenodo mints the DOI, add the DOI badge and record URL to this README,
the report README, and the Hugging Face dataset card.
""",
        encoding="utf-8",
    )
    reproducibility = dst_root / "REPRODUCIBILITY.md"
    reproducibility.write_text(
        """# Reproducibility

This repository is a curated source export. Heavy artifacts, raw datasets, and
private queue logs are intentionally excluded.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,real-data,viz]
```

## Smoke Tests

```bash
python -m pytest tests/test_model_shapes.py tests/test_field_model_smoke.py tests/test_no_label_leakage.py
python -m pytest tests/test_field_horizon_stratified_eval.py tests/test_eval_field_sentinels.py
```

## Control Plane

```bash
python scripts/wca_exp_control.py plan path/to/manifest.json
python scripts/wca_exp_control.py validate path/to/manifest.json
python scripts/wca_exp_control.py submit path/to/manifest.json
python scripts/wca_exp_control.py status path/to/manifest.json
python scripts/wca_exp_control.py fetch path/to/manifest.json
python scripts/wca_exp_control.py inventory path/to/manifest.json
python scripts/wca_exp_control.py report path/to/manifest.json
```

Remote machine details are intentionally excluded from the public repository.
Run local gates before any paid remote submission.

## TypeScript Shadow Control Plane

```bash
cd tools/wca-exp-control-ts
npm install
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
```

## Artifact Policy

Use the report and artifact repositories for PDFs, large result tables,
prediction panels, and Zenodo handoff packages.

## Archive And DOI

The Zenodo DOI should be minted from the technical-report release zip and
`zenodo_metadata.json`, then linked from the code README, report README, and
Hugging Face dataset card.
""",
        encoding="utf-8",
    )
    security = dst_root / "SECURITY_AND_PRIVACY.md"
    security.write_text(
        """# Security and Privacy

Do not commit or upload SSH commands, host credentials, API tokens, private keys,
known-host files, raw VPS logs, raw cloud cache directories, or local-only paths.

The public source export is allowlist-based. If a file is absent from
`release_manifest.json`, it is not part of the release.
""",
        encoding="utf-8",
    )
    ts_readme = dst_root / "tools" / "wca-exp-control-ts" / "README.md"
    ts_readme.write_text(
        """# WCA Experiment Control TypeScript Shadow Plane

This package is a public, non-production TypeScript shadow of selected WCA
experiment-control concepts. It is included to document and exercise the typed
planner/control-plane migration path.

It does not submit jobs, fetch artifacts, mutate queues, SSH to remote machines,
or replace `scripts/wca_exp_control.py`.

## Public Smoke

```bash
cd tools/wca-exp-control-ts
npm install
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
```

The full internal TS parity tests depend on private run inventories and are not
part of the public source export. Public users should treat this package as a
shadow planner scaffold, not as the canonical experiment launcher.
""",
        encoding="utf-8",
    )
    license_file = dst_root / "LICENSE"
    if not license_file.exists():
        license_file.write_text(
            """MIT License

Copyright (c) 2026 WCA authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
            encoding="utf-8",
        )
    citation = dst_root / "CITATION.cff"
    citation.write_text(
        """cff-version: 1.2.0
title: "World Cellular Automata"
message: "If you use this code, please cite the World Cellular Automata technical report."
type: software
authors:
  - family-names: "WCA Authors"
version: "0.1.0"
repository-code: "https://github.com/LibertychaserUS/world-cellular-automata"
license: MIT
""",
        encoding="utf-8",
    )
    manifest_entries = collect_manifest(dst_root)
    manifest = {
        "schema": "world-cellular-automata.release_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": "curated_export",
            "note": "Generated from a local working tree; private source paths are intentionally omitted.",
        },
        "policy": "allowlist_export_no_raw_artifacts",
        "file_count": len(manifest_entries),
        "files": [entry.__dict__ for entry in manifest_entries],
    }
    (dst_root / "release_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build_export(dst: Path, force: bool) -> None:
    src_root = repo_root()
    dst = dst.resolve()
    if dst.exists():
        if not force:
            raise SystemExit(f"destination exists; pass --force: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for rel in DEFAULT_INCLUDE_FILES:
        copy_file(src_root, dst, Path(rel))
    for rel_dir in DEFAULT_INCLUDE_DIRS:
        copy_dir(src_root, dst, Path(rel_dir))
    for pattern in DEFAULT_INCLUDE_GLOBS:
        for src in src_root.glob(pattern):
            if src.is_file():
                copy_file(src_root, dst, src.relative_to(src_root))

    entries = collect_manifest(dst)
    write_release_files(dst, entries, src_root)
    findings = scan_for_secrets(dst)
    if findings:
        for finding in findings:
            print(f"secret-scan: {finding}")
        raise SystemExit("public export failed secret scan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_export(args.output, args.force)
    print(f"public export ready: {args.output}")


if __name__ == "__main__":
    main()
