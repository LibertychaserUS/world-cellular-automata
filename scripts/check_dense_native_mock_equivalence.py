#!/usr/bin/env python3
"""Build and check the standalone C++ CPU dense sender-reduction mock."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"WCADPR1\0"
ATOL = 1.0e-5
RTOL = 1.0e-5
PROTECTED_OUTPUT_DIRS = (
    REPO_ROOT / "artifacts/control",
    REPO_ROOT / "artifacts/control_shadow",
    REPO_ROOT / "artifacts/queues",
    REPO_ROOT / "artifacts/fetch",
    REPO_ROOT / "artifacts/fetched",
    REPO_ROOT / "artifacts/remote",
    REPO_ROOT / "runs",
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    seed: int
    batch_size: int
    center_count: int
    receiver_count: int
    sender_count: int
    hidden_dim: int
    chunk_size: int
    residual_scale: float = 0.375


def parse_chunk_sizes(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value < 0:
            raise argparse.ArgumentTypeError("chunk sizes must be non-negative")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    return values


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def reject_protected_output_dir(output_dir: Path) -> None:
    for protected in PROTECTED_OUTPUT_DIRS:
        if output_dir == protected or is_relative_to(output_dir, protected):
            protected_rel = protected.relative_to(REPO_ROOT)
            raise ValueError(f"--output-dir may not be inside protected evidence directory: {protected_rel}")


def publish_artifacts(staging_dir: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    os.replace(staging_dir, output_dir)


def compiler() -> str:
    configured = os.environ.get("CXX")
    if configured:
        return configured
    for candidate in ("c++", "clang++", "g++"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("no C++ compiler found; set CXX")


def build_cli(output_dir: Path) -> Path:
    binary = output_dir / "dense_pair_reduce_mock_cli"
    cmd = [
        compiler(),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-I",
        str(REPO_ROOT / "native/wca_kernels/include"),
        str(REPO_ROOT / "native/wca_kernels/src/dense_pair_reduce.cpp"),
        str(REPO_ROOT / "native/wca_kernels/tools/dense_pair_reduce_mock_cli.cpp"),
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return binary


def make_fixture(case: CaseSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    local = rng.normal(
        loc=0.0,
        scale=0.5,
        size=(case.batch_size, case.center_count, case.receiver_count, case.hidden_dim),
    ).astype(np.float32)
    delta = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(case.batch_size, case.center_count, case.receiver_count, case.sender_count, case.hidden_dim),
    ).astype(np.float32)
    mask = rng.integers(0, 2, size=(case.batch_size, case.receiver_count, case.sender_count)).astype(np.float32)
    denom = rng.uniform(1.0, float(case.sender_count + 1), size=(case.batch_size, case.receiver_count)).astype(np.float32)
    reference = local + np.float32(case.residual_scale) * (
        (delta * mask[:, None, :, :, None]).sum(axis=3, dtype=np.float32) / denom[:, None, :, None]
    )
    return local, delta, mask, denom, reference.astype(np.float32)


def write_fixture(path: Path, case: CaseSpec, local: np.ndarray, delta: np.ndarray, mask: np.ndarray, denom: np.ndarray) -> None:
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<QQQQQf",
                case.batch_size,
                case.center_count,
                case.receiver_count,
                case.sender_count,
                case.hidden_dim,
                np.float32(case.residual_scale),
            )
        )
        for array in (local, delta, mask, denom):
            handle.write(np.ascontiguousarray(array, dtype=np.float32).tobytes(order="C"))


def max_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    abs_error = np.abs(actual - expected)
    rel_error = abs_error / np.maximum(np.abs(expected), np.float32(1.0e-12))
    return float(abs_error.max(initial=0.0)), float(rel_error.max(initial=0.0))


def run_case(binary: Path, output_dir: Path, case: CaseSpec) -> dict[str, object]:
    local, delta, mask, denom, reference = make_fixture(case)
    fixture_path = output_dir / f"{case.case_id}.bin"
    actual_path = output_dir / f"{case.case_id}.out.bin"
    write_fixture(fixture_path, case, local, delta, mask, denom)
    subprocess.run([str(binary), str(fixture_path), str(actual_path), str(case.chunk_size)], check=True, cwd=REPO_ROOT)
    actual = np.fromfile(actual_path, dtype=np.float32).reshape(reference.shape)
    max_abs, max_rel = max_errors(actual, reference)
    passed = max_abs <= ATOL and max_rel <= RTOL
    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "batch_size": case.batch_size,
        "center_count": case.center_count,
        "receiver_count": case.receiver_count,
        "sender_count": case.sender_count,
        "hidden_dim": case.hidden_dim,
        "chunk_size": case.chunk_size,
        "dtype": "fp32",
        "device": "cpu",
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "atol": ATOL,
        "rtol": RTOL,
        "passed": passed,
    }


def build_cases(batch_size: int, n_nodes: int, hidden_dim: int, chunk_sizes: list[int]) -> list[CaseSpec]:
    specs = [
        CaseSpec("required_B1_C4_R4_S4_D3_chunk0", 1729, 1, 4, 4, 4, 3, 0),
        CaseSpec("required_B1_C4_R4_S4_D3_chunk1", 1730, 1, 4, 4, 4, 3, 1),
        CaseSpec("required_B1_C4_R4_S4_D3_chunk2", 1731, 1, 4, 4, 4, 3, 2),
        CaseSpec("required_B1_C4_R4_S4_D3_chunk3_nondiv", 1732, 1, 4, 4, 4, 3, 3),
    ]
    for index, chunk_size in enumerate(chunk_sizes):
        specs.append(
            CaseSpec(
                f"cli_B{batch_size}_C{n_nodes}_R{n_nodes}_S{n_nodes}_D{hidden_dim}_chunk{chunk_size}",
                9000 + index,
                batch_size,
                n_nodes,
                n_nodes,
                n_nodes,
                hidden_dim,
                chunk_size,
            )
        )
    return specs


def write_results_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "seed",
        "batch_size",
        "center_count",
        "receiver_count",
        "sender_count",
        "hidden_dim",
        "chunk_size",
        "dtype",
        "device",
        "max_abs_error",
        "max_rel_error",
        "atol",
        "rtol",
        "passed",
    ]
    with (output_dir / "equivalence_results.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gate(output_dir: Path, rows: list[dict[str, object]]) -> None:
    all_passed = all(bool(row["passed"]) for row in rows)
    gate = {
        "schema_version": 1,
        "mode": "wca_kernel_equivalence_gate",
        "gate_id": "dense-pair-reduce-cpu-mock-gate",
        "generated_at_epoch_seconds": int(time.time()),
        "backend_name": "dense_pair_reduce_cpu_native_mock",
        "backend_kind": "python_chunking",
        "optimized_backend_flag": "none:standalone_cpu_mock_only",
        "mock_backend": True,
        "formal_claim_eligible": False,
        "default_backend_unchanged": True,
        "baseline_reference": "FullRecursiveWorldStateNCA",
        "optimization_role": "provisional_system",
        "equivalence_required": True,
        "equivalence_status": "pass" if all_passed else "fail",
        "promotion_requested": False,
        "evidence_status": "guardrail_only",
        "low_precision_quality_gate_status": "not_applicable",
        "cases": [
            {
                "case_id": str(row["case_id"]),
                "dtype": "fp32",
                "device": "cpu",
                "seed": int(row["seed"]),
                "shape": {
                    "batch_size": int(row["batch_size"]),
                    "node_count": int(row["receiver_count"]),
                    "center_count": int(row["center_count"]),
                    "receiver_count": int(row["receiver_count"]),
                    "sender_count": int(row["sender_count"]),
                    "hidden_dim": int(row["hidden_dim"]),
                },
                "tolerance": {"atol": ATOL, "rtol": RTOL},
                "checked_targets": ["dense_sender_reduce_output"],
                "max_abs_error": float(row["max_abs_error"]),
                "max_rel_error": float(row["max_rel_error"]),
                "passed": bool(row["passed"]),
                "diagnostics_requested": False,
            }
            for row in rows
        ],
        "notes": [
            "Standalone C++17 CPU mock for tensor-axis and sender-reduction contract only.",
            "Non-promotional: mock_backend=true, formal_claim_eligible=false, promotion_requested=false.",
            "No CUDA, autograd, model integration, speed, memory, or model-quality claim.",
        ],
    }
    (output_dir / "optimization_gate.json").write_text(f"{json.dumps(gate, indent=2, sort_keys=True)}\n", encoding="utf8")


def run_bad_fixture_check(binary: Path, output_dir: Path) -> None:
    bad_path = output_dir / "bad_truncated_fixture.bin"
    bad_path.write_bytes(MAGIC + struct.pack("<QQQQQf", 1, 1, 1, 1, 1, np.float32(1.0)))
    result = subprocess.run(
        [str(binary), str(bad_path), str(output_dir / "bad.out.bin"), "0"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        raise RuntimeError("bad truncated fixture unexpectedly passed")


def run_bad_chunk_size_check(binary: Path, output_dir: Path) -> None:
    fixture_path = next(output_dir.glob("*.bin"))
    for raw in ("+1", "-1"):
        result = subprocess.run(
            [str(binary), str(fixture_path), str(output_dir / f"bad_chunk_{raw}.out.bin"), raw],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            raise RuntimeError(f"signed chunk size {raw!r} unexpectedly passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-nodes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=7)
    parser.add_argument("--chunk-sizes", type=parse_chunk_sizes, default=parse_chunk_sizes("0,1,2,3"))
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.n_nodes <= 0 or args.hidden_dim <= 0:
        parser.error("--batch-size, --n-nodes, and --hidden-dim must be positive")

    output_dir = args.output_dir.resolve()
    try:
        reject_protected_output_dir(output_dir)
    except ValueError as error:
        parser.error(str(error))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.work-", dir=output_dir.parent)).resolve()
    try:
        binary = build_cli(staging_dir)
        rows = [
            run_case(binary, staging_dir, case)
            for case in build_cases(args.batch_size, args.n_nodes, args.hidden_dim, args.chunk_sizes)
        ]
        run_bad_fixture_check(binary, staging_dir)
        run_bad_chunk_size_check(binary, staging_dir)
        write_results_csv(staging_dir, rows)
        write_gate(staging_dir, rows)

        failed = [row["case_id"] for row in rows if not row["passed"]]
        if failed:
            print(f"dense native mock equivalence failed: {failed}", file=sys.stderr)
            return 1
        publish_artifacts(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    print(f"wrote {output_dir / 'equivalence_results.csv'}")
    print(f"wrote {output_dir / 'optimization_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
