from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/check_dense_native_mock_equivalence.py"


def test_dense_native_mock_equivalence_harness(tmp_path: Path) -> None:
    output_dir = tmp_path / "native_mock"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--n-nodes",
            "5",
            "--hidden-dim",
            "7",
            "--chunk-sizes",
            "0,1,2,3",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    gate = json.loads((output_dir / "optimization_gate.json").read_text(encoding="utf8"))
    assert gate["mock_backend"] is True
    assert gate["formal_claim_eligible"] is False
    assert gate["promotion_requested"] is False
    assert gate["equivalence_status"] == "pass"
    assert {case["device"] for case in gate["cases"]} == {"cpu"}
    assert any(case["shape"]["batch_size"] == 1 and case["shape"]["receiver_count"] == 4 for case in gate["cases"])
    assert any(case["shape"]["batch_size"] == 2 and case["shape"]["receiver_count"] == 5 for case in gate["cases"])
    chunk_sizes = {
        int(case["case_id"].rsplit("chunk", 1)[1].split("_", 1)[0])
        for case in gate["cases"]
        if "chunk" in case["case_id"]
    }
    assert {0, 1, 2, 3}.issubset(chunk_sizes)

    csv_text = (output_dir / "equivalence_results.csv").read_text(encoding="utf8")
    assert "max_abs_error" in csv_text
    assert "False" not in csv_text


def test_dense_native_mock_equivalence_rejects_protected_output_dirs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            "artifacts/control/dense_native_mock",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "protected evidence directory" in result.stderr
