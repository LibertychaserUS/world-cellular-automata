import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

import type {
  ArtifactInventory,
  ArtifactInventoryParityReport,
  ArtifactInventoryParityRow,
  FormalEvidenceStub,
  ManifestParityReport,
  ManifestParityRow,
  PythonManifestValidationResult,
  QueueStatusParityReport,
  QueueStatusParityRow,
  QueueStatusReport,
  ReportGateParityReport,
  ReportGateParityRow,
} from "./contracts.ts";
import { buildArtifactInventory } from "./inventory.ts";
import { inspectControlManifest } from "./manifest.ts";
import { buildLocalQueueStatus } from "./queue_status.ts";
import { buildFormalEvidenceStub } from "./report.ts";

const pythonValidationSnippet = `
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root))

import scripts.wca_exp_control as control

manifest = control.load_manifest(manifest_path)
errors, warnings = control.validate_manifest(
    manifest,
    local_root=repo_root,
    enforce_clean_git=False,
    enforce_prerequisites=False,
    enforce_submit_contract=True,
)
print(json.dumps({
    "ok": not errors,
    "experiment_id": manifest.get("experiment_id"),
    "errors": errors,
    "warnings": warnings,
}, sort_keys=True))
`;

const pythonInventorySnippet = `
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root))

import scripts.wca_exp_control as control

manifest = control.load_manifest(manifest_path)
inventory = control.build_artifact_inventory(manifest, local_root=repo_root)
print(json.dumps({
    "schema_version": inventory.get("schema_version"),
    "experiment_id": inventory.get("experiment_id"),
    "artifact_contract": inventory.get("artifact_contract", {}),
    "artifact_contract_source": inventory.get("artifact_contract_source"),
    "required_files": inventory.get("required_files", []),
    "present_files": inventory.get("present_files", []),
    "missing_files": inventory.get("missing_files", []),
    "validation_errors": inventory.get("validation_errors", []),
    "provenance_warnings": inventory.get("provenance_warnings", []),
    "status": inventory.get("status"),
}, sort_keys=True))
`;

const pythonReportGateSnippet = `
import hashlib
import json
import sys
import tempfile
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root))

import scripts.wca_exp_control as control

manifest = control.load_manifest(manifest_path)
inventory = control.build_artifact_inventory(manifest, local_root=repo_root)
with tempfile.TemporaryDirectory(prefix="wca-report-gate-parity-") as tmp:
    tmp_root = Path(tmp)
    report_path = control.write_formal_evidence_stub(manifest, inventory, local_root=tmp_root)
    report_text = report_path.read_text(encoding="utf-8")
    print(json.dumps({
        "schema_version": 1,
        "experiment_id": manifest.get("experiment_id"),
        "strict_gate": inventory.get("status"),
        "report_relative_path": report_path.relative_to(tmp_root).as_posix(),
        "report_text": report_text,
        "report_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
    }, sort_keys=True))
`;

const pythonQueueStatusSnippet = `
import json
import sys
from pathlib import Path

queue_dir = Path(sys.argv[1]).resolve()
status_path = queue_dir / "status.jsonl"
summary_path = queue_dir / "queue_summary.json"

missing_files = []
parse_errors = []

def read_text_optional(path):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        missing_files.append(path.as_posix())
        return None

def read_status_events():
    text = read_text_optional(status_path)
    if text is None:
        return []
    events = []
    for index, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception as exc:
            parse_errors.append(f"status.jsonl line {index}: {exc}")
            continue
        if not isinstance(event, dict):
            parse_errors.append(f"status.jsonl line {index} is not an object")
            continue
        events.append(event)
    return events

def read_summary():
    text = read_text_optional(summary_path)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except Exception as exc:
        parse_errors.append(f"queue_summary.json: {exc}")
        return None
    if not isinstance(payload, dict):
        parse_errors.append("queue_summary.json is not an object")
        return None
    return payload

events = read_status_events()
summary = read_summary()
started_jobs = {}
finished_jobs = {}
queue_name = summary.get("queue_name") if isinstance(summary, dict) else None
run_id = summary.get("run_id") if isinstance(summary, dict) else None
started_at = summary.get("started_at") if isinstance(summary, dict) else None
finished_at = summary.get("finished_at") if isinstance(summary, dict) else None
exit_code = summary.get("exit_code") if isinstance(summary, dict) and isinstance(summary.get("exit_code"), int) else None
event_job_count = None

for event in events:
    queue_name = queue_name or event.get("queue_name")
    run_id = run_id or event.get("run_id")
    if event.get("event") == "queue_started":
        started_at = started_at or event.get("at")
        if isinstance(event.get("job_count"), int):
            event_job_count = event.get("job_count")
    elif event.get("event") == "queue_finished":
        finished_at = event.get("at") or finished_at
        if isinstance(event.get("exit_code"), int):
            exit_code = event.get("exit_code")
    elif event.get("event") == "job_started" and event.get("job_id"):
        started_jobs[event["job_id"]] = event
    elif event.get("event") == "job_finished" and event.get("job_id"):
        finished_jobs[event["job_id"]] = event

current_job_ids = sorted([job_id for job_id in started_jobs if job_id not in finished_jobs])
summary_jobs = summary.get("jobs", []) if isinstance(summary, dict) and isinstance(summary.get("jobs"), list) else []
job_ids = sorted(set([job.get("id") for job in summary_jobs if isinstance(job, dict) and job.get("id")] + list(started_jobs) + list(finished_jobs)))
jobs = []
for job_id in job_ids:
    summary_job = next((job for job in summary_jobs if isinstance(job, dict) and job.get("id") == job_id), {})
    finished = finished_jobs.get(job_id, {})
    jobs.append({
        "job_id": job_id,
        "started_at": summary_job.get("started_at") or started_jobs.get(job_id, {}).get("at"),
        "finished_at": summary_job.get("finished_at") or finished.get("at"),
        "exit_code": summary_job.get("exit_code") if isinstance(summary_job.get("exit_code"), int) else finished.get("exit_code"),
        "duration_seconds": summary_job.get("duration_seconds") if isinstance(summary_job.get("duration_seconds"), (int, float)) else finished.get("duration_seconds"),
        "timed_out": summary_job.get("timed_out") if isinstance(summary_job.get("timed_out"), bool) else finished.get("timed_out"),
    })

failed_job_count = sum(1 for job in jobs if (job.get("exit_code") or 0) != 0 or job.get("timed_out") is True)
if finished_at or any(event.get("event") == "queue_finished" for event in events):
    state = "complete" if exit_code == 0 else "failed"
elif current_job_ids:
    state = "running"
elif any(event.get("event") == "queue_started" for event in events):
    state = "waiting"
else:
    state = "unknown"

job_count = event_job_count if isinstance(event_job_count, int) else len(summary_jobs) if summary_jobs else None
ok = len(parse_errors) == 0 and len(missing_files) < 2 and state != "unknown"
print(json.dumps({
    "schema_version": 1,
    "ok": ok,
    "mode": "shadow_queue_status",
    "source": "local_queue_artifacts",
    "queue_dir": queue_dir.as_posix(),
    "queue_name": queue_name or queue_dir.name,
    "run_id": run_id,
    "state": state,
    "current_job_id": current_job_ids[0] if current_job_ids else None,
    "current_job_ids": current_job_ids,
    "started_at": started_at,
    "finished_at": finished_at,
    "exit_code": exit_code,
    "job_count": job_count,
    "started_job_count": len(started_jobs),
    "finished_job_count": len(finished_jobs),
    "failed_job_count": failed_job_count,
    "status_event_count": len(events),
    "summary_job_count": len(summary_jobs),
    "jobs": jobs,
    "missing_files": missing_files,
    "parse_errors": parse_errors,
    "notes": ["Python local queue artifact parser; read-only."],
}, sort_keys=True))
`;

export async function buildManifestParityReport(input: {
  manifestPaths: string[];
  repoRoot: string;
  pythonExecutable?: string;
}): Promise<ManifestParityReport> {
  const pythonExecutable = input.pythonExecutable ?? "python3";
  const repoRoot = resolve(input.repoRoot);
  const rows: ManifestParityRow[] = [];

  for (const rawManifestPath of input.manifestPaths) {
    const manifestPath = resolve(rawManifestPath);
    const text = await readFile(manifestPath, "utf8");
    const payload = JSON.parse(text) as unknown;
    const ts = inspectControlManifest(payload);
    const python = validateWithPython({
      pythonExecutable,
      repoRoot,
      manifestPath,
    });

    if ("process_error" in python) {
      rows.push({
        manifest_path: manifestPath,
        experiment_id: ts.experiment_id,
        ts_ok: ts.ok,
        python_ok: null,
        status: "python_error",
        ts_error_count: ts.errors.length,
        python_error_count: null,
        ts_warning_count: ts.warnings.length,
        python_warning_count: null,
        ts_errors: ts.errors,
        python_errors: [python.process_error],
        ts_warnings: ts.warnings,
        python_warnings: [],
      });
      continue;
    }

    const status = ts.ok === python.ok ? "match" : "difference";
    rows.push({
      manifest_path: manifestPath,
      experiment_id: ts.experiment_id ?? python.experiment_id,
      ts_ok: ts.ok,
      python_ok: python.ok,
      status,
      ts_error_count: ts.errors.length,
      python_error_count: python.errors.length,
      ts_warning_count: ts.warnings.length,
      python_warning_count: python.warnings.length,
      ts_errors: ts.errors,
      python_errors: python.errors,
      ts_warnings: ts.warnings,
      python_warnings: python.warnings,
    });
  }

  const matchCount = rows.filter((row) => row.status === "match").length;
  const differenceCount = rows.filter((row) => row.status === "difference").length;
  const pythonErrorCount = rows.filter((row) => row.status === "python_error").length;
  return {
    schema_version: 1,
    ok: differenceCount === 0 && pythonErrorCount === 0,
    mode: "shadow_manifest_parity",
    python_validation: "validate_manifest_no_clean_git_no_prerequisites",
    manifest_count: rows.length,
    match_count: matchCount,
    difference_count: differenceCount,
    python_error_count: pythonErrorCount,
    rows,
  };
}

export async function buildArtifactInventoryParityReport(input: {
  manifestPaths: string[];
  repoRoot: string;
  pythonExecutable?: string;
}): Promise<ArtifactInventoryParityReport> {
  const pythonExecutable = input.pythonExecutable ?? "python3";
  const repoRoot = resolve(input.repoRoot);
  const rows: ArtifactInventoryParityRow[] = [];

  for (const rawManifestPath of input.manifestPaths) {
    const manifestPath = resolve(rawManifestPath);
    const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as Record<string, unknown>;
    const ts = await buildArtifactInventory({ manifest, repoRoot });
    const python = buildPythonArtifactInventory({ pythonExecutable, repoRoot, manifestPath });

    if ("process_error" in python) {
      rows.push({
        manifest_path: manifestPath,
        experiment_id: ts.experiment_id,
        ts_status: ts.status,
        python_status: null,
        status: "python_error",
        ts_required_count: ts.required_files.length,
        python_required_count: null,
        ts_present_count: ts.present_files.length,
        python_present_count: null,
        ts_missing_count: ts.missing_files.length,
        python_missing_count: null,
        ts_validation_error_count: ts.validation_errors.length,
        python_validation_error_count: null,
        missing_diff: ts.missing_files,
        validation_error_diff: ts.validation_errors,
        python_errors: [python.process_error],
      });
      continue;
    }

    const missingDiff = symmetricDiff(ts.missing_files, python.missing_files);
    const validationErrorDiff = symmetricDiff(ts.validation_errors, python.validation_errors);
    const rowMatches =
      ts.status === python.status &&
      ts.required_files.length === python.required_files.length &&
      ts.present_files.length === python.present_files.length &&
      ts.missing_files.length === python.missing_files.length &&
      ts.validation_errors.length === python.validation_errors.length &&
      missingDiff.length === 0 &&
      validationErrorDiff.length === 0;
    rows.push({
      manifest_path: manifestPath,
      experiment_id: ts.experiment_id,
      ts_status: ts.status,
      python_status: python.status,
      status: rowMatches ? "match" : "difference",
      ts_required_count: ts.required_files.length,
      python_required_count: python.required_files.length,
      ts_present_count: ts.present_files.length,
      python_present_count: python.present_files.length,
      ts_missing_count: ts.missing_files.length,
      python_missing_count: python.missing_files.length,
      ts_validation_error_count: ts.validation_errors.length,
      python_validation_error_count: python.validation_errors.length,
      missing_diff: missingDiff,
      validation_error_diff: validationErrorDiff,
      python_errors: [],
    });
  }

  const matchCount = rows.filter((row) => row.status === "match").length;
  const differenceCount = rows.filter((row) => row.status === "difference").length;
  const pythonErrorCount = rows.filter((row) => row.status === "python_error").length;
  return {
    schema_version: 1,
    ok: differenceCount === 0 && pythonErrorCount === 0,
    mode: "shadow_artifact_inventory_parity",
    python_inventory: "build_artifact_inventory_read_only",
    manifest_count: rows.length,
    match_count: matchCount,
    difference_count: differenceCount,
    python_error_count: pythonErrorCount,
    rows,
  };
}

export async function buildReportGateParityReport(input: {
  manifestPaths: string[];
  repoRoot: string;
  pythonExecutable?: string;
}): Promise<ReportGateParityReport> {
  const pythonExecutable = input.pythonExecutable ?? "python3";
  const repoRoot = resolve(input.repoRoot);
  const rows: ReportGateParityRow[] = [];

  for (const rawManifestPath of input.manifestPaths) {
    const manifestPath = resolve(rawManifestPath);
    const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as Record<string, unknown>;
    const inventory = await buildArtifactInventory({ manifest, repoRoot });
    const ts = buildFormalEvidenceStub({ manifest, inventory });
    const python = buildPythonReportGate({ pythonExecutable, repoRoot, manifestPath });

    if ("process_error" in python) {
      rows.push({
        manifest_path: manifestPath,
        experiment_id: ts.experiment_id,
        ts_strict_gate: ts.strict_gate,
        python_strict_gate: null,
        status: "python_error",
        ts_report_relative_path: ts.report_relative_path,
        python_report_relative_path: null,
        ts_report_sha256: ts.report_sha256,
        python_report_sha256: null,
        differences: [],
        python_errors: [python.process_error],
      });
      continue;
    }

    const differences = reportGateDifferences(ts, python);
    rows.push({
      manifest_path: manifestPath,
      experiment_id: ts.experiment_id,
      ts_strict_gate: ts.strict_gate,
      python_strict_gate: python.strict_gate,
      status: differences.length === 0 ? "match" : "difference",
      ts_report_relative_path: ts.report_relative_path,
      python_report_relative_path: python.report_relative_path,
      ts_report_sha256: ts.report_sha256,
      python_report_sha256: python.report_sha256,
      differences,
      python_errors: [],
    });
  }

  const matchCount = rows.filter((row) => row.status === "match").length;
  const differenceCount = rows.filter((row) => row.status === "difference").length;
  const pythonErrorCount = rows.filter((row) => row.status === "python_error").length;
  return {
    schema_version: 1,
    ok: differenceCount === 0 && pythonErrorCount === 0,
    mode: "shadow_report_gate_parity",
    python_report_gate: "write_formal_evidence_stub_tempdir",
    manifest_count: rows.length,
    match_count: matchCount,
    difference_count: differenceCount,
    python_error_count: pythonErrorCount,
    rows,
  };
}

export async function buildQueueStatusParityReport(input: {
  queueDirs: string[];
  pythonExecutable?: string;
}): Promise<QueueStatusParityReport> {
  const pythonExecutable = input.pythonExecutable ?? "python3";
  const rows: QueueStatusParityRow[] = [];

  for (const rawQueueDir of input.queueDirs) {
    const queueDir = resolve(rawQueueDir);
    const ts = await buildLocalQueueStatus(queueDir);
    const python = buildPythonQueueStatus({ pythonExecutable, queueDir });
    if ("process_error" in python) {
      rows.push({
        queue_dir: queueDir,
        queue_name: ts.queue_name,
        ts_state: ts.state,
        python_state: null,
        status: "python_error",
        differences: [],
        python_errors: [python.process_error],
      });
      continue;
    }
    const differences = queueStatusDifferences(ts, python);
    rows.push({
      queue_dir: queueDir,
      queue_name: ts.queue_name ?? python.queue_name,
      ts_state: ts.state,
      python_state: python.state,
      status: differences.length === 0 ? "match" : "difference",
      differences,
      python_errors: [],
    });
  }

  const matchCount = rows.filter((row) => row.status === "match").length;
  const differenceCount = rows.filter((row) => row.status === "difference").length;
  const pythonErrorCount = rows.filter((row) => row.status === "python_error").length;
  return {
    schema_version: 1,
    ok: differenceCount === 0 && pythonErrorCount === 0,
    mode: "shadow_queue_status_parity",
    python_status: "local_queue_status_read_only",
    queue_count: rows.length,
    match_count: matchCount,
    difference_count: differenceCount,
    python_error_count: pythonErrorCount,
    rows,
  };
}

function validateWithPython(input: {
  pythonExecutable: string;
  repoRoot: string;
  manifestPath: string;
}): PythonManifestValidationResult | { process_error: string } {
  const result = spawnSync(
    input.pythonExecutable,
    ["-c", pythonValidationSnippet, input.repoRoot, input.manifestPath],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    return {
      process_error: [
        `python validation exited with status ${result.status}`,
        result.stderr.trim(),
        result.stdout.trim(),
      ].filter(Boolean).join(": "),
    };
  }
  try {
    const payload = JSON.parse(result.stdout) as PythonManifestValidationResult;
    return {
      ok: payload.ok === true,
      experiment_id: typeof payload.experiment_id === "string" ? payload.experiment_id : null,
      errors: Array.isArray(payload.errors) ? payload.errors.filter((item): item is string => typeof item === "string") : [],
      warnings: Array.isArray(payload.warnings) ? payload.warnings.filter((item): item is string => typeof item === "string") : [],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { process_error: `python validation produced invalid JSON: ${message}: ${result.stdout.trim()}` };
  }
}

function buildPythonArtifactInventory(input: {
  pythonExecutable: string;
  repoRoot: string;
  manifestPath: string;
}): ArtifactInventory | { process_error: string } {
  const result = spawnSync(
    input.pythonExecutable,
    ["-c", pythonInventorySnippet, input.repoRoot, input.manifestPath],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    return {
      process_error: [
        `python inventory exited with status ${result.status}`,
        result.stderr.trim(),
        result.stdout.trim(),
      ].filter(Boolean).join(": "),
    };
  }
  try {
    const payload = JSON.parse(result.stdout) as ArtifactInventory;
    return {
      schema_version: 1,
      experiment_id: String(payload.experiment_id),
      artifact_contract: typeof payload.artifact_contract === "object" && payload.artifact_contract !== null
        ? payload.artifact_contract as Record<string, unknown>
        : {},
      artifact_contract_source: String(payload.artifact_contract_source),
      required_files: arrayOfStrings(payload.required_files),
      present_files: arrayOfStrings(payload.present_files),
      missing_files: arrayOfStrings(payload.missing_files),
      validation_errors: arrayOfStrings(payload.validation_errors),
      provenance_warnings: arrayOfStrings(payload.provenance_warnings),
      status: payload.status === "complete" ? "complete" : "incomplete",
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { process_error: `python inventory produced invalid JSON: ${message}: ${result.stdout.trim()}` };
  }
}

function buildPythonReportGate(input: {
  pythonExecutable: string;
  repoRoot: string;
  manifestPath: string;
}): FormalEvidenceStub | { process_error: string } {
  const result = spawnSync(
    input.pythonExecutable,
    ["-c", pythonReportGateSnippet, input.repoRoot, input.manifestPath],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    return {
      process_error: [
        `python report gate exited with status ${result.status}`,
        result.stderr.trim(),
        result.stdout.trim(),
      ].filter(Boolean).join(": "),
    };
  }
  try {
    const payload = JSON.parse(result.stdout) as FormalEvidenceStub;
    return {
      schema_version: 1,
      experiment_id: String(payload.experiment_id),
      strict_gate: payload.strict_gate === "complete" ? "complete" : "incomplete",
      report_relative_path: String(payload.report_relative_path),
      report_text: String(payload.report_text),
      report_sha256: String(payload.report_sha256),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { process_error: `python report gate produced invalid JSON: ${message}: ${result.stdout.trim()}` };
  }
}

function buildPythonQueueStatus(input: {
  pythonExecutable: string;
  queueDir: string;
}): QueueStatusReport | { process_error: string } {
  const result = spawnSync(input.pythonExecutable, ["-c", pythonQueueStatusSnippet, input.queueDir], { encoding: "utf8" });
  if (result.status !== 0) {
    return {
      process_error: [
        `python queue status exited with status ${result.status}`,
        result.stderr.trim(),
        result.stdout.trim(),
      ].filter(Boolean).join(": "),
    };
  }
  try {
    const payload = JSON.parse(result.stdout) as QueueStatusReport;
    return payload;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { process_error: `python queue status produced invalid JSON: ${message}: ${result.stdout.trim()}` };
  }
}

function queueStatusDifferences(left: QueueStatusReport, right: QueueStatusReport): string[] {
  const differences: string[] = [];
  const keys: Array<keyof Pick<
    QueueStatusReport,
    | "ok"
    | "queue_name"
    | "run_id"
    | "state"
    | "current_job_id"
    | "started_at"
    | "finished_at"
    | "exit_code"
    | "job_count"
    | "started_job_count"
    | "finished_job_count"
    | "failed_job_count"
    | "status_event_count"
    | "summary_job_count"
  >> = [
    "ok",
    "queue_name",
    "run_id",
    "state",
    "current_job_id",
    "started_at",
    "finished_at",
    "exit_code",
    "job_count",
    "started_job_count",
    "finished_job_count",
    "failed_job_count",
    "status_event_count",
    "summary_job_count",
  ];
  for (const key of keys) {
    if (left[key] !== right[key]) {
      differences.push(`${key}: ts=${String(left[key])} python=${String(right[key])}`);
    }
  }
  if (symmetricDiff(left.current_job_ids, right.current_job_ids).length > 0) {
    differences.push(`current_job_ids: ${symmetricDiff(left.current_job_ids, right.current_job_ids).join(",")}`);
  }
  if (symmetricDiff(left.jobs.map((job) => job.job_id), right.jobs.map((job) => job.job_id)).length > 0) {
    differences.push(`jobs: ${symmetricDiff(left.jobs.map((job) => job.job_id), right.jobs.map((job) => job.job_id)).join(",")}`);
  }
  return differences;
}

function reportGateDifferences(left: FormalEvidenceStub, right: FormalEvidenceStub): string[] {
  const differences: string[] = [];
  if (left.strict_gate !== right.strict_gate) {
    differences.push(`strict_gate: ts=${left.strict_gate} python=${right.strict_gate}`);
  }
  if (left.report_relative_path !== right.report_relative_path) {
    differences.push(`report_relative_path: ts=${left.report_relative_path} python=${right.report_relative_path}`);
  }
  if (left.report_sha256 !== right.report_sha256) {
    differences.push(`report_sha256: ts=${left.report_sha256} python=${right.report_sha256}`);
  }
  return differences;
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function symmetricDiff(left: string[], right: string[]): string[] {
  const leftSet = new Set(left);
  const rightSet = new Set(right);
  return [
    ...left.filter((item) => !rightSet.has(item)).map((item) => `ts_only:${item}`),
    ...right.filter((item) => !leftSet.has(item)).map((item) => `python_only:${item}`),
  ].sort();
}
