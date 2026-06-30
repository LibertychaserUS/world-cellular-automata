import { resolve } from "node:path";

import type {
  ArtifactInventoryParityReport,
  FetchPackageValidationReport,
  ManifestParityReport,
  MigrationReadinessGate,
  MigrationReadinessReport,
  QueueStatusParityReport,
  RemoteStatusSnapshotValidationReport,
  ReportGateParityReport,
  RuntimeLeaseStoreValidationReport,
} from "./contracts.ts";
import { validateFetchPackageFile } from "./fetch_package.ts";
import { validateRuntimeLeaseStoreFile } from "./lease_store.ts";
import {
  buildArtifactInventoryParityReport,
  buildManifestParityReport,
  buildQueueStatusParityReport,
  buildReportGateParityReport,
} from "./parity.ts";
import { validateRemoteStatusSnapshotFile } from "./remote_status_snapshot.ts";

export async function buildMigrationReadinessReport(input: {
  manifestPaths: string[];
  queueDirs: string[];
  repoRoot: string;
  pythonExecutable?: string;
  leaseStorePaths?: string[];
  fetchPackagePaths?: string[];
  remoteStatusSnapshotPaths?: string[];
}): Promise<MigrationReadinessReport> {
  const repoRoot = resolve(input.repoRoot);
  const manifestPaths = input.manifestPaths.map((path) => resolve(path));
  const queueDirs = input.queueDirs.map((path) => resolve(path));
  const leaseStorePaths = (input.leaseStorePaths ?? []).map((path) => resolve(path));
  const fetchPackagePaths = (input.fetchPackagePaths ?? []).map((path) => resolve(path));
  const remoteStatusSnapshotPaths = (input.remoteStatusSnapshotPaths ?? []).map((path) => resolve(path));
  const gates: MigrationReadinessGate[] = [];

  gates.push(await manifestParityGate({ manifestPaths, repoRoot, pythonExecutable: input.pythonExecutable }));
  gates.push(await inventoryParityGate({ manifestPaths, repoRoot, pythonExecutable: input.pythonExecutable }));
  gates.push(await reportGateParityGate({ manifestPaths, repoRoot, pythonExecutable: input.pythonExecutable }));
  gates.push(await queueStatusParityGate({ queueDirs, pythonExecutable: input.pythonExecutable }));
  gates.push(await runtimeLeaseStoreValidationGate({ leaseStorePaths }));
  gates.push(await fetchPackageValidationGate({ fetchPackagePaths }));
  gates.push(await remoteStatusSnapshotValidationGate({ remoteStatusSnapshotPaths }));
  gates.push(staticPassGate({
    gate_id: "active_mutation_commands_disabled",
    reason: "TS CLI rejects active run/submit/fetch/cancel/kill and remote command aliases.",
    evidence: {
      cli_mode: "shadow_read_only",
      active_commands_denied: ["run", "submit", "fetch", "cancel", "kill", "remote-status", "remote-fetch", "remote-submit"],
    },
  }));
  gates.push(staticFailGate({
    gate_id: "active_submit_fetch_not_implemented",
    reason: "Active submit/fetch remains Python-only until audited live remote parity exists.",
    evidence: { python_remains_active_authority: true },
  }));
  gates.push(staticFailGate({
    gate_id: "runtime_lease_enforcement_not_active",
    reason: "Validated runtime lease stores are local shadow contracts only and do not enforce remote GPU/process ownership.",
    evidence: { lease_enforcement: "shadow_only", validated_lease_store_count: leaseStorePaths.length },
  }));
  gates.push(staticFailGate({
    gate_id: "live_remote_read_only_status_missing",
    reason: "TS validates remote status snapshots locally but has no audited live SSH read-only status reader.",
    evidence: { status_source: "local_or_snapshot_only", validated_remote_status_snapshot_count: remoteStatusSnapshotPaths.length },
  }));
  gates.push(staticFailGate({
    gate_id: "report_first_fetch_parity_missing",
    reason: "TS validates already-fetched packages locally but has no audited live report-first fetch/archive transport matching Python.",
    evidence: { fetch_transport: "not_implemented", validated_fetch_package_count: fetchPackagePaths.length },
  }));

  const blockers = gates
    .filter((gate) => gate.required_for_active_takeover && gate.status !== "pass")
    .map((gate) => `${gate.gate_id}: ${gate.reason}`);
  const passingGateCount = gates.filter((gate) => gate.status === "pass").length;

  return {
    schema_version: 1,
    ok: true,
    mode: "shadow_migration_readiness",
    execution_state: "shadow_only",
    active_takeover_allowed: false,
    recommendation: "keep_python_active",
    manifest_count: manifestPaths.length,
    queue_count: queueDirs.length,
    gate_count: gates.length,
    passing_gate_count: passingGateCount,
    blocking_gate_count: blockers.length,
    gates,
    blockers,
    notes: [
      "This report is diagnostic evidence only; ok=true means the readiness report generated successfully.",
      "Python remains the active queue authority; this TS package cannot approve active takeover in Slice I1.",
    ],
  };
}

async function runtimeLeaseStoreValidationGate(input: {
  leaseStorePaths: string[];
}): Promise<MigrationReadinessGate> {
  if (input.leaseStorePaths.length === 0) {
    return {
      gate_id: "runtime_lease_store_validation",
      status: "fail",
      required_for_active_takeover: true,
      reason: "At least one local shadow runtime lease store validation report is required before migration review.",
      evidence: { lease_store_count: 0 },
    };
  }
  return validationFilesGate<RuntimeLeaseStoreValidationReport>({
    gate_id: "runtime_lease_store_validation",
    paths: input.leaseStorePaths,
    build: validateRuntimeLeaseStoreFile,
    summarize: (reports) => ({
      ok: reports.every((report) => report.ok),
      reason: reports.every((report) => report.ok)
        ? "All supplied local runtime lease stores passed shadow validation."
        : "One or more supplied local runtime lease stores failed shadow validation.",
      evidence: {
        mode: "shadow_runtime_lease_store_validation",
        lease_store_count: reports.length,
        failing_lease_store_count: reports.filter((report) => !report.ok).length,
        total_lease_count: sum(reports.map((report) => report.lease_count)),
        total_invalid_lease_count: sum(reports.map((report) => report.invalid_lease_count)),
        total_duplicate_runtime_lease_id_count: sum(reports.map((report) => report.duplicate_runtime_lease_id_count)),
        total_duplicate_resource_lease_id_count: sum(reports.map((report) => report.duplicate_resource_lease_id_count)),
      },
    }),
  });
}

async function fetchPackageValidationGate(input: {
  fetchPackagePaths: string[];
}): Promise<MigrationReadinessGate> {
  if (input.fetchPackagePaths.length === 0) {
    return {
      gate_id: "fetch_package_validation",
      status: "fail",
      required_for_active_takeover: true,
      reason: "At least one local report-first or raw-complete fetch package validation report is required before migration review.",
      evidence: { fetch_package_count: 0 },
    };
  }
  return validationFilesGate<FetchPackageValidationReport>({
    gate_id: "fetch_package_validation",
    paths: input.fetchPackagePaths,
    build: validateFetchPackageFile,
    summarize: (reports) => ({
      ok: reports.every((report) => report.ok),
      reason: reports.every((report) => report.ok)
        ? "All supplied local fetch packages passed shadow validation."
        : "One or more supplied local fetch packages failed shadow validation.",
      evidence: {
        mode: "shadow_fetch_package_validation",
        fetch_package_count: reports.length,
        failing_fetch_package_count: reports.filter((report) => !report.ok).length,
        report_first_count: reports.filter((report) => report.fetch_mode === "report_first").length,
        raw_complete_count: reports.filter((report) => report.fetch_mode === "raw_complete").length,
        total_missing_count: sum(reports.map((report) => report.missing_count)),
        total_missing_local_file_count: sum(reports.map((report) => report.missing_local_file_count)),
        archive_integrity_failure_count: reports.filter((report) => (
          report.archive_integrity !== null && !report.archive_integrity.extraction_allowed
        )).length,
      },
    }),
  });
}

async function remoteStatusSnapshotValidationGate(input: {
  remoteStatusSnapshotPaths: string[];
}): Promise<MigrationReadinessGate> {
  if (input.remoteStatusSnapshotPaths.length === 0) {
    return {
      gate_id: "remote_status_snapshot_validation",
      status: "fail",
      required_for_active_takeover: true,
      reason: "At least one local remote status snapshot validation report is required before live read-only status review.",
      evidence: { remote_status_snapshot_count: 0 },
    };
  }
  return validationFilesGate<RemoteStatusSnapshotValidationReport>({
    gate_id: "remote_status_snapshot_validation",
    paths: input.remoteStatusSnapshotPaths,
    build: validateRemoteStatusSnapshotFile,
    summarize: (reports) => ({
      ok: reports.every((report) => report.ok),
      reason: reports.every((report) => report.ok)
        ? "All supplied local remote status snapshots passed shadow validation."
        : "One or more supplied local remote status snapshots failed shadow validation.",
      evidence: {
        mode: "shadow_remote_status_snapshot_validation",
        remote_status_snapshot_count: reports.length,
        failing_remote_status_snapshot_count: reports.filter((report) => !report.ok).length,
        running_snapshot_count: reports.filter((report) => report.queue_state === "running").length,
        complete_snapshot_count: reports.filter((report) => report.queue_state === "complete").length,
        total_error_count: sum(reports.map((report) => report.error_count)),
        total_warning_count: sum(reports.map((report) => report.warning_count)),
      },
    }),
  });
}

async function manifestParityGate(input: {
  manifestPaths: string[];
  repoRoot: string;
  pythonExecutable?: string;
}): Promise<MigrationReadinessGate> {
  return parityGate<ManifestParityReport>({
    gate_id: "manifest_parity",
    required_for_active_takeover: true,
    emptyFailReason: "At least one control manifest is required for migration readiness.",
    build: () => buildManifestParityReport(input),
    summarize: (report) => ({
      ok: report.ok,
      reason: report.ok
        ? "TS manifest validation matches the Python oracle for all supplied manifests."
        : "TS manifest validation differs from the Python oracle or Python validation failed.",
      evidence: {
        mode: report.mode,
        manifest_count: report.manifest_count,
        match_count: report.match_count,
        difference_count: report.difference_count,
        python_error_count: report.python_error_count,
      },
    }),
  });
}

async function inventoryParityGate(input: {
  manifestPaths: string[];
  repoRoot: string;
  pythonExecutable?: string;
}): Promise<MigrationReadinessGate> {
  return parityGate<ArtifactInventoryParityReport>({
    gate_id: "artifact_inventory_parity",
    required_for_active_takeover: true,
    emptyFailReason: "At least one control manifest is required for artifact inventory parity.",
    build: () => buildArtifactInventoryParityReport(input),
    summarize: (report) => ({
      ok: report.ok,
      reason: report.ok
        ? "TS artifact inventory matches the Python read-only inventory oracle."
        : "TS artifact inventory differs from Python or Python inventory failed.",
      evidence: {
        mode: report.mode,
        manifest_count: report.manifest_count,
        match_count: report.match_count,
        difference_count: report.difference_count,
        python_error_count: report.python_error_count,
      },
    }),
  });
}

async function reportGateParityGate(input: {
  manifestPaths: string[];
  repoRoot: string;
  pythonExecutable?: string;
}): Promise<MigrationReadinessGate> {
  return parityGate<ReportGateParityReport>({
    gate_id: "report_gate_parity",
    required_for_active_takeover: true,
    emptyFailReason: "At least one control manifest is required for report-gate parity.",
    build: () => buildReportGateParityReport(input),
    summarize: (report) => ({
      ok: report.ok,
      reason: report.ok
        ? "TS formal evidence stub gate matches the Python report-gate oracle."
        : "TS formal evidence stub gate differs from Python or Python report-gate failed.",
      evidence: {
        mode: report.mode,
        manifest_count: report.manifest_count,
        match_count: report.match_count,
        difference_count: report.difference_count,
        python_error_count: report.python_error_count,
      },
    }),
  });
}

async function queueStatusParityGate(input: {
  queueDirs: string[];
  pythonExecutable?: string;
}): Promise<MigrationReadinessGate> {
  if (input.queueDirs.length === 0) {
    return {
      gate_id: "queue_status_parity",
      status: "fail",
      required_for_active_takeover: true,
      reason: "At least one already-fetched local queue directory is required for queue-status parity.",
      evidence: { queue_count: 0 },
    };
  }
  return parityGate<QueueStatusParityReport>({
    gate_id: "queue_status_parity",
    required_for_active_takeover: true,
    emptyFailReason: "At least one already-fetched local queue directory is required for queue-status parity.",
    build: () => buildQueueStatusParityReport(input),
    summarize: (report) => ({
      ok: report.ok,
      reason: report.ok
        ? "TS local queue-status parser matches the Python local artifact oracle."
        : "TS local queue-status parser differs from Python or Python queue-status failed.",
      evidence: {
        mode: report.mode,
        queue_count: report.queue_count,
        match_count: report.match_count,
        difference_count: report.difference_count,
        python_error_count: report.python_error_count,
      },
    }),
  });
}

async function parityGate<T>(input: {
  gate_id: string;
  required_for_active_takeover: boolean;
  emptyFailReason: string;
  build: () => Promise<T>;
  summarize: (report: T) => { ok: boolean; reason: string; evidence: Record<string, unknown> };
}): Promise<MigrationReadinessGate> {
  try {
    const report = await input.build();
    const summary = input.summarize(report);
    return {
      gate_id: input.gate_id,
      status: summary.ok ? "pass" : "fail",
      required_for_active_takeover: input.required_for_active_takeover,
      reason: summary.reason,
      evidence: summary.evidence,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      gate_id: input.gate_id,
      status: "fail",
      required_for_active_takeover: input.required_for_active_takeover,
      reason: `${input.emptyFailReason} ${message}`,
      evidence: { error: message },
    };
  }
}

async function validationFilesGate<T>(input: {
  gate_id: string;
  paths: string[];
  build: (path: string) => Promise<T>;
  summarize: (reports: T[]) => { ok: boolean; reason: string; evidence: Record<string, unknown> };
}): Promise<MigrationReadinessGate> {
  try {
    const reports = await Promise.all(input.paths.map((path) => input.build(path)));
    const summary = input.summarize(reports);
    return {
      gate_id: input.gate_id,
      status: summary.ok ? "pass" : "fail",
      required_for_active_takeover: true,
      reason: summary.reason,
      evidence: {
        ...summary.evidence,
        checked_paths: input.paths,
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      gate_id: input.gate_id,
      status: "fail",
      required_for_active_takeover: true,
      reason: `Local validation failed unexpectedly. ${message}`,
      evidence: { error: message, checked_paths: input.paths },
    };
  }
}

function staticPassGate(input: {
  gate_id: string;
  reason: string;
  evidence: Record<string, unknown>;
}): MigrationReadinessGate {
  return {
    gate_id: input.gate_id,
    status: "pass",
    required_for_active_takeover: true,
    reason: input.reason,
    evidence: input.evidence,
  };
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function staticFailGate(input: {
  gate_id: string;
  reason: string;
  evidence: Record<string, unknown>;
}): MigrationReadinessGate {
  return {
    gate_id: input.gate_id,
    status: "fail",
    required_for_active_takeover: true,
    reason: input.reason,
    evidence: input.evidence,
  };
}
