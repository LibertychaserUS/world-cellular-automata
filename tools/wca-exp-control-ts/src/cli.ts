#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { validateFetchPackageFile } from "./fetch_package.ts";
import { validateKernelEquivalenceGateFile } from "./kernel_equivalence.ts";
import { validateRuntimeLeaseStoreFile } from "./lease_store.ts";
import { inspectControlManifest } from "./manifest.ts";
import { runMockActiveControlFile } from "./mock_active_control.ts";
import {
  buildArtifactInventoryParityReport,
  buildManifestParityReport,
  buildQueueStatusParityReport,
  buildReportGateParityReport,
} from "./parity.ts";
import { comparePlannerStrategies } from "./planner_compare.ts";
import { planFromSpec } from "./planner.ts";
import { extractResourceProfiles } from "./profile_extract.ts";
import { auditResourceProfileLibrary } from "./profiles.ts";
import { buildLocalQueueStatus } from "./queue_status.ts";
import { validateRemoteStatusSnapshotFile } from "./remote_status_snapshot.ts";
import { buildMigrationReadinessReport } from "./readiness.ts";
import { prepareShadowOutputFile, REPO_ROOT } from "./shadow_paths.ts";
import type { PlanSpec, ResourceProfileCoverageRequirement, ResourceProfileRow } from "./contracts.ts";

interface CliArgs {
  command:
    | "plan"
    | "planner-compare"
    | "inspect-manifest"
    | "parity-report"
    | "inventory-parity"
    | "report-gate-parity"
    | "extract-profiles"
    | "queue-status"
    | "queue-status-parity"
    | "migration-readiness"
    | "audit-profiles"
    | "lease-store-validate"
    | "fetch-package-validate"
    | "remote-status-snapshot-validate"
    | "kernel-equivalence-validate"
    | "mock-active-control";
  specPaths: string[];
  queueDirs?: string[];
  outputPath?: string;
  repoRoot?: string;
  pretty: boolean;
  gpuModel?: string;
  gpuCount?: number;
  gpuSlots?: number;
  profileRowsPath?: string;
  requirementsPath?: string;
  nowEpochSeconds?: number;
  leaseStorePaths?: string[];
  fetchPackagePaths?: string[];
  remoteStatusSnapshotPaths?: string[];
  requirePromotion: boolean;
}

async function main(argv: string[]): Promise<number> {
  const args = parseArgs(argv);
  const outputPath = args.outputPath ? await prepareShadowOutputFile(args.outputPath, "--output") : undefined;
  const result = await runCommand(args);
  const output = `${JSON.stringify(result, null, args.pretty ? 2 : 0)}\n`;

  if (outputPath) {
    await writeFile(outputPath, output, "utf8");
  } else {
    process.stdout.write(output);
  }

  return resultOk(result) ? 0 : 2;
}

async function runCommand(args: CliArgs): Promise<unknown> {
  if (args.command === "parity-report") {
    return buildManifestParityReport({
      manifestPaths: args.specPaths,
      repoRoot: args.repoRoot ?? REPO_ROOT,
    });
  }
  if (args.command === "inventory-parity") {
    return buildArtifactInventoryParityReport({
      manifestPaths: args.specPaths,
      repoRoot: args.repoRoot ?? REPO_ROOT,
    });
  }
  if (args.command === "report-gate-parity") {
    return buildReportGateParityReport({
      manifestPaths: args.specPaths,
      repoRoot: args.repoRoot ?? REPO_ROOT,
    });
  }
  if (args.command === "queue-status-parity") {
    return buildQueueStatusParityReport({
      queueDirs: args.specPaths,
    });
  }
  if (args.command === "migration-readiness") {
    return buildMigrationReadinessReport({
      manifestPaths: args.specPaths,
      queueDirs: args.queueDirs ?? [],
      repoRoot: args.repoRoot ?? REPO_ROOT,
      leaseStorePaths: args.leaseStorePaths ?? [],
      fetchPackagePaths: args.fetchPackagePaths ?? [],
      remoteStatusSnapshotPaths: args.remoteStatusSnapshotPaths ?? [],
    });
  }
  if (args.command === "extract-profiles") {
    return extractResourceProfiles({
      sourceRoots: args.specPaths,
      repoRoot: args.repoRoot ?? REPO_ROOT,
      gpuModel: args.gpuModel,
      gpuCount: args.gpuCount,
      gpuSlots: args.gpuSlots,
    });
  }
  if (args.command === "audit-profiles") {
    return auditResourceProfileLibrary(
      await readProfileRowsFromPath(args.specPaths[0]),
      args.requirementsPath ? await readProfileRequirementsFromPath(args.requirementsPath) : [],
      { now_epoch_seconds: args.nowEpochSeconds },
    );
  }
  if (args.command === "lease-store-validate") {
    return validateRuntimeLeaseStoreFile(args.specPaths[0]);
  }
  if (args.command === "fetch-package-validate") {
    return validateFetchPackageFile(args.specPaths[0]);
  }
  if (args.command === "remote-status-snapshot-validate") {
    return validateRemoteStatusSnapshotFile(args.specPaths[0]);
  }
  if (args.command === "kernel-equivalence-validate") {
    const report = await validateKernelEquivalenceGateFile(args.specPaths[0]);
    if (args.requirePromotion && report.promotion_allowed !== true) {
      return {
        ...report,
        ok: false,
        errors: [
          ...report.errors,
          "--require-promotion requires evidence_status=promotion_allowed",
        ],
      };
    }
    return report;
  }
  if (args.command === "mock-active-control") {
    return runMockActiveControlFile(args.specPaths[0]);
  }
  if (args.command === "queue-status") {
    return buildLocalQueueStatus(args.specPaths[0]);
  }
  const specText = await readFile(args.specPaths[0], "utf8");
  const payload = JSON.parse(specText) as unknown;
  if (args.command === "inspect-manifest") {
    return inspectControlManifest(payload);
  }
  if (args.command === "planner-compare") {
    return comparePlannerStrategies(await withProfileRows(payload as PlanSpec, args.profileRowsPath));
  }
  return planFromSpec(await withProfileRows(payload as PlanSpec, args.profileRowsPath));
}

function resultOk(result: unknown): boolean {
  if (typeof result !== "object" || result === null) {
    return false;
  }
  if ("scheduler_decision" in result) {
    const schedulerDecision = (result as { scheduler_decision?: { ok?: unknown } }).scheduler_decision;
    return schedulerDecision?.ok === true;
  }
  return (result as { ok?: unknown }).ok === true;
}

function parseArgs(argv: string[]): CliArgs {
  const values = [...argv];
  const first = values.shift();
  if (!first || first === "--help" || first === "-h") {
    printHelp();
    process.exit(first ? 0 : 1);
  }
  assertNotActiveControlCommand(first);
  const command =
    first === "plan" ||
      first === "planner-compare" ||
      first === "inspect-manifest" ||
      first === "parity-report" ||
      first === "inventory-parity" ||
      first === "report-gate-parity" ||
      first === "extract-profiles" ||
      first === "queue-status" ||
      first === "queue-status-parity" ||
      first === "migration-readiness" ||
      first === "audit-profiles" ||
      first === "lease-store-validate" ||
      first === "fetch-package-validate" ||
      first === "remote-status-snapshot-validate" ||
      first === "kernel-equivalence-validate" ||
      first === "mock-active-control"
      ? first
      : "plan";
  const specPath = command === "plan" && first !== "plan" ? first : values.shift();
  if (!specPath) {
    throw new Error(`${command} requires a JSON path`);
  }
  const specPaths = [specPath];

  let outputPath: string | undefined;
  let repoRoot: string | undefined;
  let pretty = false;
  let gpuModel: string | undefined;
  let gpuCount: number | undefined;
  let gpuSlots: number | undefined;
  let profileRowsPath: string | undefined;
  let requirementsPath: string | undefined;
  let nowEpochSeconds: number | undefined;
  let requirePromotion = false;
  const queueDirs: string[] = [];
  const leaseStorePaths: string[] = [];
  const fetchPackagePaths: string[] = [];
  const remoteStatusSnapshotPaths: string[] = [];
  while (values.length > 0) {
    const value = values.shift();
    if (value === "--pretty") {
      pretty = true;
    } else if (value === "--output") {
      const next = values.shift();
      if (!next) {
        throw new Error("--output requires a path");
      }
      outputPath = next;
    } else if (value === "--repo-root") {
      const next = values.shift();
      if (!next) {
        throw new Error("--repo-root requires a path");
      }
      repoRoot = resolve(next);
    } else if (value === "--gpu-model") {
      gpuModel = requireNext(values, "--gpu-model");
    } else if (value === "--gpu-count") {
      gpuCount = parsePositiveInteger(requireNext(values, "--gpu-count"), "--gpu-count");
    } else if (value === "--gpu-slots") {
      gpuSlots = parsePositiveInteger(requireNext(values, "--gpu-slots"), "--gpu-slots");
    } else if (value === "--profile-rows") {
      profileRowsPath = resolve(requireNext(values, "--profile-rows"));
    } else if (value === "--requirements") {
      requirementsPath = resolve(requireNext(values, "--requirements"));
    } else if (value === "--now-epoch-seconds") {
      nowEpochSeconds = parseNonNegativeNumber(requireNext(values, "--now-epoch-seconds"), "--now-epoch-seconds");
    } else if (value === "--queue-dir") {
      queueDirs.push(resolve(requireNext(values, "--queue-dir")));
    } else if (value === "--lease-store") {
      leaseStorePaths.push(resolve(requireNext(values, "--lease-store")));
    } else if (value === "--fetch-package") {
      fetchPackagePaths.push(resolve(requireNext(values, "--fetch-package")));
    } else if (value === "--remote-status-snapshot") {
      remoteStatusSnapshotPaths.push(resolve(requireNext(values, "--remote-status-snapshot")));
    } else if (value === "--require-promotion") {
      requirePromotion = true;
    } else if (
      command === "parity-report" ||
      command === "inventory-parity" ||
      command === "report-gate-parity" ||
      command === "extract-profiles" ||
      command === "queue-status-parity" ||
      command === "migration-readiness"
    ) {
      specPaths.push(value ?? "");
    } else {
      throw new Error(`unknown argument: ${value}`);
    }
  }
  if (
    command !== "migration-readiness" &&
    (leaseStorePaths.length > 0 || fetchPackagePaths.length > 0 || remoteStatusSnapshotPaths.length > 0)
  ) {
    throw new Error("--lease-store, --fetch-package, and --remote-status-snapshot are only valid for migration-readiness");
  }
  if (requirePromotion && command !== "kernel-equivalence-validate") {
    throw new Error("--require-promotion is only valid for kernel-equivalence-validate");
  }

  return {
    command,
    specPaths: specPaths.map((path) => resolve(path)),
    queueDirs,
    outputPath,
    repoRoot,
    pretty,
    gpuModel,
    gpuCount,
    gpuSlots,
    profileRowsPath,
    requirementsPath,
    nowEpochSeconds,
    leaseStorePaths,
    fetchPackagePaths,
    remoteStatusSnapshotPaths,
    requirePromotion,
  };
}

function assertNotActiveControlCommand(command: string): void {
  const forbidden = new Set([
    "run",
    "submit",
    "fetch",
    "cancel",
    "kill",
    "remote-status",
    "remote-fetch",
    "remote-submit",
  ]);
  if (forbidden.has(command)) {
    throw new Error(
      `TS control plane is shadow/read-only only; active command '${command}' must use Python control plane until migration gates pass`,
    );
  }
}

function printHelp(): void {
  process.stdout.write(
    [
      "Usage:",
      "  node --experimental-strip-types src/cli.ts plan <plan-spec.json> [--profile-rows rows.json] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts planner-compare <plan-spec.json> [--profile-rows rows.json] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts inspect-manifest <control-manifest.json> [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts parity-report <control-manifest.json>... [--repo-root <repo>] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts inventory-parity <control-manifest.json>... [--repo-root <repo>] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts report-gate-parity <control-manifest.json>... [--repo-root <repo>] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts extract-profiles <run-root-or-summary.json>... [--repo-root <repo>] [--gpu-model <name>] [--gpu-count <n>] [--gpu-slots <n>] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts audit-profiles <profile-rows.json> [--requirements requirements.json] [--now-epoch-seconds <n>] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts lease-store-validate <lease-store.json> [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts fetch-package-validate <fetch-package.json> [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts remote-status-snapshot-validate <remote-status-snapshot.json> [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts kernel-equivalence-validate <optimization-gate.json> [--require-promotion] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts mock-active-control <mock-spec.json> [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts queue-status <local-queue-dir> [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts queue-status-parity <local-queue-dir>... [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts migration-readiness <control-manifest.json>... --queue-dir <local-queue-dir>... [--lease-store <lease-store.json>]... [--fetch-package <fetch-package.json>]... [--remote-status-snapshot <snapshot.json>]... [--repo-root <repo>] [--pretty] [--output result.json]",
      "  node --experimental-strip-types src/cli.ts <plan-spec.json> [--pretty]",
      "",
      "Dry-runs WCA GPU slot planning or inspects a control manifest.",
      "These commands do not submit remote jobs.",
      "",
    ].join("\n"),
  );
}

async function withProfileRows(spec: PlanSpec, profileRowsPath: string | undefined): Promise<PlanSpec> {
  if (!profileRowsPath) {
    return spec;
  }
  return {
    ...spec,
    resource_profile_rows: await readProfileRowsFromPath(profileRowsPath),
  };
}

async function readProfileRowsFromPath(path: string): Promise<ResourceProfileRow[]> {
  const payload = JSON.parse(await readFile(path, "utf8")) as unknown;
  const rows = Array.isArray(payload)
    ? payload
    : typeof payload === "object" && payload !== null && Array.isArray((payload as { rows?: unknown }).rows)
      ? (payload as { rows: unknown[] }).rows
      : null;
  if (!rows) {
    throw new Error("profile rows path must point to a JSON array or extract-profiles report with rows");
  }
  return rows as ResourceProfileRow[];
}

async function readProfileRequirementsFromPath(path: string): Promise<ResourceProfileCoverageRequirement[]> {
  const payload = JSON.parse(await readFile(path, "utf8")) as unknown;
  const requirements = Array.isArray(payload)
    ? payload
    : typeof payload === "object" && payload !== null && Array.isArray((payload as { requirements?: unknown }).requirements)
      ? (payload as { requirements: unknown[] }).requirements
      : null;
  if (!requirements) {
    throw new Error("--requirements must point to a JSON array or object with requirements");
  }
  return requirements as ResourceProfileCoverageRequirement[];
}

function requireNext(values: string[], flag: string): string {
  const next = values.shift();
  if (!next) {
    throw new Error(`${flag} requires a value`);
  }
  return next;
}

function parsePositiveInteger(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeNumber(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative finite number`);
  }
  return parsed;
}

main(process.argv.slice(2)).then(
  (exitCode) => {
    process.exitCode = exitCode;
  },
  (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  },
);
