import { createHash } from "node:crypto";
import path from "node:path/posix";

import type { ArtifactInventory, FormalEvidenceStub } from "./contracts.ts";

export function buildFormalEvidenceStub(input: {
  manifest: Record<string, unknown>;
  inventory: ArtifactInventory;
}): FormalEvidenceStub {
  const experimentId = requiredString(input.manifest.experiment_id, "experiment_id");
  const artifactContract = input.inventory.artifact_contract ?? asRecord(input.manifest.artifacts) ?? {};
  const reportDirs = arrayOfStrings(artifactContract.report_dirs);
  const reportDir = reportDirs.length > 0 ? stripTrailingSlash(reportDirs[0]) : `artifacts/reports/${experimentId}`;
  const reportRelativePath = path.join(reportDir, "formal_evidence.md");
  const lines = [
    `# ${experimentId} Formal Evidence`,
    "",
    `strict_gate: ${input.inventory.status}`,
    "",
    "This is an automatically generated control-plane stub.",
    "Final model capability analysis must wait for complete fetched artifacts and paired statistics.",
    "",
    "## Missing Files",
    "",
    ...sectionLines(input.inventory.missing_files, true),
    "",
    "## Validation Errors",
    "",
    ...sectionLines(input.inventory.validation_errors, false),
    "",
    "## Provenance Warnings",
    "",
    ...sectionLines(input.inventory.provenance_warnings, false),
  ];
  const reportText = `${lines.join("\n")}\n`;
  return {
    schema_version: 1,
    experiment_id: experimentId,
    strict_gate: input.inventory.status,
    report_relative_path: reportRelativePath,
    report_text: reportText,
    report_sha256: createHash("sha256").update(reportText, "utf8").digest("hex"),
  };
}

function sectionLines(items: string[], codeFormat: boolean): string[] {
  if (items.length === 0) {
    return ["- none"];
  }
  return codeFormat ? items.map((item) => `- \`${item}\``) : items.map((item) => `- ${item}`);
}

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/u, "");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${field} must be a non-empty string`);
  }
  return value;
}
