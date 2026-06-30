import { readdir, readFile, stat } from "node:fs/promises";
import { basename, dirname, relative, resolve } from "node:path";

import type {
  ResourceProfileExtractionReport,
  ResourceProfileExtractionSkip,
  ResourceProfileRow,
} from "./contracts.ts";
import { validateResourceProfileRow } from "./profiles.ts";

export async function extractResourceProfiles(input: {
  sourceRoots: string[];
  repoRoot: string;
  gpuModel?: string;
  gpuCount?: number;
  gpuSlots?: number;
}): Promise<ResourceProfileExtractionReport> {
  const repoRoot = resolve(input.repoRoot);
  const summaryPaths = (await Promise.all(input.sourceRoots.map((root) => findSummaryFiles(resolve(repoRoot, root)))))
    .flat()
    .sort();
  const rows: ResourceProfileRow[] = [];
  const skipped: ResourceProfileExtractionSkip[] = [];

  for (const summaryPath of summaryPaths) {
    const result = await extractProfileFromSummary({
      summaryPath,
      repoRoot,
      gpuModel: input.gpuModel ?? "unknown",
      gpuCount: input.gpuCount ?? 1,
      gpuSlots: input.gpuSlots ?? 1,
    });
    if ("reason" in result) {
      skipped.push({ summary_path: relative(repoRoot, summaryPath), reason: result.reason });
      continue;
    }
    const validation = validateResourceProfileRow(result.row);
    if (!validation.ok) {
      skipped.push({
        summary_path: relative(repoRoot, summaryPath),
        reason: `invalid extracted profile row: ${validation.errors.join("; ")}`,
      });
      continue;
    }
    rows.push(result.row);
  }

  return {
    schema_version: 1,
    ok: true,
    source_roots: input.sourceRoots,
    scanned_summary_count: summaryPaths.length,
    extracted_count: rows.length,
    skipped_count: skipped.length,
    rows,
    skipped,
  };
}

async function extractProfileFromSummary(input: {
  summaryPath: string;
  repoRoot: string;
  gpuModel: string;
  gpuCount: number;
  gpuSlots: number;
}): Promise<{ row: ResourceProfileRow } | { reason: string }> {
  const summary = await readJsonObject(input.summaryPath);
  const summaryStat = await stat(input.summaryPath);
  const config = asRecord(summary.config);
  if (!config) {
    return { reason: "summary.config is missing" };
  }

  const trainLogPath = resolve(dirname(input.summaryPath), "train_log.csv");
  const trainLog = await readTrainLogMetrics(trainLogPath);
  if (!trainLog) {
    return { reason: "train_log.csv is missing or has no rows" };
  }

  const peakMemoryMb = trainLog.maxNumeric("cuda_peak_memory_allocated_mb");
  const reservedMemoryMb = trainLog.maxNumeric("cuda_peak_memory_reserved_mb");
  const stepSeconds = trainLog.numericValues("step_seconds");
  if (!isPositiveFinite(peakMemoryMb)) {
    return { reason: "cuda_peak_memory_allocated_mb is missing from train_log.csv" };
  }
  if (stepSeconds.length === 0) {
    return { reason: "step_seconds is missing from train_log.csv" };
  }

  const runDir = stringValue(summary.run_dir) ?? stringValue(config.run_dir) ?? relative(input.repoRoot, dirname(input.summaryPath));
  const modelFamily = modelFamilyFromSummary(summary, config);
  const modelVariant = modelVariantFromSummary(summary, config, modelFamily);
  const datasetId = datasetIdFromConfig(config);
  const splitId = splitIdFromConfig(config);
  const nodesOrTokens = numberValue(config.n_nodes) ?? numberValue(summary.final_metrics?.field_patch_count);
  const batchSize = numberValue(config.batch_size);
  const precision = stringValue(config.precision);
  if (!isPositiveFinite(nodesOrTokens)) {
    return { reason: "n_nodes or final field_patch_count is missing" };
  }
  if (!isPositiveFinite(batchSize)) {
    return { reason: "batch_size is missing" };
  }
  if (!precision) {
    return { reason: "precision is missing" };
  }

  const trainStepCount = stepSeconds.length;
  const wallClockSeconds = sum(stepSeconds);
  const meanStepSeconds = wallClockSeconds / trainStepCount;
  const samplesPerSecond = batchSize / meanStepSeconds;
  const sourceRunId = normalizeId(runDir);
  const row: ResourceProfileRow = {
    schema_version: 1,
    profile_id: `${sourceRunId}:profile`,
    source_run_id: runDir,
    observed_at_epoch_seconds: Math.floor(summaryStat.mtimeMs / 1000),
    recipe_id: recipeIdFromParts({
      modelFamily,
      modelVariant,
      datasetId,
      nodesOrTokens,
      batchSize,
      precision,
      supportMode: supportModeFromConfig(config),
      pairKernel: pairKernelFromSummary(summary, config, modelFamily),
    }),
    model_family: modelFamily,
    model_variant: modelVariant,
    dataset_id: datasetId,
    split_id: splitId,
    gpu_model: input.gpuModel,
    gpu_count: input.gpuCount,
    gpu_slots: input.gpuSlots,
    nodes_or_tokens: nodesOrTokens,
    batch_size: batchSize,
    precision,
    peak_memory_mb: peakMemoryMb,
    wall_clock_seconds: wallClockSeconds,
    train_step_count: trainStepCount,
    step_time_ms: meanStepSeconds * 1000,
    samples_per_second: samplesPerSecond,
    oom: false,
    notes: `extracted_from=train_log.csv; summary_path=${relative(input.repoRoot, input.summaryPath)}`,
  };

  const steadyMemoryMb = trainLog.lastNumeric("cuda_memory_allocated_mb");
  if (isPositiveFinite(steadyMemoryMb)) {
    row.steady_memory_mb = steadyMemoryMb;
  } else if (isPositiveFinite(reservedMemoryMb)) {
    row.steady_memory_mb = reservedMemoryMb;
  }
  const hiddenDim = modelFamily === "wca" ? numberValue(config.hidden_dim) : undefined;
  if (isPositiveFinite(hiddenDim)) {
    row.hidden_dim = hiddenDim;
  }
  const outerSteps = modelFamily === "wca" ? numberValue(config.outer_steps) : undefined;
  if (isPositiveFinite(outerSteps)) {
    row.outer_steps = outerSteps;
  }
  const innerSteps = modelFamily === "wca" ? numberValue(config.inner_steps) : undefined;
  if (isPositiveFinite(innerSteps)) {
    row.inner_steps = innerSteps;
  }
  const supportMode = supportModeFromConfig(config);
  if (supportMode) {
    row.support_mode = supportMode;
  }
  const pairKernel = pairKernelFromSummary(summary, config, modelFamily);
  if (pairKernel) {
    row.pair_kernel = pairKernel;
  }
  return { row };
}

async function findSummaryFiles(root: string): Promise<string[]> {
  if (!(await pathExists(root))) {
    return [];
  }
  const rootStat = await stat(root);
  if (rootStat.isFile()) {
    return basename(root) === "summary.json" ? [root] : [];
  }
  const entries = await readdir(root, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const fullPath = resolve(root, entry.name);
    if (entry.isDirectory()) {
      return findSummaryFiles(fullPath);
    }
    return entry.isFile() && entry.name === "summary.json" ? [fullPath] : [];
  }));
  return nested.flat();
}

async function readJsonObject(path: string): Promise<Record<string, unknown>> {
  return JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
}

interface TrainLogMetrics {
  numericValues(column: string): number[];
  maxNumeric(column: string): number | undefined;
  lastNumeric(column: string): number | undefined;
}

async function readTrainLogMetrics(path: string): Promise<TrainLogMetrics | null> {
  if (!(await pathExists(path))) {
    return null;
  }
  const text = await readFile(path, "utf8");
  const rows = parseCsv(text);
  if (rows.length === 0) {
    return null;
  }
  return {
    numericValues: (column: string) => rows.map((row) => parseNumber(row[column])).filter(isPositiveFinite),
    maxNumeric: (column: string) => {
      const values = rows.map((row) => parseNumber(row[column])).filter(isPositiveFinite);
      return values.length > 0 ? Math.max(...values) : undefined;
    },
    lastNumeric: (column: string) => {
      for (let index = rows.length - 1; index >= 0; index -= 1) {
        const value = parseNumber(rows[index][column]);
        if (isPositiveFinite(value)) {
          return value;
        }
      }
      return undefined;
    },
  };
}

function parseCsv(text: string): Array<Record<string, string>> {
  const lines = text.split(/\r?\n/u).filter((line) => line.length > 0);
  if (lines.length < 2) {
    return [];
  }
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function modelFamilyFromSummary(summary: Record<string, unknown>, config: Record<string, unknown>): string {
  const baselineModel = stringValue(config.baseline_model);
  if (baselineModel) {
    return baselineModel;
  }
  const model = stringValue(summary.model)?.toLowerCase() ?? "";
  return model.includes("wca") || model.includes("recursiveworld") ? "wca" : "unknown";
}

function modelVariantFromSummary(summary: Record<string, unknown>, config: Record<string, unknown>, modelFamily: string): string {
  if (modelFamily === "wca") {
    const checkpointed = Boolean(config.activation_checkpoint_inner);
    return `dense_o${numberValue(config.outer_steps) ?? "na"}_i${numberValue(config.inner_steps) ?? "na"}${checkpointed ? "_checkpointed" : ""}`;
  }
  const width = numberValue(config.baseline_width) ?? "na";
  const depth = numberValue(config.baseline_depth) ?? "na";
  const modes = numberValue(config.fno_modes);
  return modes ? `w${width}_d${depth}_m${modes}` : `w${width}_d${depth}`;
}

function datasetIdFromConfig(config: Record<string, unknown>): string {
  const dataset = stringValue(config.field_dataset) ?? stringValue(config.task) ?? "unknown";
  const pathValue = stringValue(config.field_data_path);
  if (!pathValue) {
    return dataset;
  }
  return `${dataset}:${basename(pathValue)}`;
}

function splitIdFromConfig(config: Record<string, unknown>): string {
  const trainStart = numberValue(config.field_train_start) ?? 0;
  const trainSize = numberValue(config.field_train_size) ?? 0;
  const evalStart = numberValue(config.field_eval_start) ?? 0;
  const evalSize = numberValue(config.field_eval_size) ?? 0;
  return `train${trainStart}_${trainSize}:eval${evalStart}_${evalSize}`;
}

function supportModeFromConfig(config: Record<string, unknown>): string {
  const adjacency = stringValue(config.field_adjacency_mode) ?? "unknown";
  const scope = stringValue(config.field_input_scope) ?? "unknown";
  return `${adjacency}_${scope}`;
}

function pairKernelFromSummary(summary: Record<string, unknown>, config: Record<string, unknown>, modelFamily: string): string {
  if (modelFamily !== "wca") {
    return modelFamily;
  }
  const structural = stringValue(summary.structural_invariant)?.toLowerCase() ?? "";
  if (structural.includes("dense")) {
    return `dense_mlp_chunk${numberValue(config.pair_chunk_size) ?? "na"}`;
  }
  return "unknown_wca_pair_kernel";
}

function recipeIdFromParts(input: {
  modelFamily: string;
  modelVariant: string;
  datasetId: string;
  nodesOrTokens: number;
  batchSize: number;
  precision: string;
  supportMode: string;
  pairKernel: string;
}): string {
  return normalizeId([
    input.modelFamily,
    input.modelVariant,
    input.datasetId,
    `n${input.nodesOrTokens}`,
    `b${input.batchSize}`,
    input.precision,
    input.supportMode,
    input.pairKernel,
  ].join("_"));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function parseNumber(value: unknown): number | undefined {
  if (typeof value !== "string" || value.length === 0 || value.toLowerCase() === "nan") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function isPositiveFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

function normalizeId(value: string): string {
  return value
    .replace(/[^a-zA-Z0-9._-]+/gu, "_")
    .replace(/_+/gu, "_")
    .replace(/^_+|_+$/gu, "");
}
