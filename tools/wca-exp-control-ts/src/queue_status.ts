import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

import type { QueueStatusJobSummary, QueueStatusReport, QueueStatusState } from "./contracts.ts";

interface StatusEvent {
  event?: string;
  at?: string;
  queue_name?: string;
  run_id?: string;
  job_count?: number;
  job_id?: string;
  exit_code?: number;
  duration_seconds?: number;
  timed_out?: boolean;
}

interface QueueSummaryJob {
  id?: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number;
  duration_seconds?: number;
  timed_out?: boolean;
}

interface QueueSummaryPayload {
  queue_name?: string;
  run_id?: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number;
  jobs?: QueueSummaryJob[];
}

export async function buildLocalQueueStatus(queueDirInput: string): Promise<QueueStatusReport> {
  const queueDir = resolve(queueDirInput);
  const missingFiles: string[] = [];
  const parseErrors: string[] = [];
  const statusPath = join(queueDir, "status.jsonl");
  const summaryPath = join(queueDir, "queue_summary.json");
  const statusEvents = await readStatusEvents(statusPath, missingFiles, parseErrors);
  const summary = await readQueueSummary(summaryPath, missingFiles, parseErrors);

  const startedJobs = new Map<string, StatusEvent>();
  const finishedJobs = new Map<string, StatusEvent>();
  let queueName: string | null = summary?.queue_name ?? null;
  let runId: string | null = summary?.run_id ?? null;
  let startedAt: string | null = summary?.started_at ?? null;
  let finishedAt: string | null = summary?.finished_at ?? null;
  let exitCode: number | null = typeof summary?.exit_code === "number" ? summary.exit_code : null;
  let eventJobCount: number | null = null;

  for (const event of statusEvents) {
    if (event.queue_name && !queueName) {
      queueName = event.queue_name;
    }
    if (event.run_id && !runId) {
      runId = event.run_id;
    }
    if (event.event === "queue_started") {
      startedAt = startedAt ?? event.at ?? null;
      eventJobCount = typeof event.job_count === "number" ? event.job_count : eventJobCount;
    } else if (event.event === "queue_finished") {
      finishedAt = event.at ?? finishedAt;
      exitCode = typeof event.exit_code === "number" ? event.exit_code : exitCode;
    } else if (event.event === "job_started" && event.job_id) {
      startedJobs.set(event.job_id, event);
    } else if (event.event === "job_finished" && event.job_id) {
      finishedJobs.set(event.job_id, event);
    }
  }

  const currentJobIds = [...startedJobs.keys()].filter((jobId) => !finishedJobs.has(jobId)).sort();
  const summaryJobs = Array.isArray(summary?.jobs) ? summary.jobs : [];
  const jobs = mergeJobSummaries(summaryJobs, startedJobs, finishedJobs);
  const failedJobCount = jobs.filter((job) => (job.exit_code ?? 0) !== 0 || job.timed_out === true).length;
  const state = determineState({
    statusEvents,
    currentJobIds,
    finishedAt,
    exitCode,
    missingFiles,
  });
  const jobCount = typeof eventJobCount === "number" ? eventJobCount : summaryJobs.length > 0 ? summaryJobs.length : null;
  const ok = parseErrors.length === 0 && missingFiles.length < 2 && state !== "unknown";

  return {
    schema_version: 1,
    ok,
    mode: "shadow_queue_status",
    source: "local_queue_artifacts",
    queue_dir: queueDir,
    queue_name: queueName ?? basename(queueDir),
    run_id: runId,
    state,
    current_job_id: currentJobIds[0] ?? null,
    current_job_ids: currentJobIds,
    started_at: startedAt,
    finished_at: finishedAt,
    exit_code: exitCode,
    job_count: jobCount,
    started_job_count: startedJobs.size,
    finished_job_count: finishedJobs.size,
    failed_job_count: failedJobCount,
    status_event_count: statusEvents.length,
    summary_job_count: summaryJobs.length,
    jobs,
    missing_files: missingFiles,
    parse_errors: parseErrors,
    notes: [
      "This is a local queue artifact parser. It does not SSH, fetch, submit, cancel, or mutate remote state.",
    ],
  };
}

async function readStatusEvents(
  statusPath: string,
  missingFiles: string[],
  parseErrors: string[],
): Promise<StatusEvent[]> {
  const text = await readOptionalText(statusPath, missingFiles);
  if (text === null) {
    return [];
  }
  const events: StatusEvent[] = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const payload = JSON.parse(trimmed) as unknown;
      if (typeof payload !== "object" || payload === null) {
        parseErrors.push(`status.jsonl line ${index + 1} is not an object`);
        continue;
      }
      events.push(payload as StatusEvent);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      parseErrors.push(`status.jsonl line ${index + 1}: ${message}`);
    }
  }
  return events;
}

async function readQueueSummary(
  summaryPath: string,
  missingFiles: string[],
  parseErrors: string[],
): Promise<QueueSummaryPayload | null> {
  const text = await readOptionalText(summaryPath, missingFiles);
  if (text === null) {
    return null;
  }
  try {
    const payload = JSON.parse(text) as unknown;
    if (typeof payload !== "object" || payload === null) {
      parseErrors.push("queue_summary.json is not an object");
      return null;
    }
    return payload as QueueSummaryPayload;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    parseErrors.push(`queue_summary.json: ${message}`);
    return null;
  }
}

async function readOptionalText(path: string, missingFiles: string[]): Promise<string | null> {
  try {
    return await readFile(path, "utf8");
  } catch (error) {
    if (isNodeErrorWithCode(error, "ENOENT")) {
      missingFiles.push(path);
      return null;
    }
    throw error;
  }
}

function mergeJobSummaries(
  summaryJobs: QueueSummaryJob[],
  startedJobs: Map<string, StatusEvent>,
  finishedJobs: Map<string, StatusEvent>,
): QueueStatusJobSummary[] {
  const ids = new Set<string>();
  for (const job of summaryJobs) {
    if (job.id) {
      ids.add(job.id);
    }
  }
  for (const jobId of startedJobs.keys()) {
    ids.add(jobId);
  }
  for (const jobId of finishedJobs.keys()) {
    ids.add(jobId);
  }

  const summaryById = new Map(summaryJobs.filter((job): job is QueueSummaryJob & { id: string } => Boolean(job.id)).map((job) => [job.id, job]));
  return [...ids].sort().map((jobId) => {
    const summary = summaryById.get(jobId);
    const started = startedJobs.get(jobId);
    const finished = finishedJobs.get(jobId);
    return {
      job_id: jobId,
      started_at: summary?.started_at ?? started?.at ?? null,
      finished_at: summary?.finished_at ?? finished?.at ?? null,
      exit_code: typeof summary?.exit_code === "number" ? summary.exit_code : typeof finished?.exit_code === "number" ? finished.exit_code : null,
      duration_seconds:
        typeof summary?.duration_seconds === "number"
          ? summary.duration_seconds
          : typeof finished?.duration_seconds === "number"
            ? finished.duration_seconds
            : null,
      timed_out:
        typeof summary?.timed_out === "boolean"
          ? summary.timed_out
          : typeof finished?.timed_out === "boolean"
            ? finished.timed_out
            : null,
    };
  });
}

function determineState(input: {
  statusEvents: StatusEvent[];
  currentJobIds: string[];
  finishedAt: string | null;
  exitCode: number | null;
  missingFiles: string[];
}): QueueStatusState {
  if (input.finishedAt || input.statusEvents.some((event) => event.event === "queue_finished")) {
    return input.exitCode === 0 ? "complete" : "failed";
  }
  if (input.currentJobIds.length > 0) {
    return "running";
  }
  if (input.statusEvents.some((event) => event.event === "queue_started")) {
    return "waiting";
  }
  if (input.missingFiles.length >= 2 && input.statusEvents.length === 0) {
    return "unknown";
  }
  return "unknown";
}

function isNodeErrorWithCode(error: unknown, code: string): boolean {
  return typeof error === "object" && error !== null && (error as { code?: unknown }).code === code;
}

