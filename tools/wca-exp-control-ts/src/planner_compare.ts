import type {
  PlanSpec,
  PlannerResult,
  PlannerStrategyComparisonReport,
  PlannerStrategyComparisonRow,
  SchedulingStrategy,
} from "./contracts.ts";
import { planFromSpec } from "./planner.ts";

const comparedStrategies: SchedulingStrategy[] = ["priority_lpt", "resource_best_fit_v1"];

export function comparePlannerStrategies(spec: PlanSpec): PlannerStrategyComparisonReport {
  const rows = comparedStrategies.map((strategy) => comparisonRow(strategy, planFromSpec(withStrategy(spec, strategy))));
  const okRows = rows.filter((row) => row.ok);
  const warnings: string[] = [
    "planner-compare is a local shadow report; it does not submit jobs, reserve GPUs, or prove global optimality.",
  ];
  if (!spec.resource_profile_rows || spec.resource_profile_rows.length === 0) {
    warnings.push("comparison uses declared job estimates only because no resource_profile_rows were provided");
  }

  return {
    schema_version: 1,
    ok: rows.every((row) => row.ok),
    mode: "shadow_planner_strategy_comparison",
    compared_strategy_count: rows.length,
    baseline_strategy: "priority_lpt",
    best_by_makespan: bestStrategy(okRows, (row) => row.expected_makespan_seconds),
    best_by_memory_waste: bestStrategy(
      okRows.filter((row) => row.expected_memory_waste_mb_slot_seconds !== null),
      (row) => row.expected_memory_waste_mb_slot_seconds ?? Number.POSITIVE_INFINITY,
    ),
    rows,
    warnings,
  };
}

function withStrategy(spec: PlanSpec, strategy: SchedulingStrategy): PlanSpec {
  return {
    ...spec,
    policy: {
      ...(spec.policy ?? {}),
      scheduling_strategy: strategy,
    },
    manifest: spec.manifest
      ? {
          ...spec.manifest,
          planner: spec.manifest.planner
            ? {
                ...spec.manifest.planner,
                policy: {
                  ...(spec.manifest.planner.policy ?? {}),
                  scheduling_strategy: strategy,
                },
              }
            : spec.manifest.planner,
        }
      : spec.manifest,
  };
}

function comparisonRow(strategy: SchedulingStrategy, result: PlannerResult): PlannerStrategyComparisonRow {
  const decision = result.scheduler_decision;
  const memoryWasteMetric = decision.planner_objective.metrics.find(
    (metric) => metric.name === "expected_memory_waste_mb_slot_seconds",
  );
  return {
    strategy,
    ok: decision.ok,
    planned_job_count: decision.planned_job_count,
    rejected_job_count: decision.rejected_job_count,
    resource_lock_conflict_count: decision.resource_lock_conflict_count,
    expected_makespan_seconds: decision.expected_makespan_seconds,
    expected_idle_slot_seconds: decision.expected_idle_slot_seconds,
    expected_idle_fraction: decision.expected_idle_fraction,
    expected_mean_gpu_utilization: decision.expected_mean_gpu_utilization,
    expected_min_memory_headroom_mb: decision.expected_min_memory_headroom_mb,
    expected_memory_waste_mb_slot_seconds: memoryWasteMetric?.value ?? null,
    objective_id: decision.planner_objective.objective_id,
    ...(decision.reason ? { reason: decision.reason } : {}),
  };
}

function bestStrategy(
  rows: PlannerStrategyComparisonRow[],
  metric: (row: PlannerStrategyComparisonRow) => number,
): SchedulingStrategy | null {
  if (rows.length === 0) {
    return null;
  }
  return [...rows].sort((left, right) => {
    const metricDelta = metric(left) - metric(right);
    if (metricDelta !== 0) {
      return metricDelta;
    }
    return strategyTieBreak(left.strategy) - strategyTieBreak(right.strategy);
  })[0].strategy;
}

function strategyTieBreak(strategy: SchedulingStrategy): number {
  return comparedStrategies.indexOf(strategy);
}
