/**
 * The Plan view: which produced tensors are divided into tasks, what each
 * family of tasks and the whole plan cost, what one task reads and computes,
 * and which producer tasks supply it.
 *
 * Every figure here is a function of the graph and the declared tiling. It is
 * exact for the plan as given, or an upper bound with the reason named. The
 * plan evaluation declines a family above its budget instead of reporting part
 * of a sum.
 */

import React, { useEffect, useMemo, useState } from "react";
import { nodeById, producerNode, type Node } from "../../core/graph";
import { byteFigure, regionSliceExprs, type Figure } from "../../core/metrics";
import { opLabel } from "../../core/ops/index";
import {
  planReport,
  supplyOf,
  type BoundaryDemand,
  type Computed,
  type Demand,
  type InterfaceReport,
  type PlanReport,
  type ProducerNeed,
  type Work,
} from "../../core/plan/interfaces";
import type { TaskRef, TilePlan } from "../../core/plan/plan";
import { tileBox, tileOrdinal, tiles, type TileFamily } from "../../core/plan/tile-family";
import { count, formatBoxIndices, unionOf } from "../../core/region";
import { FIGURE_MARK, fmt, formatBytes, formatFigure, formatIntensity } from "../../view/format";
import { demandDetail, demandSummary } from "../../view/demand";
import { describeTiling, WORK_ROWS, workCell, workCellTitle } from "../../view/plan-work";
import { boxColor, rgbCss } from "../../view/palette";
import { useDark, useStore } from "../../state/store";
import { DraftField } from "../chrome/DraftField";
import { sameTiles } from "../../state/history";
import { MAX_KEPT_PLANS } from "../../state/plan";
import {
  analysisWorkerAvailable,
  isAnalysisCancelled,
  planInWorker,
} from "../../state/analysis-worker-client";

/** Plans up to this many tasks in all are evaluated as soon as they are shown; larger ones on request. */
export const AUTO_PLAN_TASKS = 256;
/** The dependency matrix is drawn when both sides have at most this many tiles. */
export const MATRIX_MAX_SIDE = 64;

/**
 * Parse tile extents written as `64 × 64`, `64x64`, `64, 64` or `64 64`.
 * Returns null unless there is one positive integer per axis.
 */
/** @internal Pure parsing seam exported for tests. */
export function parseTileExtents(text: string, rank: number): number[] | null {
  if (rank === 0) return /^(?:scalar)?$/i.test(text.trim()) ? [] : null;
  const fields = text.trim().split(/\s*[×x,]\s*|\s+/i).filter(Boolean);
  if (fields.length !== rank || !fields.every((f) => /^\d+$/.test(f))) return null;
  const extents = fields.map(Number);
  return extents.every((e) => Number.isSafeInteger(e) && e >= 1) ? extents : null;
}

export const formatTileExtents = (tile: readonly number[]): string =>
  tile.length ? tile.join(" × ") : "scalar";

/** A task by tensor name and tile coordinate. The element slice goes beside it. */
export const taskName = (name: string, coord: readonly number[]): string =>
  coord.length ? `${name} (${coord.join(", ")})` : name;

function TileExtentsInput({
  tensorId,
  name,
  shape,
  tile,
}: {
  tensorId: string;
  name: string;
  shape: number[];
  tile: number[];
}): React.ReactElement {
  const setPlanTile = useStore((s) => s.setPlanTile);
  return (
    <DraftField
      value={formatTileExtents(tile)}
      parse={(text) => parseTileExtents(text, shape.length)}
      apply={(parsed) => {
        if (parsed.some((e, axis) => e !== tile[axis])) setPlanTile(tensorId, parsed);
      }}
      label={`${name} tile extents`}
      title="tile extents per axis; Enter or leaving the field applies, Escape abandons"
      invalidTitle={`expected ${shape.length} positive whole number${shape.length === 1 ? "" : "s"}, one per axis`}
      className="box-range"
    />
  );
}

/** The plan's evaluation: every family and their total, computed once per tiling. */
type PlanRun = {
  report: PlanReport | null;
  evaluating: boolean;
  error: string | null;
  /** Tasks across every family. */
  tasks: number;
  /** Whether the plan is evaluated without being asked. */
  auto: boolean;
  evaluate: () => void;
};

function usePlanRun(plan: TilePlan | null): PlanRun {
  const workerGraphId = useStore((s) => s.workerGraphId);
  const tasks = plan ? [...plan.families.values()].reduce((n, f) => n + f.count, 0) : 0;
  const auto = tasks <= AUTO_PLAN_TASKS;
  const workerEnabled = analysisWorkerAvailable();
  const tiles = useMemo(
    () => Object.fromEntries(
      [...(plan?.families ?? [])].map(([tensorId, family]) => [tensorId, [...family.tile]])
    ),
    [plan]
  );
  const analysisKey = useMemo(
    () => plan && JSON.stringify([workerGraphId ?? plan.graph.nodes.map((node) => node.id), tiles]),
    [plan, tiles, workerGraphId]
  );
  const automatic = useMemo(
    () => (plan && !workerEnabled && auto ? planReport(plan) : null),
    [auto, plan, workerEnabled]
  );
  const [run, setRun] = useState<{ plan: TilePlan; report: PlanReport } | null>(null);
  const [requestedKey, setRequestedKey] = useState<string | null>(null);
  const [workerRun, setWorkerRun] = useState<{
    key: string;
    report: PlanReport | null;
    error: string | null;
  } | null>(null);
  const shouldRunInWorker = !!plan && workerEnabled && (auto || requestedKey === analysisKey);
  /* A cancellation is the lane dropping this work, not an answer about it, and
     the question is unchanged - so ask again rather than leaving `evaluating`
     true with no dependency left that could retry it. This terminates: only
     one query holds the lane at a time, and the reuse sweep that shares it
     lives on a mutually exclusive inspector tab, so what cancels a plan run
     is a build finishing, once per build. */
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!shouldRunInWorker || !plan || !analysisKey) return;
    let live = true;
    setWorkerRun({ key: analysisKey, report: null, error: null });
    void planInWorker({ graphId: workerGraphId, graph: plan.graph, tiles }).then((report) => {
      if (live) setWorkerRun({ key: analysisKey, report, error: null });
    }).catch((error) => {
      if (!live) return;
      if (isAnalysisCancelled(error)) setAttempt((previous) => previous + 1);
      else setWorkerRun({
        key: analysisKey,
        report: null,
        error: error instanceof Error ? error.message : String(error),
      });
    });
    return () => { live = false; };
  }, [analysisKey, attempt, shouldRunInWorker, workerGraphId]);

  const requested = run && run.plan === plan ? run.report : null;
  const currentWorkerRun = workerRun?.key === analysisKey ? workerRun : null;
  return {
    report: automatic ?? requested ?? currentWorkerRun?.report ?? null,
    evaluating: shouldRunInWorker &&
      (!currentWorkerRun || (currentWorkerRun.report === null && !currentWorkerRun.error)),
    error: currentWorkerRun?.error ?? null,
    tasks,
    auto,
    evaluate: () => {
      if (!plan) return;
      if (workerEnabled) setRequestedKey(analysisKey);
      else setRun({ plan, report: planReport(plan) });
    },
  };
}

const flopText = (f: Figure) => formatFigure(f, fmt);

/** One family's work on a line: what it computes, moves, and the ratio of the two. */
function WorkLine({ work }: { work: Work }): React.ReactElement {
  const recomputed = (work.recomputed.value ?? 0) !== 0;
  return (
    <span className="plan-work muted">
      <span title={work.flops.reasons.join("; ") || undefined}>{flopText(work.flops)} FLOP</span>
      {recomputed && (
        <span title="work more than one task does">, {flopText(work.recomputed)} recomputed</span>
      )}
      {" · "}
      <span title={`each task's reads summed; ${formatFigure(work.readDistinct, formatBytes)} distinct`}>
        reads {formatFigure(work.read, formatBytes)}
      </span>
      {" · "}
      <span>writes {formatFigure(work.written, formatBytes)}</span>
      {" · "}
      <span title="FLOPs per byte read or written">{formatIntensity(work.intensity)}</span>
    </span>
  );
}

/**
 * The whole plan in the figures plans are compared by, beside the plans kept
 * for comparison. Every row is always shown, so the columns line up.
 */
function PlanTotals({ report }: { report: PlanReport | null }): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const planTiles = useStore((s) => s.planTiles);
  const kept = useStore((s) => s.keptPlans);
  const keepPlan = useStore((s) => s.keepPlan);
  const dropKeptPlan = useStore((s) => s.dropKeptPlan);
  const restoreKeptPlan = useStore((s) => s.restoreKeptPlan);
  const nameOf = (tensorId: string) => resolved.tensors[tensorId].name;
  const total = report?.total ?? null;

  const same = kept.find((k) => sameTiles(k.tiles, planTiles));
  const full = kept.length >= MAX_KEPT_PLANS;
  const keepTitle = !total
    ? "evaluate the plan to keep its figures"
    : same
      ? `this tiling is kept as P${same.id}`
      : full
        ? `at most ${MAX_KEPT_PLANS} plans are kept: remove one to keep this one`
        : "keep this plan's figures, to compare it with the next tiling";
  const columns = [
    ...kept.map((k) => ({ key: `P${k.id}`, work: k.total as Work | null, unwritten: k.unwritten })),
    { key: "current", work: total, unwritten: report?.unwritten ?? [] },
  ];
  const works = columns.flatMap((c) => (c.work ? [c.work] : []));
  const figures = works.flatMap((w) => WORK_ROWS.map((row) => row.figure(w)));

  return (
    <>
      <div className="plan-total-head">
        <span className="muted">whole plan</span>
        <button
          className="mini"
          onClick={() => report && keepPlan(report)}
          disabled={!total || !!same || full}
          title={keepTitle}
        >
          keep
        </button>
      </div>
      <div
        className="plan-compare"
        style={{ gridTemplateColumns: `auto repeat(${columns.length}, auto)` }}
      >
        {kept.length > 0 && (
          <>
            <span />
            {columns.map((c) => (
              <span key={c.key} className="plan-compare-head">{c.key}</span>
            ))}
          </>
        )}
        {WORK_ROWS.map((row) => (
          <React.Fragment key={row.label}>
            <span className="muted" title={row.title}>{row.label}</span>
            {columns.map((c) => (
              <span
                key={c.key}
                className="num"
                title={c.work ? workCellTitle(row, c.work, total) : undefined}
              >
                {c.work ? workCell(row, c.work) : "–"}
              </span>
            ))}
          </React.Fragment>
        ))}
        {/* A plan that skips an output looks cheaper for doing less of the
            program. The row is shown whenever any plan here does, so that
            reading stays beside the figures it qualifies. */}
        {columns.some((c) => c.work && c.unwritten.length > 0) && (
          <>
            <span
              className="muted"
              title="graph outputs the plan does not tile: its figures leave out their work"
            >
              not written
            </span>
            {columns.map((c) => (
              <span key={c.key} className="num">
                {c.work ? (c.unwritten.length ? c.unwritten.map(nameOf).join(", ") : "none") : "–"}
              </span>
            ))}
          </>
        )}
      </div>
      {kept.length > 0 && (
        <ul className="plan-list plan-kept">
          {kept.map((k) => (
            <li key={k.id}>
              <span className="plan-compare-head">P{k.id}</span>
              <span className="muted plan-kept-tiling">{describeTiling(k.tiles, nameOf)}</span>
              <button
                className="mini"
                onClick={() => restoreKeptPlan(k.id)}
                disabled={sameTiles(k.tiles, planTiles)}
                title="make this tiling the current plan"
              >
                restore
              </button>
              <button
                className="mini danger"
                aria-label={`stop keeping P${k.id}`}
                title={`stop keeping P${k.id}`}
                onClick={() => dropKeptPlan(k.id)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="hint">
        Logical figures for each plan: each task reads its own demand and each tiled tensor is
        written once. Not measured traffic; FLOPs follow each operation's cost formula.
      </p>
      {works.some((w) => w.flops.status === "unknown") ? (
        <p className="hint overlap">
          No FLOP total: some task computes an operation whose arithmetic is not modelled. The byte
          figures still hold.
        </p>
      ) : figures.some((f) => f.status === "upper" || f.status === "approximate") ? (
        <p className="hint overlap">
          Some reads are widened: ≤ is “no more than”, ~ moved in an unknown direction.
        </p>
      ) : null}
    </>
  );
}

function TiledTensors({ plan, run }: { plan: TilePlan | null; run: PlanRun }): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const setPlanTile = useStore((s) => s.setPlanTile);
  const families = plan ? [...plan.families.values()] : [];
  const reports = new Map(run.report?.families.map((r) => [r.tensorId, r]) ?? []);
  const keptCount = useStore((s) => s.keptPlans.length);
  // A report is about the plan it was evaluated for, and there is none to
  // show while nothing is tiled; kept plans are still shown, to restore one.
  const total = plan ? run.report?.total ?? null : null;
  const unwritten = run.report?.unwritten ?? [];
  return (
    <div className="ins-section">
      <div className="ins-title with-action">
        Tiled tensors
        {plan && !run.auto && !run.report && (
          <button
            className="mini"
            onClick={run.evaluate}
            disabled={run.evaluating}
            title="run one bounded query per task of every tiled tensor"
          >
            {run.evaluating ? "evaluating…" : "evaluate"}
          </button>
        )}
      </div>
      {families.length === 0 ? (
        <p className="hint">
          Draw a rectangle on a produced tensor to divide it, or click one to divide it at the size
          the canvas is drawing. Graph inputs have no tasks and cannot be tiled.
        </p>
      ) : (
        <>
        <p className="hint">
          Click a tile to inspect its task; the arrow keys step it. A tiled tensor is written to
          memory. Clear one and the tasks that read it compute it themselves; type new extents to
          divide it differently.
        </p>
        <ul className="plan-list">
          {families.map((family) => {
            const tensor = resolved.tensors[family.tensorId];
            const report = reports.get(family.tensorId);
            return (
              <li key={family.tensorId}>
                <code>{tensor.name}</code>
                <TileExtentsInput
                  tensorId={family.tensorId}
                  name={tensor.name}
                  shape={[...family.shape]}
                  tile={[...family.tile]}
                />
                <span className="muted">
                  {family.count} task{family.count === 1 ? "" : "s"}
                </span>
                <button
                  className="mini danger"
                  title={`stop tiling ${tensor.name}: the tasks that read it compute it instead`}
                  aria-label={`stop tiling ${tensor.name}`}
                  onClick={() => setPlanTile(family.tensorId, null)}
                >
                  ×
                </button>
                {families.length > 1 && report?.status === "evaluated" && (
                  <WorkLine work={report.work} />
                )}
                {report?.status === "over-budget" && (
                  <span className="plan-work muted">over the budget of {report.budget} tasks</span>
                )}
              </li>
            );
          })}
        </ul>
        {run.error ? (
          <p className="hint overlap">Plan analysis failed: {run.error}</p>
        ) : run.evaluating ? (
          <p className="hint">Evaluating one bounded query per task…</p>
        ) : !run.report ? (
          <p className="hint">
            {run.tasks} tasks in all · plans above {AUTO_PLAN_TASKS} tasks are evaluated on request.
          </p>
        ) : !total ? (
          <p className="hint">
            No totals: a tensor is over the task budget, and a total over part of the plan would
            understate it.
          </p>
        ) : null}
        {run.report && unwritten.length > 0 && (
          <p className="hint">
            Not written by this plan:{" "}
            {unwritten.map((id) => resolved.tensors[id].name).join(", ")}. Tile a graph output to
            count its work.
          </p>
        )}
        </>
      )}
      {(total || keptCount > 0) && <PlanTotals report={plan ? run.report : null} />}
    </div>
  );
}

function ProducerRow({
  need,
  name,
  family,
  onSelect,
}: {
  need: ProducerNeed;
  name: string;
  family: TileFamily;
  onSelect: (task: TaskRef) => void;
}): React.ReactElement {
  const used = need.used.value ?? 0;
  const share = need.volume ? (100 * used) / need.volume : 0;
  const slice = `${name}[${formatBoxIndices(tileBox(family, need.task.coord))}]`;
  const reasons = [...new Set(need.witnesses.flatMap((w) => w.region.reasons))];
  return (
    <li>
      <button
        className="plan-task"
        title={`inspect this task: ${slice}`}
        onClick={() => onSelect(need.task)}
      >
        {taskName(name, need.task.coord)}
      </button>
      <span className="muted" title={`reads ${formatFigure(need.used, String)} of the ${need.volume} elements in ${slice}`}>
        {FIGURE_MARK[need.used.status]}
        {share === 100 ? "whole tile" : `${share.toFixed(share < 10 ? 1 : 0)}% of tile`}
      </span>
      {!need.definite && (
        <span className="badge approx" title={`possible: reached only through widened demand (${reasons.join(", ")})`}>
          ≈
        </span>
      )}
    </li>
  );
}

/** Producer tasks listed before the rest are folded behind a control. */
export const PRODUCERS_SHOWN = 8;

/**
 * One tensor a task reads, with a line per operand slot that reads it.
 *
 * Grouped by tensor rather than by slot because the producers and the supplier
 * are facts about the tensor. Listed per slot, an operation reading one tensor
 * twice printed its whole producer list twice.
 */
function DemandGroup({
  tensorId,
  demands,
  node,
  plan,
  producers,
  onSelect,
}: {
  tensorId: string;
  demands: Demand[];
  node: Node;
  plan: TilePlan;
  producers: ProducerNeed[];
  onSelect: (task: TaskRef) => void;
}): React.ReactElement {
  const tensor = plan.graph.tensors[tensorId];
  const [all, setAll] = useState(false);
  const family = plan.families.get(tensorId);
  const mine = producers.filter((p) => p.task.tensorId === tensorId);
  const shown = all ? mine : mine.slice(0, PRODUCERS_SHOWN);
  // A task that computes an untiled tensor reads through that tensor's
  // operation, and such a slot is named by the tensor it computes: two matmuls
  // are both `matmul`, and the tensor is what the reader can find on the graph.
  const own = demands.every((d) => d.node === node.id);
  const named = demands.length > 1 || !own || node.inputs.length > 1;
  const slotLabel = (d: Demand) => {
    if (d.node === node.id) return `arg${d.slot}`;
    const reader = nodeById(plan.graph, d.node)!;
    return `arg${d.slot} of ${plan.graph.tensors[reader.outputs[0]].name}`;
  };

  // One element counts once however many slots read it, as the family figures
  // count it, so the bytes here and under the family mean the same thing.
  const union = unionOf(demands.map((d) => d.region));

  return (
    <div className="plan-demand">
      <div className="plan-demand-head">
        <code>{tensor.name}</code>
        <span className="muted">
          {named ? `${demands.map(slotLabel).join(", ")} · ` : ""}
          {formatFigure(byteFigure(tensor, count(union), union), formatBytes)}
        </span>
        {!union.exact && (
          <span className="badge approx" title={union.reasons.join("; ")}>≈</span>
        )}
      </div>
      {demands.map((d) => (
        <pre className="plan-slice" key={d.slot}>
          {regionSliceExprs(tensor.name, d.region).join("\n")}
        </pre>
      ))}
      {demands[0].supplier === "input" ? (
        <p className="hint">Graph input: read from memory, produced by no task.</p>
      ) : (
        family && (
          <>
            <p className="plan-need">
              needs {mine.length} of {family.count} {tensor.name} task{family.count === 1 ? "" : "s"}
            </p>
            <ul className="plan-list">
              {shown.map((need) => (
                <ProducerRow
                  key={tileOrdinal(family, need.task.coord)}
                  need={need}
                  name={tensor.name}
                  family={family}
                  onSelect={onSelect}
                />
              ))}
            </ul>
            {mine.length > PRODUCERS_SHOWN && (
              <button className="mini" onClick={() => setAll(!all)}>
                {all ? "show fewer" : `show all ${mine.length}`}
              </button>
            )}
          </>
        )
      )}
    </div>
  );
}

/**
 * Consumer tasks down, producer tiles across; a mark where the consumer reads
 * the producer. Solid for a definite dependency, hollow for a possible one.
 */
function DependencyMatrix({
  plan,
  consumer,
  producer,
  current,
  onSelect,
}: {
  plan: TilePlan;
  consumer: TileFamily;
  producer: TileFamily;
  current: number;
  onSelect: (task: TaskRef) => void;
}): React.ReactElement {
  const dark = useDark();
  const rows = useMemo(
    () =>
      [...tiles(consumer)].map((coord) => ({
        coord,
        needs: new Map(
          supplyOf(plan, { tensorId: consumer.tensorId, coord }).producers
            .filter((p) => p.task.tensorId === producer.tensorId)
            .map((p) => [tileOrdinal(producer, p.task.coord), p.definite] as const)
        ),
      })),
    [plan, consumer, producer]
  );
  const cell = Math.max(3, Math.min(10, Math.floor(260 / producer.count)));
  const width = cell * producer.count;
  const height = cell * consumer.count;
  const hue = rgbCss(boxColor(0, dark));
  const consumerName = plan.graph.tensors[consumer.tensorId].name;
  const producerName = plan.graph.tensors[producer.tensorId].name;
  const links = rows.reduce((n, row) => n + row.needs.size, 0);

  return (
    <svg
      className="plan-matrix"
      width={width}
      height={height}
      role="img"
      aria-label={`${consumer.count} ${consumerName} tasks by ${producer.count} ${producerName} tiles, ${links} dependencies`}
    >
      {rows.map((row, r) => (
        <g key={r} onClick={() => onSelect({ tensorId: consumer.tensorId, coord: row.coord })}>
          <title>{taskName(consumerName, row.coord)}: {row.needs.size} {producerName} tiles</title>
          <rect
            className={`plan-matrix-row${r === current ? " current" : ""}`}
            x={0}
            y={r * cell}
            width={width}
            height={cell}
          />
          {[...row.needs].map(([p, definite]) => (
            <rect
              key={p}
              x={p * cell + 0.5}
              y={r * cell + 0.5}
              width={cell - 1}
              height={cell - 1}
              fill={definite ? hue : "none"}
              stroke={hue}
              strokeWidth={definite ? 0 : 1}
            />
          ))}
        </g>
      ))}
    </svg>
  );
}

function BoundaryRow({
  row,
  plan,
  tasks,
}: {
  row: BoundaryDemand;
  plan: TilePlan;
  tasks: number;
}): React.ReactElement {
  const tensor = plan.graph.tensors[row.tensorId];
  const family = plan.families.get(row.tensorId);
  const fan = row.fanOut ? [...row.fanOut.values()] : [];
  const unread = family && row.fanOut ? family.count - row.fanOut.size : 0;
  return (
    <li>
      <code>{tensor.name}</code>
      <span className="muted" title={demandDetail(row, `${row.readers} of ${tasks} tasks`)}>
        {demandSummary(row)}
      </span>
      {!row.exact && (
        <span className="badge approx" title={row.reasons.join("; ")}>≈</span>
      )}
      {fan.length > 0 && (
        <span className="plan-fan muted">
          each tile read by {Math.min(...fan) === Math.max(...fan) ? fan[0] : `${Math.min(...fan)}–${Math.max(...fan)}`}{" "}
          task{Math.max(...fan) === 1 ? "" : "s"}
          {unread > 0 && `, ${unread} by none`}
        </span>
      )}
    </li>
  );
}

function FamilySection({
  plan,
  task,
  run,
}: {
  plan: TilePlan;
  task: TaskRef;
  run: PlanRun;
}): React.ReactElement {
  const selectPlanTask = useStore((s) => s.selectPlanTask);
  const family = plan.families.get(task.tensorId)!;
  const name = plan.graph.tensors[task.tensorId].name;
  const report: InterfaceReport | undefined =
    run.report?.families.find((r) => r.tensorId === task.tensorId);

  const matrixFor = (row: BoundaryDemand) => {
    const producer = plan.families.get(row.tensorId);
    return row.supplier === "tasks" &&
      producer &&
      producer.count <= MATRIX_MAX_SIDE &&
      family.count <= MATRIX_MAX_SIDE
      ? producer
      : null;
  };

  return (
    <div className="ins-section">
      <div className="ins-title">
        Across all {family.count} {name} task{family.count === 1 ? "" : "s"}
      </div>
      {run.error ? (
        <p className="hint overlap">Plan analysis failed: {run.error}</p>
      ) : run.evaluating ? (
        <p className="hint">Evaluating one bounded query per task…</p>
      ) : !report ? (
        <p className="hint">Evaluate the plan above to see what the family reads.</p>
      ) : report.status === "over-budget" ? (
        <p className="hint">
          {report.tasks} tasks is over the budget of {report.budget}. No totals are shown: a sum over
          part of the family would understate them.
        </p>
      ) : (
        <>
          <ul className="reuse-list">
            {report.boundary.map((row) => (
              <BoundaryRow key={row.tensorId} row={row} plan={plan} tasks={report.tasks} />
            ))}
          </ul>
          {report.boundary.map((row) => {
            const producer = matrixFor(row);
            return (
              producer && (
                <div className="plan-matrix-wrap" key={row.tensorId}>
                  <div className="plan-matrix-label muted">
                    {name} tasks ↓ · {plan.graph.tensors[row.tensorId].name} tiles →
                  </div>
                  <DependencyMatrix
                    plan={plan}
                    consumer={family}
                    producer={producer}
                    current={tileOrdinal(family, task.coord)}
                    onSelect={selectPlanTask}
                  />
                </div>
              )
            );
          })}
          <p className="hint">
            Demand is the mean number of tasks reading each element they read. It is not a cache hit
            rate or a count of memory transfers.
          </p>
        </>
      )}
    </div>
  );
}

/**
 * The untiled tensors a task computes on the way to its tile. Tiling one here
 * writes it to memory instead, and the task reads it from its producer tasks.
 */
function ComputedSection({ plan, computes }: { plan: TilePlan; computes: Computed[] }): React.ReactElement {
  const tilePlanTensor = useStore((s) => s.tilePlanTensor);
  return (
    <section className="ins-section">
      <div className="ins-title">Computed in this task</div>
      <p className="hint">Not tiled, so this task computes them itself rather than reading them.</p>
      {computes.map(({ tensorId, region }) => {
        const tensor = plan.graph.tensors[tensorId];
        return (
          <div className="plan-demand" key={tensorId}>
            <div className="plan-demand-head">
              <code>{tensor.name}</code>
              <span className="muted">{fmt(count(region))} elements</span>
              {!region.exact && (
                <span className="badge approx" title={region.reasons.join("; ")}>≈</span>
              )}
              <button
                className="mini"
                title={`write ${tensor.name} to memory: its tasks compute it, and this task reads it`}
                onClick={() => tilePlanTensor(tensorId)}
              >
                tile {tensor.name}
              </button>
            </div>
            <pre className="plan-slice">{regionSliceExprs(tensor.name, region).join("\n")}</pre>
          </div>
        );
      })}
    </section>
  );
}

export function PlanPanel(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const plan = useStore((s) => s.plan);
  const task = useStore((s) => s.planTask);
  const supply = useStore((s) => s.planSupply);
  const selectPlanTask = useStore((s) => s.selectPlanTask);
  const setFocusNode = useStore((s) => s.setFocusNode);
  const dark = useDark();
  const run = usePlanRun(plan);

  /** Following a producer makes it the task, and brings its tensor into view. */
  const follow = (next: TaskRef) => {
    selectPlanTask(next);
    setFocusNode({ kind: "tensor", id: next.tensorId });
  };

  const family = plan && task ? plan.families.get(task.tensorId) : undefined;
  const tensor = task ? resolved.tensors[task.tensorId] : undefined;
  const node = task ? producerNode(resolved, task.tensorId) : undefined;

  return (
    <div className="ins-tabpanel" role="region" id="ins-panel-plan" aria-labelledby="ins-tab-plan">
      <p className="tab-note">
        A plan divides produced tensors into tiles, one task per tile, and writes each tiled tensor
        to memory. A task computes any untiled tensor it needs itself. Figures are exact for the
        plan as given, or bounded with the reason named.
      </p>
      <TiledTensors plan={plan} run={run} />
      {plan && task && family && tensor && node && supply ? (
        <>
          <div className="ins-section">
            <div className="ins-title">Task {taskName(tensor.name, task.coord)}</div>
            <p className="plan-op">
              <code>{`${tensor.name}[${formatBoxIndices(tileBox(family, task.coord))}]`}</code>
              <span className="muted">
                {" "}computed by {opLabel(node)} ·{" "}
                <span title={supply.flops.reasons.join("; ") || undefined}>
                  {flopText(supply.flops)} FLOP
                </span>
              </span>
            </p>
          </div>
          <section className="ins-section">
            <div className="ins-title">
              <span className="cone-key needs" style={{ color: rgbCss(boxColor(0, dark)) }} aria-hidden />{" "}
              Reads
            </div>
            {supply.demand.length === 0 ? (
              <p className="hint">This task reads nothing.</p>
            ) : (
              [...new Map(supply.demand.map((d) => [d.tensorId, d])).keys()].map((tensorId) => (
                <DemandGroup
                  key={tensorId}
                  tensorId={tensorId}
                  demands={supply.demand.filter((d) => d.tensorId === tensorId)}
                  node={node}
                  plan={plan}
                  producers={supply.producers}
                  onSelect={follow}
                />
              ))
            )}
          </section>
          {supply.computes.length > 0 && <ComputedSection plan={plan} computes={supply.computes} />}
          <FamilySection plan={plan} task={task} run={run} />
        </>
      ) : (
        plan && <p className="hint">Click a tile of a tiled tensor to inspect its task.</p>
      )}
    </div>
  );
}
