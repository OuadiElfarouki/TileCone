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
import { operandReads } from "../../core/multiplicity";
import { opLabel } from "../../core/ops/index";
import {
  planReport,
  producerPage,
  supplyOf,
  type BoundaryDemand,
  type Computed,
  type Demand,
  type InterfaceReport,
  type PlanReport,
  type ProducerNeed,
  type TaskSupply,
  type Work,
} from "../../core/plan/interfaces";
import type { TaskRef, TilePlan } from "../../core/plan/plan";
import { coordinatePage, tileBox, tileOrdinal, tiles, type TileFamily } from "../../core/plan/tile-family";
import { taskLineage, type LineageRow } from "../../core/plan/replay";
import { count, formatBoxIndices, unionOf } from "../../core/region";
import { FIGURE_MARK, fmt, formatBytes, formatFigure, formatIntensity } from "../../view/format";
import { demandDetail, demandSummary } from "../../view/demand";
import { describePlanChange, describeTiling, readOrder, WORK_ROWS, workCell, workCellTitle } from "../../view/plan-work";
import { boxColor, rgbCss } from "../../view/palette";
import { useDark, useStore } from "../../state/store";
import { DraftField } from "../chrome/DraftField";
import { OperandReuse } from "./operand-reuse";
import type { PlanReplayControl } from "./usePlanReplay";
import type { ReplayPlayback } from "../../view/workspace";
import { sameTiles } from "../../state/history";
import { activePlanTrail, MAX_KEPT_PLANS } from "../../state/plan";
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
     one query holds the lane at a time, a build cancels a plan run once per
     build, and the family replay that shares the lane is not offered while a
     run is evaluating and gives way after one retry when a run interrupts it. */
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
  const isOutput = (tensorId: string) => !resolved.consumers[tensorId]?.length;
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
              <span className="hint plan-change">
                {describePlanChange(
                  `P${k.id}`,
                  { tiles: k.tiles, work: k.total },
                  { tiles: planTiles, work: total },
                  isOutput,
                  nameOf
                )}
              </span>
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
          memory. Fuse one and the tasks that read it compute it themselves; type new extents to
          divide it differently.
        </p>
        <ul className="plan-list">
          {families.map((family) => {
            const tensor = resolved.tensors[family.tensorId];
            const report = reports.get(family.tensorId);
            // Untiling an intermediate fuses it into its readers; untiling a
            // graph output leaves it unwritten. Different results, so different words.
            const output = !resolved.consumers[family.tensorId]?.length;
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
                  className="mini"
                  title={output
                    ? `stop writing ${tensor.name}: this plan no longer computes the graph output`
                    : `fuse ${tensor.name} into its consumers: the tasks that read it compute it instead of reading it from memory`}
                  aria-label={output ? `stop writing ${tensor.name}` : `fuse ${tensor.name} into its consumers`}
                  onClick={() => setPlanTile(family.tensorId, null)}
                >
                  {output ? "stop writing" : "fuse"}
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

/** Producer details per page; the compact set and totals are never truncated. */
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
  supply,
  onSelect,
}: {
  tensorId: string;
  demands: Demand[];
  node: Node;
  plan: TilePlan;
  supply: TaskSupply;
  onSelect: (task: TaskRef) => void;
}): React.ReactElement {
  const tensor = plan.graph.tensors[tensorId];
  const [paging, setPaging] = useState({ supply, page: 0 });
  const page = paging.supply === supply ? paging.page : 0;
  const setPage = (page: number) => setPaging({ supply, page });
  const family = plan.families.get(tensorId);
  const group = supply.producerGroups.find((g) => g.tensorId === tensorId);
  const total = group?.count ?? 0;
  const lastPage = Math.max(0, Math.ceil(total / PRODUCERS_SHOWN) - 1);
  const currentPage = Math.min(page, lastPage);
  const shown = useMemo(() => producerPage(plan, supply, tensorId, currentPage * PRODUCERS_SHOWN, PRODUCERS_SHOWN),
    [plan, supply, tensorId, currentPage]);
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
        {demands[0].supplier === "input" && (
          <span className="badge input" title="read from memory; no task produces it">graph input</span>
        )}
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
      {demands[0].supplier === "tasks" && (
        family && (
          <>
            <p className="plan-need" title={`${total} of the ${family.count} tasks of ${tensor.name}`}>
              requires {group && group.definiteCount < total ? FIGURE_MARK.upper : ""}{total}/{family.count} task{family.count === 1 ? "" : "s"}
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
            {total > PRODUCERS_SHOWN && (
              <div className="plan-producer-pages">
                <button className="mini" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>previous</button>
                <span className="muted">{currentPage * PRODUCERS_SHOWN + 1}–{Math.min((currentPage + 1) * PRODUCERS_SHOWN, total)} of {total}</span>
                <button className="mini" disabled={currentPage === lastPage} onClick={() => setPage(currentPage + 1)}>next</button>
              </div>
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
  replay,
}: {
  plan: TilePlan;
  task: TaskRef;
  run: PlanRun;
  replay: PlanReplayControl;
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
      <div className="ins-title with-action">
        Across all {family.count} {name} task{family.count === 1 ? "" : "s"}
        <button
          className="mini"
          onClick={replay.run}
          disabled={replay.pending || run.evaluating}
          title={`replay this family's tasks on the cards: what each one reads, and what it shares with task ${taskName(name, task.coord)}`}
        >
          {replay.pending ? "preparing…" : "replay"}
        </button>
      </div>
      {replay.depthLimit > 1 && (
        <ReplayDepth
          tensorId={task.tensorId}
          name={name}
          nameOf={(tensorId) => plan.graph.tensors[tensorId].name}
          replay={replay}
        />
      )}
      {replay.error && <p className="hint overlap">Replay failed: {replay.error}</p>}
      {replay.playback && (
        <ReplayStatus
          playback={replay.playback}
          name={name}
          nameOf={(tensorId) => plan.graph.tensors[tensorId].name}
          replay={replay}
        />
      )}
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
                title={`materialize ${tensor.name}: write it to memory in tiles of its own tasks, and this task reads it`}
                onClick={() => tilePlanTensor(tensorId)}
              >
                materialize
              </button>
            </div>
            <pre className="plan-slice">{regionSliceExprs(tensor.name, region).join("\n")}</pre>
          </div>
        );
      })}
    </section>
  );
}

/**
 * Where a family replay is, and what it shows. The listing order is said every
 * time: a replay walks the grid row by row, and read as an execution order it
 * would claim a schedule nobody declared.
 */
function ReplayStatus({
  playback,
  name,
  nameOf,
  replay,
}: {
  playback: ReplayPlayback;
  name: string;
  nameOf: (tensorId: string) => string;
  replay: PlanReplayControl;
}): React.ReactElement {
  const { frames, tasks, depth, steps, stepTensors } = playback.replay;
  const current = frames[playback.visited - 1];
  const runs = frames.length < steps;
  const stepped = depth > 1;
  const unit = runs ? "run" : stepped ? "step" : "task";
  /* A step deeper than 1 is a path, read as what feeds what: the family's task,
     then the producer task feeding it, and so on upstream. */
  const pathName = (path: TaskRef[]) => path.map((t) => taskName(nameOf(t.tensorId), t.coord)).join(" ← ");
  const feeders = stepTensors.map(nameOf).join(" or ");
  const last = playback.visited >= frames.length;
  return (
    <>
      <div className="plan-replay-controls tc-slider" role="group" aria-label="replay controls">
        <button
          className="mini"
          onClick={() => replay.seek(playback.visited - 1)}
          disabled={playback.visited <= 1}
          aria-label={`previous ${unit}`}
          title={`previous ${unit}`}
        >
          ‹
        </button>
        {playback.paused ? (
          <button
            className="mini"
            onClick={replay.play}
            aria-label={last ? "play again" : "play"}
            title={last ? "play again from the first " + unit : "play on from here"}
          >
            ▶
          </button>
        ) : (
          <button className="mini" onClick={replay.pause} aria-label="pause" title="pause on this run">
            ❚❚
          </button>
        )}
        <button
          className="mini"
          onClick={() => replay.seek(playback.visited + 1)}
          disabled={last}
          aria-label={`next ${unit}`}
          title={`next ${unit}`}
        >
          ›
        </button>
        <input
          type="range"
          min={1}
          max={frames.length}
          step={1}
          value={playback.visited}
          onChange={(e) => replay.seek(Number(e.currentTarget.value))}
          aria-label={`replay ${unit}`}
          aria-valuetext={`${playback.visited} of ${frames.length}`}
        />
        <button className="mini" onClick={replay.clear} aria-label="clear replay" title="clear the replay from the cards">
          clear
        </button>
      </div>
      {current && (
        <p className="hint">
          {stepped
            ? current.steps === 1
              ? pathName(current.firstPath)
              : `${pathName(current.firstPath)} to ${pathName(current.lastPath)}`
            : current.tasks === 1
              ? `Task ${taskName(name, current.first.coord)}`
              : `Tasks ${taskName(name, current.first.coord)} to ${taskName(name, current.last.coord)}`}{" "}
          · {playback.visited} of {frames.length}
        </p>
      )}
      <p className="hint">
        {stepped
          ? `${runs ? `All ${steps} steps in ${frames.length} runs` : `All ${steps} steps`}, each a ${feeders} task feeding the ${name} tasks ${depth > 2 ? `through ${depth - 2} level${depth > 3 ? "s" : ""} of producers ` : ""}in listing order, which is not an execution order. A step moves with the parts of the other operands it is combined with in the task it feeds, and what its task reads is drawn fainter.`
          : `${runs ? `All ${tasks} tasks in ${frames.length} runs,` : `All ${tasks} tasks`} in listing order, which is not an execution order.`}{" "}
        Full strength marks what each {unit} shares with this task; outlines are the producer tiles.
      </p>
    </>
  );
}

/**
 * How far upstream a replay of this family follows the plan: 1 is the
 * family's own tasks, and each level adds the tasks producing the tiles the
 * last one reads. Offered only where the plan has such tasks, and remembered
 * per family.
 *
 * The depth changes what one step is, not how much is shown around it, so the
 * control says which tasks a step visits. The producers are named once a
 * replay at this depth has found them.
 */
function ReplayDepth({ tensorId, name, nameOf, replay }: {
  tensorId: string;
  name: string;
  nameOf: (tensorId: string) => string;
  replay: PlanReplayControl;
}): React.ReactElement {
  const setReplayDepth = useStore((s) => s.setReplayDepth);
  const found = replay.playback?.replay.depth === replay.depth ? replay.playback.replay.stepTensors : null;
  const levels = replay.depth - 1;
  const unit = replay.depth === 1
    ? `${name} tasks`
    : `${found?.length ? found.map(nameOf).join(" or ") : "producer"} tasks ${levels} level${levels === 1 ? "" : "s"} up, each with the ${levels === 1 ? `${name} ` : ""}task it feeds`;
  return (
    <div className="plan-replay-depth" role="group" aria-label="replay depth">
      <span className="muted">depth</span>
      {Array.from({ length: replay.depthLimit }, (_, i) => i + 1).map((level) => (
        <button
          key={level}
          className={`mini toggle${level === replay.depth ? " on" : ""}`}
          aria-pressed={level === replay.depth}
          onClick={() => setReplayDepth(tensorId, level)}
          title={level === 1
            ? "step through this family's tasks: what each reads and computes"
            : `step through the tasks ${level - 1} level${level > 2 ? "s" : ""} of producers up, each with what it is combined with in the task it feeds`}
        >
          {level}
        </button>
      ))}
      <span className="muted">· steps through {unit}</span>
    </div>
  );
}

export function PlanPanel({ replay }: { replay: PlanReplayControl }): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const plan = useStore((s) => s.plan);
  const task = useStore((s) => s.planTask);
  const supply = useStore((s) => s.planSupply);
  const followPlanTask = useStore((s) => s.followPlanTask);
  const backPlanTask = useStore((s) => s.backPlanTask);
  const trail = useStore(activePlanTrail);
  const setFocusNode = useStore((s) => s.setFocusNode);
  const dark = useDark();
  const run = usePlanRun(plan);
  /* Counted for the inspected task only, over what it computes: one task's
     operations, where a family evaluation would be every task's. */
  const reads = useMemo(
    () => (plan && supply ? operandReads(plan.graph, supply.work) : null),
    [plan, supply]
  );

  /** Following a producer makes it the task, remembers where it came from, and
   *  brings its tensor into view. */
  const follow = (next: TaskRef) => {
    followPlanTask(next);
    setFocusNode({ kind: "tensor", id: next.tensorId });
  };
  const back = (index: number) => {
    const target = trail?.from[index];
    if (!target) return;
    backPlanTask(index);
    setFocusNode({ kind: "tensor", id: target.tensorId });
  };
  const nameOf = (tensorId: string) => resolved.tensors[tensorId]?.name ?? tensorId;

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
            {/* Where this task was reached from, by following producer lists:
                any step back returns there, Alt+← to the last. */}
            {trail && (
              <nav className="plan-trail" aria-label="followed from">
                {trail.from.map((step, i) => (
                  <React.Fragment key={i}>
                    <button
                      className="mini"
                      onClick={() => back(i)}
                      title={i === trail.from.length - 1 ? "back to this task (Alt+←)" : "back to this task"}
                    >
                      {taskName(nameOf(step.tensorId), step.coord)}
                    </button>
                    <span className="muted" aria-hidden>›</span>
                  </React.Fragment>
                ))}
                <span className="muted">{taskName(tensor.name, task.coord)}</span>
              </nav>
            )}
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
              readOrder(supply.demand, node.id, plan.graph.topo).map((tensorId) => (
                <DemandGroup
                  key={tensorId}
                  tensorId={tensorId}
                  demands={supply.demand.filter((d) => d.tensorId === tensorId)}
                  node={node}
                  plan={plan}
                  supply={supply}
                  onSelect={follow}
                />
              ))
            )}
          </section>
          {supply.computes.length > 0 && <ComputedSection plan={plan} computes={supply.computes} />}
          <LineageSection plan={plan} task={task} onFollow={follow} />
          {reads && (
            <OperandReuse
              reads={reads}
              title="Operand reuse in this task"
              empty="The operations this task computes read nothing for its tile."
              nameOf={(tensorId) => resolved.tensors[tensorId]?.name ?? tensorId}
              view="plan"
            />
          )}
          <FamilySection plan={plan} task={task} run={run} replay={replay} />
        </>
      ) : (
        plan && <p className="hint">Click a tile of a tiled tensor to inspect its task.</p>
      )}
    </div>
  );
}

const levelNote = (level: number) =>
  level === 1 ? "this task" : level === 2 ? "the tasks producing those tiles" : `${level - 1} levels of producers up`;

/**
 * Everything the inspected task depends on through the plan, to the graph's
 * inputs (`taskLineage`): per upstream tensor, the producer tasks it needs, the
 * input elements it reads, or the fused computation its producers do, level by
 * level. An output task relates to the tasks of any tensor above it here
 * without replaying its family. Pointing at a row outlines those tasks on the
 * tensor's card; a row's first task can be followed into, keeping the trail.
 */
function LineageSection({
  plan,
  task,
  onFollow,
}: {
  plan: TilePlan;
  task: TaskRef;
  onFollow: (task: TaskRef) => void;
}): React.ReactElement | null {
  const setLineageFocus = useStore((s) => s.setLineageFocus);
  const rows = useMemo(() => taskLineage(plan, task), [plan, task]);
  // A focus left behind by a row that went away with its task would outline
  // tiles nothing on screen is pointing at.
  useEffect(() => () => setLineageFocus(null), [plan, task, setLineageFocus]);
  if (!rows.length) return null;
  const levels = [...new Set(rows.map((row) => row.level))];
  const name = (row: LineageRow) => plan.graph.tensors[row.tensorId].name;
  return (
    <section className="ins-section" aria-label="upstream to the inputs">
      <div className="ins-title">Upstream, to the inputs</div>
      {levels.map((level) => (
        <div className="plan-lineage-level" key={level}>
          <div className="plan-lineage-head muted">
            level {level} · {levelNote(level)}
          </div>
          <ul className="reuse-list">
            {rows.filter((row) => row.level === level).map((row) => {
              const tensor = plan.graph.tensors[row.tensorId];
              const bytes = formatFigure(byteFigure(tensor, count(row.region), row.region), formatBytes);
              if (row.kind === "tasks") {
                const first = [...coordinatePage(row.coordinates, 0, 1)][0];
                return (
                  <li
                    key={row.tensorId}
                    onMouseEnter={() => setLineageFocus({
                      tensorId: row.tensorId, coordinates: row.coordinates, definite: row.definiteCoordinates,
                    })}
                    onMouseLeave={() => setLineageFocus(null)}
                  >
                    <code>{name(row)}</code>
                    <span
                      className="muted"
                      title={`${row.tasks} of the ${row.of} tasks of ${name(row)}${
                        row.definiteTasks < row.tasks ? `, ${row.definiteTasks} of them definite and the rest only possible` : ""
                      }; point to outline them`}
                    >
                      requires {row.definiteTasks < row.tasks ? FIGURE_MARK.upper : ""}{row.tasks}/{row.of} task{row.of === 1 ? "" : "s"} · {bytes}
                    </span>
                    {!row.region.exact && (
                      <span className="badge approx" title={row.region.reasons.join("; ")}>≈</span>
                    )}
                    {first && (
                      <button
                        className="mini reads-figures"
                        title={`inspect ${taskName(name(row), first)}, the first of them, keeping the way back`}
                        onClick={() => onFollow({ tensorId: row.tensorId, coord: first })}
                      >
                        {taskName(name(row), first)}
                      </button>
                    )}
                  </li>
                );
              }
              return (
                <li key={row.tensorId}>
                  <code>{name(row)}</code>
                  {row.kind === "input" ? (
                    <span className="badge input" title="read from memory; no task produces it">graph input</span>
                  ) : (
                    <span className="muted">fused, computed by {levelNote(level)}</span>
                  )}
                  <span className="muted">{bytes}</span>
                  {!row.region.exact && (
                    <span className="badge approx" title={row.region.reasons.join("; ")}>≈</span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
      <p className="hint">
        Level 1 is what this task reads; each further level is what the tasks producing those tiles
        read, which is whole tiles. Point at a row to outline its tasks on the card.
      </p>
    </section>
  );
}
