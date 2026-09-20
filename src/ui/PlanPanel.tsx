/**
 * The Plan view: which produced tensors are divided into tasks, what one task
 * reads, which producer tasks supply it, and the same questions asked of the
 * task's whole family.
 *
 * Every figure here is a function of the graph and the declared tiling. It is
 * exact for the plan as given, or an upper bound with the reason named. The
 * family evaluation declines above its budget instead of reporting part of a
 * sum.
 */

import React, { useEffect, useMemo, useState } from "react";
import type { Node } from "../core/graph";
import { byteFigure, regionSliceExprs } from "../core/metrics";
import { opLabel } from "../core/ops/index";
import {
  interfaceOf,
  supplyOf,
  type BoundaryDemand,
  type Demand,
  type InterfaceReport,
  type ProducerNeed,
} from "../core/plan/interfaces";
import type { TaskRef, TilePlan } from "../core/plan/plan";
import type { Region } from "../core/region";
import { tileBox, tileOrdinal, tiles, type TileFamily } from "../core/plan/tile-family";
import { count, formatBoxIndices } from "../core/region";
import { formatBytes, formatFigure } from "./format";
import { boxColor, rgbCss } from "./palette";
import { useDark, useStore } from "./store";

/** Families up to this many tasks are evaluated as soon as they are shown; larger ones on request. */
export const AUTO_FAMILY_TASKS = 256;
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
  const formatted = formatTileExtents(tile);
  const [draft, setDraft] = useState(formatted);
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    setDraft(formatted);
    setInvalid(false);
  }, [formatted]);

  const commit = () => {
    const parsed = parseTileExtents(draft, shape.length);
    if (!parsed) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    if (parsed.some((e, axis) => e !== tile[axis])) setPlanTile(tensorId, parsed);
  };

  return (
    <input
      className={`box-range${invalid ? " invalid" : ""}`}
      value={draft}
      aria-label={`${name} tile extents`}
      aria-invalid={invalid}
      title={
        invalid
          ? `expected ${shape.length} positive whole number${shape.length === 1 ? "" : "s"}, one per axis`
          : "tile extents per axis; Enter or blur applies"
      }
      spellCheck={false}
      onChange={(event) => {
        setDraft(event.target.value);
        setInvalid(false);
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter") commit();
        if (event.key === "Escape") {
          setDraft(formatted);
          setInvalid(false);
        }
      }}
    />
  );
}

function TiledTensors({ plan }: { plan: TilePlan | null }): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const setPlanTile = useStore((s) => s.setPlanTile);
  const families = plan ? [...plan.families.values()] : [];
  return (
    <div className="ins-section">
      <div className="ins-title">Tiled tensors</div>
      {families.length === 0 ? (
        <p className="hint">
          Draw a rectangle on a produced tensor to divide it, or click one to divide it at the size
          the canvas is drawing. Graph inputs have no tasks and cannot be tiled.
        </p>
      ) : (
        <>
        <p className="hint">
          Click a tile to inspect its task; the arrow keys step it. A tensor is divided once: clear
          it here, or type new extents, to divide it differently.
        </p>
        <ul className="plan-list">
          {families.map((family) => {
            const tensor = resolved.tensors[family.tensorId];
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
                  title={`stop tiling ${tensor.name}`}
                  aria-label={`stop tiling ${tensor.name}`}
                  onClick={() => setPlanTile(family.tensorId, null)}
                >
                  ×
                </button>
              </li>
            );
          })}
        </ul>
        </>
      )}
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
        {need.used.status === "exact" ? "" : "≤ "}
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
 * Grouped by tensor rather than by slot because the producers, the supplier and
 * the action that tiles it are facts about the tensor. Listed per slot, an
 * operation reading one tensor twice printed its whole producer list twice.
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
  const tilePlanTensor = useStore((s) => s.tilePlanTensor);
  const [all, setAll] = useState(false);
  const family = plan.families.get(tensorId);
  const mine = producers.filter((p) => p.task.tensorId === tensorId);
  const shown = all ? mine : mine.slice(0, PRODUCERS_SHOWN);
  const named = demands.length > 1 || node.inputs.length > 1;

  // One element counts once however many slots read it, as the family figures
  // count it, so the bytes here and under the family mean the same thing.
  const union: Region = {
    boxes: demands.flatMap((d) => d.region.boxes),
    exact: demands.every((d) => d.region.exact),
    reasons: [...new Set(demands.flatMap((d) => d.region.reasons))],
  };

  return (
    <div className="plan-demand">
      <div className="plan-demand-head">
        <code>{tensor.name}</code>
        <span className="muted">
          {named ? `${demands.map((d) => `arg${d.slot}`).join(", ")} · ` : ""}
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
      ) : demands[0].supplier === "unplanned" ? (
        <p className="hint">
          {tensor.name} is not tiled, so the tasks that produce it are not named.{" "}
          <button className="mini" onClick={() => tilePlanTensor(tensorId)}>
            tile {tensor.name}
          </button>
        </p>
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
      <span
        className="muted"
        title={`${formatFigure(row.summed, formatBytes)} read in total by ${row.readers} of ${tasks} tasks; ${formatFigure(row.distinct, formatBytes)} distinct`}
      >
        {formatFigure(row.duplication, (v) => `${v.toFixed(2)}×`)} demand ·{" "}
        {formatFigure(row.distinct, formatBytes)} distinct
      </span>
      {!row.exact && (
        <span className="badge approx" title="some tasks' demand on this tensor is widened">≈</span>
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

function FamilySection({ plan, task }: { plan: TilePlan; task: TaskRef }): React.ReactElement {
  const selectPlanTask = useStore((s) => s.selectPlanTask);
  const family = plan.families.get(task.tensorId)!;
  const name = plan.graph.tensors[task.tensorId].name;
  const auto = family.count <= AUTO_FAMILY_TASKS;
  const automatic = useMemo(
    () => (auto ? interfaceOf(plan, task.tensorId) : null),
    [auto, plan, task.tensorId]
  );
  const [run, setRun] = useState<{ plan: TilePlan; tensorId: string; report: InterfaceReport } | null>(null);
  const requested = run && run.plan === plan && run.tensorId === task.tensorId ? run.report : null;
  const report = automatic ?? requested;

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
        {!auto && (
          <button
            className="mini"
            onClick={() => setRun({ plan, tensorId: task.tensorId, report: interfaceOf(plan, task.tensorId) })}
            title="run one bounded query per task of this family"
          >
            evaluate
          </button>
        )}
      </div>
      {!report ? (
        <p className="hint">One query per task · run on request above {AUTO_FAMILY_TASKS} tasks.</p>
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

export function PlanPanel(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const plan = useStore((s) => s.plan);
  const task = useStore((s) => s.planTask);
  const supply = useStore((s) => s.planSupply);
  const selectPlanTask = useStore((s) => s.selectPlanTask);
  const setFocusNode = useStore((s) => s.setFocusNode);
  const dark = useDark();

  /** Following a producer makes it the task, and brings its tensor into view. */
  const follow = (next: TaskRef) => {
    selectPlanTask(next);
    setFocusNode({ kind: "tensor", id: next.tensorId });
  };

  const family = plan && task ? plan.families.get(task.tensorId) : undefined;
  const tensor = task ? resolved.tensors[task.tensorId] : undefined;
  const node =
    tensor?.producer && resolved.nodes.find((n) => n.id === tensor.producer!.nodeId);

  return (
    <div className="ins-tabpanel" role="region" id="ins-panel-plan" aria-labelledby="ins-tab-plan">
      <p className="tab-note">
        A plan divides produced tensors into tiles, one task per tile, each computing its tile
        completely. Figures are exact for the plan as given, or bounded with the reason named.
      </p>
      <TiledTensors plan={plan} />
      {plan && task && family && tensor && node && supply ? (
        <>
          <div className="ins-section">
            <div className="ins-title">Task {taskName(tensor.name, task.coord)}</div>
            <p className="plan-op">
              <code>{`${tensor.name}[${formatBoxIndices(tileBox(family, task.coord))}]`}</code>
              <span className="muted"> computed by {opLabel(node)}</span>
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
            {!supply.complete && (
              <p className="hint overlap">
                Some of what this task reads is produced by a tensor the plan does not tile, so the
                list of tasks it waits on is incomplete.
              </p>
            )}
          </section>
          <FamilySection plan={plan} task={task} />
        </>
      ) : (
        plan && <p className="hint">Click a tile of a tiled tensor to inspect its task.</p>
      )}
    </div>
  );
}
