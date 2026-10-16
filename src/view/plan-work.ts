/**
 * The whole-plan figures as rows, shared by one plan's totals and a comparison
 * of several. The rows are fixed in number and order, so plans read side by
 * side line up.
 */

import { ratioFigure, type Figure } from "../core/metrics";
import type { Work } from "../core/plan/interfaces";
import { FIGURE_MARK, fmt, formatBytes, formatFigure } from "./format";

export type WorkRow = {
  label: string;
  /** What the row means, for its tooltip. */
  title: string;
  /** The row's figure for one plan. */
  figure: (work: Work) => Figure;
  /** The figure as a cell, without a unit: the label carries it. */
  format: (value: number) => string;
};

export const WORK_ROWS: readonly WorkRow[] = [
  {
    label: "tasks",
    title: "one task per tile of every tiled tensor",
    figure: (w) => ({ value: w.tasks, status: "exact", reasons: [] }),
    format: String,
  },
  {
    label: "dependencies",
    title: "producer tasks each task reads from, summed over tasks",
    figure: (w) => w.dependencies,
    format: String,
  },
  { label: "FLOPs", title: "work summed over tasks", figure: (w) => w.flops, format: fmt },
  {
    label: "recomputed",
    title:
      "work more than one task does: an untiled tensor several tasks compute, or a row statistic every tile across a normalised axis computes again",
    figure: (w) => w.recomputed,
    format: fmt,
  },
  {
    label: "read",
    title: "what the tasks read, each task counted separately: no reuse between tasks",
    figure: (w) => w.read,
    format: formatBytes,
  },
  {
    label: "distinct read",
    title: "what the tasks read, each element counted once",
    figure: (w) => w.readDistinct,
    format: formatBytes,
  },
  { label: "written", title: "every tiled tensor, once", figure: (w) => w.written, format: formatBytes },
  {
    label: "FLOP / byte",
    title: "FLOPs per byte read or written",
    figure: (w) => w.intensity,
    format: (v) => v.toFixed(2),
  },
];

/** One row's cell for one plan, with the mark its status earns. */
export const workCell = (row: WorkRow, work: Work): string => formatFigure(row.figure(work), row.format);

/**
 * How a kept plan's figure compares with the current plan's, as a signed
 * percentage, or null when either has no number or the current one is zero.
 *
 * The change is a ratio, so it takes the ratio's status: marked `~` when
 * either side is a bound, since two bounds can hide any difference between the
 * quantities, including one where they read the same.
 */
export function workChange(row: WorkRow, kept: Work, current: Work): string | null {
  const now = row.figure(current);
  const ratio = ratioFigure(row.figure(kept), now);
  if (ratio.value === null || now.value === 0) return null;
  const change = 100 * (ratio.value - 1);
  const text =
    Math.abs(change) < 0.05
      ? "same as the current plan"
      : `${change > 0 ? "+" : ""}${change.toFixed(Math.abs(change) < 10 ? 1 : 0)}% against the current plan`;
  return FIGURE_MARK[ratio.status] + text;
}

/**
 * A cell's tooltip: its change against the current plan when it is a kept
 * plan's, and the reasons its own figure is qualified.
 */
export function workCellTitle(row: WorkRow, work: Work, current: Work | null): string | undefined {
  const parts: string[] = [];
  const change = current && work !== current ? workChange(row, work, current) : null;
  if (change) parts.push(change);
  const reasons = row.figure(work).reasons;
  if (reasons.length) parts.push(reasons.join("; "));
  return parts.length ? parts.join(" · ") : undefined;
}

/** A tiling in words: each tiled tensor with its extents, by name. */
export function describeTiling(
  tiles: Readonly<Record<string, readonly number[]>>,
  nameOf: (tensorId: string) => string
): string {
  const parts = Object.entries(tiles)
    .map(([id, tile]) => ({ name: nameOf(id), tile }))
    .sort((a, b) => a.name.localeCompare(b.name))
    .map(({ name, tile }) => `${name} ${tile.length ? tile.join("×") : "scalar"}`);
  return parts.length ? parts.join(", ") : "nothing tiled";
}

/**
 * What a plan changes from another, as the plan's own decisions: which
 * tensors it fuses, materializes, retiles, or starts or stops writing. Names
 * the decisions and not their effects - the figures report those, and one
 * change can move several figures for reasons a sentence would only guess at.
 */
export function describeTilingChange(
  from: Readonly<Record<string, readonly number[]>>,
  to: Readonly<Record<string, readonly number[]>>,
  isOutput: (tensorId: string) => boolean,
  nameOf: (tensorId: string) => string
): string[] {
  const extents = (tile: readonly number[]) => (tile.length ? tile.join("×") : "scalar");
  return [...new Set([...Object.keys(from), ...Object.keys(to)])]
    .sort((a, b) => nameOf(a).localeCompare(nameOf(b)))
    .flatMap((id) => {
      const before = from[id];
      const after = to[id];
      const name = nameOf(id);
      if (before && !after) return [isOutput(id) ? `stops writing ${name}` : `fuses ${name} into its consumers`];
      if (!before && after) return [isOutput(id) ? `writes ${name}` : `materializes ${name}`];
      if (before.length === after!.length && before.every((extent, axis) => extent === after![axis])) return [];
      return [`retiles ${name} ${extents(before)} → ${extents(after!)}`];
    });
}

/**
 * The rows whose figure is lower, and higher, in `to` than in `from`. Only
 * figures exact on both sides are compared: two bounds do not say which
 * quantity is the larger.
 */
export function workDirection(from: Work, to: Work): { lower: string[]; higher: string[] } {
  const lower: string[] = [];
  const higher: string[] = [];
  for (const row of WORK_ROWS) {
    const a = row.figure(from);
    const b = row.figure(to);
    if (a.status !== "exact" || b.status !== "exact" || a.value === null || b.value === null) continue;
    if (b.value < a.value) lower.push(row.label);
    else if (b.value > a.value) higher.push(row.label);
  }
  return { lower, higher };
}

/**
 * A kept plan against the current one in a sentence: what the current plan
 * changes from it, then which figures that leaves lower and higher.
 */
export function describePlanChange(
  label: string,
  from: { tiles: Readonly<Record<string, readonly number[]>>; work: Work },
  to: { tiles: Readonly<Record<string, readonly number[]>>; work: Work | null },
  isOutput: (tensorId: string) => boolean,
  nameOf: (tensorId: string) => string
): string {
  const changes = describeTilingChange(from.tiles, to.tiles, isOutput, nameOf);
  if (!changes.length) return `${label} is the current plan.`;
  const list = (items: string[]) =>
    items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
  let text = `From ${label}, the current plan ${list(changes)}.`;
  if (!to.work) return text;
  const { lower, higher } = workDirection(from.work, to.work);
  if (lower.length) text += ` Lower: ${lower.join(", ")}.`;
  if (higher.length) text += ` Higher: ${higher.join(", ")}.`;
  return text;
}
