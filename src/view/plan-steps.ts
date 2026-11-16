/**
 * How steps, chunks and the budget are worded and typed in the Plan view.
 * Pure: the panel renders what these return, and the tests read them directly.
 */

import { nodeById } from "../core/graph";
import type { StepSpec, TilePlan } from "../core/plan/plan";
import type { Staged } from "../core/plan/steps";
import type { Budget } from "../core/plan/sweep";
import { formatBytes } from "./format";

/**
 * What the author would call each reduced axis of a contraction, parallel to
 * `staging.reduced`.
 *
 * The engine keys a chunk by the loop label, and a `matmul`'s label is an
 * internal `k` the source never wrote. The reader knows the axis by the name
 * they gave it on an operand, or failing that by the symbol that sizes it, so
 * those come first; the label is the fallback, and also what two axes fall
 * back to when their preferred names are the same.
 */
export function reducedAxisNames(plan: TilePlan, staging: Staged): string[] {
  const node = nodeById(plan.graph, staging.node)!;
  const preferred = staging.reduced.map((axis) => {
    const names = new Set<string>();
    const symbols = new Set<string>();
    node.inputs.forEach((tensorId, slot) => {
      const tensor = plan.graph.tensors[tensorId];
      staging.domain.inputs[slot].forEach((loopAxis, tensorAxis) => {
        if (loopAxis !== axis.axis) return;
        const name = tensor.axisNames?.[tensorAxis];
        if (name) names.add(name);
        const symbol = tensor.symShape[tensorAxis];
        if (typeof symbol === "string") symbols.add(symbol);
      });
    });
    if (names.size === 1) return [...names][0];
    if (symbols.size === 1) return [...symbols][0];
    return axis.label;
  });
  return preferred.map((name, i) =>
    preferred.filter((other) => other === name).length === 1 ? name : staging.reduced[i].label
  );
}

/**
 * A chunk as typed: a whole number from 1 to the axis, or nothing, `all`,
 * `whole` or the axis's own extent for the whole axis in one step.
 */
export function parseChunk(text: string, extent: number): number | "whole" | null {
  const trimmed = text.trim().toLowerCase();
  if (trimmed === "" || trimmed === "all" || trimmed === "whole") return "whole";
  if (!/^\d+$/.test(trimmed)) return null;
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value) || value < 1 || value > extent) return null;
  return value === extent ? "whole" : value;
}

/**
 * A step declaration with one axis's chunk set or cleared, or null when that
 * leaves nothing declared. The loop order and accumulator are kept.
 */
export function withChunk(spec: StepSpec | undefined, label: string, chunk: number | "whole"): StepSpec | null {
  const chunks = { ...(spec?.chunk ?? {}) };
  if (chunk === "whole") delete chunks[label];
  else chunks[label] = chunk;
  const next: { chunk?: Record<string, number>; order?: readonly string[]; accumulator?: StepSpec["accumulator"] } = {};
  if (Object.keys(chunks).length) next.chunk = chunks;
  if (spec?.order?.length) next.order = spec.order;
  if (spec?.accumulator) next.accumulator = spec.accumulator;
  return Object.keys(next).length ? next : null;
}

const UNITS: Record<string, number> = { b: 1, kb: 1024, kib: 1024, mb: 1 << 20, mib: 1 << 20, gb: 1 << 30, gib: 1 << 30 };

/** Editable budgets must survive a focus/blur round trip without rounding. */
export function formatBudgetBytes(bytes: number): string {
  const compact = formatBytes(bytes);
  return parseBytes(compact) === bytes ? compact : `${bytes} B`;
}

/** Keep an unchanged restored value even when it is outside the input parser's
 * whole-byte range. New entries still follow the ordinary byte-count rules. */
export function parseBudgetBytes(text: string, current: number): number | null {
  return text.trim() === formatBudgetBytes(current) ? current : parseBytes(text);
}

/**
 * A byte count as typed: `99 KB`, `99KiB`, `1.5 MB`, `4096`, `4096 B`. Units
 * are binary, as `formatBytes` prints them. At least one byte, rounded to a
 * whole number of them.
 */
export function parseBytes(text: string): number | null {
  const match = /^\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*$/i.exec(text);
  if (!match) return null;
  const unit = UNITS[match[2].toLowerCase() || "b"];
  if (unit === undefined) return null;
  const bytes = Math.round(Number(match[1]) * unit);
  return Number.isSafeInteger(bytes) && bytes >= 1 ? bytes : null;
}

/** Where a payload stands against the stated budget. A payload that is only an
 * upper bound can prove "within" and cannot prove "over". */
export type Verdict = "within" | "over" | "unknown";

export function budgetVerdict(bytes: { value: number | null; status: string }, limit: number | undefined): Verdict | null {
  if (limit === undefined) return null;
  if (bytes.value === null) return "unknown";
  if (bytes.value <= limit) return bytes.status === "exact" || bytes.status === "upper" ? "within" : "unknown";
  return bytes.status === "exact" || bytes.status === "lower" ? "over" : "unknown";
}

export const sameBudgetBytes = (a: Budget, b: Budget): boolean =>
  a.stepBytes === b.stepBytes && a.accumulatorBytes === b.accumulatorBytes;
