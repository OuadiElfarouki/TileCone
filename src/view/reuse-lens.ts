/**
 * The reuse lens: one operand's read multiplicity drawn on its own card.
 *
 * The operand-reuse rows say how often an operation's outputs use each element
 * of an operand, as figures. The lens draws the same count per element, so two
 * footprints of one size that play different roles - a GEMM band every element
 * of which is used 64 times, a convolution halo whose edges are used once and
 * whose middle nine times - look different where they sit.
 *
 * The scope is the row's, and only the row's: one operation, the outputs the
 * analysis has of it (the drawn tiles' cone under Dependencies, the inspected
 * task under Plan), one operand. Counts are not multiplied along paths and are
 * not memory loads, as the rows' tooltips already say.
 */

import type { ResolvedGraph } from "../core/graph";
import { operandTallyOf } from "../core/multiplicity";
import type { Box, Region } from "../core/region";
import { measureTally, tallyLevels, type Piece } from "../core/tally";
import type { RGB } from "./palette";
import { analysisTarget, groupPropResult, type BoxProp, type InspectorTab, type SelPart } from "./workspace";
import type { PropResult } from "../core/propagate";

/** Which row the lens draws: the view it was opened in, the operation, the operand. */
export type ReuseLens = { view: InspectorTab; nodeId: string; tensorId: string };

export const sameLens = (a: ReuseLens | null, b: ReuseLens | null): boolean =>
  !!a && !!b && a.view === b.view && a.nodeId === b.nodeId && a.tensorId === b.tensorId;

/**
 * The regions the reuse rows of `view` are counted over, derived from the store
 * by the inspector's own rules so the lens and the row can never disagree:
 * under Dependencies the backward cone of the analysed group, narrowed to a
 * focused tile in it; under Plan what the inspected task computes.
 */
export function lensRegions(
  view: InspectorTab,
  state: {
    selection: { parts: SelPart[] } | null;
    perBox: BoxProp[] | null;
    byTensorRes: Record<string, { backward: PropResult | null; forward: PropResult | null }> | null;
    hiddenBoxes: Set<number>;
    focusedBox: number | null;
    analysisGroup: string | null;
    planSupply: { work: ReadonlyMap<string, { region: Region }> } | null;
  }
): ReadonlyMap<string, { region: Region }> | null {
  if (view === "plan") return state.planSupply?.work ?? null;
  const parts = state.selection?.parts ?? [];
  const tensorId = analysisTarget(parts, state.analysisGroup, null).tensorId;
  const raw = state.perBox ? state.focusedBox : null;
  const focus = raw !== null && parts[raw]?.tensorId === tensorId ? raw : null;
  return groupPropResult(state.byTensorRes, state.perBox, parts, state.hiddenBoxes, focus, tensorId, "backward")?.tensors ?? null;
}

export type LensPaint = {
  /** Disjoint pieces with the count at each, lowest first; empty when not `complete`. */
  pieces: Piece[];
  /** False when the count changes too often to draw at this size. */
  complete: boolean;
  /** The fewest and most outputs using one element read at all. */
  min: number;
  max: number;
  /** An inexact count is an upper bound at every element, and is drawn hatched. */
  exact: boolean;
  reasons: string[];
};

/** The lens for one row, or null when the row has nothing to draw under this scope. */
export function lensPaint(
  graph: ResolvedGraph,
  regions: ReadonlyMap<string, { region: Region }>,
  lens: ReuseLens
): LensPaint | null {
  const tally = operandTallyOf(graph, regions, lens.nodeId, lens.tensorId);
  if (!tally) return null;
  const measure = measureTally(tally);
  if (!measure.max) return null;
  const levels = tallyLevels(tally);
  return {
    pieces: levels.pieces,
    complete: levels.complete,
    min: measure.min ?? 0,
    max: measure.max,
    exact: tally.exact,
    reasons: [...tally.reasons],
  };
}

/**
 * One violet hue, light to dark, validated as an ordinal ramp with the dataviz
 * validator against the card surfaces (`CARD_SURFACE`): monotone lightness,
 * every step at least 0.06 OKLCH L from the next, and the end nearest the
 * surface clear of it at 2:1, since one use differs from none.
 *
 *   light (surface #f3f4f6): lightest #978cd7 at 2.71:1 : PASS
 *   dark  (surface #1c1f24): darkest  #6451ac at 2.61:1 : PASS
 *
 * Violet because every other hue here already means something: blue, orange
 * and aqua are tile identities, the accent marks what is inspected, and amber
 * and green read as status. Dark mode runs the other way, so the most-used
 * elements stand furthest from the surface in both themes.
 */
const LENS_RAMP: Record<"light" | "dark", RGB[]> = {
  light: [[151, 140, 215], [128, 113, 200], [106, 85, 184], [86, 55, 167], [67, 16, 150]],
  dark: [[100, 81, 172], [125, 111, 194], [151, 140, 215], [179, 170, 237], [208, 201, 255]],
};

export const lensRamp = (dark: boolean): readonly RGB[] => LENS_RAMP[dark ? "dark" : "light"];

/**
 * The colour of a count. The scale runs from zero to the largest count, not
 * from the smallest: a band used 32 times beside one used 64 should look half
 * as strong, and a uniform count takes the strongest step whatever it is,
 * which the legend states as a number.
 */
export function lensColor(count: number, max: number, dark: boolean): RGB {
  const ramp = lensRamp(dark);
  const t = Math.max(0, Math.min(1, count / max)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(t));
  const f = t - i;
  return ramp[i].map((c, k) => Math.round(c + (ramp[i + 1][k] - c) * f)) as RGB;
}

/**
 * The counts within `window`: one element under a slice, every hidden index
 * of a column under projection. `lo` is 0 when part of the window is not read.
 * Null when none of it is read.
 */
export function lensReading(paint: LensPaint, window: Box): { lo: number; hi: number } | null {
  if (!paint.complete) return null;
  const volume = (box: Box) => box.reduce((n, I) => n * Math.max(0, I.hi - I.lo), 1);
  let covered = 0;
  let lo = Infinity;
  let hi = 0;
  for (const piece of paint.pieces) {
    const overlap = piece.box.map((I, axis) => ({
      lo: Math.max(I.lo, window[axis].lo),
      hi: Math.min(I.hi, window[axis].hi),
    }));
    const v = volume(overlap);
    if (v <= 0) continue;
    covered += v;
    lo = Math.min(lo, piece.count);
    hi = Math.max(hi, piece.count);
  }
  if (!hi) return null;
  return { lo: covered < volume(window) ? 0 : lo, hi };
}

/** A reading as the card's tooltip states it. */
export function lensReadingText(reading: { lo: number; hi: number } | null): string {
  if (!reading) return "not read";
  const uses = (n: number) => `${n} use${n === 1 ? "" : "s"}`;
  return reading.lo === reading.hi ? uses(reading.hi) : `${reading.lo}–${uses(reading.hi)}`;
}
