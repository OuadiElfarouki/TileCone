/**
 * The workspace's shared vocabulary: selection parts, directions, views and
 * the pure helpers that read them. Pure, so the card and inspector builders
 * can use it without depending on the store.
 */

import { ResolvedGraph } from "../core/graph";
import { mergeProps, PropResult } from "../core/propagate";
import { Box } from "../core/region";
import { ReuseSweepFrame } from "../core/reuse";

/** Which independently toggled views are active in the workspace. `none` is
 * the explicit figures-only state: analysis remains live while paint and rows hide. */
export type Direction = "none" | "backward" | "forward" | "both";
export type ConeDirection = "backward" | "forward";
export type PanelSide = "left" | "right";
export type Theme = "light" | "dark";
/** The two classes of question the inspector answers; see `inspectorTab`. */
export type InspectorTab = "dependencies" | "execution" | "plan";
export type ExecutionScope = {
  hidden: SelPart[];
  focused: SelPart | null;
  pinned: SelPart | null;
};
export type ExecutionPlayback = {
  tensorId: string;
  anchorBox: Box;
  tile: number[];
  colorIndex: number;
  frames: ReuseSweepFrame[];
  /** Number of frames already visited; the active one is `visited - 1`. */
  visited: number;
  phase: "playing" | "settled";
  /**
   * The overlay is on its way out and `opacity` is being driven to zero.
   *
   * Beside `phase` rather than a third value of it, because a departure has to
   * leave the sweep reading as whatever it was: folded into `phase`, a fade
   * that began mid-sweep dropped the probe being walked and jumped the input
   * cards from that one pulse to the union of every probe so far - a change of
   * subject on the way out, in the frames the reader was still watching.
   */
  exiting: boolean;
  opacity: number;
};
/** Defensive share-state bound; far beyond any usable graph arrangement while
 * preventing finite-but-overflowing coordinates from poisoning scene bounds. */
export const MAX_TENSOR_OFFSET = 1_000_000;

/** User-authored tensor IDs are dictionary keys, so these records must not
 * inherit magic names such as `__proto__` or `toString`. */
export const idRecord = <T>(source?: Record<string, T>): Record<string, T> =>
  Object.assign(Object.create(null) as Record<string, T>, source);
/**
 * One drawn tile. The tensor travels with the part rather than sitting above
 * the list, so tiles on different tensors coexist: comparing what two tensors
 * pull from a shared input is the reason the tool exists, and it cannot be done
 * if drawing on B discards the tile on A.
 */
export type SelPart = { tensorId: string; box: Box };

/**
 * Scope an aggregate result to enabled tiles, optionally to one focused tile.
 * Per-tile propagation stays cached; toggling visibility only re-merges those
 * results and never reruns the symbolic executor.
 */
export function enabledPropResult(
  aggregate: PropResult | null,
  perBox: BoxProp[] | null,
  hiddenBoxes: Set<number>,
  focusedBox: number | null,
  direction: ConeDirection
): PropResult | null {
  if (!perBox) return aggregate;
  if (focusedBox !== null && !hiddenBoxes.has(focusedBox))
    return perBox[focusedBox]?.[direction] ?? null;
  return mergeProps(
    perBox.flatMap((prop, index) =>
      hiddenBoxes.has(index) || !prop[direction] ? [] : [prop[direction]!]
    )
  );
}
/**
 * Which tensors a highlight should reach: the drawn tiles' own, plus the cones
 * the direction filter is actually showing.
 *
 * Both cones are computed whatever the filter says, so this has to be derived
 * from what was *asked for* rather than from what was computed - otherwise
 * hiding a cone leaves its tensors lit. The graph canvas and the operations
 * list are two views of one answer, and they were drifting: the canvas followed
 * the filter while the list stayed on the union of both directions and never
 * changed. With both cones off, what stays lit is what the reader drew.
 */
export function involvedTensorIds(
  selection: Selection,
  backwardRes: PropResult | null,
  forwardRes: PropResult | null,
  perBox: BoxProp[] | null,
  hiddenBoxes: Set<number>,
  direction: Direction
): Set<string> {
  const involved = new Set<string>(selectedTensorIds(selection));
  const shown = [
    direction === "backward" || direction === "both"
      ? enabledPropResult(backwardRes, perBox, hiddenBoxes, null, "backward")
      : null,
    direction === "forward" || direction === "both"
      ? enabledPropResult(forwardRes, perBox, hiddenBoxes, null, "forward")
      : null,
  ];
  for (const res of shown) if (res) for (const id of res.tensors.keys()) involved.add(id);
  return involved;
}

/**
 * The user's ordered parts (identity-stable, may overlap, may span tensors),
 * never a canonicalized set. See the note in core/region.ts.
 */
export type Selection = { parts: SelPart[] } | null;

/**
 * One tensor's enabled tiles as a single cone, for the inspector's analysis.
 *
 * `enabledPropResult` merges every enabled tile whatever it sits on, which is
 * the right scope for the canvas and the operations list: those describe the
 * selection. The inspector deliberately scopes cost to one output tensor's
 * tiles. Multi-output jobs could be modeled too, but require explicit output
 * and materialization semantics; union itself does not double-count work.
 *
 * Past `MAX_PER_BOX_PROPS` there are no per-tile cones to filter and the
 * grouped per-tensor query stands in - coarser, never mixed.
 */
export function groupPropResult(
  byTensorRes: Record<string, { backward: PropResult | null; forward: PropResult | null }> | null,
  perBox: BoxProp[] | null,
  parts: SelPart[],
  hiddenBoxes: Set<number>,
  focusedBox: number | null,
  tensorId: string | null,
  direction: ConeDirection
): PropResult | null {
  if (!tensorId) return null;
  if (!perBox) return byTensorRes?.[tensorId]?.[direction] ?? null;
  if (
    focusedBox !== null &&
    !hiddenBoxes.has(focusedBox) &&
    parts[focusedBox]?.tensorId === tensorId
  )
    return perBox[focusedBox]?.[direction] ?? null;
  return mergeProps(
    perBox.flatMap((prop, index) =>
      hiddenBoxes.has(index) || parts[index]?.tensorId !== tensorId || !prop[direction]
        ? []
        : [prop[direction]!]
    )
  );
}

/** Parts drawn on one tensor, carrying the global index each one keeps. */
export function partsOn(
  selection: Selection,
  tensorId: string
): { index: number; box: Box }[] {
  if (!selection) return [];
  const out: { index: number; box: Box }[] = [];
  selection.parts.forEach((p, index) => {
    if (p.tensorId === tensorId) out.push({ index, box: p.box });
  });
  return out;
}

/** Distinct tensors carrying at least one part, in first-drawn order. */
export function selectedTensorIds(selection: Selection): string[] {
  const out: string[] = [];
  for (const p of selection?.parts ?? []) if (!out.includes(p.tensorId)) out.push(p.tensorId);
  return out;
}

/**
 * The tensor a whole-selection action applies to: the focused part's tensor,
 * else the most recently drawn one. Arrow keys resolve their axis indices
 * against one shape, and with parts on tensors of different rank there is no
 * single axis that means the same thing everywhere.
 */
export function anchorTensorId(selection: Selection, focusedBox: number | null): string | null {
  const parts = selection?.parts ?? [];
  if (!parts.length) return null;
  if (focusedBox !== null && parts[focusedBox]) return parts[focusedBox].tensorId;
  return parts[parts.length - 1].tensorId;
}

/** Shared target for inspector analysis, movement, and hidden-axis controls. */
export function analysisTarget(parts: SelPart[], group: string | null, focus: number | null) {
  const tensorId = group && parts.some((part) => part.tensorId === group)
    ? group : anchorTensorId({ parts }, null);
  const focusedBox = focus !== null && parts[focus]?.tensorId === tensorId ? focus : null;
  const index = focusedBox ?? parts.reduce((last, part, i) => part.tensorId === tensorId ? i : last, -1);
  return { tensorId, focusedBox, index };
}
/**
 * The tile a reuse sweep is about: the focused one when it is in the analysis
 * group and still enabled, else the last enabled tile drawn there.
 *
 * Lives here rather than in the inspector because entering Execution has to
 * know which tile it is scoping to before the inspector renders, and the two
 * choosing differently would scope the panel to one tile and sweep another.
 */
export function sweepAnchorIndex(
  parts: SelPart[],
  group: string | null,
  focus: number | null,
  hidden: Set<number>
): number | null {
  const { tensorId } = analysisTarget(parts, group, null);
  if (!tensorId) return null;
  const enabled = (index: number) =>
    parts[index]?.tensorId === tensorId && !hidden.has(index);
  if (focus !== null && enabled(focus)) return focus;
  return parts.reduce<number | null>(
    (last, _part, index) => (enabled(index) ? index : last),
    null
  );
}

/**
 * The operation a tile on this tensor is "at", for the operations list.
 *
 * Its producer, because that is the operation the tensor *is* the result of. A
 * graph input has no producer, and then the only honest answer is its consumer
 * when there is exactly one - with several, no single row is the one the reader
 * is looking at, and lighting an arbitrary one would be a guess presented as a
 * fact. `null` leaves the list unhighlighted, which is a true statement.
 */
export function operationForTensor(
  graph: ResolvedGraph | null,
  tensorId: string | null
): string | null {
  if (!graph || !tensorId) return null;
  const producer = graph.tensors[tensorId]?.producer;
  if (producer) return producer.nodeId;
  const consumers = graph.consumers[tensorId] ?? [];
  const distinct = [...new Set(consumers.map((c) => c.nodeId))];
  return distinct.length === 1 ? distinct[0] : null;
}

/** Panel geometry. VS Code semantics: drag to resize between the bounds, drag
 * far enough inward to collapse, click the rail to bring it back. */
export const PANEL_MIN = 232;
export const PANEL_MAX = 560;
/** Release below this and the panel collapses rather than clamping to the min. */
export const PANEL_COLLAPSE_AT = 168;
/** Width of the collapsed rail. */
export const PANEL_RAIL = 30;

/** Per-part propagation results, aligned with `selection.parts`. */
export type BoxProp = { backward: PropResult | null; forward: PropResult | null };

/** Above this many boxes, per-box attribution costs more than it is worth. */
export const MAX_PER_BOX_PROPS = 12;

/**
 * Above this many nodes, hovering does not compute a preview cone.
 *
 * The bound is a frame budget, not a guess. A bidirectional query costs roughly
 * 3us per node, so a thousand nodes is about 6ms - comfortably inside a frame,
 * with the rest of it left for painting. The cap was previously less than half
 * this because the query ran once per *pointer event* rather than once per
 * frame, which on a high-polling-rate mouse is an order of magnitude more work
 * for the same picture; `useFrameThrottle` is what removed that multiplier.
 */
export const MAX_PREVIEW_NODES = 1000;
