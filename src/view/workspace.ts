/**
 * The workspace's shared vocabulary: selection parts, directions, views and
 * the pure helpers that read them. Pure, so the card and inspector builders
 * can use it without depending on the store.
 */

import { ResolvedGraph } from "../core/graph";
import { mergeProps, PropResult } from "../core/propagate";
import { Box } from "../core/region";
import { MAX_DISTINCT_HUES } from "./palette";
import type { TilePlan } from "../core/plan/plan";
import type { FamilyReplay } from "../core/plan/replay";
import { sameParameterizedSource } from "./params";

/** Which independently toggled views are active in the workspace. `none` is
 * the explicit figures-only state: analysis remains live while paint and rows hide. */
export type Direction = "none" | "backward" | "forward" | "both";
export type ConeDirection = "backward" | "forward";
export type PanelSide = "left" | "right";
export type Theme = "light" | "dark";
/** What a press on a graph node does: `select` draws tiles on a card and picks
 *  an operation, `move` drags either from anywhere on it. */
export type CanvasTool = "select" | "move";
/** The two classes of question the inspector answers; see `inspectorTab`. */
export type InspectorTab = "dependencies" | "plan";

/**
 * Whether cards and operations can be moved. The Plan view divides and inspects
 * tasks on the arrangement as it stands, so nothing there moves a node: not the
 * move tool, not a Ctrl/Cmd drag, not the arrow keys, not a layout reset.
 */
export const layoutMovable = (tab: InspectorTab): boolean => tab !== "plan";

/**
 * The tool a press on the graph uses: the remembered one, except in the Plan
 * view, which always selects. The preference itself is kept, and returns with
 * the Dependencies view.
 */
export const activeCanvasTool = (state: { canvasTool: CanvasTool; inspectorTab: InspectorTab }): CanvasTool =>
  layoutMovable(state.inspectorTab) ? state.canvasTool : "select";

/**
 * What a press on the canvas does right now, in one line. The two views give
 * the same gesture different meanings - a drag draws a tile in one and divides
 * a tensor into tasks in the other - so the canvas says which one is in force,
 * where the gesture is made rather than in a panel that may be collapsed.
 */
export function canvasHint(state: {
  inspectorTab: InspectorTab;
  canvasTool: CanvasTool;
  /** Whether the plan divides any tensor into tasks yet. */
  planTiled: boolean;
}): string {
  if (state.inspectorTab === "plan")
    return state.planTiled
      ? "click a tile to inspect its task · drag on an untiled tensor to divide it"
      : "drag on a produced tensor to divide it into tasks, or click it to use the drawn grid";
  return activeCanvasTool(state) === "move"
    ? "drag a card or operation to move it · Esc returns to drawing tiles"
    : "drag on a tensor to draw a tile · Shift adds one, Alt subtracts";
}

/** Parameter edits keep the built graph interactive, including while resized
 * source builds. Other source changes leave a navigable snapshot. */
export function graphIsInteractive(state: {
  resolved: ResolvedGraph | null;
  dslText: string;
  draftText: string;
  compiling: boolean;
}): boolean {
  if (state.resolved === null) return false;
  // An unchanged-source build is also used for full reloads/restores.
  if (state.draftText === state.dslText) return !state.compiling;
  return sameParameterizedSource(state.dslText, state.draftText);
}
/**
 * A family replay on the cards (`core/plan/replay.ts`), and how far through it
 * the playback is. Transient view state, never an edit: it overlays the plan,
 * plays once and holds its last run until cleared or departed from, leaving
 * the plan unchanged.
 */
export type ReplayPlayback = {
  replay: FamilyReplay;
  /** The plan the frames were computed for. Another plan makes them stale. */
  plan: TilePlan;
  /** Runs walked so far; the last of them is the active one. */
  visited: number;
  /** The walk is stopped at `visited`: paused, stepped, or held at its end. */
  paused: boolean;
  /**
   * The overlay is on its way out and `opacity` is being driven to zero.
   *
   * The trail fades as it stands, so a departure mid-replay reads as the
   * same picture going away rather than a jump to some other one.
   */
  exiting: boolean;
  opacity: number;
};
/** Defensive share-state bound; far beyond any usable graph arrangement while
 * preventing finite-but-overflowing coordinates from poisoning scene bounds. */
export const MAX_NODE_OFFSET = 1_000_000;

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
export type SelPart = {
  tensorId: string;
  box: Box;
  /**
   * The tile's colour slot (`boxColor`), given once and kept while the tile
   * lives, so removing a tile does not repaint the others. Parts read from
   * older links or built by tests may lack one; `withHues` fills it in, and
   * readers fall back to the part's index.
   */
  hue?: number;
};

/** A part's colour slot, or its index for a part that has none. */
export const partHue = (parts: readonly SelPart[] | undefined, index: number): number =>
  parts?.[index]?.hue ?? index;

/**
 * The slot a new tile takes: the lowest one no tile holds, so the colour of a
 * removed tile is the next one used. With every slot held, the least used,
 * which wraps the palette round in order.
 */
export function nextHue(parts: readonly SelPart[]): number {
  const uses = Array.from({ length: MAX_DISTINCT_HUES }, () => 0);
  parts.forEach((part, index) => {
    const hue = part.hue ?? index;
    uses[((hue % MAX_DISTINCT_HUES) + MAX_DISTINCT_HUES) % MAX_DISTINCT_HUES]++;
  });
  return uses.indexOf(Math.min(...uses));
}

/** Give every part without a slot the next free one, in order. Parts that
 * already have one are returned as they are, keeping their identity. */
export function withHues(parts: SelPart[]): SelPart[] {
  if (parts.every((part) => part.hue !== undefined)) return parts;
  const held = parts.filter((part) => part.hue !== undefined);
  return parts.map((part) => {
    if (part.hue !== undefined) return part;
    const given = { ...part, hue: nextHue(held) };
    held.push(given);
    return given;
  });
}

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
  const index = tensorId === null ? -1 : viewAnchorIndex(parts, tensorId, focus) ?? -1;
  return { tensorId, focusedBox, index };
}

/**
 * The tile the arrows, Delete and `h` act on: one hovered in the tiles list,
 * else the keyboard's tile (`editBox`: the tile just drawn or clicked in the
 * list), else none. A pin alone does not count: it scopes the analysis, and a
 * press elsewhere releases the keyboard's tile while leaving the pin, so
 * nothing moves that the reader has stopped pointing at. Nothing is a real
 * answer - the arrows then move no tile rather than every tile on a tensor.
 */
export function keyboardTile(state: {
  selection: { parts: SelPart[] } | null;
  perBox: unknown[] | null;
  focusedBox: number | null;
  pinnedBox: number | null;
  editBox: number | null;
}): number | null {
  const count = state.selection?.parts.length ?? 0;
  const hovered = state.perBox && state.pinnedBox === null ? state.focusedBox : null;
  const index = hovered ?? state.editBox;
  return index !== null && index < count ? index : null;
}

/**
 * The tile a tensor's own view is about: the focused tile when it is on that
 * tensor, else the last tile drawn there, or null when it has none. A change to
 * how the tensor is drawn keeps this tile on screen.
 */
export function viewAnchorIndex(parts: SelPart[], tensorId: string, focus: number | null): number | null {
  if (focus !== null && parts[focus]?.tensorId === tensorId) return focus;
  const last = parts.reduce((found, part, i) => (part.tensorId === tensorId ? i : found), -1);
  return last >= 0 ? last : null;
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
