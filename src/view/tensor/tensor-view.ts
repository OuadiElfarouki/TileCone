import { foldRun, MAX_UNFOLD_H, MAX_UNFOLD_W, splitHiddenAxes } from "./fold-layout";
import { cardPx, planeExtents } from "./tiling";

/**
 * How an unfolded card lays its hidden axes beside the drawn plane: every one
 * across, every one down, or the fastest across and the rest down.
 */
export type UnfoldDir = "across" | "down" | "grid";
export const UNFOLD_DIRS: readonly UnfoldDir[] = ["across", "down", "grid"];
/** How a card and its menu mark each direction. */
export const UNFOLD_MARK: Record<UnfoldDir, string> = { across: "→", down: "↓", grid: "▦" };

/** Per-tensor controls for viewing ranks above the two-dimensional card plane. */
export type ViewCfg = {
  sliders: number[]; // index per hidden axis (full rank length; row/col entries ignored)
  projection: boolean; // union over hidden axes vs slice at slider
  /**
   * Draw every hidden index at once, one panel of the plane per index, laid
   * out in stride order (`view/tensor/fold.ts`). Set only with `projection`
   * false: an unfolded card shows elements, not unions, so every reader of
   * `projection` already treats its hidden axes one index at a time.
   */
  unfold?: UnfoldDir;
  /**
   * The tile this tensor is worked in, one extent per axis. Absent means the
   * canvas default: the square display tile on the visible axes, and on the
   * hidden axes whatever the view mode selects. See `view/tensor/tile-spec.ts`.
   */
  tile?: number[];
  /**
   * The axes drawn as the card's rows and columns. Absent means the row-major
   * default, the last two. This is presentation: it changes what the canvas
   * shows, never the graph, and the card names the axes whenever it is set.
   */
  axes?: [number, number];
};

/**
 * Whether a hidden axis has a position the reader can set, by its slider or
 * by the scrub keys. In slice mode it is the slice drawn. In projection the
 * card draws the union over the axis, so a position means something only when
 * the tensor's tile is narrower than the axis: it says which tile a gesture
 * takes. The slider and the keys read this one rule, so a slider that moves
 * always has keys that move it, and a disabled one has none.
 */
export function hiddenAxisPositioned(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "projection" | "tile" | "unfold">,
  axis: number
): boolean {
  // An unfolded card draws every index, so there is no position to hold.
  if (cfg.unfold) return false;
  return !cfg.projection || (!!cfg.tile && cfg.tile[axis] < shape[axis]);
}

/**
 * How a card reads the axes it does not draw as rows and columns: as one
 * union (`project`), one index at a time (`slice`), or all indices side by
 * side (`unfold`). A slice card is the only one with sliders.
 */
export type HiddenMode = "project" | "slice" | "unfold";

export function hiddenMode(cfg: Partial<Pick<ViewCfg, "projection" | "unfold">> | undefined): HiddenMode {
  if (cfg?.unfold) return "unfold";
  return cfg?.projection === false ? "slice" : "project";
}

/** The axes a card does not draw, slowest first. */
export function hiddenAxes(shape: readonly number[], cfg?: Pick<ViewCfg, "axes">): number[] {
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  return shape.map((_, axis) => axis).filter((axis) => axis !== rowAxis && axis !== colAxis);
}

/** Projected axes whose gesture tile needs a position control. */
export function projectedTileAxes(
  shape: readonly number[],
  cfg?: Partial<Pick<ViewCfg, "axes" | "projection" | "unfold" | "tile">>
): number[] {
  if (hiddenMode(cfg) !== "project" || !cfg?.tile) return [];
  return hiddenAxes(shape, cfg).filter((axis) => cfg.tile![axis] < shape[axis]);
}

/**
 * The slice a card shows as one position in row-major order over its hidden
 * axes, and how many there are. This is the order a slice card's deck is
 * stacked in: the slices still to come lie behind the one shown.
 */
export function sliceOrdinal(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "sliders" | "axes">
): { index: number; count: number } {
  let index = 0;
  let count = 1;
  for (const axis of hiddenAxes(shape, cfg)) {
    const extent = Math.max(1, shape[axis]);
    index = index * extent + Math.min(extent - 1, cfg.sliders[axis] ?? 0);
    count *= extent;
  }
  return { index, count };
}

export function defaultViewCfg(shape: number[]): ViewCfg {
  return { sliders: shape.map(() => 0), projection: true };
}

/** Structural validation for serialized view state; bounds need the graph. */
export function isViewCfg(value: unknown): value is ViewCfg {
  if (!value || typeof value !== "object") return false;
  const cfg = value as Partial<ViewCfg>;
  return typeof cfg.projection === "boolean" && Array.isArray(cfg.sliders) &&
    (cfg.unfold === undefined || (UNFOLD_DIRS.includes(cfg.unfold) && cfg.projection === false)) &&
    cfg.sliders.every((v) => Number.isSafeInteger(v) && v >= 0) &&
    (cfg.tile === undefined ||
      (Array.isArray(cfg.tile) && cfg.tile.every((v) => Number.isSafeInteger(v) && v >= 1))) &&
    (cfg.axes === undefined ||
      (Array.isArray(cfg.axes) && cfg.axes.length === 2 &&
        cfg.axes.every((v) => Number.isSafeInteger(v) && v >= 0) && cfg.axes[0] !== cfg.axes[1]));
}

export function viewCfgFits(shape: number[], value: unknown): value is ViewCfg {
  return isViewCfg(value) && value.sliders.length === shape.length &&
    value.sliders.every((v, axis) => v < Math.max(1, shape[axis])) &&
    (value.tile === undefined ||
      (value.tile.length === shape.length &&
        value.tile.every((v, axis) => v <= Math.max(1, shape[axis])))) &&
    (value.axes === undefined || value.axes.every((axis) => axis < shape.length)) &&
    (value.unfold === undefined || unfoldRefusal(shape, value) === null);
}

/**
 * The most panels an unfolded card draws. Every panel is a copy of the plane,
 * so past this the card is either too large to place or too small to read,
 * and slicing serves the reader better.
 */
export const MAX_UNFOLD_PANELS = 256;

/** Why a tensor cannot be unfolded, or null when it can. */
export function unfoldRefusal(shape: readonly number[], cfg?: Pick<ViewCfg, "axes" | "unfold">): string | null {
  const hidden = hiddenAxes(shape, cfg);
  if (!hidden.length) return "every axis is already drawn";
  const panels = hidden.reduce((n, axis) => n * Math.max(1, shape[axis]), 1);
  if (panels > MAX_UNFOLD_PANELS)
    return `${panels} panels: more than the ${MAX_UNFOLD_PANELS} an unfolded card draws`;
  if (cfg?.unfold) {
    const { rowAxis, colAxis } = viewAxes(shape, cfg);
    const { rows, cols } = planeExtents([...shape], rowAxis, colAxis);
    // Even at scale zero, singleton sides and the gaps retain their size.
    const panel = cardPx(rows, cols, 0);
    const { across, down } = splitHiddenAxes(hidden, cfg.unfold);
    if (foldRun(across, shape, panel.w).total > MAX_UNFOLD_W || foldRun(down, shape, panel.h).total > MAX_UNFOLD_H)
      return "panels and gaps cannot fit in this direction; choose another direction or slice";
  }
  return null;
}

/**
 * Which axes the grid draws. By default the last axis (fastest-varying) is
 * columns and the one before it is rows. A tensor's view may choose another
 * pair (`ViewCfg.axes`), which is presentation only: a transpose that changes
 * the program is still a `transpose` node in the graph. A chosen pair is
 * never silent - the card prints the axes it draws whenever they are not the
 * default - so the canvas cannot be mistaken for a transposed tensor.
 */
export function viewAxes(
  shape: readonly number[],
  cfg?: Pick<ViewCfg, "axes">
): { rowAxis: number; colAxis: number } {
  const rank = shape.length;
  const chosen = cfg?.axes;
  if (chosen && rank >= 2 && chosen[0] !== chosen[1] && chosen.every((axis) => axis >= 0 && axis < rank))
    return { rowAxis: chosen[0], colAxis: chosen[1] };
  return { rowAxis: rank >= 2 ? rank - 2 : -1, colAxis: rank >= 1 ? rank - 1 : -1 };
}

/** Whether a view draws a pair other than the default one. */
export function remapped(shape: readonly number[], cfg?: Pick<ViewCfg, "axes">): boolean {
  const chosen = viewAxes(shape, cfg);
  const plain = viewAxes(shape);
  return chosen.rowAxis !== plain.rowAxis || chosen.colAxis !== plain.colAxis;
}

/**
 * The pair a card draws after `axis` takes the given role. A card draws two
 * distinct axes, so taking the axis that holds the other role swaps the two;
 * any other axis replaces the one in that role.
 */
export function axesWith(
  shape: readonly number[],
  cfg: Pick<ViewCfg, "axes"> | undefined,
  axis: number,
  role: "rows" | "cols"
): [number, number] {
  const { rowAxis, colAxis } = viewAxes(shape, cfg);
  return role === "rows"
    ? [axis, axis === colAxis ? rowAxis : colAxis]
    : [axis === rowAxis ? colAxis : rowAxis, axis];
}
