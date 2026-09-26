/** Per-tensor controls for viewing ranks above the two-dimensional card plane. */
export type ViewCfg = {
  sliders: number[]; // index per hidden axis (full rank length; row/col entries ignored)
  projection: boolean; // union over hidden axes vs slice at slider
  /**
   * The tile this tensor is worked in, one extent per axis. Absent means the
   * canvas default: the square display tile on the visible axes, and on the
   * hidden axes whatever the view mode selects. See `ui/tile-spec.ts`.
   */
  tile?: number[];
  /**
   * The axes drawn as the card's rows and columns. Absent means the row-major
   * default, the last two. This is presentation: it changes what the canvas
   * shows, never the graph, and the card names the axes whenever it is set.
   */
  axes?: [number, number];
};

export function defaultViewCfg(shape: number[]): ViewCfg {
  return { sliders: shape.map(() => 0), projection: true };
}

/** Structural validation for serialized view state; bounds need the graph. */
export function isViewCfg(value: unknown): value is ViewCfg {
  if (!value || typeof value !== "object") return false;
  const cfg = value as Partial<ViewCfg>;
  return typeof cfg.projection === "boolean" && Array.isArray(cfg.sliders) &&
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
    (value.axes === undefined || value.axes.every((axis) => axis < shape.length));
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
