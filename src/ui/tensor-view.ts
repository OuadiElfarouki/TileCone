/** Per-tensor controls for viewing ranks above the two-dimensional card plane. */
export type ViewCfg = {
  sliders: number[]; // index per hidden axis (full rank length; row/col entries ignored)
  projection: boolean; // union over hidden axes vs slice at slider
};

export function defaultViewCfg(shape: number[]): ViewCfg {
  return { sliders: shape.map(() => 0), projection: true };
}

/** Structural validation for serialized view state; bounds need the graph. */
export function isViewCfg(value: unknown): value is ViewCfg {
  if (!value || typeof value !== "object") return false;
  const cfg = value as Partial<ViewCfg>;
  return typeof cfg.projection === "boolean" && Array.isArray(cfg.sliders) &&
    cfg.sliders.every((v) => Number.isSafeInteger(v) && v >= 0);
}

export function viewCfgFits(shape: number[], value: unknown): value is ViewCfg {
  return isViewCfg(value) && value.sliders.length === shape.length &&
    value.sliders.every((v, axis) => v < Math.max(1, shape[axis]));
}

/**
 * Which axes the grid draws, fixed row-major for every tensor: the last axis
 * (fastest-varying) is columns, the one before it is rows. There is no per-card
 * axis remapping - a different view of a tensor is a `transpose` node in the
 * graph, where it is part of the computation being explained rather than a
 * display setting that silently disagrees with the DSL.
 */
export function viewAxes(shape: number[]): { rowAxis: number; colAxis: number } {
  const rank = shape.length;
  return { rowAxis: rank >= 2 ? rank - 2 : -1, colAxis: rank >= 1 ? rank - 1 : -1 };
}
