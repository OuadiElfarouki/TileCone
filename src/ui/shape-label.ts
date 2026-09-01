/**
 * How a tensor's shape reads on a card.
 *
 * The compact card header chooses between semantic labels and numeric extents.
 * A label is intentionally not treated as the extent itself: the details view
 * can show the axis identity (`emb`), symbolic extent (`H*D`), and bound extent
 * (`512`) as three related but distinct facts.
 */

import { Tensor } from "../core/graph";

export type AxisMode = "symbolic" | "numeric";

/**
 * The symbolic reading of one axis: the name the source gave it, else the
 * dimension it was declared with, else its extent. The fallbacks matter — a
 * produced tensor has no declared shape, and propagated names have holes where
 * an operation invented an axis — so a symbolic shape may be partly numeric.
 * That is honest: those axes have no symbol to show.
 */
export function axisLabel(tensor: Tensor, axis: number): string {
  const named = tensor.axisNames?.[axis];
  if (named !== undefined) return named;
  // `symShape` is what resolution established and verified: a declared
  // tensor's own dimensions, or whatever its producing operation could carry
  // across. It is full length, so it also holds the literal where nothing did.
  const symbolic = tensor.symShape?.[axis] ?? tensor.shape[axis];
  if (typeof symbolic === "string") return symbolic;
  return String(tensor.resolved?.[axis] ?? symbolic ?? "");
}

export function shapeLabel(tensor: Tensor, mode: AxisMode): string {
  const extents = tensor.resolved ?? [];
  const parts =
    mode === "numeric"
      ? extents.map(String)
      : extents.map((_, axis) => axisLabel(tensor, axis));
  return `[${parts.join(" × ")}]`;
}

/** The verified symbolic extent expression for every axis, without allowing an
 * axis name to hide it. This is the middle reading between semantic labels and
 * resolved numeric extents in the tensor details. */
export function symbolicExtentLabel(tensor: Tensor): string {
  const extents = tensor.resolved ?? [];
  return `[${extents.map((extent, axis) => {
    const symbolic = tensor.symShape?.[axis];
    return typeof symbolic === "string" ? symbolic : String(extent);
  }).join(" × ")}]`;
}

/**
 * The readings worth listing in a tensor's details, most semantic first.
 *
 * A reading is dropped when a *less* symbolic one below it says exactly the
 * same thing, which keeps the surviving row named after what it actually is:
 * an unnamed, unparameterised tensor lists one row called `extents`, not three
 * identical rows the first of which claims to be labels.
 */
/** @internal Pure seam exported so the collapse rule is asserted directly. */
export function shapeReadings(tensor: Tensor): { label: string; value: string }[] {
  const readings = [
    { label: "labels", value: shapeLabel(tensor, "symbolic") },
    { label: "symbols", value: symbolicExtentLabel(tensor) },
    { label: "extents", value: shapeLabel(tensor, "numeric") },
  ];
  return readings.filter(
    (reading, i) => !readings.slice(i + 1).some((later) => later.value === reading.value)
  );
}

/** True when the symbolic reading says nothing the numeric one does not, so a
 * caller can skip offering a toggle that would change nothing. */
export function hasSymbolicShape(tensor: Tensor): boolean {
  return shapeLabel(tensor, "symbolic") !== shapeLabel(tensor, "numeric");
}
