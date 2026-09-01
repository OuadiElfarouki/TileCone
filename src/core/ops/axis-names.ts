/**
 * Shared axis-name mappings for `OpSpec.inferAxisNames`.
 *
 * The rule every helper here obeys: an output axis takes a name only when it is
 * genuinely the same axis as the input one it came from. A name is shown to the
 * reader as the source's own word for that axis, so inventing one — or letting a
 * broadcast axis lend its name to the axis it was stretched to match — is worse
 * than leaving it unnamed.
 */

import { AxisNames } from "./types";
import { Sym } from "../shapes";

/** The output has the input's own axes in order: rank-preserving operations
 * that change extents or values but not what each axis means. */
export function sameAxisNames(inNames: AxisNames[]): AxisNames[] {
  return [inNames[0].slice()];
}

/**
 * Trailing-aligned inheritance, the alignment broadcasting already uses. An
 * input axis may name an output axis only where their extents agree: an axis
 * stretched from extent 1 is a different axis that happened to fit, so it does
 * not get to name the one it was stretched against.
 */
export function broadcastAxisNames(
  inNames: AxisNames[],
  inShapes: number[][],
  outShape: number[]
): AxisNames {
  return outShape.map((extent, axis) => {
    for (let slot = 0; slot < inNames.length; slot++) {
      const aligned = axis - (outShape.length - inShapes[slot].length);
      if (aligned < 0 || inShapes[slot][aligned] !== extent) continue;
      const name = inNames[slot][aligned];
      if (name !== undefined) return name;
    }
    return undefined;
  });
}

/** The first name any input gives each axis, for operations whose inputs all
 * share one layout and differ only in extent along the axis being joined. */
export function firstNamedAxis(inNames: AxisNames[], rank: number): AxisNames {
  return Array.from({ length: rank }, (_, axis) =>
    inNames.map((names) => names[axis]).find((name) => name !== undefined)
  );
}

/**
 * The word a note should use for one axis: its semantic name, then its verified
 * symbolic extent, then its position. Callers keep their own "axis"/"axes"
 * prefix, so this returns a bare token that reads correctly either way —
 * "full axis seq", "full axis K", "full axis 3".
 */
export function axisWord(
  names: AxisNames | undefined,
  dims: Sym[] | undefined,
  axis: number
): string {
  const dim = dims?.[axis];
  return names?.[axis] ?? (typeof dim === "string" ? dim : String(axis));
}
