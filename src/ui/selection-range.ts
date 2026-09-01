import { Box, formatBoxIndices } from "../core/region";

/** Compact, rank-preserving selection syntax used by the inspector. This is the
 * printed form `parseSelectionBox` below reads back, so it delegates the terms
 * rather than spelling them a second time. */
export function formatSelectionBox(box: Box): string {
  return `[${formatBoxIndices(box)}]`;
}

/**
 * Parse `[index, lo:hi, ...]` against a concrete tensor shape.
 * Returns null instead of repairing invalid input: silently clamping a range
 * would make the displayed probe disagree with what the user typed.
 */
export function parseSelectionBox(text: string, shape: number[]): Box | null {
  const trimmed = text.trim();
  const body = trimmed.startsWith("[") && trimmed.endsWith("]")
    ? trimmed.slice(1, -1).trim()
    : trimmed;
  const fields = body === "" ? [] : body.split(",").map((part) => part.trim());
  if (fields.length !== shape.length) return null;

  const box: Box = [];
  for (let axis = 0; axis < fields.length; axis++) {
    const match = /^(-?\d+)(?:\s*:\s*(-?\d+))?$/.exec(fields[axis]);
    if (!match) return null;
    const lo = Number(match[1]);
    const hi = match[2] === undefined ? lo + 1 : Number(match[2]);
    if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi)) return null;
    if (lo < 0 || hi <= lo || hi > shape[axis]) return null;
    box.push({ lo, hi });
  }
  return box;
}
