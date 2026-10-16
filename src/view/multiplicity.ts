/**
 * A read-multiplicity row (`core/multiplicity.ts`) in words: what of an
 * operand is read once, and what is read by several outputs of the operation.
 */

import type { OperandReads } from "../core/multiplicity";
import { fmt, formatBytes, formatFigure, formatRatio } from "./format";

/**
 * The operation a row is about, named by what it computes: two operations of
 * one kind share a name, and attention reads `X` through three `linear`s.
 */
export function readsOperation(r: OperandReads, nameOf: (tensorId: string) => string): string {
  return `${r.op} → ${r.outputIds.map(nameOf).join(", ")}`;
}

/**
 * How the operand's elements split between used by one output and used by
 * several. Worded as uses, not reads: "read 64×" suggests 64 memory loads,
 * and the count is of output elements depending on an element, not traffic.
 */
export function readsSummary(r: OperandReads): string {
  if (r.reused.value === 0) return `used once · ${formatFigure(r.onceBytes, formatBytes)}`;
  if (r.once.value === 0) {
    const uses = r.min.value === r.max.value
      ? formatFigure(r.max, fmt)
      : `${formatFigure(r.min, fmt)}–${formatFigure(r.max, fmt)}`;
    return `${uses} uses per element · ${formatFigure(r.reusedBytes, formatBytes)}`;
  }
  return (
    `${formatFigure(r.reusedBytes, formatBytes)} at up to ${formatFigure(r.max, fmt)} uses per element · ` +
    `${formatFigure(r.onceBytes, formatBytes)} used once`
  );
}

/**
 * Operands read once, on one line: each named with what it becomes, since the
 * operations - mostly reshapes and transposes - would repeat on every entry.
 * `detail` names them, for the line's tooltip.
 */
export function readOnceLine(
  rows: readonly OperandReads[],
  nameOf: (tensorId: string) => string
): { text: string; detail: string } {
  return {
    text: rows.map((r) => `${nameOf(r.tensorId)} → ${r.outputIds.map(nameOf).join(", ")}`).join(", "),
    detail: rows.map((r) => `${nameOf(r.tensorId)} via ${readsOperation(r, nameOf)}`).join("\n"),
  };
}

/** The counts the summary rests on, and what a use is, for its tooltip. */
export function readsDetail(r: OperandReads): string {
  return (
    `${formatFigure(r.reads, fmt)} uses of ${formatFigure(r.elements, fmt)} elements, ` +
    `${formatFigure(r.mean, formatRatio)} per element on average. A use is one output element ` +
    `of the operation depending on an operand element; it is not a memory load.`
  );
}
