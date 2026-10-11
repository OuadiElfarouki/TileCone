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

/** How the operand's elements split between read once and read by several outputs. */
export function readsSummary(r: OperandReads): string {
  if (r.reused.value === 0) return `read once · ${formatFigure(r.onceBytes, formatBytes)}`;
  if (r.once.value === 0) {
    const times = r.min.value === r.max.value
      ? formatFigure(r.max, fmt)
      : `${formatFigure(r.min, fmt)}–${formatFigure(r.max, fmt)}`;
    return `each read ${times}× · ${formatFigure(r.reusedBytes, formatBytes)}`;
  }
  return (
    `${formatFigure(r.reusedBytes, formatBytes)} read up to ${formatFigure(r.max, fmt)}× · ` +
    `${formatFigure(r.onceBytes, formatBytes)} once`
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

/** The counts the summary rests on, for its tooltip. */
export function readsDetail(r: OperandReads): string {
  return (
    `${formatFigure(r.reads, fmt)} reads of ${formatFigure(r.elements, fmt)} elements, ` +
    `${formatFigure(r.mean, (v) => `${formatRatio(v)}×`)} on average`
  );
}
