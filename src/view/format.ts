import type { Figure, FigureStatus } from "../core/metrics";

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1 << 30) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / (1 << 30)).toFixed(2)} GB`;
}

/**
 * The mark a figure's status earns, in the vocabulary the rest of the app uses.
 *
 * `≤` is a one-sided bound: no more than this, never less. `~` is a number that
 * moved in an unknown direction, which is what a ratio of two widened sums
 * does. An exact figure is unmarked, because a mark on a count would teach a
 * reader to ignore the marks. `unknown` has no mark because it has no number to
 * put one on.
 */
export const FIGURE_MARK: Record<FigureStatus, string> = {
  exact: "",
  upper: "≤ ",
  approximate: "~ ",
  unknown: "",
};

/**
 * A figure, written the way it may honestly be read.
 *
 * An unknown figure prints the word rather than a number. That is the whole
 * point of the status: a FLOP total spanning an operation nobody described is
 * not a smaller number or a looser bound, it is an answer this engine does not
 * have, and printing anything numeric there invites it to be read as one.
 */
export function formatFigure(f: Figure, format: (value: number) => string): string {
  return f.value === null ? "unknown" : `${FIGURE_MARK[f.status]}${format(f.value)}`;
}
