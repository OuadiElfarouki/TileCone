/**
 * A shared-demand row (`core/demand.ts`) in words. The Dependencies view's
 * input sharing and the plan's boundary rows read the same figures, so they
 * read the same way.
 */

import type { SharedDemand } from "../core/demand";
import { formatBytes, formatFigure, formatRatio } from "./format";

/** How many readers demand each element, and how much is demanded twice or more. */
export function demandSummary(d: SharedDemand): string {
  if (d.duplicate.value === 0) return "no duplicate demand";
  return `${formatFigure(d.duplication, (v) => `${formatRatio(v)}×`)} demand · ${formatFigure(d.duplicate, formatBytes)} duplicate`;
}

/** The sums the summary rests on, for its tooltip; `readers` names who reads. */
export function demandDetail(d: SharedDemand, readers: string): string {
  return `${readers} demand ${formatFigure(d.summed, formatBytes)} in total; ${formatFigure(d.distinct, formatBytes)} distinct`;
}
