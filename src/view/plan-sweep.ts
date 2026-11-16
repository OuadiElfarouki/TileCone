/**
 * How a sweep's rows and limits are worded in the Plan view. Pure.
 */

import type { TilePlan } from "../core/plan/plan";
import { stagingOf } from "../core/plan/steps";
import type { Sweep, SweepRow } from "../core/plan/sweep";

const plural = (n: number, one: string, many = `${one}s`): string => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

/**
 * What a sweep left out, one sentence per limit that bit, most consequential
 * first. Empty when the rows are the whole answer for the sets searched.
 *
 * A reader acts on the rows, so each sentence says what its limit means for
 * them: whether a better plan could be among what was not looked at.
 */
export function sweepCaveats(counts: Sweep["counts"]): string[] {
  const out: string[] = [];
  const unsearched = counts.fusionSetsTotal - counts.fusionSets - counts.setsInexact - counts.setsSkipped;
  if (counts.combinationsDropped)
    out.push(
      `${plural(counts.combinationsDropped, "partial combination")} dropped at the plan limit: the rows are unbeaten among the plans kept, not among all.`
    );
  if (counts.setsSkipped)
    out.push(`${plural(counts.setsSkipped, "fusion choice")} not reached within the work limit.`);
  if (counts.setsInexact)
    out.push(
      `${plural(counts.setsInexact, "fusion choice")} not searched: a region in ${counts.setsInexact === 1 ? "it" : "them"} is over-approximated, and the sweep ranks exact figures only.`
    );
  if (unsearched > 0)
    out.push(`${plural(unsearched, "fusion choice")} not considered: with many intermediates only none, each alone and all are tried.`);
  if (counts.overTasks)
    out.push(`${plural(counts.overTasks, "tile extent")} not measured: more tasks than the sweep measures per tensor.`);
  if (counts.overSteps)
    out.push(`${plural(counts.overSteps, "tile extent")} left out: too many steps to measure, so within the budget or not is not known.`);
  if (counts.cappedUnresolved)
    out.push(`${plural(counts.cappedUnresolved, "tile extent")} left out: only a chunking the search did not try fits.`);
  if (counts.chunkingsCapped)
    out.push(`${plural(counts.chunkingsCapped, "tile extent")} had a capped chunk search: fewer steps or a smaller step payload may exist among untried chunkings.`);
  return out;
}

/** What a sweep did measure, in one line. */
export function sweepSummary(sweep: Sweep): string {
  const { counts, rows } = sweep;
  const sets = `${plural(counts.fusionSets, "fusion choice")}`;
  const shown = counts.unbeaten > rows.length ? `the first ${rows.length} of ${counts.unbeaten.toLocaleString("en-US")}` : plural(rows.length, "plan");
  return `${shown} unbeaten · ${plural(counts.measured, "tile extent")} measured over ${sets} · ${plural(counts.overBudget, "extent")} over the budget`;
}

/** Compare effective steps, including defaults, without measuring either plan. */
export function rowIsCurrent(
  row: Pick<SweepRow, "tiles">,
  current: TilePlan,
  rowPlan: TilePlan
): boolean {
  const ids = Object.keys(row.tiles);
  if (ids.length !== current.families.size) return false;
  return ids.every((id) => {
    const tile = current.families.get(id)?.tile;
    if (!tile || tile.length !== row.tiles[id].length || tile.some((extent, axis) => extent !== row.tiles[id][axis])) return false;
    try {
      const a = stagingOf(current, id);
      const b = stagingOf(rowPlan, id);
      if (!a.staged || !b.staged) return !a.staged && !b.staged;
      return a.accumulator === b.accumulator && a.reduced.length === b.reduced.length &&
        a.reduced.every((axis, i) => axis.label === b.reduced[i].label && axis.chunk === b.reduced[i].chunk);
    } catch {
      // Invalid declarations on an unstaged tensor cannot match an installable row.
      return false;
    }
  });
}
