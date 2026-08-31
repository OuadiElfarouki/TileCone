import { executeQuery } from "./executor";
import { ResolvedGraph } from "./graph";
import { PropResult } from "./propagate";
import { Box, boundingBox, count, isEmpty, Region, subtract } from "./region";

export type Contribution = {
  tensorId: string;
  /** True when the selection alone does not determine this tensor's region. */
  partial: boolean;
  /**
   * False when the probe ran on an over-approximated region, so `partial` may be
   * an over-warning. Never the reverse — see the note on `contributions`.
   */
  exact: boolean;
  /** What the downstream region needs beyond this tile. Null when complete. */
  detail: string | null;
};

export type ContributionReport = {
  byTensor: Map<string, Contribution>;
  /** True when the cone was too wide to probe, so nothing could be classified. */
  capped: boolean;
};

/**
 * Each probe is a full backward propagation over the graph, and the report is
 * rebuilt on every selection change, so the count is bounded up front rather
 * than after the first slow graph.
 */
export const MAX_CONTRIBUTION_PROBES = 24;

/** The axes on which `needed` reaches outside `seed`, and their extents. */
function escapingAxes(needed: Box, seed: Box): number[] {
  const axes: number[] = [];
  for (let axis = 0; axis < needed.length; axis++)
    if (needed[axis].lo < seed[axis].lo || needed[axis].hi > seed[axis].hi) axes.push(axis);
  return axes;
}

/** Name what is missing, preferring the axis-shaped statement over a count. */
function residueClause(
  downstream: string,
  seedName: string,
  shape: number[],
  needed: Region,
  seed: Region,
  residue: Region
): string {
  const neededBox = boundingBox(needed);
  const seedBox = boundingBox(seed);
  if (neededBox && seedBox) {
    const spanned = escapingAxes(neededBox, seedBox).filter(
      (axis) => neededBox[axis].lo === 0 && neededBox[axis].hi === shape[axis]
    );
    if (spanned.length) {
      const list = spanned
        .map((axis) => `all ${shape[axis]} along axis ${axis}`)
        .join(" and ");
      return `${downstream} accumulates over ${list} of ${seedName}`;
    }
  }
  const missing = count(residue);
  return `${downstream} also needs ${missing} element${missing === 1 ? "" : "s"} of ${seedName} outside this tile`;
}

/**
 * Which downstream tensors this selection *completes*, and which it only
 * contributes to.
 *
 * Being in the forward cone means a tile influences a tensor, not that it
 * produces it: for `O = P @ V`, a tile spanning 128 of P's 384 columns reaches
 * `O[0:64, 0:64]` without determining a single element of it. The test asks the
 * question backwards — what does that downstream region actually read? — and
 * subtracts the tile. Anything left over is what the tile does not supply.
 *
 * The asymmetry matters: a backward region may be an over-approximation, so the
 * residue may be phantom and `partial` may over-warn. It can never under-warn,
 * because an empty residue proves the true residue is empty too. Rows carry
 * `exact` so an over-warning can be shown as one.
 */
export function contributions(
  graph: ResolvedGraph,
  forward: PropResult,
  seeds: Map<string, Region>,
  cap = MAX_CONTRIBUTION_PROBES
): ContributionReport {
  const byTensor = new Map<string, Contribution>();
  const downstream = [...forward.tensors.keys()].filter((id) => !seeds.has(id));
  if (downstream.length > cap) return { byTensor, capped: true };

  for (const tensorId of downstream) {
    const region = forward.tensors.get(tensorId)!.region;
    const back = executeQuery(graph, { tensorId, region, direction: "backward" }).backward!;
    const name = graph.tensors[tensorId].name;

    let exact = region.exact;
    const clauses: string[] = [];
    for (const [seedId, seed] of seeds) {
      const needed = back.tensors.get(seedId)?.region;
      if (!needed) continue;
      exact = exact && needed.exact;
      const residue = subtract(needed, seed);
      if (isEmpty(residue)) continue;
      clauses.push(
        residueClause(
          name,
          graph.tensors[seedId].name,
          graph.tensors[seedId].resolved ?? [],
          needed,
          seed,
          residue
        )
      );
    }
    byTensor.set(tensorId, {
      tensorId,
      partial: clauses.length > 0,
      exact,
      detail: clauses.length ? clauses.join("; ") : null,
    });
  }

  return { byTensor, capped: false };
}
