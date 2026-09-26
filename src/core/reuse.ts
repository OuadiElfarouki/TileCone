import { entangledWith } from "./entangle";
import { executeQuery, validateSelection } from "./executor";
import { ResolvedGraph } from "./graph";
import { PropResult, Selection } from "./propagate";
import { Box, Region, canonicalize, count, fromBox, intersect, isEmpty } from "./region";
import { DTYPE_BYTES } from "./dtypes";

export type ReuseEstimate = {
  tensorId: string;
  probes: number;
  totalTiles: number;
  estimatedTiles: number;
  exhaustive: boolean;
  /** Average fraction of the anchor footprint shared by overlapping grid tiles. */
  meanSharedFraction: number | null;
  /** Precision of the propagated regions, independent of sampling error. */
  geometryExact: boolean;
  reasons: string[];
  /** Local probes displaced by one tile extent, not a claim of global invariance. */
  neighbors: {
    axis: number;
    delta: number;
    sharedFraction: number;
    exact: boolean;
    reasons: string[];
  }[];
};

export type InputSharing = {
  tensorId: string;
  selectedTiles: number;
  contributingTiles: number;
  summedDemandBytes: number;
  distinctDemandBytes: number;
  duplicateDemandBytes: number;
  geometryExact: boolean;
  reasons: string[];
};

/** Element-granular leaf demand only; excludes memory transactions, caches,
 * internal reads and output traffic. This is shareable footprint, not measured traffic.
 * Count a represented union directly, without introducing a box-cap approximation. */
export function inputSharing(graph: ResolvedGraph, cones: PropResult[]): InputSharing[] {
  if (new Set(cones.flatMap((cone) => cone.roots)).size > 1)
    throw new Error("input sharing requires tiles on one tensor");
  return Object.values(graph.tensors).filter((t) => !t.producer).flatMap((input) => {
    const regions = cones.flatMap((cone) => {
      const region = cone.tensors.get(input.id)?.region;
      return region && !isEmpty(region) ? [region] : [];
    });
    const boxes = regions.flatMap((region) => region.boxes);
    if (!boxes.length) return [];
    const exact = regions.every((region) => region.exact);
    const reasons = [...new Set(regions.flatMap((region) => region.reasons))];
    const bytes = DTYPE_BYTES[input.dtype];
    const summedDemandBytes = regions.reduce((sum, region) => sum + count(region) * bytes, 0);
    const distinctDemandBytes = count({ boxes, exact, reasons }) * bytes;
    /* This difference remains an upper bound when regions are conservative.
       At each element, widening can only increase the number of tile regions
       containing it; max(membership - 1, 0) is monotone in that membership. */
    const duplicateDemandBytes = Math.max(0, summedDemandBytes - distinctDemandBytes);
    return [{
      tensorId: input.id,
      selectedTiles: cones.length,
      contributingTiles: regions.length,
      summedDemandBytes,
      distinctDemandBytes,
      duplicateDemandBytes,
      geometryExact: exact,
      reasons,
    }];
  });
}

/**
 * A relation a probe can be reported on, named after the three the canvas
 * already paints: a solid fill, a ruling, and a stipple.
 *
 * Asked for rather than assumed. The backward walk happens either way because
 * the estimate is a statement about backward demand, but reporting it per
 * tensor is paint; `forward` and `entangled` cost a query per probe on top.
 */
export type ReuseSurface = "backward" | "forward" | "entangled";

/** Where one probe lands on one tensor through one relation. */
export type ReuseReach = {
  /** What the probe reaches there. */
  region: Region;
  /** The part of it the anchor reaches too, or null where they do not meet. */
  shared: Region | null;
};

export type ReuseOptions = {
  sampleCap?: number;
  seed?: number;
  /** Relations to report per probe. `backward` is always reported on the graph
   *  inputs whether or not it is listed, because the estimate is made of it. */
  surfaces?: readonly ReuseSurface[];
};

/**
 * One real probe performed by the reuse estimator.
 *
 * `surfaces` carries a relation only when it was asked for, so an unpainted
 * relation is absent rather than empty - on a card those read differently.
 * All of it is paint: the estimate is computed from the same walk but is not
 * read back out of here. Keeping both in one record is what makes the playback
 * the work the estimator did rather than a second derivation of it.
 */
export type ReuseSweepFrame = {
  box: Box;
  weight: number;
  surfaces: Partial<Record<ReuseSurface, Record<string, ReuseReach>>>;
};

export type ReuseSweep = {
  estimates: ReuseEstimate[];
  frames: ReuseSweepFrame[];
  /** The relations the frames carry, so a consumer can tell an unpainted
   *  relation from one that is painted and happens to be empty. */
  surfaces: ReuseSurface[];
};

/** A propagation result reduced to the one region it holds per tensor. */
const regionsOf = (tensors: Map<string, { region: Region }>): Map<string, Region> =>
  new Map([...tensors].map(([tensorId, reached]) => [tensorId, reached.region]));

/** Every tensor an entanglement query reached, as one region each. A tensor
 *  read through two slots has two entries, and their union is what it meets. */
function entangledByTensor(graph: ResolvedGraph, tensorId: string, region: Region): Map<string, Region> {
  const byTensor = new Map<string, Region>();
  for (const entry of entangledWith(graph, tensorId, validateSelection(graph, { tensorId, region }).region)) {
    const found = byTensor.get(entry.tensorId);
    byTensor.set(entry.tensorId, found
      ? canonicalize({
          boxes: [...found.boxes, ...entry.region.boxes],
          exact: found.exact && entry.region.exact,
          reasons: [...new Set([...found.reasons, ...entry.region.reasons])],
        })
      : entry.region);
  }
  return byTensor;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic, non-repeating picks with one sample from each equal stratum. */
function sampledTileIndices(total: number, cap: number, seed: number): number[] {
  if (!Number.isSafeInteger(total) || total < 1)
    throw new Error("reuse tile count must be a positive safe integer");
  if (!Number.isSafeInteger(cap) || cap < 1) throw new Error("sample cap must be a positive safe integer");
  const n = Math.max(1, Math.min(total, Math.floor(cap)));
  if (n === total) return Array.from({ length: total }, (_, index) => index);
  const random = mulberry32(seed);
  return Array.from({ length: n }, (_, index) => {
    const lo = Math.floor((index * total) / n);
    const hi = Math.floor(((index + 1) * total) / n);
    return lo + Math.floor(random() * (hi - lo));
  });
}

/**
 * Estimate how many same-sized tiles reuse each input region touched by one
 * anchored selection. Sampling is seeded and checked through the public
 * executor, so repeated estimates of the same workspace are reproducible.
 */
export function estimateInputReuseSweep(
  graph: ResolvedGraph,
  root: Selection,
  { sampleCap = 48, seed = 0x5eedc0de, surfaces = ["backward"] }: ReuseOptions = {}
): ReuseSweep {
  const painted = new Set<ReuseSurface>(surfaces);
  const reported = [...painted];
  const tensor = graph.tensors[root.tensorId];
  if (!tensor) return { estimates: [], frames: [], surfaces: reported };
  const checkedRoot = executeQuery(graph, {
    ...root,
    direction: painted.has("forward") ? "both" : "backward",
  });
  const current = checkedRoot.backward!;
  if (checkedRoot.selection.region.boxes.length !== 1)
    throw new Error("reuse estimation requires exactly one selection box");
  const rootBox = checkedRoot.selection.region.boxes[0];
  /* The anchor's own side of each painted relation, computed once. A probe's
     `shared` is measured against these, so the emphasis the canvas draws is
     the same overlap the estimate counts rather than a second reading of it. */
  const anchorBackward = regionsOf(current.tensors);
  const anchorForward = checkedRoot.forward ? regionsOf(checkedRoot.forward.tensors) : null;
  const anchorEntangled = painted.has("entangled")
    ? entangledByTensor(graph, root.tensorId, checkedRoot.selection.region)
    : null;

  const shape = tensor.resolved!;
  const tileExtents = rootBox.map((interval) => interval.hi - interval.lo);
  const grid = shape.map((extent, axis) => Math.ceil(extent / tileExtents[axis]));
  const totalTiles = grid.reduce((product, extent) => product * extent, 1);
  const flatIndices = sampledTileIndices(totalTiles, sampleCap, seed);
  const inputs = Object.values(graph.tensors).filter((candidate) =>
    !candidate.producer && current.tensors.has(candidate.id));
  const estimated = new Map(inputs.map((input) => [input.id, 0]));
  const overlapTotals = new Map(inputs.map((input) => [input.id, 0]));
  const exact = new Map(inputs.map((input) => [input.id, current.tensors.get(input.id)!.region.exact]));
  const reasons = new Map(inputs.map((input) => [input.id, new Set(current.tensors.get(input.id)!.region.reasons)]));
  const neighbors = new Map(inputs.map((input) => [input.id, [] as ReuseEstimate["neighbors"]]));
  const frames: ReuseSweepFrame[] = [];
  const overlap = (
    inputId: string,
    probe: PropResult
  ): { region: Region | null; fraction: number; exact: boolean; reasons: string[] } => {
    const anchor = current.tensors.get(inputId)!.region;
    const other = probe.tensors.get(inputId)?.region;
    if (!other) return { region: null, fraction: 0, exact: true, reasons: [] };
    const shared = intersect(anchor, other);
    const size = count(anchor);
    return {
      region: isEmpty(shared) ? null : shared,
      fraction: size ? Math.min(1, count(shared) / size) : 0,
      exact: anchor.exact && other.exact && shared.exact,
      reasons: [...other.reasons, ...shared.reasons],
    };
  };

  /** Pair every tensor a probe reached with the part the anchor reached too. */
  const reachOf = (
    probeSide: Map<string, Region>,
    anchorSide: Map<string, Region> | null
  ): Record<string, ReuseReach> => {
    const out: Record<string, ReuseReach> = {};
    for (const [tensorId, region] of probeSide) {
      const mine = anchorSide?.get(tensorId);
      const meeting = mine ? intersect(region, mine) : null;
      out[tensorId] = { region, shared: meeting && !isEmpty(meeting) ? meeting : null };
    }
    return out;
  };

  for (const [sample, flat] of flatIndices.entries()) {
    // Strata can differ in size. Each sample represents its stratum, not an
    // equal fraction of the grid; otherwise a 5-tile/2-probe sweep is biased.
    const weight = Math.floor((sample + 1) * totalTiles / flatIndices.length) -
      Math.floor(sample * totalTiles / flatIndices.length);
    const tileIndex = new Array<number>(shape.length);
    let rest = flat;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      tileIndex[axis] = rest % grid[axis];
      rest = Math.floor(rest / grid[axis]);
    }
    const probeBox = shape.map((extent, axis) => ({
      lo: tileIndex[axis] * tileExtents[axis],
      hi: Math.min((tileIndex[axis] + 1) * tileExtents[axis], extent),
    }));
    const probeRegion = fromBox(probeBox);
    const probed = executeQuery(graph, {
      tensorId: root.tensorId,
      region: probeRegion,
      direction: painted.has("forward") ? "both" : "backward",
    });
    const probe = probed.backward!;

    for (const input of inputs) {
      const result = overlap(input.id, probe);
      if (result.fraction <= 0) continue; // see NOTE below: a reported miss is a true miss
      estimated.set(input.id, estimated.get(input.id)! + weight);
      overlapTotals.set(input.id, overlapTotals.get(input.id)! + result.fraction * weight);
      if (!result.exact) {
        exact.set(input.id, false);
        for (const reason of result.reasons) reasons.get(input.id)!.add(reason);
      }
    }
    frames.push({
      box: probeBox,
      weight,
      surfaces: {
        ...(painted.has("backward")
          ? { backward: reachOf(regionsOf(probe.tensors), anchorBackward) }
          : {}),
        ...(anchorForward
          ? { forward: reachOf(regionsOf(probed.forward!.tensors), anchorForward) }
          : {}),
        ...(anchorEntangled
          ? {
              entangled: reachOf(
                entangledByTensor(graph, root.tensorId, probeRegion),
                anchorEntangled
              ),
            }
          : {}),
      },
    });
  }

  /* NOTE: only a probe that reported a touch can have reported it falsely.
     Regions are over-approximations, so an empty intersection of two of them is
     an empty intersection of the truth. Region widening therefore makes an
     exhaustive count an upper bound. A sampled sweep remains an estimate in
     either direction because a sample stands in for every tile in its stratum.

     Probe local spatial sharing independently of the global sample. Keep the
     other coordinates fixed, and skip out-of-bounds neighbors (no edge
     resizing). These probes are not part of the sampled count, so their
     precision is reported per neighbor and does not travel into the
     estimate's own `geometryExact`. */
  shape.forEach((extent, axis) => {
    for (const delta of [-1, 1]) {
      const lo = rootBox[axis].lo + delta * tileExtents[axis];
      const hi = rootBox[axis].hi + delta * tileExtents[axis];
      if (lo < 0 || hi > extent) continue;
      const box = rootBox.map((interval, i) => i === axis ? { lo, hi } : interval);
      const probe = executeQuery(graph, { tensorId: root.tensorId, region: fromBox(box), direction: "backward" }).backward!;
      for (const input of inputs) {
        const result = overlap(input.id, probe);
        neighbors.get(input.id)!.push({
          axis,
          delta,
          sharedFraction: result.fraction,
          exact: result.exact,
          reasons: result.reasons,
        });
      }
    }
  });

  return {
    estimates: inputs.map((input) => ({
      tensorId: input.id,
      probes: flatIndices.length,
      totalTiles,
      estimatedTiles: estimated.get(input.id)!,
      exhaustive: flatIndices.length === totalTiles,
      meanSharedFraction: estimated.get(input.id)! > 0 ? overlapTotals.get(input.id)! / estimated.get(input.id)! : null,
      geometryExact: exact.get(input.id)!,
      reasons: [...reasons.get(input.id)!],
      neighbors: neighbors.get(input.id)!,
    })),
    frames,
    surfaces: reported,
  };
}
