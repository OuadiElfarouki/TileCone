import { executeQuery } from "./executor";
import { ResolvedGraph } from "./graph";
import { PropResult, Selection } from "./propagate";
import { count, fromBox, intersect } from "./region";
import { DTYPE_BYTES } from "./dtypes";

export type ReuseEstimate = {
  tensorId: string;
  touches: number;
  probes: number;
  totalTiles: number;
  estimatedTiles: number;
  exhaustive: boolean;
  /** Average fraction of the anchor footprint shared by overlapping grid tiles. */
  meanSharedFraction: number | null;
  exact: boolean;
  reasons: string[];
  /** Local probes displaced by one tile extent, not a claim of global invariance. */
  neighbors: { axis: number; delta: number; sharedFraction: number; exact: boolean }[];
};

export type InputSharing = {
  tensorId: string;
  tiles: number;
  independentBytes: number;
  unionBytes: number;
  duplicateBytes: number | null;
  exact: boolean;
  reasons: string[];
};

/** Leaf footprints only; excludes cache effects, internal rereads and output traffic.
 * Count a represented union directly, without introducing a box-cap approximation. */
export function inputSharing(graph: ResolvedGraph, cones: PropResult[]): InputSharing[] {
  if (new Set(cones.flatMap((cone) => cone.roots)).size > 1)
    throw new Error("input sharing requires tiles on one tensor");
  return Object.values(graph.tensors).filter((t) => !t.producer).flatMap((input) => {
    const regions = cones.flatMap((cone) => {
      const region = cone.tensors.get(input.id)?.region;
      return region ? [region] : [];
    });
    const boxes = regions.flatMap((region) => region.boxes);
    if (!boxes.length) return [];
    const exact = regions.every((region) => region.exact);
    const reasons = [...new Set(regions.flatMap((region) => region.reasons))];
    const bytes = DTYPE_BYTES[input.dtype];
    const independentBytes = regions.reduce((sum, region) => sum + count(region) * bytes, 0);
    const unionBytes = count({ boxes, exact, reasons }) * bytes;
    return [{ tensorId: input.id, tiles: cones.length, independentBytes, unionBytes,
      duplicateBytes: exact ? Math.max(0, independentBytes - unionBytes) : null, exact, reasons }];
  });
}

export type ReuseOptions = { sampleCap?: number; seed?: number };

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
export function estimateInputReuse(
  graph: ResolvedGraph,
  root: Selection,
  { sampleCap = 48, seed = 0x5eedc0de }: ReuseOptions = {}
): ReuseEstimate[] {
  const tensor = graph.tensors[root.tensorId];
  if (!tensor) return [];
  const checkedRoot = executeQuery(graph, { ...root, direction: "backward" });
  const current = checkedRoot.backward!;
  if (checkedRoot.selection.region.boxes.length !== 1)
    throw new Error("reuse estimation requires exactly one selection box");
  const rootBox = checkedRoot.selection.region.boxes[0];

  const shape = tensor.resolved!;
  const tileExtents = rootBox.map((interval) => interval.hi - interval.lo);
  const grid = shape.map((extent, axis) => Math.ceil(extent / tileExtents[axis]));
  const totalTiles = grid.reduce((product, extent) => product * extent, 1);
  const flatIndices = sampledTileIndices(totalTiles, sampleCap, seed);
  const inputs = Object.values(graph.tensors).filter((candidate) =>
    !candidate.producer && current.tensors.has(candidate.id));
  const touches = new Map(inputs.map((input) => [input.id, 0]));
  const estimated = new Map(inputs.map((input) => [input.id, 0]));
  const overlapTotals = new Map(inputs.map((input) => [input.id, 0]));
  const exact = new Map(inputs.map((input) => [input.id, current.tensors.get(input.id)!.region.exact]));
  const reasons = new Map(inputs.map((input) => [input.id, new Set(current.tensors.get(input.id)!.region.reasons)]));
  const neighbors = new Map(inputs.map((input) => [input.id, [] as ReuseEstimate["neighbors"]]));
  const overlap = (
    inputId: string,
    probe: PropResult
  ): { fraction: number; exact: boolean; reasons: string[] } => {
    const anchor = current.tensors.get(inputId)!.region;
    const other = probe.tensors.get(inputId)?.region;
    if (!other) return { fraction: 0, exact: true, reasons: [] };
    const shared = intersect(anchor, other);
    const size = count(anchor);
    return {
      fraction: size ? Math.min(1, count(shared) / size) : 0,
      exact: anchor.exact && other.exact && shared.exact,
      reasons: [...other.reasons, ...shared.reasons],
    };
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
    const probe = executeQuery(graph, {
      tensorId: root.tensorId,
      region: fromBox(probeBox),
      direction: "backward",
    }).backward!;
    for (const input of inputs) {
      const probed = overlap(input.id, probe);
      if (probed.fraction <= 0) continue; // see NOTE below: a reported miss is a true miss
      touches.set(input.id, touches.get(input.id)! + 1);
      estimated.set(input.id, estimated.get(input.id)! + weight);
      overlapTotals.set(input.id, overlapTotals.get(input.id)! + probed.fraction * weight);
      if (!probed.exact) {
        exact.set(input.id, false);
        for (const reason of probed.reasons) reasons.get(input.id)!.add(reason);
      }
    }
  }

  /* NOTE: only a probe that reported a touch can have reported it falsely.
     Regions are over-approximations, so an empty intersection of two of them is
     an empty intersection of the truth, and the count stays an upper bound
     rather than an unknown.

     Probe local spatial sharing independently of the global sample. Keep the
     other coordinates fixed, and skip out-of-bounds neighbors (no edge
     resizing). These probes are not part of the sampled count, so their
     precision is reported per neighbor and does not travel into the
     estimate's own `exact`. */
  shape.forEach((extent, axis) => {
    for (const delta of [-1, 1]) {
      const lo = rootBox[axis].lo + delta * tileExtents[axis];
      const hi = rootBox[axis].hi + delta * tileExtents[axis];
      if (lo < 0 || hi > extent) continue;
      const box = rootBox.map((interval, i) => i === axis ? { lo, hi } : interval);
      const probe = executeQuery(graph, { tensorId: root.tensorId, region: fromBox(box), direction: "backward" }).backward!;
      for (const input of inputs) {
        const result = overlap(input.id, probe);
        neighbors.get(input.id)!.push({ axis, delta, sharedFraction: result.fraction, exact: result.exact });
      }
    }
  });

  return inputs.map((input) => ({
    tensorId: input.id,
    touches: touches.get(input.id)!,
    probes: flatIndices.length,
    totalTiles,
    estimatedTiles: estimated.get(input.id)!,
    exhaustive: flatIndices.length === totalTiles,
    meanSharedFraction: estimated.get(input.id)! > 0 ? overlapTotals.get(input.id)! / estimated.get(input.id)! : null,
    exact: exact.get(input.id)!,
    reasons: [...reasons.get(input.id)!],
    neighbors: neighbors.get(input.id)!,
  }));
}
