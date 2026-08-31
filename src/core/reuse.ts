import { executeQuery } from "./executor";
import { ResolvedGraph } from "./graph";
import { Selection } from "./propagate";
import { fromBox, intersect, isEmpty } from "./region";

export type ReuseEstimate = {
  tensorId: string;
  touches: number;
  probes: number;
  totalTiles: number;
  estimatedTiles: number;
};

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
export function sampledTileIndices(total: number, count: number, seed: number): number[] {
  if (!Number.isSafeInteger(total) || total < 1)
    throw new Error("reuse tile count must be a positive safe integer");
  const n = Math.max(1, Math.min(total, Math.floor(count)));
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
  const inputs = Object.values(graph.tensors).filter((candidate) => !candidate.producer);
  const touches = new Map(inputs.map((input) => [input.id, 0]));

  for (const flat of flatIndices) {
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
      const anchorRegion = current.tensors.get(input.id)?.region;
      const probeRegion = probe.tensors.get(input.id)?.region;
      if (anchorRegion && probeRegion && !isEmpty(intersect(anchorRegion, probeRegion)))
        touches.set(input.id, touches.get(input.id)! + 1);
    }
  }

  return inputs.map((input) => ({
    tensorId: input.id,
    touches: touches.get(input.id)!,
    probes: flatIndices.length,
    totalTiles,
    estimatedTiles: (touches.get(input.id)! / flatIndices.length) * totalTiles,
  }));
}
