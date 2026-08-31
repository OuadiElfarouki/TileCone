import { expect } from "vitest";
import { Graph, ResolvedGraph, resolveGraph, Tensor } from "../core/graph";
import { DType } from "../core/dtypes";
import { propagateBackward, propagateForward } from "../core/propagate";
import { Region, fromBox, points } from "../core/region";
import { computeOracle, regionToFlatSet, truthBackward, truthForward, unflatIndex, Oracle } from "./oracle";

export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const randInt = (r: () => number, lo: number, hi: number) =>
  lo + Math.floor(r() * (hi - lo));

/** Terse graph builder for tests. */
export function G(
  inputs: Record<string, number[]>,
  nodes: [id: string, op: string, ins: string[], outs: string[], attrs?: Record<string, unknown>][],
  extraTensors: string[] = []
): Graph {
  const tensors: Record<string, Tensor> = {};
  for (const [id, shape] of Object.entries(inputs))
    tensors[id] = { id, name: id, shape, dtype: "f32" as DType };
  for (const [, , , outs] of nodes)
    for (const o of outs) if (!tensors[o]) tensors[o] = { id: o, name: o, shape: [], dtype: "f32" };
  for (const id of extraTensors)
    if (!tensors[id]) tensors[id] = { id, name: id, shape: [], dtype: "f32" };
  return {
    nodes: nodes.map(([id, op, ins, outs, attrs]) => ({ id, op, inputs: ins, outputs: outs, attrs: attrs ?? {} })),
    tensors,
    params: {},
  };
}

function volume(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function sampleElements(shape: number[], cap: number, r: () => number): number[][] {
  const n = volume(shape);
  if (n <= cap) {
    const out: number[][] = [];
    for (let f = 0; f < n; f++) out.push(unflatIndex(f, shape));
    return out;
  }
  const out: number[][] = [];
  for (let k = 0; k < cap; k++) out.push(unflatIndex(randInt(r, 0, n), shape));
  return out;
}

function randomBox(shape: number[], r: () => number): Region {
  return fromBox(
    shape.map((e) => {
      const lo = randInt(r, 0, e);
      const hi = randInt(r, lo + 1, e + 1);
      return { lo, hi };
    })
  );
}

function assertAgainstTruth(
  g: ResolvedGraph,
  analytic: Map<string, { region: Region }>,
  truthFor: (onTensor: string) => Set<number>,
  label: string
) {
  for (const t of Object.values(g.tensors)) {
    const truth = truthFor(t.id);
    const entry = analytic.get(t.id);
    const got = entry ? regionToFlatSet(entry.region, t.resolved!) : new Set<number>();
    if (!entry || entry.region.exact) {
      // exact => equal
      expect(got, `${label}: tensor ${t.id} (exact) mismatch`).toEqual(truth);
    } else {
      // inexact => superset, and NEVER a strict subset
      for (const f of truth)
        expect(got.has(f), `${label}: tensor ${t.id} approx region misses element ${f}`).toBe(true);
    }
  }
}

export type CheckOpts = {
  seed?: number;
  perTensorElementCap?: number;
  boxSelections?: number;
  forward?: boolean;
  backward?: boolean;
};

/** Exhaustive-ish oracle check of backward and forward propagation on a small graph. */
export function checkGraph(graph: Graph, opts: CheckOpts = {}): void {
  const {
    seed = 42,
    perTensorElementCap = 24,
    boxSelections = 2,
    forward = true,
    backward = true,
  } = opts;
  const g = resolveGraph(graph);
  const oracle: Oracle = computeOracle(g);
  const r = rng(seed);

  for (const t of Object.values(g.tensors)) {
    const shape = t.resolved!;
    if (volume(shape) === 0) continue;
    const sels: Region[] = [];
    if (shape.length === 0) sels.push(fromBox([]));
    else {
      for (const idx of sampleElements(shape, perTensorElementCap, r))
        sels.push(fromBox(idx.map((v) => ({ lo: v, hi: v + 1 }))));
      for (let k = 0; k < boxSelections; k++) sels.push(randomBox(shape, r));
    }
    for (const sel of sels) {
      const selDesc = JSON.stringify(sel.boxes);
      if (backward) {
        const res = propagateBackward(g, { tensorId: t.id, region: sel });
        assertAgainstTruth(
          g,
          res.tensors,
          (on) => truthBackward(g, oracle, t.id, sel, on),
          `backward from ${t.id} ${selDesc}`
        );
      }
      if (forward) {
        const res = propagateForward(g, { tensorId: t.id, region: sel });
        assertAgainstTruth(
          g,
          res.tensors,
          (on) => truthForward(g, oracle, t.id, sel, on),
          `forward from ${t.id} ${selDesc}`
        );
      }
    }
  }
}

export { points };
