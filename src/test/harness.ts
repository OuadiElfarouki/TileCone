import { expect } from "vitest";
import { Graph, ResolvedGraph, resolveGraph, Tensor } from "../core/graph";
import { DType } from "../core/dtypes";
import { propagateBackward, propagateForward, propagateWithin, Selection } from "../core/propagate";
import { Region, fromBox, points } from "../core/region";
import { DEFAULT_LIMITS, type Limits } from "../core/ops/limits";
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
  /** Lowered fallback thresholds, so the conservative branches are reachable
   * at shapes a brute-force oracle can still enumerate. */
  limits?: Partial<Limits>;
  seed?: number;
  perTensorElementCap?: number;
  boxSelections?: number;
  forward?: boolean;
  backward?: boolean;
  /** Check bounded cones instead: every walk stops at these tensors, and the
   * oracle cuts its paths at the same ones. */
  frontier?: readonly string[];
};

/**
 * How many walks ran, and how many of them a frontier shortened. A bounded
 * check whose frontiers never cut a path would pass while testing nothing
 * beyond the transitive case, so callers assert on `cut`.
 */
export type CheckStats = { walks: number; cut: number };

/** Exhaustive-ish oracle check of backward and forward propagation on a small graph. */
export function checkGraph(graph: Graph, opts: CheckOpts = {}): CheckStats {
  const {
    limits: limitOverrides,
    seed = 42,
    perTensorElementCap = 24,
    boxSelections = 2,
    forward = true,
    backward = true,
    frontier,
  } = opts;
  const g = resolveGraph(graph);
  const oracle: Oracle = computeOracle(g, new Set(frontier));
  const r = rng(seed);
  const limits = limitOverrides ? { ...DEFAULT_LIMITS, ...limitOverrides } : undefined;
  const stats: CheckStats = { walks: 0, cut: 0 };
  const within = frontier ? ` within [${frontier.join(", ")}]` : "";

  const run = (dir: "backward" | "forward", sel: Selection) => {
    const open =
      dir === "backward" ? propagateBackward(g, sel, limits) : propagateForward(g, sel, limits);
    stats.walks++;
    if (!frontier) return open.tensors;
    const bounded = propagateWithin(g, sel, dir, frontier, limits);
    if (bounded.tensors.size < open.tensors.size) stats.cut++;
    // Crossings split a stopped tensor's region by slot: each lands on a stop,
    // and together they cover exactly the region the oracle checks below.
    for (const c of bounded.crossings) expect(bounded.stoppedAt).toContain(c.tensorId);
    for (const id of bounded.stoppedAt) {
      const shape = g.tensors[id].resolved!;
      const joined = new Set<number>();
      for (const c of bounded.crossings)
        if (c.tensorId === id) for (const f of regionToFlatSet(c.region, shape)) joined.add(f);
      expect(joined, `crossings on ${id}`).toEqual(regionToFlatSet(bounded.tensors.get(id)!.region, shape));
    }
    return bounded.tensors;
  };

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
      if (backward)
        assertAgainstTruth(
          g,
          run("backward", { tensorId: t.id, region: sel }),
          (on) => truthBackward(g, oracle, t.id, sel, on),
          `backward from ${t.id} ${selDesc}${within}`
        );
      if (forward)
        assertAgainstTruth(
          g,
          run("forward", { tensorId: t.id, region: sel }),
          (on) => truthForward(oracle, t.id, sel, on),
          `forward from ${t.id} ${selDesc}${within}`
        );
    }
  }
  return stats;
}

/**
 * A random composed graph: transposes, reductions, reshapes, scans, concats,
 * matmuls, slices, pads, splits, broadcasts and normalizations, with diamonds
 * (an op reading one tensor twice). Small enough for the oracle to enumerate.
 */
export function randomGraph(r: () => number, nNodes: number) {
  type T = { id: string; shape: number[] };
  const inputs: Record<string, number[]> = {};
  const pool: T[] = [];
  let tid = 0;
  const newInput = (shape: number[]) => {
    const id = `in${tid++}`;
    inputs[id] = shape;
    const t = { id, shape };
    pool.push(t);
    return t;
  };
  newInput([randInt(r, 2, 5), randInt(r, 2, 5)]);
  newInput([randInt(r, 2, 5), randInt(r, 2, 5), randInt(r, 2, 4)]);
  const nodes: [string, string, string[], string[], Record<string, unknown>?][] = [];
  let nid = 0;
  const emit = (op: string, ins: T[], outShape: number[], attrs?: Record<string, unknown>) => {
    const id = `t${tid++}`;
    nodes.push([`n${nid++}`, op, ins.map((x) => x.id), [id], attrs]);
    const t = { id, shape: outShape };
    pool.push(t);
    return t;
  };
  const pick = () => pool[randInt(r, 0, pool.length)];
  for (let k = 0; k < nNodes; k++) {
    const choice = randInt(r, 0, 14);
    const t = pick();
    const sh = t.shape;
    if (choice === 0 && sh.length >= 2) {
      const perm = sh.map((_, i) => i);
      for (let i = perm.length - 1; i > 0; i--) {
        const j = randInt(r, 0, i + 1);
        [perm[i], perm[j]] = [perm[j], perm[i]];
      }
      emit("transpose", [t], perm.map((p) => sh[p]), { perm });
    } else if (choice === 1) {
      // diamond: t + t
      emit("elementwise", [t, t], sh.slice(), { fn: "add", nary: 2 });
    } else if (choice === 2 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      emit("softmax", [t], sh.slice(), { axis });
    } else if (choice === 3 && sh.length >= 2) {
      const axis = randInt(r, 0, sh.length);
      const out = sh.filter((_, i) => i !== axis);
      emit("reduce", [t], out, { fn: "sum", axes: [axis], keepdim: false });
    } else if (choice === 4) {
      // reshape to random re-factorization
      const vol = sh.reduce((a, b) => a * b, 1);
      const dims: number[] = [];
      let rest = vol;
      while (rest > 1 && dims.length < 3) {
        const divisors: number[] = [];
        for (let d = 2; d <= rest; d++) if (rest % d === 0) divisors.push(d);
        const d = divisors[randInt(r, 0, divisors.length)];
        dims.push(d);
        rest /= d;
      }
      if (rest > 1) dims.push(rest);
      if (dims.length === 0) dims.push(1);
      emit("reshape", [t], dims, { shape: dims });
    } else if (choice === 5 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      emit("cumsum", [t], sh.slice(), { axis, reverse: r() < 0.5 });
    } else if (choice === 6 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      const out = sh.slice();
      out[axis] *= 2;
      emit("concat", [t, t], out, { axis });
    } else if (choice === 7 && sh.length === 2) {
      const other = newInput([sh[1], randInt(r, 2, 4)]);
      emit("matmul", [t, other], [sh[0], other.shape[1]]);
    } else if (choice === 8 && sh.length >= 1) {
      // slice, sometimes strided
      const step = randInt(r, 1, 3);
      const starts = sh.map(() => 0);
      const stops = sh.slice();
      const steps = sh.map(() => 1);
      const axis = randInt(r, 0, sh.length);
      steps[axis] = step;
      starts[axis] = randInt(r, 0, Math.max(1, sh[axis] - 1));
      const out = sh.map((e, i) =>
        i === axis ? Math.max(1, Math.ceil((e - starts[i]) / steps[i])) : e
      );
      emit("slice", [t], out, { starts, stops, steps });
    } else if (choice === 9 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      const pads: [number, number][] = sh.map(() => [0, 0]);
      const mode = ["constant", "replicate", "reflect"][randInt(r, 0, 3)];
      // reflect cannot pad wider than extent-1
      const cap = mode === "reflect" ? Math.max(0, sh[axis] - 1) : 2;
      pads[axis] = [randInt(r, 0, Math.min(2, cap) + 1), randInt(r, 0, Math.min(2, cap) + 1)];
      emit("pad", [t], sh.map((e, i) => e + pads[i][0] + pads[i][1]), { pads, mode });
    } else if (choice === 10 && sh.length >= 1) {
      const axis = randInt(r, 0, sh.length);
      if (sh[axis] < 2) {
        emit("elementwise", [t], sh.slice(), { fn: "relu", nary: 1 });
      } else {
        const cut = randInt(r, 1, sh[axis]);
        const sizes = [cut, sh[axis] - cut];
        const outs = sizes.map((sz) => sh.map((e, i) => (i === axis ? sz : e)));
        // split is the only multi-output op; take the first piece onward
        const id = `t${tid++}`;
        const id2 = `t${tid++}`;
        nodes.push([`n${nid++}`, "split", [t.id], [id, id2], { axis, sizes }]);
        pool.push({ id, shape: outs[0] });
        pool.push({ id: id2, shape: outs[1] });
      }
    } else if (choice === 11 && sh.length >= 1) {
      // expand a fresh degenerate axis up to t's shape
      const axis = randInt(r, 0, sh.length);
      const src = newInput(sh.map((e, i) => (i === axis ? 1 : e)));
      emit("expand", [src], sh.slice(), { shape: sh.slice() });
    } else if (choice === 12 && sh.length >= 1) {
      const perm = sh.map((_, i) => i);
      emit("transpose", [t], perm.map((p) => sh[p]), { perm });
    } else if (choice === 13 && sh.length >= 1) {
      const axes = [randInt(r, 0, sh.length)];
      const wShape = axes.map((a) => sh[a]);
      // normalize's affine params must match the normalized axes, and
      // expansion requires them to be trailing
      if (axes[0] === sh.length - 1) {
        const w = newInput(wShape);
        emit("normalize", [t, w], sh.slice(), {
          kind: r() < 0.5 ? "layernorm" : "rmsnorm",
          axes,
          hasWeight: true,
          hasBias: false,
        });
      } else emit("elementwise", [t], sh.slice(), { fn: "relu", nary: 1 });
    } else {
      emit("elementwise", [t], sh.slice(), { fn: "relu", nary: 1 });
    }
  }
  const graph = G(inputs, nodes);
  // declared shapes for intermediates are unknown; leave empty (inferred)
  return graph;
}

export { points };
