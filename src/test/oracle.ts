/**
 * Brute-force ground truth (IDEA.md §8): every element of EVERY tensor gets a
 * fresh global id; id-sets are propagated forward through the graph using each
 * op's pointwise `oracleDeps` semantics (never its box-level backward rule).
 * An element's id-set = its own id + all transitive upstream ids, so the true
 * dependency region of a selection on any tensor falls out by membership tests.
 */

import { ResolvedGraph } from "../core/graph";
import { getOp } from "../core/ops/index";
import { OpCtx } from "../core/ops/types";
import { Region, points } from "../core/region";

export type Oracle = {
  /** tensorId -> flat element index -> set of global element ids (incl. own) */
  sets: Map<string, Set<number>[]>;
  idOf: (tensorId: string, index: number[]) => number;
  ownerOf: (globalId: number) => { tensorId: string; flat: number };
};

export function flatIndex(index: number[], shape: number[]): number {
  let f = 0;
  for (let i = 0; i < shape.length; i++) f = f * shape[i] + index[i];
  return f;
}

export function unflatIndex(flat: number, shape: number[]): number[] {
  const idx = new Array(shape.length).fill(0);
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = flat % shape[i];
    flat = Math.floor(flat / shape[i]);
  }
  return idx;
}

export function computeOracle(g: ResolvedGraph): Oracle {
  const base = new Map<string, number>();
  const owners: { tensorId: string; flat: number }[] = [];
  let next = 0;
  for (const t of Object.values(g.tensors)) {
    base.set(t.id, next);
    const n = t.resolved!.reduce((a, b) => a * b, 1);
    for (let f = 0; f < n; f++) owners.push({ tensorId: t.id, flat: f });
    next += n;
  }
  const sets = new Map<string, Set<number>[]>();
  for (const t of Object.values(g.tensors)) {
    const n = t.resolved!.reduce((a, b) => a * b, 1);
    const arr: Set<number>[] = [];
    for (let f = 0; f < n; f++) arr.push(new Set([base.get(t.id)! + f]));
    sets.set(t.id, arr);
  }
  for (const node of g.topo) {
    const spec = getOp(node.op)!;
    const ctx: OpCtx = {
      inShapes: g.shapesOf(node.inputs),
      outShapes: g.shapesOf(node.outputs),
      attrs: node.attrs,
    };
    node.outputs.forEach((outId, slot) => {
      const outShape = g.tensors[outId].resolved!;
      const outN = outShape.reduce((a, b) => a * b, 1);
      const outSets = sets.get(outId)!;
      for (let f = 0; f < outN; f++) {
        const outIdx = unflatIndex(f, outShape);
        const deps = spec.oracleDeps(slot, outIdx, ctx);
        deps.forEach((tuples, inSlot) => {
          const inId = node.inputs[inSlot];
          const inShape = g.tensors[inId].resolved!;
          const inSets = sets.get(inId)!;
          for (const tup of tuples) {
            for (const id of inSets[flatIndex(tup, inShape)]) outSets[f].add(id);
          }
        });
      }
    });
  }
  return {
    sets,
    idOf: (tensorId, index) => base.get(tensorId)! + flatIndex(index, g.tensors[tensorId].resolved!),
    ownerOf: (gid) => owners[gid],
  };
}

/** True dependency set of `selection` (region on selTensor) restricted to tensor `onTensor`. */
export function truthBackward(
  g: ResolvedGraph,
  oracle: Oracle,
  selTensor: string,
  selRegion: Region,
  onTensor: string
): Set<number> {
  const selShape = g.tensors[selTensor].resolved!;
  const selSets = oracle.sets.get(selTensor)!;
  const wanted = new Set<number>();
  for (const p of points(selRegion))
    for (const id of selSets[flatIndex(p, selShape)]) wanted.add(id);
  const out = new Set<number>();
  for (const id of wanted) {
    const o = oracle.ownerOf(id);
    if (o.tensorId === onTensor) out.add(o.flat);
  }
  return out;
}

/** True influence set of `selection` on tensor `onTensor` (forward direction). */
export function truthForward(
  g: ResolvedGraph,
  oracle: Oracle,
  selTensor: string,
  selRegion: Region,
  onTensor: string
): Set<number> {
  const selIds = new Set<number>();
  for (const p of points(selRegion)) selIds.add(oracle.idOf(selTensor, p));
  const onSets = oracle.sets.get(onTensor)!;
  const out = new Set<number>();
  onSets.forEach((s, f) => {
    for (const id of selIds)
      if (s.has(id)) {
        out.add(f);
        break;
      }
  });
  return out;
}

/** Flat element indices covered by a region. */
export function regionToFlatSet(r: Region, shape: number[]): Set<number> {
  const out = new Set<number>();
  for (const p of points(r)) out.add(flatIndex(p, shape));
  return out;
}
