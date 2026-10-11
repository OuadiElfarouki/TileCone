/**
 * Read multiplicity over whole cones, against brute force.
 *
 * `multiplicity.test.ts` checks each operation's count on boxes chosen by the
 * test. This checks what the inspector shows: `operandReads` over a merged
 * backward cone of several tiles, on random composed graphs. The outputs each
 * operation must compute are taken from the oracle's own cone rather than the
 * propagator's, and counted from `oracleDeps`, so a count taken over the wrong
 * region fails here even when every operation's hook is right.
 *
 * Two properties are asserted. No row is lost: every operation and operand the
 * true cone reaches has a row. And every row is right: exact figures equal the
 * truth, and the figures a widened count may only raise are at least it.
 */

import { describe, expect, it } from "vitest";
import { ResolvedGraph, resolveGraph } from "../../src/core/graph";
import { operandReads } from "../../src/core/multiplicity";
import { getOp } from "../../src/core/ops/index";
import type { Limits } from "../../src/core/ops/limits";
import { OpCtx } from "../../src/core/ops/types";
import { mergeProps, propagateBackward } from "../../src/core/propagate";
import { Box, Region, iv } from "../../src/core/region";
import { G, randInt, randomGraph, rng } from "./harness";
import { Oracle, computeOracle, flatIndex, truthBackward, unflatIndex } from "./oracle";

type Truth = { reads: number; elements: number; once: number; reused: number; min: number; max: number };

/** Per (node, tensor read), the true counts over the outputs the true cone needs. */
function trueReads(g: ResolvedGraph, oracle: Oracle, seed: string, region: Region): Map<string, Truth> {
  const out = new Map<string, Truth>();
  for (const node of g.topo) {
    const spec = getOp(node.op)!;
    const ctx: OpCtx = { inShapes: g.shapesOf(node.inputs), outShapes: g.shapesOf(node.outputs), attrs: node.attrs };
    const counts = new Map<string, Map<number, number>>();
    node.outputs.forEach((outId, outSlot) => {
      const shape = g.tensors[outId].resolved!;
      for (const flat of truthBackward(g, oracle, seed, region, outId)) {
        const deps = spec.oracleDeps(outSlot, unflatIndex(flat, shape), ctx);
        node.inputs.forEach((inId, inSlot) => {
          const byElement = counts.get(inId) ?? new Map<number, number>();
          counts.set(inId, byElement);
          const inShape = ctx.inShapes[inSlot];
          for (const f of new Set(deps[inSlot].map((idx) => flatIndex(idx, inShape))))
            byElement.set(f, (byElement.get(f) ?? 0) + 1);
        });
      }
    });
    for (const [tensorId, byElement] of counts) {
      const values = [...byElement.values()];
      if (!values.length) continue;
      out.set(`${node.id}:${tensorId}`, {
        reads: values.reduce((a, b) => a + b, 0),
        elements: values.length,
        once: values.filter((v) => v === 1).length,
        reused: values.filter((v) => v > 1).length,
        min: Math.min(...values),
        max: Math.max(...values),
      });
    }
  }
  return out;
}

function randomBox(shape: number[], r: () => number): Box {
  return shape.map((e) => {
    const lo = randInt(r, 0, e);
    return iv(lo, randInt(r, lo + 1, e + 1));
  });
}

function checkCones(seeds: number, nodes: number, limits?: Limits): { exact: number; inexact: number } {
  const tally = { exact: 0, inexact: 0 };
  for (let seed = 0; seed < seeds; seed++) {
    const r = rng(9000 + seed);
    const g = resolveGraph(randomGraph(r, nodes));
    const oracle = computeOracle(g);
    const produced = Object.values(g.tensors).filter((t) => t.producer);
    if (!produced.length) continue;
    const target = produced[randInt(r, 0, produced.length)];
    // One to three tiles, merged the way the inspector merges a group.
    const boxes = Array.from({ length: randInt(r, 1, 4) }, () => randomBox(target.resolved!, r));
    const cones = boxes.map((b) =>
      propagateBackward(g, { tensorId: target.id, region: { boxes: [b], exact: true, reasons: [] } }, limits)
    );
    const merged = mergeProps(cones)!;
    const union: Region = { boxes, exact: true, reasons: [] };
    const truth = trueReads(g, oracle, target.id, union);
    const rows = operandReads(g, merged.tensors);
    const label = `seed ${seed} on ${target.id} ${JSON.stringify(boxes)}`;

    const shown = new Set(rows.map((row) => `${row.nodeId}:${row.tensorId}`));
    for (const key of truth.keys()) expect(shown.has(key), `${label}: no row for ${key}`).toBe(true);

    for (const row of rows) {
      const want = truth.get(`${row.nodeId}:${row.tensorId}`);
      const at = `${label}: ${row.op} reading ${row.tensorId}`;
      if (row.exact) {
        expect(want, `${at}: an exact row the true cone never reaches`).toBeDefined();
        expect(
          {
            reads: row.reads.value, elements: row.elements.value, once: row.once.value,
            reused: row.reused.value, min: row.min.value, max: row.max.value,
          },
          at
        ).toEqual(want);
        tally.exact++;
      } else {
        const base = want ?? { reads: 0, elements: 0, reused: 0, max: 0 };
        expect(row.reads.value!, at).toBeGreaterThanOrEqual(base.reads);
        expect(row.elements.value!, at).toBeGreaterThanOrEqual(base.elements);
        expect(row.reused.value!, at).toBeGreaterThanOrEqual(base.reused);
        expect(row.max.value!, at).toBeGreaterThanOrEqual(base.max);
        expect(row.reads.status).toBe("upper");
        expect(row.once.status).toBe("approximate");
        tally.inexact++;
      }
    }
  }
  return tally;
}

describe("read multiplicity over merged cones on random graphs", () => {
  it("names every operand the cone reads, with the true counts", () => {
    const { exact } = checkCones(160, 6);
    expect(exact).toBeGreaterThan(200);
  });

  it("keeps every count a bound where the cone itself is widened", () => {
    // Fallbacks are rare at oracle sizes, so this runs longer graphs than the
    // case above, and asserts that some rows were widened so it cannot pass
    // vacuously.
    const { inexact } = checkCones(240, 10, { stridedEnum: 2, diagEnum: 2, reshapeRuns: 1, maxBoxes: 2, tallyPieces: 2 });
    expect(inexact).toBeGreaterThan(0);
  });

  it("marks an operation counted over a widened output region, and bounds it", () => {
    // The second reshape's backward is widened under one run, so the region the
    // cone needs of R is a superset, and the first reshape is counted over it.
    const g = resolveGraph(G({ X: [4, 4] }, [
      ["r", "reshape", ["X"], ["R"], { shape: [2, 8] }],
      ["s", "reshape", ["R"], ["S"], { shape: [4, 4] }],
      ["t", "elementwise", ["S"], ["T"], { fn: "relu", nary: 1 }],
    ]));
    const region: Region = { boxes: [[iv(1, 3), iv(1, 3)]], exact: true, reasons: [] };
    const back = propagateBackward(g, { tensorId: "T", region }, { stridedEnum: 2, diagEnum: 2, reshapeRuns: 1, maxBoxes: 2, tallyPieces: 2 });
    const rows = operandReads(g, back.tensors);
    const truth = trueReads(g, computeOracle(g), "T", region);
    const viaFirst = rows.find((row) => row.nodeId === "r")!;
    expect(back.tensors.get("R")!.region.exact).toBe(false);
    expect(viaFirst.exact).toBe(false);
    expect(viaFirst.elements.value!).toBeGreaterThan(truth.get("r:X")!.elements);
    expect(viaFirst.reads.value!).toBeGreaterThanOrEqual(truth.get("r:X")!.reads);
    // The operations downstream of the widening are counted over exact regions.
    expect(rows.find((row) => row.nodeId === "t")!.exact).toBe(true);
  });
});
