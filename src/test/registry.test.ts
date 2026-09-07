/**
 * Properties that must hold for EVERY registered operation, driven off the
 * registry itself rather than a hand-kept list.
 *
 * Two are checked here:
 *
 * 1. Coverage — the fixture table names every registered op, so a new op cannot
 *    ship untested.
 * 2. Adjointness — `forward` and `backward` are two implementations of one
 *    relation, and nothing else in the suite makes them agree directly. The
 *    oracle checks each against truth, but only where the oracle runs; this law
 *    needs no oracle and so runs on every op at every fixture size.
 */

import { describe, expect, it } from "vitest";
import { listOps } from "../core/ops/index";
import { OpCtx } from "../core/ops/types";
import { resolveGraph } from "../core/graph";
import { Box, Region, intersect, isEmpty, iv } from "../core/region";
import { OP_FIXTURES } from "./op-fixtures";
import { checkGraph, rng, randInt } from "./harness";

describe("registry coverage", () => {
  it("every registered op has a fixture", () => {
    const registered = listOps().map((spec) => spec.name).sort();
    const covered = [...new Set(OP_FIXTURES.map((fixture) => fixture.op))].sort();
    // Reported as a set difference rather than a length check, so the failure
    // names the op you just registered instead of a number.
    expect(registered.filter((name) => !covered.includes(name))).toEqual([]);
  });

  it("no fixture names an op that is not registered", () => {
    const registered = listOps().map((spec) => spec.name);
    expect(OP_FIXTURES.map((f) => f.op).filter((op) => !registered.includes(op))).toEqual([]);
  });

  it("every op declares an attribute schema and a positive input arity", () => {
    for (const spec of listOps()) {
      expect(spec.attrSchema, `${spec.name} has no attrSchema`).toBeDefined();
      const min =
        typeof spec.arity.inputs === "number" ? spec.arity.inputs : spec.arity.inputs.min;
      expect(min, `${spec.name} accepts zero inputs`).toBeGreaterThan(0);
    }
  });
});

/**
 * Every fixture against brute-force truth.
 *
 * `ops.test.ts` keeps a richer hand-written corpus with several variants per
 * operation, and that is where a specific tricky case belongs. This one exists
 * for a different reason: it is driven off the same table the coverage test
 * checks, so "registered" implies "oracle-checked" without anyone remembering
 * to add a case.
 */
describe("oracle corpus: every registered op, via its fixture", () => {
  for (const fixture of OP_FIXTURES)
    it(fixture.op, () => checkGraph(fixture.graph, { perTensorElementCap: 12 }));
});

function randomBox(shape: number[], r: () => number): Box {
  return shape.map((e) => {
    const lo = randInt(r, 0, e);
    const hi = randInt(r, lo + 1, e + 1);
    return iv(lo, hi);
  });
}

function fullBox(shape: number[]): Box {
  return shape.map((e) => iv(0, e));
}

/**
 * The dependency relation restricted to `inBox x outBox` is either empty or it
 * is not, and both directions must agree about which:
 *
 *   forward(inBox)[out] meets outBox   <=>   backward(outBox)[in] meets inBox
 *
 * Both sides say "some element of outBox depends on some element of inBox".
 * An over-approximation may report a meeting that truth does not have, so the
 * law is only asserted when both regions involved are exact; the counter below
 * exists so that a fixture quietly going inexact cannot make this vacuous.
 */
describe("adjointness: forward and backward describe one relation", () => {
  for (const fixture of OP_FIXTURES) {
    it(`${fixture.op}${fixture.note ? ` (${fixture.note.split(":")[0]})` : ""}`, () => {
      const g = resolveGraph(fixture.graph);
      const node = g.nodes.find((n) => n.id === fixture.nodeId)!;
      const spec = listOps().find((s) => s.name === node.op)!;
      const ctx: OpCtx = {
        inShapes: g.shapesOf(node.inputs),
        outShapes: g.shapesOf(node.outputs),
        attrs: node.attrs,
      };
      const r = rng(1234);
      let asserted = 0;

      for (let inSlot = 0; inSlot < ctx.inShapes.length; inSlot++) {
        for (let outSlot = 0; outSlot < ctx.outShapes.length; outSlot++) {
          const cases: [Box, Box][] = [
            [fullBox(ctx.inShapes[inSlot]), fullBox(ctx.outShapes[outSlot])],
          ];
          for (let k = 0; k < 24; k++)
            cases.push([
              randomBox(ctx.inShapes[inSlot], r),
              randomBox(ctx.outShapes[outSlot], r),
            ]);

          for (const [inBox, outBox] of cases) {
            const fwd: Region | undefined = spec.forward(inSlot, inBox, ctx)[outSlot];
            const bwd: Region | undefined = spec.backward(outSlot, outBox, ctx)[inSlot];
            if (!fwd || !bwd || !fwd.exact || !bwd.exact) continue;

            const forwardMeets = !isEmpty(
              intersect(fwd, { boxes: [outBox], exact: true, reasons: [] })
            );
            const backwardMeets = !isEmpty(
              intersect(bwd, { boxes: [inBox], exact: true, reasons: [] })
            );
            expect(
              forwardMeets,
              `${fixture.op}: in slot ${inSlot} ${JSON.stringify(inBox)} vs out slot ${outSlot} ` +
                `${JSON.stringify(outBox)} — forward says ${forwardMeets}, backward says ${backwardMeets}`
            ).toBe(backwardMeets);
            asserted++;
          }
        }
      }
      // A fixture that produced only inexact regions would pass every check
      // above without testing anything.
      expect(asserted, `${fixture.op}: no exact pair was available to assert on`).toBeGreaterThan(0);
    });
  }
});
