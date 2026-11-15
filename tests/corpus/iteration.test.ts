/**
 * Iteration domains against the terms themselves.
 *
 * An operation that declares `iteration` claims its arithmetic is a loop nest
 * and that every relation it reports is a projection of a box of that nest.
 * Two things are checked for each declaring operation:
 *
 * 1. Against truth. `oracleTerms` lists every term of every output element,
 *    from the operation's definition. The terms inside an iteration box read a
 *    set of operand elements, and `projectInput` must equal that set when exact
 *    and contain it when not. This holds for boxes narrower than a reduced
 *    axis, which is the case no other hook can state and a staged plan rests on.
 *    The diagonal cases run a second time with `diagEnum` lowered, so the
 *    blocked-staircase bound is checked as a superset and not only enumerated.
 * 2. Against the operation's own hooks. `backward`, `forward`, `coaccess`,
 *    `multiplicity` and `flopsFor` must be the projections the domain gives.
 *    For einsum that is true by construction; the law is here for the next
 *    operation to declare a domain without rewriting its hooks over it.
 */

import { describe, expect, it } from "vitest";
import { getOp, listOps } from "../../src/core/ops/index";
import type { OpCtx, OpSpec } from "../../src/core/ops/types";
import {
  type IterationDomain,
  fibreOf,
  fullIterationBox,
  iterationBoxOfInput,
  iterationBoxOfOutput,
  iterationVolume,
  projectInput,
  projectOutput,
  spansReduced,
} from "../../src/core/iteration";
import { Box, empty, fromBox, isEmpty, iv, points } from "../../src/core/region";
import { measureTally } from "../../src/core/tally";
import { DEFAULT_LIMITS, limitsOf } from "../../src/core/ops/limits";
import { rng, randInt } from "./harness";

type Case = { name: string; spec: OpSpec; ctx: OpCtx; expectInexact?: true };

function caseOf(op: string, inShapes: number[][], attrs: Record<string, unknown> = {}): Case {
  const spec = getOp(op)!;
  const label = attrs.equation ? `${op} ${attrs.equation}` : `${op} rank ${inShapes[0].length}`;
  return { name: label, spec, ctx: { inShapes, outShapes: spec.inferShapes(inShapes, attrs), attrs } };
}

const eq = (equation: string, ...inShapes: number[][]) => caseOf("einsum", inShapes, { equation });

/** The same equation with the diagonal enumeration cut short, so a diagonal
 * wider than `diagEnum` takes the staircase bound. */
function bounded(c: Case, diagEnum: number): Case {
  return {
    ...c,
    name: `${c.name}, diagEnum ${diagEnum}`,
    ctx: { ...c.ctx, limits: { ...DEFAULT_LIMITS, diagEnum } },
    expectInexact: true,
  };
}

const CASES: Case[] = [
  eq("mk,kn->mn", [3, 4], [4, 5]),
  eq("bmk,bkn->bmn", [2, 3, 4], [2, 4, 3]),
  eq("ij,jk,kl->il", [2, 3], [3, 2], [2, 3]),
  eq("ij,ij->i", [3, 4], [3, 4]),
  eq("i,j->ij", [3], [4]),
  eq("ij->i", [3, 4]),
  eq("ij->ji", [3, 4]),
  eq("ij,kj->ik", [3, 4], [2, 4]),
  eq("ii->i", [4, 4]),
  eq("ii->", [4, 4]),
  eq("iij,jk->ik", [3, 3, 2], [2, 4]),
  caseOf("matmul", [[3, 4], [4, 5]]),
  caseOf("matmul", [[2, 3, 4], [2, 4, 3]]),
  caseOf("matmul", [[2, 2, 3, 2], [2, 2, 2, 3]]),
  caseOf("linear", [[3, 4], [5, 4]]),
  caseOf("linear", [[2, 3, 4], [5, 4]]),
  bounded(eq("ii->i", [4, 4]), 2),
  bounded(eq("ii->", [4, 4]), 1),
  bounded(eq("iij,jk->ik", [3, 3, 2], [2, 4]), 2),
];

function randomBox(extents: readonly number[], r: () => number): Box {
  return extents.map((e) => {
    const lo = randInt(r, 0, e);
    return iv(lo, randInt(r, lo + 1, e + 1));
  });
}

const within = (index: readonly number[], box: Box) =>
  index.every((v, ax) => box[ax].lo <= v && v < box[ax].hi);

const fullOf = (shape: readonly number[]): Box => shape.map((e) => iv(0, e));

/**
 * Every term of the operation, as the element each operand contributes and the
 * output element it belongs to. A term lies in an iteration box when every
 * index it uses lies in the box's interval for the loop axis that index is
 * addressed by, so membership needs only the access maps, not a label
 * assignment the oracle does not carry.
 */
function termsOf(c: Case): { out: number[]; reads: number[][] }[] {
  const terms: { out: number[]; reads: number[][] }[] = [];
  for (const out of points(fromBox(fullOf(c.ctx.outShapes[0]))))
    for (const term of c.spec.oracleTerms!(0, out, c.ctx))
      terms.push({ out, reads: term.map((t) => t!) });
  return terms;
}

function inIterBox(d: IterationDomain, term: { out: number[]; reads: number[][] }, box: Box) {
  const ok = (index: number[], access: readonly number[]) =>
    index.every((v, ax) => box[access[ax]].lo <= v && v < box[access[ax]].hi);
  return ok(term.out, d.output) && term.reads.every((index, slot) => ok(index, d.inputs[slot]));
}

describe("the operations that declare a domain", () => {
  it("einsum, matmul and linear", () => {
    const declaring = listOps().filter((s) => s.iteration).map((s) => s.name).sort();
    expect(declaring).toEqual(["einsum", "linear", "matmul"]);
    for (const spec of listOps().filter((s) => s.iteration)) {
      expect(spec.flopsForIteration, `${spec.name}: iteration without flopsForIteration`).toBeDefined();
      expect(spec.oracleTerms, `${spec.name}: iteration without oracleTerms`).toBeDefined();
    }
  });
});

describe("iteration boxes against the terms", () => {
  for (const c of CASES)
    it(c.name, () => {
      const d = c.spec.iteration!(c.ctx);
      const terms = termsOf(c);
      const r = rng(4321);
      const diagEnum = limitsOf(c.ctx).diagEnum;
      let inexact = 0;

      // One point of the domain is one term.
      expect(iterationVolume(fullIterationBox(d))).toBe(terms.length);

      const boxes = [fullIterationBox(d)];
      for (let k = 0; k < 40; k++) boxes.push(randomBox(d.axes.map((a) => a.extent), r));

      for (const box of boxes) {
        const inside = terms.filter((t) => inIterBox(d, t, box));
        expect(iterationVolume(box), `terms in ${JSON.stringify(box)}`).toBe(inside.length);

        // The output elements the box contributes to.
        const outs = new Set(inside.map((t) => t.out.join()));
        const outBox = projectOutput(d, box);
        expect(outs).toEqual(new Set([...points(fromBox(outBox))].map((p) => p.join())));

        for (let slot = 0; slot < d.inputs.length; slot++) {
          const truth = new Map<string, Set<string>>();
          for (const t of inside) {
            const key = t.reads[slot].join();
            if (!truth.has(key)) truth.set(key, new Set());
            truth.get(key)!.add(t.out.join());
          }
          const region = projectInput(d, slot, box, diagEnum);
          if (!region.exact) inexact++;
          const claimed = new Set([...points(region)].map((p) => p.join()));
          if (region.exact) expect(claimed, `slot ${slot} of ${JSON.stringify(box)}`).toEqual(new Set(truth.keys()));
          else for (const key of truth.keys()) expect(claimed.has(key)).toBe(true);

          // The fibre: how many outputs in the box read each element read.
          const fibre = fibreOf(d, slot, box);
          for (const [key, readers] of truth)
            expect(readers.size, `readers of slot ${slot} element ${key}`).toBe(fibre);
        }

        // A box completes its outputs exactly when it holds all their terms.
        const all = terms.filter((t) => within(t.out, outBox)).length;
        expect(spansReduced(d, box)).toBe(inside.length === all);

        expect(c.spec.flopsForIteration!(box, c.ctx)).toBe(
          (inside.length * c.spec.flopsFor(0, fullOf(c.ctx.outShapes[0]), c.ctx)) / terms.length
        );
      }
      // A lowered limit that never bit would leave the superset rule untested.
      if (c.expectInexact) expect(inexact).toBeGreaterThan(0);
      else expect(inexact).toBe(0);

      // No terms: nothing is read, by a plain operand or along a diagonal.
      const none = d.axes.map(() => iv(0, 0));
      expect(iterationVolume(none)).toBe(0);
      expect(c.spec.flopsForIteration!(none, c.ctx)).toBe(0);
      d.inputs.forEach((access, slot) => {
        if (access.length) expect(isEmpty(projectInput(d, slot, none, diagEnum))).toBe(true);
      });
      c.ctx.inShapes.forEach((shape, slot) => {
        if (shape.length) expect(iterationBoxOfInput(d, slot, shape.map(() => iv(0, 0)))).toBeNull();
      });
    });
});

describe("an operation's hooks are the projections of its domain", () => {
  for (const c of CASES)
    it(c.name, () => {
      const d = c.spec.iteration!(c.ctx);
      const diagEnum = limitsOf(c.ctx).diagEnum;
      const r = rng(99);
      const outShape = c.ctx.outShapes[0];

      for (let k = 0; k < 24; k++) {
        const outBox = k === 0 ? fullOf(outShape) : randomBox(outShape, r);
        const iterBox = iterationBoxOfOutput(d, outBox);
        expect(spansReduced(d, iterBox)).toBe(true);
        expect(projectOutput(d, iterBox)).toEqual(outBox);

        const back = c.spec.backward(0, outBox, c.ctx);
        back.forEach((region, slot) => {
          expect(region).toEqual(projectInput(d, slot, iterBox, diagEnum));
          const tally = c.spec.multiplicity!(0, outBox, slot, c.ctx);
          if (tally.exact && tally.pieces.length)
            expect(measureTally(tally).max).toBe(fibreOf(d, slot, iterBox));
        });
        expect(c.spec.flopsForIteration!(iterBox, c.ctx)).toBe(c.spec.flopsFor(0, outBox, c.ctx));

        // Splitting a reduced axis anywhere splits the work and nothing else.
        d.axes.forEach((axis, i) => {
          if (axis.role !== "reduced" || axis.extent < 2) return;
          const cut = randInt(r, 1, axis.extent);
          const lo = iterBox.map((I, j) => (j === i ? iv(0, cut) : I));
          const hi = iterBox.map((I, j) => (j === i ? iv(cut, axis.extent) : I));
          expect(
            c.spec.flopsForIteration!(lo, c.ctx) + c.spec.flopsForIteration!(hi, c.ctx)
          ).toBe(c.spec.flopsFor(0, outBox, c.ctx));
          expect(spansReduced(d, lo)).toBe(false);
        });
      }

      c.ctx.inShapes.forEach((shape, slot) => {
        for (let k = 0; k < 24; k++) {
          const inBox = randomBox(shape, r);
          const iterBox = iterationBoxOfInput(d, slot, inBox);
          expect(c.spec.forward(slot, inBox, c.ctx)).toEqual([
            iterBox ? fromBox(projectOutput(d, iterBox)) : empty(outShape.length),
          ]);
          c.ctx.inShapes.forEach((other, otherSlot) => {
            if (otherSlot === slot) return;
            expect(c.spec.coaccess!(slot, inBox, otherSlot, c.ctx)).toEqual(
              iterBox ? projectInput(d, otherSlot, iterBox, diagEnum) : empty(other.length)
            );
          });
        }
      });
    });
});
