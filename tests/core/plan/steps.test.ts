/**
 * Steps of a task: the reduction divided along its reduced axes.
 *
 * Three kinds of check. Fixtures with figures worked by hand. Identities
 * against the plan as it was before steps existed: a task's steps together
 * read what the task reads, their terms cost what the contraction costs, and a
 * plan that declares nothing has one step per task. And a brute-force oracle:
 * the terms inside a step's box, followed pointwise to the plan's frontier,
 * read exactly what the step reports. The oracle takes the step's box as
 * given; that the boxes are the right ones rests on the identities and on the
 * step-order test, and a region past an enumeration limit is checked as a
 * marked superset in a test of its own.
 */

import { describe, expect, it } from "vitest";
import { compileDSL } from "../../../src/parse/compiler";
import { getOp } from "../../../src/core/ops/index";
import { nodeById } from "../../../src/core/graph";
import { flopsOver } from "../../../src/core/metrics";
import { box, count, fromBox, points, canonicalize, unionOf, type Region } from "../../../src/core/region";
import { tilePlan, taskDemand, type TilePlan, type TaskRef, type StepSpec } from "../../../src/core/plan/plan";
import { supplyOf } from "../../../src/core/plan/interfaces";
import { PlanError } from "../../../src/core/plan/tile-family";
import {
  checkSteps,
  stagingOf,
  stepOf,
  stepPayload,
  stepsOf,
  taskSteps,
  type Staged,
  type TaskSteps,
} from "../../../src/core/plan/steps";
import { computeOracle, flatIndex } from "../../corpus/oracle";

const graphOf = (src: string) => compileDSL(src).resolved;

const GEMM = `M = 256
N = 256
K = 512
A = Tensor(M, K, dtype=float16)
B = Tensor(K, N, dtype=float16)
C = matmul(A, B)
`;

const staged = (plan: TilePlan, task: TaskRef): TaskSteps => {
  const steps = taskSteps(plan, task);
  if (!("staging" in steps)) throw new Error(`unstaged: ${steps.reason}`);
  return steps;
};

const code = (fn: () => unknown) => {
  try {
    fn();
  } catch (e) {
    return (e as PlanError).code;
  }
  return null;
};

const elements = (r: Region) => new Set([...points(r)].map((p) => p.join()));

describe("the GEMM fixture", () => {
  const plan = tilePlan(graphOf(GEMM), { C: [64, 64] }, { C: { chunk: { k: 32 } } });
  const task = { tensorId: "C", coord: [1, 0] };
  const steps = staged(plan, task);

  it("has sixteen steps and a 16 KiB float32 accumulator", () => {
    expect(steps.staging.steps).toBe(16);
    expect(steps.staging.reduced).toEqual([{ axis: 1, label: "k", extent: 512, chunk: 32, count: 16 }]);
    expect(steps.accumulator).toEqual({ dtype: "f32", elements: 4096, bytes: { value: 16384, status: "exact", reasons: [] } });
    expect(steps.epilogue).toEqual([]);
  });

  it("reads 4 KiB of each operand per step", () => {
    const step = stepOf(plan, steps, 3);
    expect(step.coord).toEqual([3]);
    expect(step.reads.map((r) => [r.tensorId, r.slot, r.region.boxes])).toEqual([
      ["A", 0, [box([64, 128], [96, 128])]],
      ["B", 1, [box([96, 128], [0, 64])]],
    ]);
    expect(step.needs).toEqual([]);
    const payload = stepPayload(plan, step);
    expect(payload.tensors.map((t) => [t.tensorId, t.bytes.value])).toEqual([["A", 4096], ["B", 4096]]);
    expect(payload.bytes).toMatchObject({ value: 8192, status: "exact" });
    expect(step.flops).toMatchObject({ value: 2 * 64 * 64 * 32, status: "exact" });
  });

  it("gives the last chunk of an axis the chunk does not divide what is left", () => {
    const ragged = tilePlan(graphOf(GEMM), { C: [64, 64] }, { C: { chunk: { k: 200 } } });
    const s = staged(ragged, task);
    expect(s.staging.steps).toBe(3);
    expect(stepOf(ragged, s, 2).box[1]).toEqual({ lo: 400, hi: 512 });
  });
});

/** The identities every staged task must satisfy against the unstaged plan. */
function checkAgainstTask(plan: TilePlan, task: TaskRef) {
  const steps = staged(plan, task);
  const supply = supplyOf(plan, task);
  const all = [...stepsOf(plan, steps)];
  expect(all.length).toBe(steps.staging.steps);

  // Together, the steps and the epilogue read what the task reads.
  const stepReads = new Map<string, Region[]>();
  for (const read of [...all.flatMap((s) => s.reads), ...steps.epilogue])
    stepReads.set(read.tensorId, [...(stepReads.get(read.tensorId) ?? []), read.region]);
  const taskReads = new Map<string, Region[]>();
  for (const d of supply.demand)
    taskReads.set(d.tensorId, [...(taskReads.get(d.tensorId) ?? []), d.region]);
  expect([...stepReads.keys()].sort()).toEqual([...taskReads.keys()].sort());
  for (const [id, regions] of taskReads)
    expect(elements(unionOf(stepReads.get(id)!)), `reads of ${id}`).toEqual(elements(unionOf(regions)));

  // The steps' terms cost what the contraction costs over the tile, and the
  // rest of the task's work is counted once whichever step needs it.
  const contraction = new Map([[steps.staging.output, { region: fromBox(steps.outBox) }]]);
  const terms = all.reduce((n, s) => n + s.flops.value!, 0);
  expect(terms).toBe(flopsOver(plan.graph, contraction).flops.value);
  const rest = new Map([...supply.work].filter(([id]) => id !== steps.staging.output));
  expect(terms + flopsOver(plan.graph, rest).flops.value!).toBe(supply.flops.value);

  // What the steps need of each untiled tensor is what the task computes of it.
  const needs = new Map<string, Region[]>();
  for (const s of all)
    for (const n of s.needs) needs.set(n.tensorId, [...(needs.get(n.tensorId) ?? []), n.region]);
  for (const [id, regions] of needs)
    expect(elements(unionOf(regions)), `needs of ${id}`).toEqual(elements(supply.work.get(id)!.region));
  // Every tensor the task computes is the tile, the contraction's result, an
  // epilogue tensor, or something a step needs.
  const after = new Set([task.tensorId, steps.staging.output]);
  for (const id of steps.staging.epilogue) after.add(nodeById(plan.graph, id)!.outputs[0]);
  expect([...needs.keys()].sort()).toEqual([...supply.work.keys()].filter((id) => !after.has(id)).sort());
  // One row per operation, slot and tensor, as a task's demand has.
  for (const s of all) {
    const rows = s.reads.map((r) => `${r.node}/${r.slot}/${r.tensorId}`);
    expect(new Set(rows).size).toBe(rows.length);
  }
  return { steps, all, supply };
}

const SMALL = {
  gemm: `A = Tensor(6, 8, dtype=float16)
B = Tensor(8, 4, dtype=float16)
C = matmul(A, B)
`,
  mlp: `X = Tensor(6, 4, dtype=float16)
W1 = Parameter(4, 8, dtype=float16)
W2 = Parameter(8, 6, dtype=float16)
H = matmul(X, W1)
Y = matmul(H, W2)
`,
  epilogue: `A = Tensor(6, 8, dtype=float16)
B = Tensor(8, 4, dtype=float16)
Bias = Parameter(4, dtype=float16)
C = matmul(A, B)
D = add(C, Bias)
E = relu(D)
Y = astype(E, float16)
`,
  prologue: `A = Tensor(6, 8, dtype=float16)
B = Tensor(4, 8, dtype=float16)
S = Parameter(8, dtype=float16)
Ap = relu(A)
Ar = multiply(Ap, S)
Bt = permute_dims(B, [1, 0])
C = matmul(Ar, Bt)
`,
  batched: `Q = Tensor(2, 3, 4, 5, dtype=float16)
Kt = Tensor(2, 3, 6, 5, dtype=float16)
Sc = einsum("bhqd,bhkd->bhqk", Q, Kt)
`,
  twoAxes: `A = Tensor(4, 3, 5, dtype=float16)
B = Tensor(3, 5, 6, dtype=float16)
C = einsum("ijk,jkl->il", A, B)
`,
  shared: `A = Tensor(6, 6, dtype=float16)
C = matmul(A, A)
`,
  sharedComputed: `X = Tensor(6, 6, dtype=float16)
H = relu(X)
C = matmul(H, H)
`,
  directAndBehind: `X = Tensor(6, 6, dtype=float16)
H = relu(X)
C = matmul(X, H)
`,
  diagonal: `A = Tensor(3, 3, 4, dtype=float16)
B = Tensor(4, 5, dtype=float16)
C = einsum("iik,kj->ij", A, B)
`,
  ragged: `A = Tensor(7, 9, dtype=float16)
B = Tensor(9, 5, dtype=float16)
C = matmul(A, B)
`,
};

describe("a plan that declares no steps", () => {
  for (const [name, src, tiles] of [
    ["gemm", SMALL.gemm, { C: [3, 2] }],
    ["mlp with H fused", SMALL.mlp, { Y: [3, 3] }],
    ["epilogue", SMALL.epilogue, { Y: [3, 2] }],
    ["prologue", SMALL.prologue, { C: [3, 2] }],
    ["one computed tensor in both slots", SMALL.sharedComputed, { C: [3, 2] }],
    ["a tensor read directly and behind a prologue", SMALL.directAndBehind, { C: [3, 2] }],
  ] as const)
    it(`${name}: one step per task, reading what the task reads`, () => {
      const plan = tilePlan(graphOf(src), tiles);
      const tensorId = Object.keys(tiles)[0];
      const { steps, all, supply } = checkAgainstTask(plan, { tensorId, coord: [1, 0] });
      expect(steps.staging.steps).toBe(1);
      expect(all.length).toBe(1);
      // The single step's reads are the task's own demand rows, less the epilogue's.
      const epilogue = new Set(steps.staging.epilogue);
      expect(all[0].reads).toEqual(supply.demand.filter((d) => !epilogue.has(d.node)).map(({ supplier: _s, ...d }) => d));
    });
});

describe("steps against the task they divide", () => {
  const cases: [string, string, Record<string, number[]>, Record<string, StepSpec>, TaskRef][] = [
    ["gemm", SMALL.gemm, { C: [3, 2] }, { C: { chunk: { k: 3 } } }, { tensorId: "C", coord: [1, 1] }],
    ["mlp with H fused", SMALL.mlp, { Y: [3, 3] }, { Y: { chunk: { k: 3 } } }, { tensorId: "Y", coord: [1, 0] }],
    ["mlp with H tiled", SMALL.mlp, { H: [3, 4], Y: [3, 3] }, { Y: { chunk: { k: 3 } }, H: { chunk: { k: 2 } } }, { tensorId: "Y", coord: [0, 1] }],
    ["epilogue", SMALL.epilogue, { Y: [3, 2] }, { Y: { chunk: { k: 5 } } }, { tensorId: "Y", coord: [1, 0] }],
    ["prologue and a view", SMALL.prologue, { C: [3, 2] }, { C: { chunk: { k: 3 } } }, { tensorId: "C", coord: [0, 1] }],
    ["batched", SMALL.batched, { Sc: [1, 2, 2, 3] }, { Sc: { chunk: { d: 2 } } }, { tensorId: "Sc", coord: [1, 0, 1, 1] }],
    ["two reduced axes", SMALL.twoAxes, { C: [2, 3] }, { C: { chunk: { j: 2, k: 2 } } }, { tensorId: "C", coord: [1, 1] }],
    ["two reduced axes, reordered", SMALL.twoAxes, { C: [2, 3] }, { C: { chunk: { j: 2, k: 2 }, order: ["k", "j"] } }, { tensorId: "C", coord: [0, 0] }],
    ["shared operand", SMALL.shared, { C: [3, 2] }, { C: { chunk: { k: 4 } } }, { tensorId: "C", coord: [1, 2] }],
    ["one computed tensor in both slots", SMALL.sharedComputed, { C: [3, 2] }, { C: { chunk: { k: 4 } } }, { tensorId: "C", coord: [1, 0] }],
    ["a tensor read directly and behind a prologue", SMALL.directAndBehind, { C: [3, 2] }, { C: { chunk: { k: 4 } } }, { tensorId: "C", coord: [0, 2] }],
    ["a diagonal operand", SMALL.diagonal, { C: [2, 2] }, { C: { chunk: { k: 3 } } }, { tensorId: "C", coord: [1, 2] }],
    ["ragged tiles and chunks, the last of each", SMALL.ragged, { C: [3, 2] }, { C: { chunk: { k: 4 } } }, { tensorId: "C", coord: [2, 2] }],
    ["epilogue, another task", SMALL.epilogue, { Y: [4, 3] }, { Y: { chunk: { k: 3 } } }, { tensorId: "Y", coord: [1, 1] }],
  ];

  for (const [name, src, tiles, specs, task] of cases)
    it(name, () => {
      const plan = tilePlan(graphOf(src), tiles, specs);
      checkSteps(plan);
      const { steps, all } = checkAgainstTask(plan, task);

      // Brute force: the terms in each step's box, followed pointwise to the
      // frontier, read exactly what the step reports.
      const g = plan.graph;
      const frontier = new Set(plan.frontier);
      const oracle = computeOracle(g, frontier);
      const node = nodeById(g, steps.staging.node)!;
      const spec = getOp(node.op)!;
      const ctx = { inShapes: g.shapesOf(node.inputs), outShapes: g.shapesOf(node.outputs), attrs: node.attrs };
      const d = steps.staging.domain;
      const inBox = (index: number[], access: readonly number[], b: (typeof all)[0]["box"]) =>
        index.every((v, ax) => b[access[ax]].lo <= v && v < b[access[ax]].hi);

      for (const step of all) {
        const truth = new Map<string, Set<string>>();
        const add = (tensorId: string, key: string) => {
          if (!truth.has(tensorId)) truth.set(tensorId, new Set());
          truth.get(tensorId)!.add(key);
        };
        for (const out of points(fromBox(steps.outBox)))
          for (const term of spec.oracleTerms!(0, out, ctx)) {
            if (!term.every((index, slot) => inBox(index!, d.inputs[slot], step.box))) continue;
            term.forEach((index, slot) => {
              const tensorId = node.inputs[slot];
              if (frontier.has(tensorId)) return add(tensorId, index!.join());
              const shape = g.tensors[tensorId].resolved;
              for (const id of oracle.sets.get(tensorId)![flatIndex(index!, shape)]) {
                const owner = oracle.ownerOf(id);
                if (!frontier.has(owner.tensorId)) continue;
                const o = g.tensors[owner.tensorId].resolved;
                const idx: number[] = [];
                let f = owner.flat;
                for (let i = o.length - 1; i >= 0; i--) {
                  idx[i] = f % o[i];
                  f = Math.floor(f / o[i]);
                }
                add(owner.tensorId, idx.join());
              }
            });
          }
        const claimed = new Map<string, Region[]>();
        for (const read of step.reads)
          claimed.set(read.tensorId, [...(claimed.get(read.tensorId) ?? []), read.region]);
        expect([...claimed.keys()].sort(), `step ${step.index}`).toEqual([...truth.keys()].sort());
        for (const [id, regions] of claimed) {
          expect(regions.every((r) => r.exact)).toBe(true);
          expect(elements(unionOf(regions)), `step ${step.index} reads of ${id}`).toEqual(truth.get(id));
        }
        // The payload counts each element of a tensor once.
        for (const t of stepPayload(plan, step).tensors)
          expect(count(t.region)).toBe(truth.get(t.tensorId)!.size);
      }
    });
});

describe("step order", () => {
  const g = graphOf(SMALL.twoAxes);
  const coords = (order?: string[]) => {
    const plan = tilePlan(g, { C: [2, 3] }, { C: { chunk: { j: 2, k: 2 }, order } });
    const steps = staged(plan, { tensorId: "C", coord: [0, 0] });
    return [...stepsOf(plan, steps)].map((s) => [s.box[1], s.box[2]].map((I) => `${I.lo}:${I.hi}`).join(" "));
  };

  it("numbers steps with the last axis of the order fastest", () => {
    // j has extent 3 (chunks 0:2, 2:3) and k extent 5 (0:2, 2:4, 4:5).
    expect(coords()).toEqual(["0:2 0:2", "0:2 2:4", "0:2 4:5", "2:3 0:2", "2:3 2:4", "2:3 4:5"]);
    expect(coords(["k", "j"])).toEqual(["0:2 0:2", "2:3 0:2", "0:2 2:4", "2:3 2:4", "0:2 4:5", "2:3 4:5"]);
  });
});

describe("a diagonal past the enumeration limit", () => {
  it("is read as a marked superset, and its bytes as an upper bound", () => {
    const g = graphOf(`A = Tensor(300, 300, 4, dtype=float16)
B = Tensor(4, 2, dtype=float16)
C = einsum("iik,kj->ij", A, B)
`);
    const plan = tilePlan(g, { C: [300, 2] }, { C: { chunk: { k: 2 } } });
    const steps = staged(plan, { tensorId: "C", coord: [0, 0] });
    const step = stepOf(plan, steps, 1);
    const a = step.reads.find((r) => r.tensorId === "A")!;
    expect(a.region.exact).toBe(false);
    // Every diagonal element the step's terms read is inside the bound.
    const claimed = elements(a.region);
    for (let i = 0; i < 300; i++) for (const k of [2, 3]) expect(claimed.has(`${i},${i},${k}`)).toBe(true);
    expect(stepPayload(plan, step).bytes.status).toBe("upper");
    expect(step.flops.status).toBe("exact");
  });
});

describe("what a step of a fused contraction needs", () => {
  it("re-reads X's rows in every step, and a different block of W1", () => {
    const plan = tilePlan(graphOf(SMALL.mlp), { Y: [3, 3] }, { Y: { chunk: { k: 4 } } });
    const steps = staged(plan, { tensorId: "Y", coord: [1, 0] });
    const [first, second] = [...stepsOf(plan, steps)];
    const on = (s: typeof first, id: string) => canonicalize(unionOf(s.reads.filter((r) => r.tensorId === id).map((r) => r.region))).boxes;
    expect(on(first, "X")).toEqual([box([3, 6], [0, 4])]);
    expect(on(second, "X")).toEqual([box([3, 6], [0, 4])]);
    expect(on(first, "W1")).toEqual([box([0, 4], [0, 4])]);
    expect(on(second, "W1")).toEqual([box([0, 4], [4, 8])]);
    expect(on(first, "W2")).toEqual([box([0, 4], [0, 3])]);
    expect(first.needs).toEqual([{ tensorId: "H", region: fromBox(box([3, 6], [0, 4])) }]);
  });
});

describe("the epilogue", () => {
  const plan = tilePlan(graphOf(SMALL.epilogue), { Y: [3, 2] }, { Y: { chunk: { k: 4 } } });
  const steps = staged(plan, { tensorId: "Y", coord: [1, 1] });

  it("finds the contraction behind the cast, the activation and the bias", () => {
    const s = steps.staging as Staged;
    expect(s.output).toBe("C");
    expect(s.epilogue.map((id) => nodeById(plan.graph, id)!.outputs[0])).toEqual(["Y", "E", "D"]);
    expect(s.accumulator).toBe("f32");
    expect(steps.outBox).toEqual(box([3, 6], [2, 4]));
  });

  it("reads the bias once per task and in no step", () => {
    expect(steps.epilogue.map((r) => [r.tensorId, r.region.boxes])).toEqual([["Bias", [box([2, 4])]]]);
    for (const step of stepsOf(plan, steps))
      expect(step.reads.map((r) => r.tensorId)).toEqual(["A", "B"]);
  });
});

describe("the accumulator dtype", () => {
  it("follows an integer result and takes a declared dtype", () => {
    const ints = graphOf(`A = Tensor(4, 4, dtype=int32)
B = Tensor(4, 4, dtype=int32)
C = matmul(A, B)
`);
    expect((stagingOf(tilePlan(ints, { C: [2, 2] }), "C") as Staged).accumulator).toBe("i32");
    const plan = tilePlan(graphOf(SMALL.gemm), { C: [3, 2] }, { C: { accumulator: "f16" } });
    expect((stagingOf(plan, "C") as Staged).accumulator).toBe("f16");
    expect(staged(plan, { tensorId: "C", coord: [0, 0] }).accumulator.bytes.value).toBe(12);
  });

  it("must be a dtype, and the result's kind of number", () => {
    const g = graphOf(SMALL.gemm);
    const bad = (accumulator: string) =>
      code(() => checkSteps(tilePlan(g, { C: [3, 2] }, { C: { accumulator: accumulator as "f32" } })));
    expect(bad("float32")).toBe("PLAN_STEPS");
    expect(bad("bool")).toBe("PLAN_STEPS");
    expect(bad("i32")).toBe("PLAN_STEPS");
    expect(bad("f32")).toBeNull();
  });
});

describe("tensors without steps", () => {
  it("names the operation that is not a contraction", () => {
    const g = graphOf(`X = Tensor(4, 6, dtype=float16)
P = softmax(X, axis=-1)
`);
    const s = stagingOf(tilePlan(g, { P: [2, 3] }), "P");
    expect(s).toEqual({ staged: false, tensorId: "P", reason: "P is computed by softmax, which is not a contraction" });
    expect(taskSteps(tilePlan(g, { P: [2, 3] }), { tensorId: "P", coord: [0, 0] })).toEqual(s);
  });

  it("treats an equation that reduces nothing as no contraction", () => {
    const g = graphOf(`A = Tensor(4, 6, dtype=float16)
B = Tensor(6, 4, dtype=float16)
C = matmul(A, B)
Y = einsum("ij->ji", C)
`);
    expect(stagingOf(tilePlan(g, { Y: [2, 2] }), "Y")).toMatchObject({ staged: false, reason: "Y is computed by einsum, which reduces no axis" });
  });

  it("names two computed operands meeting, and an epilogue with nothing behind it", () => {
    const two = graphOf(`A = Tensor(4, 4, dtype=float16)
B = Tensor(4, 4, dtype=float16)
C = matmul(A, B)
D = matmul(B, A)
Y = add(C, D)
`);
    expect(stagingOf(tilePlan(two, { Y: [2, 2] }), "Y")).toMatchObject({ staged: false, reason: "add combines C and D, each computed in the task" });
    // With C and D tiled, Y's own tasks only add.
    expect(stagingOf(tilePlan(two, { C: [2, 2], D: [2, 2], Y: [2, 2] }), "Y")).toMatchObject({ staged: false, reason: "no contraction is fused into the tasks of Y" });
    // With D tiled, C is the contraction and D is read by the epilogue.
    const one = tilePlan(two, { D: [2, 2], Y: [2, 2] });
    expect(stagingOf(one, "Y")).toMatchObject({ staged: true, output: "C" });
    expect(staged(one, { tensorId: "Y", coord: [0, 1] }).epilogue.map((r) => r.tensorId)).toEqual(["D"]);
  });
});

describe("step declarations the graph does not support", () => {
  const g = graphOf(SMALL.gemm);
  const bad = (spec: StepSpec) => code(() => checkSteps(tilePlan(g, { C: [3, 2] }, { C: spec })));

  it("are refused with PLAN_STEPS", () => {
    expect(bad({ chunk: { z: 2 } })).toBe("PLAN_STEPS");
    expect(bad({ chunk: { m: 2 } })).toBe("PLAN_STEPS");
    expect(bad({ chunk: { k: 0 } })).toBe("PLAN_STEPS");
    expect(bad({ chunk: { k: 9 } })).toBe("PLAN_STEPS");
    expect(bad({ chunk: { k: 2.5 } })).toBe("PLAN_STEPS");
    expect(bad({ order: [] })).toBe("PLAN_STEPS");
    expect(bad({ order: ["k", "k"] })).toBe("PLAN_STEPS");
    expect(bad({ chunk: { k: 8 }, order: ["k"] })).toBeNull();
  });

  it("are refused on a tensor that is not tiled or has no contraction", () => {
    expect(code(() => tilePlan(g, {}, { C: { chunk: { k: 2 } } }))).toBe("PLAN_UNPLANNED");
    const soft = graphOf(`X = Tensor(4, 6, dtype=float16)
P = softmax(X, axis=-1)
`);
    expect(code(() => checkSteps(tilePlan(soft, { P: [2, 3] }, { P: { chunk: { k: 2 } } })))).toBe("PLAN_STEPS");
    // A declaration that declares nothing is not one.
    expect(code(() => checkSteps(tilePlan(soft, { P: [2, 3] }, { P: { chunk: {}, order: [] } })))).toBeNull();
    const plan = tilePlan(g, { C: [3, 2] });
    expect(() => taskSteps(plan, { tensorId: "C", coord: [0, 0] }, taskDemand(plan, { tensorId: "C", coord: [1, 0] }))).toThrow(/another task/);
    expect(code(() => stepOf(tilePlan(g, { C: [3, 2] }), staged(tilePlan(g, { C: [3, 2] }), { tensorId: "C", coord: [0, 0] }), 1))).toBe("PLAN_STEPS");
  });
});
