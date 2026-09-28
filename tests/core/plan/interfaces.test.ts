import { describe, expect, it } from "vitest";
import { DTYPE_BYTES } from "../../../src/core/dtypes";
import { resolveGraph } from "../../../src/core/graph";
import { DEFAULT_LIMITS, Limits } from "../../../src/core/ops/limits";
import { getOp } from "../../../src/core/ops/index";
import { propagateWithin } from "../../../src/core/propagate";
import { interfaceOf, joinDemand, planReport, supplyOf, Supply, Work } from "../../../src/core/plan/interfaces";
import { TaskRef, tilePlan, TilePlan } from "../../../src/core/plan/plan";
import { PlanError, tileBox, tileOrdinal, tiles } from "../../../src/core/plan/tile-family";
import { Box, fromBox, points, Region } from "../../../src/core/region";
import { compileDSL } from "../../../src/parse/compiler";
import { randInt, randomGraph, rng } from "../../corpus/harness";
import { flatIndex, regionToFlatSet, unflatIndex } from "../../corpus/oracle";

/* NEXT_FEATS §8: two matmuls whose tilings meet at C. */
const chain = () =>
  compileDSL(`A = Tensor(256, 256, dtype=fp16)
B = Tensor(256, 256, dtype=fp16)
W = Tensor(256, 128, dtype=fp16)
C = matmul(A, B)
Y = matmul(C, W)
`).resolved;

const keys = (s: Supply) => s.producers.map((p) => `${p.task.tensorId}[${p.task.coord.join(",")}]`);
const code = (fn: () => unknown) => {
  try {
    fn();
  } catch (e) {
    return (e as PlanError).code;
  }
  return null;
};

describe("the two-matmul chain", () => {
  it("rejects invalid task budgets instead of disabling the guard", () => {
    const plan = tilePlan(chain(), { Y: [64, 64] });
    for (const budget of [0, -1, NaN, Infinity])
      expect(code(() => interfaceOf(plan, "Y", { budget }))).toBe("PLAN_SIZE");
  });

  it("has sixteen C tasks and eight Y tasks at 64x64", () => {
    const plan = tilePlan(chain(), { C: [64, 64], Y: [64, 64] });
    expect(plan.families.get("C")!.count).toBe(16);
    expect(plan.families.get("Y")!.count).toBe(8);
  });

  it("supplies each Y task from the four C tasks in its row band", () => {
    const plan = tilePlan(chain(), { C: [64, 64], Y: [64, 64] });
    const s = supplyOf(plan, { tensorId: "Y", coord: [1, 0] });
    expect(keys(s)).toEqual(["C[1,0]", "C[1,1]", "C[1,2]", "C[1,3]"]);
    for (const p of s.producers) {
      expect(p.definite).toBe(true);
      expect(p.used).toMatchObject({ value: 4096, status: "exact" });
      expect(p.volume).toBe(4096);
      expect(p.witnesses).toEqual([
        { node: expect.any(String), slot: 0, region: expect.objectContaining({ exact: true }) },
      ]);
    }
    expect(s.demand.map((d) => [d.tensorId, d.slot, d.supplier])).toEqual([
      ["C", 0, "tasks"],
      ["W", 1, "input"],
    ]);
    expect(s.computes).toEqual([]);
  });

  it("lets both Y tasks in a band share its C tasks, and no other band's", () => {
    const plan = tilePlan(chain(), { C: [64, 64], Y: [64, 64] });
    for (let band = 0; band < 4; band++) {
      const left = supplyOf(plan, { tensorId: "Y", coord: [band, 0] });
      const right = supplyOf(plan, { tensorId: "Y", coord: [band, 1] });
      expect(keys(left)).toEqual(keys(right));
      for (const p of left.producers) expect(p.task.coord[0]).toBe(band);
    }
  });

  it("needs eight C tasks when the Y row tile doubles to 128", () => {
    const plan = tilePlan(chain(), { C: [64, 64], Y: [128, 64] });
    expect(supplyOf(plan, { tensorId: "Y", coord: [0, 0] }).producers).toHaveLength(8);
  });

  it("measures demand duplication and fan-out across the Y family", () => {
    const report = interfaceOf(tilePlan(chain(), { C: [64, 64], Y: [64, 64] }), "Y");
    if (report.status !== "evaluated") throw new Error(report.status);
    const [c, w] = report.boundary;

    // Eight tasks each read a 64x256 fp16 band of C: twice the 256x256 of C there is.
    expect(c).toMatchObject({ tensorId: "C", supplier: "tasks", readers: 8, exact: true });
    expect(c.summed).toMatchObject({ value: 8 * 64 * 256 * 2, status: "exact" });
    expect(c.distinct).toMatchObject({ value: 256 * 256 * 2, status: "exact" });
    expect(c.duplication).toMatchObject({ value: 2, status: "exact" });
    expect([...c.fanOut!.values()]).toEqual(new Array(16).fill(2));

    // Each reads a 256x64 column block of W, which is 256x128 in total.
    expect(w).toMatchObject({ tensorId: "W", supplier: "input", fanOut: null });
    expect(w.duplication).toMatchObject({ value: 4, status: "exact" });
  });

  it("records partial use where the tilings do not align, and a shorter tail", () => {
    const plan = tilePlan(chain(), { C: [64, 64], Y: [96, 64] });
    const first = supplyOf(plan, { tensorId: "Y", coord: [0, 0] });
    // Rows 0-96 read all of C's first row band and half of its second.
    expect(keys(first)).toEqual([
      "C[0,0]", "C[0,1]", "C[0,2]", "C[0,3]",
      "C[1,0]", "C[1,1]", "C[1,2]", "C[1,3]",
    ]);
    expect(first.producers.map((p) => p.used.value)).toEqual([...new Array(4).fill(4096), ...new Array(4).fill(2048)]);

    // The third row tile is the 64-row tail, rows 192-256.
    const tail = supplyOf(plan, { tensorId: "Y", coord: [2, 0] });
    expect(keys(tail)).toEqual(["C[3,0]", "C[3,1]", "C[3,2]", "C[3,3]"]);
  });
});

describe("operand slots and suppliers", () => {
  it("lists a tile read through two slots once, with a witness per slot", () => {
    const { resolved } = compileDSL(`A = Tensor(8, 8)
X = relu(A)
C = matmul(X, X)
`);
    const plan = tilePlan(resolved, { X: [4, 4], C: [4, 4] });
    const s = supplyOf(plan, { tensorId: "C", coord: [0, 1] });
    // Slot 0 reads X's row band 0-4, slot 1 its column band 4-8; X[0,1] lies in both.
    expect(s.demand.map((d) => d.slot)).toEqual([0, 1]);
    expect(keys(s)).toEqual(["X[0,0]", "X[0,1]", "X[1,1]"]);
    const shared = s.producers.find((p) => p.task.coord.join() === "0,1")!;
    expect(shared.witnesses.map((w) => w.slot)).toEqual([0, 1]);
    expect(shared.used.value).toBe(16); // distinct elements, not 32
  });

  it("gives a graph input read by two operations no producer", () => {
    const { resolved } = compileDSL(`X = Tensor(4, 4)
P = relu(X)
Q = exp(X)
`);
    const s = supplyOf(tilePlan(resolved, { P: [2, 2], Q: [2, 2] }), { tensorId: "Q", coord: [0, 0] });
    expect(s.producers).toEqual([]);
    expect(s.demand).toEqual([expect.objectContaining({ tensorId: "X", supplier: "input" })]);
  });

  it("marks producers possible when the demand over-approximates", () => {
    // Without index values a lookup may read any row, so every row tile is a
    // possible producer and none is definite.
    const { resolved } = compileDSL(`E = Tensor(16, 8)
I = Tensor(4, dtype=int32)
F = relu(E)
Y = gather(F, I, axis=0)
`);
    const plan = tilePlan(resolved, { F: [4, 8], Y: [2, 8] });
    const s = supplyOf(plan, { tensorId: "Y", coord: [0, 0] });
    expect(keys(s)).toEqual(["F[0,0]", "F[1,0]", "F[2,0]", "F[3,0]"]);
    for (const p of s.producers) {
      expect(p.definite).toBe(false);
      expect(p.used.status).toBe("upper");
    }

    const report = interfaceOf(plan, "Y");
    if (report.status !== "evaluated") throw new Error(report.status);
    const f = report.boundary.find((b) => b.tensorId === "F")!;
    expect(f.exact).toBe(false);
    expect(f.summed.status).toBe("upper");
    expect(f.duplication.status).toBe("approximate");
  });
});

/* The chain's C, 256x256 from K=256, and Y, 256x128 from K=256: two FLOPs per term. */
const C_FLOPS = 2 * 256 * 256 * 256;
const Y_FLOPS = 2 * 256 * 128 * 256;

describe("untiled tensors are computed by the tasks that read them", () => {
  it("reads through an untiled tensor to what computes it", () => {
    const g = chain();
    const s = supplyOf(tilePlan(g, { Y: [64, 64] }), { tensorId: "Y", coord: [1, 0] });
    const cNode = g.tensors.C.producer!.nodeId;
    const yNode = g.tensors.Y.producer!.nodeId;
    expect(s.demand.map((d) => [d.tensorId, d.node, d.slot, d.supplier])).toEqual([
      ["A", cNode, 0, "input"],
      ["B", cNode, 1, "input"],
      ["W", yNode, 1, "input"],
    ]);
    expect(s.producers).toEqual([]);
    expect(s.computes.map((c) => [c.tensorId, c.region.boxes])).toEqual([
      ["C", [[{ lo: 64, hi: 128 }, { lo: 0, hi: 256 }]]],
    ]);
    // Its own 64x64 tile of Y and the 64x256 band of C it needs.
    expect(s.flops).toEqual({ value: 2 * 256 * (64 * 64 + 64 * 256), status: "exact", reasons: [] });
  });

  it("counts nothing as recomputed when every task computes only its own tile", () => {
    const report = planReport(tilePlan(chain(), { C: [64, 64], Y: [64, 64] }));
    const [c, y] = report.families;
    if (c.status !== "evaluated" || y.status !== "evaluated") throw new Error("not evaluated");
    expect(c.work).toMatchObject({
      tasks: 16,
      dependencies: { value: 0, status: "exact" },
      flops: { value: C_FLOPS, status: "exact" },
      recomputed: { value: 0, status: "exact" },
      read: { value: 16 * (64 * 256 + 256 * 64) * 2, status: "exact" },
      readDistinct: { value: 2 * 256 * 256 * 2, status: "exact" },
      written: { value: 256 * 256 * 2, status: "exact" },
    });
    // Each Y task reads the four C tasks in its row band.
    expect(y.work).toMatchObject({
      tasks: 8,
      dependencies: { value: 32, status: "exact" },
      flops: { value: Y_FLOPS, status: "exact" },
      recomputed: { value: 0, status: "exact" },
      written: { value: 256 * 128 * 2, status: "exact" },
    });
    expect(report.total).toMatchObject({
      tasks: 24,
      dependencies: { value: 32 },
      flops: { value: C_FLOPS + Y_FLOPS },
      recomputed: { value: 0 },
      written: { value: (256 * 256 + 256 * 128) * 2 },
    });
    expect(report.total!.intensity.value).toBeCloseTo(
      (C_FLOPS + Y_FLOPS) / (report.total!.read.value! + report.total!.written.value!)
    );
  });

  it("counts an untiled tensor computed by several tasks as recomputed", () => {
    // Two Y tasks share each row band, and each computes that band of C.
    const narrow = planReport(tilePlan(chain(), { Y: [64, 64] })).total!;
    expect(narrow).toMatchObject({
      tasks: 8,
      dependencies: { value: 0 },
      flops: { value: 2 * C_FLOPS + Y_FLOPS, status: "exact" },
      recomputed: { value: C_FLOPS, status: "exact" },
      // A 64x256 band of A, all of B and a 256x64 block of W per task; C is never written.
      read: { value: 8 * (64 * 256 + 256 * 256 + 256 * 64) * 2 },
      written: { value: 256 * 128 * 2 },
    });

    // Full-width Y tiles compute each band of C once.
    const wide = planReport(tilePlan(chain(), { Y: [64, 128] })).total!;
    expect(wide.flops.value).toBe(C_FLOPS + Y_FLOPS);
    expect(wide.recomputed.value).toBe(0);
  });

  it("counts a normalised axis's statistics again in every tile across it", () => {
    const { resolved } = compileDSL(`X = Tensor(8, 64)
P = softmax(X, axis=-1)
`);
    const whole = planReport(tilePlan(resolved, { P: [8, 64] })).total!;
    const split = planReport(tilePlan(resolved, { P: [8, 16] })).total!;
    expect(whole.recomputed.value).toBe(0);
    // Four tiles across each row each compute the row's max and sum.
    expect(split.recomputed.value).toBe(split.flops.value! - whole.flops.value!);
    expect(split.recomputed.value).toBeGreaterThan(0);
  });
});

describe("a whole plan", () => {
  const fork = () =>
    compileDSL(`A = Tensor(8, 8)
B = Tensor(8, 8)
C = matmul(A, B)
P = relu(C)
Q = exp(C)
`).resolved;

  it("measures work and reads two families share once, so the overlap is recomputed", () => {
    const report = planReport(tilePlan(fork(), { P: [8, 8], Q: [8, 8] }));
    const works = report.families.map((f) => (f.status === "evaluated" ? f.work : null)!);
    // Within each family C is computed once; across the plan it is computed twice.
    for (const w of works) expect(w.recomputed.value).toBe(0);
    expect(report.total!.recomputed).toMatchObject({ value: 2 * 8 * 8 * 8, status: "exact" });
    // Both read all of A and B; the plan reads them once each at best.
    expect(report.total!.read.value).toBe(2 * 2 * 64 * 4);
    expect(report.total!.readDistinct.value).toBe(2 * 64 * 4);
    expect(report.unwritten).toEqual([]);
  });

  it("names the graph outputs it never writes", () => {
    expect(planReport(tilePlan(fork(), { P: [8, 8] })).unwritten).toEqual(["Q"]);
    expect(planReport(tilePlan(fork(), { C: [8, 8] })).unwritten).toEqual(["P", "Q"]);
  });

  it("gives no total when a family is over budget, rather than part of one", () => {
    const report = planReport(tilePlan(chain(), { C: [64, 64], Y: [64, 64] }), { budget: 8 });
    expect(report.families.map((f) => f.status)).toEqual(["over-budget", "evaluated"]);
    expect(report.total).toBeNull();
  });

  it("has no FLOP total when a task computes an operation nobody described", () => {
    const { resolved } = compileDSL(`X = Tensor(4, 4)
H = opaque(X, op="Mystery", shapes=[[4, 4]])
Y = relu(H)
`);
    const report = planReport(tilePlan(resolved, { Y: [2, 2] }));
    const total = report.total!;
    expect(supplyOf(tilePlan(resolved, { Y: [2, 2] }), { tensorId: "Y", coord: [0, 0] }).computes
      .map((c) => c.tensorId)).toEqual(["H"]);
    expect(total.flops.status).toBe("unknown");
    expect(total.recomputed.status).toBe("unknown");
    expect(total.intensity.status).toBe("unknown");
    // The bytes are still figures: the barrier's output shape is declared.
    expect(total.written).toMatchObject({ value: 16 * 4, status: "exact" });
    expect(total.read.status).toBe("upper");
  });
});

describe("plan and task validation", () => {
  it("refuses tensors that are unknown, graph inputs, or given a malformed tile", () => {
    expect(code(() => tilePlan(chain(), { Q: [1, 1] }))).toBe("PLAN_UNKNOWN_TENSOR");
    expect(code(() => tilePlan(chain(), { A: [64, 64] }))).toBe("PLAN_GRAPH_INPUT");
    expect(code(() => tilePlan(chain(), { C: [64] }))).toBe("PLAN_TILE");
  });

  it("refuses tasks on an unplanned tensor or outside the grid", () => {
    const plan = tilePlan(chain(), { Y: [64, 64] });
    expect(code(() => supplyOf(plan, { tensorId: "C", coord: [0, 0] }))).toBe("PLAN_UNPLANNED");
    expect(code(() => interfaceOf(plan, "C"))).toBe("PLAN_UNPLANNED");
    expect(code(() => supplyOf(plan, { tensorId: "Y", coord: [4, 0] }))).toBe("PLAN_TASK");
    expect(code(() => supplyOf(plan, { tensorId: "Y", coord: [0] }))).toBe("PLAN_TASK");
  });

  it("declines a family with more tasks than its budget instead of summing part of it", () => {
    const report = interfaceOf(tilePlan(chain(), { C: [64, 64], Y: [64, 64] }), "Y", { budget: 4 });
    expect(report).toEqual({ status: "over-budget", tensorId: "Y", tasks: 8, budget: 4 });
  });
});

/* ------------------------------------------------------------------------ */

/**
 * Ground truth for one task from `oracleDeps` alone: walk back from every
 * element of the tile, element by element, through every untiled tensor, and
 * stop at tiled tensors and graph inputs. What the walk stops on is read, per
 * operation and operand slot; what it passes through is computed by the task.
 */
type TaskTruth = {
  /** `node#slot` -> the tensor read through that slot, and its flat indices. */
  reads: Map<string, { tensorId: string; flats: Set<number> }>;
  /** Untiled tensor -> the flat indices the task computes. */
  computes: Map<string, Set<number>>;
};

function truthTask(plan: TilePlan, tensorId: string, box: Box): TaskTruth {
  const g = plan.graph;
  const frontier = new Set(plan.frontier);
  const reads: TaskTruth["reads"] = new Map();
  const computes: TaskTruth["computes"] = new Map();
  const queue: [string, number[]][] = [...points(fromBox(box))].map((p) => [tensorId, p]);
  while (queue.length) {
    const [id, index] = queue.pop()!;
    const producer = g.tensors[id].producer!;
    const node = g.nodes.find((n) => n.id === producer.nodeId)!;
    const ctx = { inShapes: g.shapesOf(node.inputs), outShapes: g.shapesOf(node.outputs), attrs: node.attrs };
    getOp(node.op)!.oracleDeps(producer.slot, index, ctx).forEach((tuples, slot) => {
      const inId = node.inputs[slot];
      for (const tuple of tuples) {
        const flat = flatIndex(tuple, ctx.inShapes[slot]);
        if (frontier.has(inId)) {
          const key = `${node.id}#${slot}`;
          const entry = reads.get(key) ?? { tensorId: inId, flats: new Set<number>() };
          reads.set(key, entry);
          entry.flats.add(flat);
        } else {
          const seen = computes.get(inId) ?? new Set<number>();
          computes.set(inId, seen);
          if (!seen.has(flat)) {
            seen.add(flat);
            queue.push([inId, tuple]);
          }
        }
      }
    });
  }
  return { reads, computes };
}

/** The producer tiles a set of flat indices on a planned tensor falls in. */
function tilesOf(plan: TilePlan, tensorId: string, flats: Iterable<number>): Set<number> {
  const f = plan.families.get(tensorId)!;
  const out = new Set<number>();
  for (const flat of flats)
    out.add(tileOrdinal(f, unflatIndex(flat, [...f.shape]).map((i, axis) => Math.floor(i / f.tile[axis]))));
  return out;
}

/** The task's supply, from the checked path or from a cone under the given limits. */
function supplyUnder(plan: TilePlan, task: TaskRef, limits?: Limits): Supply {
  if (!limits) return supplyOf(plan, task);
  const seed = { tensorId: task.tensorId, region: fromBox(tileBox(plan.families.get(task.tensorId)!, task.coord)) };
  return joinDemand(plan, task, propagateWithin(plan.graph, seed, "backward", plan.frontier, limits));
}

/** Truth keys for the producer tiles a task reads. */
function truthProducers(plan: TilePlan, truth: TaskTruth): Set<string> {
  const keys = new Set<string>();
  for (const { tensorId, flats } of truth.reads.values())
    if (plan.families.has(tensorId))
      for (const o of tilesOf(plan, tensorId, flats)) keys.add(`${tensorId}#${o}`);
  return keys;
}

/**
 * Check one task's supply against the oracle. Returns the supply and how many
 * of its producers the truth does not have, which is zero whenever the demand
 * is exact.
 */
function checkTask(plan: TilePlan, task: TaskRef, limits?: Limits): { supply: Supply; extra: number } {
  const g = plan.graph;
  const family = plan.families.get(task.tensorId)!;
  const truth = truthTask(plan, task.tensorId, tileBox(family, task.coord));
  const supply = supplyUnder(plan, task, limits);
  const label = `${task.tensorId}[${task.coord.join(",")}]`;
  const matches = (region: Region, flats: Set<number>, shape: number[], what: string) => {
    const got = regionToFlatSet(region, shape);
    if (region.exact) expect(got, what).toEqual(flats);
    else for (const f of flats) expect(got.has(f), `${what} misses ${f}`).toBe(true);
  };

  // Reads per operation and slot: equal when exact, a superset when not.
  const keys = new Set([...truth.reads.keys(), ...supply.demand.map((d) => `${d.node}#${d.slot}`)]);
  for (const key of keys) {
    const d = supply.demand.find((x) => `${x.node}#${x.slot}` === key);
    const want = truth.reads.get(key);
    expect(d, `${label} reads nothing through ${key}`).toBeDefined();
    if (want) expect(d!.tensorId).toBe(want.tensorId);
    matches(d!.region, want?.flats ?? new Set(), g.tensors[d!.tensorId].resolved!, `${label} ${key}`);
  }

  // What it computes on the way: the same untiled tensors, the same elements.
  const computed = new Set([...truth.computes.keys(), ...supply.computes.map((c) => c.tensorId)]);
  for (const id of computed) {
    const c = supply.computes.find((x) => x.tensorId === id);
    expect(c, `${label} does not compute ${id}`).toBeDefined();
    matches(c!.region, truth.computes.get(id) ?? new Set(), g.tensors[id].resolved!, `${label} computes ${id}`);
  }

  // Producers: none missed, every definite one real, all of them real when exact.
  const truthKeys = truthProducers(plan, truth);
  const key = (p: Supply["producers"][number]) =>
    `${p.task.tensorId}#${tileOrdinal(plan.families.get(p.task.tensorId)!, p.task.coord)}`;
  const all = new Set(supply.producers.map(key));
  for (const k of truthKeys) expect(all.has(k), `${label} misses producer ${k}`).toBe(true);
  for (const p of supply.producers) if (p.definite) expect(truthKeys.has(key(p)), `${label} ${key(p)}`).toBe(true);
  if (supply.demand.every((d) => d.region.exact)) expect(all).toEqual(truthKeys);
  return { supply, extra: [...all].filter((k) => !truthKeys.has(k)).length };
}

function checkInterface(plan: TilePlan, tensorId: string): boolean {
  const report = interfaceOf(plan, tensorId, { budget: 48 });
  if (report.status !== "evaluated") return false;
  const g = plan.graph;
  const family = plan.families.get(tensorId)!;

  // Truth per boundary tensor: each task's distinct demand, then summed and joined.
  const summed = new Map<string, number>();
  const union = new Map<string, Set<number>>();
  const readers = new Map<string, number>();
  const fanOut = new Map<string, Map<number, number>>();
  let edges = 0;
  for (const coord of tiles(family)) {
    const truth = truthTask(plan, tensorId, tileBox(family, coord));
    edges += truthProducers(plan, truth).size;
    const perTensor = new Map<string, Set<number>>();
    for (const { tensorId: id, flats } of truth.reads.values()) {
      const set = perTensor.get(id) ?? new Set<number>();
      perTensor.set(id, set);
      flats.forEach((f) => set.add(f));
    }
    for (const [id, set] of perTensor) {
      summed.set(id, (summed.get(id) ?? 0) + set.size);
      readers.set(id, (readers.get(id) ?? 0) + 1);
      const u = union.get(id) ?? new Set<number>();
      union.set(id, u);
      set.forEach((f) => u.add(f));
      if (plan.families.has(id)) {
        const fan = fanOut.get(id) ?? new Map<number, number>();
        fanOut.set(id, fan);
        for (const o of tilesOf(plan, id, set)) fan.set(o, (fan.get(o) ?? 0) + 1);
      }
    }
  }

  for (const [id, total] of summed) {
    const b = report.boundary.find((x) => x.tensorId === id);
    expect(b, `${tensorId}: no boundary entry for ${id}`).toBeDefined();
    const bytes = DTYPE_BYTES[g.tensors[id].dtype];
    const cmp = (got: number, want: number, what: string) =>
      b!.exact ? expect(got, what).toBe(want) : expect(got, what).toBeGreaterThanOrEqual(want);
    cmp(b!.summed.value! / bytes, total, `${tensorId}->${id} summed`);
    cmp(b!.distinct.value! / bytes, union.get(id)!.size, `${tensorId}->${id} distinct`);
    cmp(b!.readers, readers.get(id)!, `${tensorId}->${id} readers`);
    for (const [o, n] of fanOut.get(id) ?? []) cmp(b!.fanOut!.get(o) ?? 0, n, `${tensorId}->${id} fan-out ${o}`);
    if (b!.exact) expect(b!.fanOut?.size ?? 0).toBe(fanOut.get(id)?.size ?? 0);
  }

  // The family's work: its tasks, its links, what it writes and reads.
  const { work } = report;
  const tensor = g.tensors[tensorId];
  expect(work.tasks).toBe(family.count);
  expect(work.written.value).toBe(tensor.resolved!.reduce((n, e) => n * e, 1) * DTYPE_BYTES[tensor.dtype]);
  if (work.dependencies.status === "exact") expect(work.dependencies.value).toBe(edges);
  else expect(work.dependencies.value).toBeGreaterThanOrEqual(edges);
  expect(work.read.value).toBe(report.boundary.reduce((n, b) => n + b.summed.value!, 0));
  if (work.recomputed.status === "exact") expect(work.recomputed.value).toBeGreaterThanOrEqual(0);
  return true;
}

describe("plans against a task oracle", () => {
  /* Random graphs with most produced tensors tiled at random extents, some
     wider than their axis; the rest are computed by the tasks that read them.
     Tasks are checked one at a time, and small families as a whole, against
     truth taken from `oracleDeps` alone. */
  it("agrees on random graphs and random tilings", () => {
    const r = rng(21);
    let tasks = 0;
    let joined = 0;
    let fused = 0;
    let families = 0;
    for (let trial = 0; trial < 30; trial++) {
      const g = resolveGraph(randomGraph(r, randInt(r, 5, 14)));
      const tiling: Record<string, number[]> = {};
      for (const t of Object.values(g.tensors))
        if (t.producer && r() < 0.7) tiling[t.id] = t.resolved!.map((e) => randInt(r, 1, e + 2));
      if (!Object.keys(tiling).length) continue;
      const plan = tilePlan(g, tiling);

      for (const f of plan.families.values()) {
        const all = [...tiles(f)];
        const sample = all.length <= 8 ? all : Array.from({ length: 8 }, () => all[randInt(r, 0, all.length)]);
        for (const coord of sample) {
          const { supply } = checkTask(plan, { tensorId: f.tensorId, coord });
          tasks++;
          if (supply.producers.length) joined++;
          if (supply.computes.length) fused++;
        }
        if (checkInterface(plan, f.tensorId)) families++;
      }
    }
    // Agreement only means something if tasks actually had producers to find,
    // and untiled tensors to compute.
    expect(joined).toBeGreaterThan(tasks / 4);
    expect(fused).toBeGreaterThan(tasks / 10);
    expect(families).toBeGreaterThan(30);
  });

  /* The same corpus with demand computed under lowered fallback thresholds,
     so reshapes, strided slices and coarsened regions over-approximate. This
     is where a producer could be missed; it must never be, and a producer
     reached only through widened demand must not be called definite. */
  it("never misses a producer when demand over-approximates", () => {
    const limits: Limits = { ...DEFAULT_LIMITS, stridedEnum: 2, diagEnum: 2, reshapeRuns: 1, maxBoxes: 2 };
    const r = rng(22);
    let widened = 0;
    let spurious = 0;
    for (let trial = 0; trial < 60; trial++) {
      const g = resolveGraph(randomGraph(r, randInt(r, 5, 14)));
      const tiling: Record<string, number[]> = {};
      for (const t of Object.values(g.tensors))
        if (t.producer) tiling[t.id] = t.resolved!.map((e) => randInt(r, 1, e + 2));
      const plan = tilePlan(g, tiling);
      for (const f of plan.families.values())
        for (const coord of tiles(f)) {
          const { supply, extra } = checkTask(plan, { tensorId: f.tensorId, coord }, limits);
          if (supply.producers.some((p) => !p.definite)) widened++;
          spurious += extra;
        }
    }
    // Widened demand is rare even at these thresholds: about 1 task in 100.
    // Require enough of it, including producers the truth does not have, for
    // the labelling of possible producers to have been tested.
    expect(widened).toBeGreaterThanOrEqual(10);
    expect(spurious).toBeGreaterThan(0);

    // With some tensors untiled, widening is also carried through what a task
    // computes itself, and must stay a superset there too.
    const q = rng(26);
    let widenedFused = 0;
    for (let trial = 0; trial < 60; trial++) {
      const g = resolveGraph(randomGraph(q, randInt(q, 5, 14)));
      const tiling: Record<string, number[]> = {};
      for (const t of Object.values(g.tensors))
        if (t.producer && q() < 0.4) tiling[t.id] = t.resolved!.map((e) => randInt(q, 1, e + 2));
      if (!Object.keys(tiling).length) continue;
      const plan = tilePlan(g, tiling);
      for (const f of plan.families.values())
        for (const coord of tiles(f)) {
          const { supply } = checkTask(plan, { tensorId: f.tensorId, coord }, limits);
          if (supply.computes.some((c) => !c.region.exact)) widenedFused++;
        }
    }
    // Rarer still: a widened region on a tensor the task computes. This seed
    // gives eight.
    expect(widenedFused).toBeGreaterThanOrEqual(4);
  });

  /* A whole plan measures shared work and reads once across families, so its
     distinct figures can only be smaller than the families' summed. */
  it("totals families without undercounting what they share", () => {
    const r = rng(23);
    let shared = 0;
    for (let trial = 0; trial < 30; trial++) {
      const g = resolveGraph(randomGraph(r, randInt(r, 5, 14)));
      const tiling: Record<string, number[]> = {};
      for (const t of Object.values(g.tensors))
        if (t.producer && r() < 0.5) tiling[t.id] = t.resolved!.map((e) => randInt(r, 1, e + 2));
      if (!Object.keys(tiling).length) continue;
      const report = planReport(tilePlan(g, tiling), { budget: 48 });
      if (!report.total) continue;
      const works = report.families.map((f) => (f.status === "evaluated" ? f.work : null)!);
      const sum = (pick: (w: Work) => number | null) => works.reduce((n, w) => n + pick(w)!, 0);
      const { total } = report;
      expect(total.tasks).toBe(sum((w) => w.tasks));
      expect(total.read.value).toBe(sum((w) => w.read.value));
      expect(total.written.value).toBe(sum((w) => w.written.value));
      expect(total.readDistinct.value!).toBeLessThanOrEqual(sum((w) => w.readDistinct.value));
      if (total.flops.status !== "unknown") {
        expect(total.flops.value).toBe(sum((w) => w.flops.value));
        if (total.recomputed.status === "exact") {
          expect(total.recomputed.value!).toBeGreaterThanOrEqual(sum((w) => w.recomputed.value));
          if (total.recomputed.value! > sum((w) => w.recomputed.value)) shared++;
        }
      }
    }
    // Some plans must have had an untiled tensor that two families compute.
    expect(shared).toBeGreaterThan(0);
  });
});
