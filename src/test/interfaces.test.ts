import { describe, expect, it } from "vitest";
import { DTYPE_BYTES } from "../core/dtypes";
import { resolveGraph, ResolvedGraph } from "../core/graph";
import { DEFAULT_LIMITS, Limits } from "../core/ops/limits";
import { getOp } from "../core/ops/index";
import { propagateWithin } from "../core/propagate";
import { interfaceOf, joinDemand, supplyOf, Supply } from "../core/plan/interfaces";
import { TaskRef, tilePlan, TilePlan } from "../core/plan/plan";
import { PlanError, tileBox, tileOrdinal, tiles } from "../core/plan/tile-family";
import { Box, fromBox, points } from "../core/region";
import { compileDSL } from "../parse/compiler";
import { randInt, randomGraph, rng } from "./harness";
import { flatIndex, regionToFlatSet, unflatIndex } from "./oracle";

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
      expect(p.witnesses).toEqual([{ slot: 0, region: expect.objectContaining({ exact: true }) }]);
    }
    expect(s.demand.map((d) => [d.tensorId, d.slot, d.supplier])).toEqual([
      ["C", 0, "tasks"],
      ["W", 1, "input"],
    ]);
    expect(s.complete).toBe(true);
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

  it("marks demand on a tensor the plan does not tile as unplanned", () => {
    const s = supplyOf(tilePlan(chain(), { Y: [64, 64] }), { tensorId: "Y", coord: [0, 0] });
    expect(s.demand[0]).toMatchObject({ tensorId: "C", supplier: "unplanned" });
    expect(s.producers).toEqual([]);
    expect(s.complete).toBe(false);
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
 * Ground truth for one task from the producing operation's `oracleDeps` alone:
 * for each operand slot the task reads, the flat indices of the input elements.
 */
function truthDemand(g: ResolvedGraph, tensorId: string, box: Box): Map<number, Set<number>> {
  const producer = g.tensors[tensorId].producer!;
  const node = g.nodes.find((n) => n.id === producer.nodeId)!;
  const ctx = { inShapes: g.shapesOf(node.inputs), outShapes: g.shapesOf(node.outputs), attrs: node.attrs };
  const bySlot = new Map<number, Set<number>>();
  for (const p of points(fromBox(box)))
    getOp(node.op)!.oracleDeps(producer.slot, p, ctx).forEach((tuples, slot) => {
      if (!tuples.length) return;
      const set = bySlot.get(slot) ?? new Set<number>();
      bySlot.set(slot, set);
      for (const tuple of tuples) set.add(flatIndex(tuple, ctx.inShapes[slot]));
    });
  return bySlot;
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
  const g = plan.graph;
  const node = g.nodes.find((n) => n.id === g.tensors[task.tensorId].producer!.nodeId)!;
  const seed = { tensorId: task.tensorId, region: fromBox(tileBox(plan.families.get(task.tensorId)!, task.coord)) };
  return joinDemand(plan, task, propagateWithin(g, seed, "backward", node.inputs, limits).crossings);
}

/**
 * Check one task's supply against the oracle. Returns the supply and how many
 * of its producers the truth does not have, which is zero whenever the demand
 * is exact.
 */
function checkTask(plan: TilePlan, task: TaskRef, limits?: Limits): { supply: Supply; extra: number } {
  const g = plan.graph;
  const family = plan.families.get(task.tensorId)!;
  const node = g.nodes.find((n) => n.id === g.tensors[task.tensorId].producer!.nodeId)!;
  const truth = truthDemand(g, task.tensorId, tileBox(family, task.coord));
  const supply = supplyUnder(plan, task, limits);
  const label = `${task.tensorId}[${task.coord.join(",")}]`;

  // Demand per slot: equal when exact, a superset when not.
  const slots = new Set([...truth.keys(), ...supply.demand.map((d) => d.slot)]);
  for (const slot of slots) {
    const tensorId = node.inputs[slot];
    const d = supply.demand.find((x) => x.slot === slot);
    const got = d ? regionToFlatSet(d.region, g.tensors[tensorId].resolved!) : new Set<number>();
    const want = truth.get(slot) ?? new Set<number>();
    if (!d || d.region.exact) expect(got, `${label} slot ${slot}`).toEqual(want);
    else for (const f of want) expect(got.has(f), `${label} slot ${slot} misses ${f}`).toBe(true);
  }

  // Producers: none missed, every definite one real, all of them real when exact.
  const truthKeys = new Set<string>();
  for (const [slot, flats] of truth) {
    const tensorId = node.inputs[slot];
    if (plan.families.has(tensorId))
      for (const o of tilesOf(plan, tensorId, flats)) truthKeys.add(`${tensorId}#${o}`);
  }
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
  const node = g.nodes.find((n) => n.id === g.tensors[tensorId].producer!.nodeId)!;

  // Truth per boundary tensor: each task's distinct demand, then summed and joined.
  const summed = new Map<string, number>();
  const union = new Map<string, Set<number>>();
  const readers = new Map<string, number>();
  const fanOut = new Map<string, Map<number, number>>();
  for (const coord of tiles(family)) {
    const perTensor = new Map<string, Set<number>>();
    for (const [slot, flats] of truthDemand(g, tensorId, tileBox(family, coord))) {
      const set = perTensor.get(node.inputs[slot]) ?? new Set<number>();
      perTensor.set(node.inputs[slot], set);
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
  return true;
}

describe("plans against a task oracle", () => {
  /* Random graphs with most produced tensors tiled at random extents, some
     wider than their axis. Tasks are checked one at a time, and small
     families as a whole, against truth taken from `oracleDeps` alone. */
  it("agrees on random graphs and random tilings", () => {
    const r = rng(21);
    let tasks = 0;
    let joined = 0;
    let families = 0;
    for (let trial = 0; trial < 30; trial++) {
      const g = resolveGraph(randomGraph(r, randInt(r, 5, 14)));
      const tiling: Record<string, number[]> = {};
      for (const t of Object.values(g.tensors))
        if (t.producer && r() < 0.8) tiling[t.id] = t.resolved!.map((e) => randInt(r, 1, e + 2));
      const plan = tilePlan(g, tiling);

      for (const f of plan.families.values()) {
        const all = [...tiles(f)];
        const sample = all.length <= 8 ? all : Array.from({ length: 8 }, () => all[randInt(r, 0, all.length)]);
        for (const coord of sample) {
          const { supply } = checkTask(plan, { tensorId: f.tensorId, coord });
          tasks++;
          if (supply.producers.length) joined++;
        }
        if (checkInterface(plan, f.tensorId)) families++;
      }
    }
    // Agreement only means something if tasks actually had producers to find.
    expect(joined).toBeGreaterThan(tasks / 4);
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
  });
});
