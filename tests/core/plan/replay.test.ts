import { describe, expect, it, vi } from "vitest";
import { taskDemand, tilePlan } from "../../../src/core/plan/plan";
import { familyReplay, replayDepthLimit, replayRuns } from "../../../src/core/plan/replay";
import { tileBox, tileCoord } from "../../../src/core/plan/tile-family";
import { box, count, fromBox, intersect, unionOf } from "../../../src/core/region";
import { compileDSL } from "../../../src/parse/compiler";

const GEMM = "A = Tensor(256, 128, dtype=fp16)\nB = Tensor(128, 256, dtype=fp16)\nC = matmul(A, B)\n";
const coordOf = (frame: { first: { coord: readonly number[] } }) => frame.first.coord.join(",");

describe("a family replay", () => {
  const { resolved } = compileDSL(GEMM);
  const plan = tilePlan(resolved, { C: [64, 64] });
  const replay = familyReplay(plan, { tensorId: "C", coord: [0, 1] });

  it("walks a short family one task per run, in listing order", () => {
    expect(replay.tasks).toBe(16);
    expect(replay.frames.map(coordOf)).toEqual(
      Array.from({ length: 16 }, (_, i) => `${Math.floor(i / 4)},${i % 4}`)
    );
    expect(replay.frames.every((f) => f.tasks === 1 && coordOf(f) === f.last.coord.join(","))).toBe(true);
    expect(replay.frames[5].region.boxes).toEqual([box([64, 128], [64, 128])]);
  });

  it("reaches what each task's bounded cone reaches, and nothing of its own tensor", () => {
    for (const frame of replay.frames) {
      const cone = taskDemand(plan, frame.first);
      expect([...frame.reach.keys()].sort()).toEqual(["A", "B"]);
      for (const [tensorId, { region }] of frame.reach)
        expect(count(region)).toBe(count(cone.tensors.get(tensorId)!.region));
    }
  });

  /* The inspected task C(0, 1) reads A's first row band and B's second column
     band. A task in the same row shares that A band and none of B's. */
  it("marks what each task shares with the inspected one", () => {
    const at = (coord: string) => replay.frames.find((f) => coordOf(f) === coord)!;
    expect(at("0,3").reach.get("A")!.shared!.boxes).toEqual([box([0, 64], [0, 128])]);
    expect(at("0,3").reach.get("B")!.shared).toBeNull();
    expect(at("2,1").reach.get("A")!.shared).toBeNull();
    expect(count(at("2,1").reach.get("B")!.shared!)).toBe(128 * 64);
    expect(at("3,0").reach.get("A")!.shared).toBeNull();
    expect(at("3,0").reach.get("B")!.shared).toBeNull();
  });

  /* A long family is cut into contiguous runs rather than sampled, so the
     trail on the cards has no gaps and no task is skipped. */
  describe("over a long family", () => {
    const fine = tilePlan(resolved, { C: [8, 8] });
    const family = fine.families.get("C")!;
    const long = familyReplay(fine, { tensorId: "C", coord: [0, 0] });

    it("covers every task exactly once, in contiguous runs", () => {
      expect(long.tasks).toBe(1024);
      expect(long.frames).toHaveLength(48);
      expect(long.frames.reduce((n, f) => n + f.tasks, 0)).toBe(1024);
      // Disjoint, and together the whole tensor: the trail ends as one solid stretch.
      const all = unionOf(long.frames.map((f) => f.region));
      expect(count(all)).toBe(256 * 256);
      expect(long.frames.reduce((n, f) => n + count(f.region), 0)).toBe(256 * 256);
    });

    it("names each run by its first and last task in listing order", () => {
      const runs = replayRuns(1024, 48);
      long.frames.forEach((frame, i) => {
        expect(frame.first.coord).toEqual(tileCoord(family, runs[i][0]));
        expect(frame.last.coord).toEqual(tileCoord(family, runs[i][1] - 1));
      });
    });

    it("reaches what the run's tasks reach together", () => {
      for (const i of [0, 17, 47]) {
        const frame = long.frames[i];
        const [first, end] = replayRuns(1024, 48)[i];
        const tasks = Array.from({ length: end - first }, (_, k) => ({ tensorId: "C", coord: tileCoord(family, first + k) }));
        for (const tensorId of ["A", "B"]) {
          const together = unionOf(tasks.map((t) => taskDemand(fine, t).tensors.get(tensorId)!.region));
          expect(count(frame.reach.get(tensorId)!.region)).toBe(count(together));
        }
        // Each task's own tile lies inside the run's region.
        for (const t of tasks)
          expect(count(intersect(frame.region, fromBox(tileBox(family, t.coord))))).toBe(64);
      }
    });
  });
});

describe("replay runs", () => {
  it("are one position each up to the cap, and contiguous near-equal stretches beyond it", () => {
    expect(replayRuns(5, 48)).toEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]);
    expect(replayRuns(10, 4)).toEqual([[0, 2], [2, 5], [5, 7], [7, 10]]);
    for (const [total, cap] of [[1000, 48], [49, 48], [7, 3], [4096, 48]]) {
      const runs = replayRuns(total, cap);
      expect(runs).toHaveLength(Math.min(total, cap));
      expect(runs[0][0]).toBe(0);
      expect(runs[runs.length - 1][1]).toBe(total);
      runs.forEach(([first, end], i) => {
        if (i > 0) expect(first).toBe(runs[i - 1][1]);
        expect(end - first).toBeGreaterThanOrEqual(Math.floor(total / runs.length));
        expect(end - first).toBeLessThanOrEqual(Math.ceil(total / runs.length));
      });
    }
  });
});

/* Y's tasks read C, and C's tasks read A and B. C's 16-row tiles are coarser
   than Y's 8-row ones, which is where following the plan differs from the
   element-level cone: a producer computes its whole tile. */
describe("replaying through the plan's own producers", () => {
  const CHAIN = "A = Tensor(64, 32, dtype=fp16)\nB = Tensor(32, 64, dtype=fp16)\n" +
    "W = Tensor(64, 32, dtype=fp16)\nC = matmul(A, B)\nY = matmul(C, W)\n";
  const { resolved } = compileDSL(CHAIN);
  const plan = tilePlan(resolved, { C: [16, 16], Y: [8, 8] });
  const at = (replay: ReturnType<typeof familyReplay>, coord: string) =>
    replay.frames.find((f) => coordOf(f) === coord)!;

  it("knows how many levels of tasks lie upstream of each family", () => {
    expect(replayDepthLimit(plan, "Y")).toBe(2);
    expect(replayDepthLimit(plan, "C")).toBe(1);
    // Untiled, C is computed inside Y's own tasks: nothing produces it upstream.
    expect(replayDepthLimit(tilePlan(resolved, { Y: [8, 8] }), "Y")).toBe(1);
    const deeper = compileDSL(CHAIN.replace("Y = matmul(C, W)", "D = relu(C)\nY = matmul(D, W)")).resolved;
    expect(replayDepthLimit(tilePlan(deeper, { C: [16, 16], D: [8, 8], Y: [8, 8] }), "Y")).toBe(3);
  });

  it("stops at the family's own reads at depth 1", () => {
    const replay = familyReplay(plan, { tensorId: "Y", coord: [1, 0] });
    expect(replay.depth).toBe(1);
    expect([...at(replay, "1,0").reach.keys()].sort()).toEqual(["C", "W"]);
    expect(at(replay, "1,0").reach.get("C")!.level).toBe(1);
  });

  it("follows the tasks producing what a run reads, whole tiles, at depth 2", () => {
    const replay = familyReplay(plan, { tensorId: "Y", coord: [1, 0] }, { depth: 2 });
    expect(replay.depth).toBe(2);
    const frame = at(replay, "1,0");
    expect([...frame.reach.keys()].sort()).toEqual(["A", "B", "C", "W"]);
    expect(frame.reach.get("A")!.level).toBe(2);
    // Y(1, 0) reads C's rows 8:16. C's tasks compute 16-row tiles, so the ones
    // it needs are rows 0:16, and they read A's rows 0:16 - not 8:16.
    expect(frame.reach.get("C")!.region.boxes).toEqual([box([8, 16], [0, 64])]);
    expect(frame.reach.get("C")!.producers!.boxes).toEqual([box([0, 16], [0, 64])]);
    expect(count(frame.reach.get("A")!.region)).toBe(16 * 32);
    expect(count(intersect(frame.reach.get("A")!.region, fromBox(box([0, 16], [0, 32]))))).toBe(16 * 32);
    // The same as the producer tasks' own demand, taken one task at a time.
    const producers = [0, 1, 2, 3].map((c) => taskDemand(plan, { tensorId: "C", coord: [0, c] }));
    for (const tensorId of ["A", "B"])
      expect(count(frame.reach.get(tensorId)!.region)).toBe(
        count(unionOf(producers.map((cone) => cone.tensors.get(tensorId)!.region)))
      );
  });

  it("measures the overlap against the inspected task followed to the same depth", () => {
    const replay = familyReplay(plan, { tensorId: "Y", coord: [1, 0] }, { depth: 2 });
    // Y(0, 0) reads C's rows 0:8, none of the inspected task's 8:16 - but the
    // same C tasks produce both, so upstream they read the same A rows.
    const neighbour = at(replay, "0,0");
    expect(neighbour.reach.get("C")!.shared).toBeNull();
    expect(count(neighbour.reach.get("A")!.shared!)).toBe(16 * 32);
  });

  it("clamps a depth beyond what the plan has", () => {
    expect(familyReplay(plan, { tensorId: "Y", coord: [0, 0] }, { depth: 5 }).depth).toBe(2);
    expect(familyReplay(plan, { tensorId: "Y", coord: [0, 0] }, { depth: 0 }).depth).toBe(1);
  });
});


describe("replay depth on large and shared graphs", () => {
  it("handles a deep tiled chain without recursion, including a depth-one replay", () => {
    const length = 20000;
    const lines = ["X = Tensor(1)", "P0 = relu(X)"];
    for (let i = 1; i < length; i++) lines.push(`P${i} = relu(P${i - 1})`);
    const { resolved } = compileDSL(lines.join("\n"));
    const plan = tilePlan(resolved, Object.fromEntries(
      Array.from({ length }, (_, i) => [`P${i}`, [1]])
    ));
    expect(replayDepthLimit(plan, `P${length - 1}`)).toBe(length);
    const replay = familyReplay(plan, { tensorId: `P${length - 1}`, coord: [0] });
    expect(replay.depth).toBe(1);
    expect([...replay.frames[0].reach.keys()]).toEqual([`P${length - 2}`]);
  });

  it("visits a shared untiled prefix once and reuses depths across family queries", () => {
    const length = 1000;
    const lines = ["X = Tensor(1)", "H0 = relu(X)"];
    for (let i = 1; i < length; i++) lines.push(`H${i} = relu(H${i - 1})`);
    lines.push(`P0 = relu(H${length - 1})`);
    for (let i = 1; i < length; i++) lines.push(`P${i} = add(H${length - 1}, P${i - 1})`);
    const { resolved } = compileDSL(lines.join("\n"));
    const tiles = Object.fromEntries(Array.from({ length }, (_, i) => [`P${i}`, [1]]));
    const plan = tilePlan(resolved, tiles);
    const shared = resolved.topo.find((node) => node.outputs.includes(`H${length - 1}`))!;
    const inputs = shared.inputs;
    const readInputs = vi.fn(() => inputs);
    Object.defineProperty(shared, "inputs", { get: readInputs, configurable: true });

    expect(replayDepthLimit(plan, `P${length - 1}`)).toBe(length);
    expect(readInputs).toHaveBeenCalledOnce();
    for (let i = 0; i < length; i++) expect(replayDepthLimit(plan, `P${i}`)).toBe(i + 1);
    // Playback ticks and changes of inspected family reuse the same plan index.
    expect(readInputs).toHaveBeenCalledOnce();

    const retiled = tilePlan(resolved, { ...tiles, [`H${length - 1}`]: [1] });
    expect(replayDepthLimit(retiled, `P${length - 1}`)).toBe(length + 1);
    expect(readInputs).toHaveBeenCalledTimes(2);
    expect(replayDepthLimit(plan, `P${length - 1}`)).toBe(length);
    expect(readInputs).toHaveBeenCalledTimes(2);
  });

  it("counts each tiled output separately across branches of a multi-output operation", () => {
    const { resolved } = compileDSL("X = Tensor(8)\nA = relu(X)\nB = relu(A)\n" +
      "U, V = split(B, axis=0, sizes=[4, 4])\nT = relu(V)\nY = concat(U, T, axis=0)\n");
    const tiles = { A: [1], B: [1], U: [1], T: [1], Y: [1] };
    const plan = tilePlan(resolved, tiles);
    expect(replayDepthLimit(plan, "U")).toBe(3);
    expect(replayDepthLimit(plan, "T")).toBe(3);
    expect(replayDepthLimit(plan, "Y")).toBe(4);
    expect(replayDepthLimit(tilePlan(resolved, { ...tiles, V: [1] }), "Y")).toBe(5);
  });
});
