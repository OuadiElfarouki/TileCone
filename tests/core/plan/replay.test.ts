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

  /* Deeper than 1 the family's tasks are no longer the step: each Y task needs
     C tasks first, so a step is one C task feeding one Y task. */
  const stepped = familyReplay(plan, { tensorId: "Y", coord: [1, 0] }, { depth: 2 });
  const stepsOf = (coord: string) => stepped.frames.filter((f) => f.firstPath[0].coord.join(",") === coord);

  it("steps through the producer tasks at depth 2, one C task per step, in listing order", () => {
    expect(stepped.depth).toBe(2);
    // One frame per step, well past the depth-1 run cap: a run of steps would
    // paint their union and lose the granularity a deeper replay is for.
    expect(stepped.frames.length).toBe(stepped.steps);
    expect(stepped.frames.every((f) => f.steps === 1)).toBe(true);
    // 32 Y tasks, each reading a row band of C through the 4 C tasks across it.
    expect(stepped.steps).toBe(32 * 4);
    expect(stepped.stepTensors).toEqual(["C"]);
    expect(stepsOf("1,0").map((f) => f.lastPath.map((t) => `${t.tensorId}${t.coord.join(",")}`).join(">")))
      .toEqual(["Y1,0>C0,0", "Y1,0>C0,1", "Y1,0>C0,2", "Y1,0>C0,3"]);
  });

  it("moves each step with the block of the other operand it meets inside the task it feeds", () => {
    stepsOf("1,0").forEach((frame, j) => {
      const c = frame.reach.get("C")!;
      expect(c.region.boxes).toEqual([box([0, 16], [16 * j, 16 * j + 16])]);
      expect(c.producers!.boxes).toEqual([box([0, 16], [16 * j, 16 * j + 16])]);
      expect(c.level).toBe(1);
      // C[8:16, k-block] meets W[k-block, 0:8] within Y(1, 0), not all of W's columns.
      const w = frame.reach.get("W")!;
      expect(w.region).toMatchObject({ boxes: [box([16 * j, 16 * j + 16], [0, 8])], exact: true });
      expect(w.level).toBe(1);
      // What the C task itself reads, one level further: A's rows and one band of B.
      expect(frame.reach.get("A")!).toMatchObject({ level: 2, region: { boxes: [box([0, 16], [0, 32])] } });
      expect(frame.reach.get("B")!).toMatchObject({ level: 2, region: { boxes: [box([0, 32], [16 * j, 16 * j + 16])] } });
    });
  });

  it("measures each step's overlap against the inspected task followed to the same depth", () => {
    // Y(0, 0)'s first step is C(0, 0), which reads the same A rows as Y(1, 0)'s producers.
    const [first] = stepsOf("0,0");
    expect(count(first.reach.get("A")!.shared!)).toBe(16 * 32);
  });

  /* Fused operations between the frontier and the consumer's tile: the step's
     block is carried through them, so what it becomes and what it meets later
     move with it. */
  it("carries a step through fused operations to the operands it meets there", () => {
    const fused = compileDSL(CHAIN.replace("Y = matmul(C, W)", "D = relu(C)\nY = matmul(D, W)")).resolved;
    const replay = familyReplay(tilePlan(fused, { C: [16, 16], Y: [8, 8] }), { tensorId: "Y", coord: [1, 0] }, { depth: 2 });
    const steps = replay.frames.filter((f) => f.firstPath[0].coord.join(",") === "1,0");
    expect(steps).toHaveLength(4);
    steps.forEach((frame, j) => {
      // C[0:16, k-block] reaches D[8:16, k-block] inside Y(1, 0), and D meets W[k-block, 0:8].
      expect(frame.reach.get("D")!).toMatchObject({ level: 1, region: { boxes: [box([8, 16], [16 * j, 16 * j + 16])] } });
      expect(frame.reach.get("W")!).toMatchObject({ level: 1, region: { boxes: [box([16 * j, 16 * j + 16], [0, 8])], exact: true } });
    });
  });

  it("carries the second split output even when the first is materialized", () => {
    const { resolved } = compileDSL(
      "X = Tensor(8)\nC = relu(X)\nU, V = split(C, axis=0, sizes=[4,4])\nW = Tensor(4)\nY = add(V,W)\n"
    );
    const tilings: Record<string, number[]>[] = [{ C: [1], Y: [1] }, { C: [1], U: [1], Y: [1] }];
    for (const tiles of tilings) {
      const replay = familyReplay(tilePlan(resolved, tiles), { tensorId: "Y", coord: [0] }, { depth: 2 });
      expect(replay.frames).toHaveLength(4);
      replay.frames.forEach((frame, j) => {
        expect(frame.firstPath).toEqual([{ tensorId: "Y", coord: [j] }, { tensorId: "C", coord: [j + 4] }]);
        for (const id of ["V", "W"])
          expect(frame.reach.get(id)).toMatchObject({ level: 1, region: { boxes: [box([j, j + 1])], exact: true } });
        expect(frame.reach.has("U")).toBe(false);
      });
    }
  });

  it("declines a depth whose steps need more queries than the budget", () => {
    expect(() => familyReplay(plan, { tensorId: "Y", coord: [1, 0] }, { depth: 2, budget: 10 }))
      .toThrow(/more than 10 task queries; choose a smaller depth/);
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

describe("what a deeper step walks", () => {
  // A row of C per D task: the matmul is fused into D, and D2 reads D tile for tile.
  const SOURCE = `M = 16\nN = 16\nK = 128\nE = M * N\nA = Tensor(M, K, dtype=fp16)\nB = Tensor(K, N, dtype=fp16)
C = matmul(A, B)\nD = reshape(C, shape=[E])\nD2 = add(D, D)\n`;
  const { resolved } = compileDSL(SOURCE);
  const plan = tilePlan(resolved, { D2: [16], D: [16] });

  it("walks nothing below the family at depth 1", () => {
    const replay = familyReplay(plan, { tensorId: "D2", coord: [8] });
    expect(replay.frames.every((f) => f.walked.size === 0)).toBe(true);
  });

  it("walks one producer tile per step at depth 2, with that task's own reads", () => {
    const replay = familyReplay(plan, { tensorId: "D2", coord: [8] }, { depth: 2 });
    expect(replay.stepTensors).toEqual(["D"]);
    replay.frames.forEach((frame, i) => {
      expect(frame.walked.get("D")?.boxes).toEqual([box([i * 16, i * 16 + 16])]);
      expect(frame.reach.get("A")?.region.boxes).toEqual([box([i, i + 1], [0, 128])]);
      expect(frame.reach.get("C")?.region.boxes).toEqual([box([i, i + 1], [0, 16])]);
    });
    // Only B is read by every D task, so only B is shared with the inspected one.
    const last = replay.frames[15];
    expect([...last.reach].filter(([, r]) => r.shared).map(([id]) => id)).toEqual(["B"]);
  });
});
