import { describe, expect, it } from "vitest";
import { taskDemand, tilePlan } from "../../../src/core/plan/plan";
import { familyReplay, replayRuns } from "../../../src/core/plan/replay";
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
