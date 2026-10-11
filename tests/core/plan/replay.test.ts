import { describe, expect, it } from "vitest";
import { taskDemand, tilePlan } from "../../../src/core/plan/plan";
import { familyReplay, replayOrdinals } from "../../../src/core/plan/replay";
import { box, count, fromBox, intersect } from "../../../src/core/region";
import { compileDSL } from "../../../src/parse/compiler";

const GEMM = "A = Tensor(256, 128, dtype=fp16)\nB = Tensor(128, 256, dtype=fp16)\nC = matmul(A, B)\n";

describe("a family replay", () => {
  const { resolved } = compileDSL(GEMM);
  const plan = tilePlan(resolved, { C: [64, 64] });
  const replay = familyReplay(plan, { tensorId: "C", coord: [0, 1] });

  it("walks every task of a short family in listing order", () => {
    expect(replay.tasks).toBe(16);
    expect(replay.frames.map((f) => f.task.coord.join(","))).toEqual(
      Array.from({ length: 16 }, (_, i) => `${Math.floor(i / 4)},${i % 4}`)
    );
    expect(replay.frames[5].box).toEqual(box([64, 128], [64, 128]));
  });

  it("reaches what each task's bounded cone reaches, and nothing of its own tensor", () => {
    for (const frame of replay.frames) {
      const cone = taskDemand(plan, frame.task);
      expect([...frame.reach.keys()].sort()).toEqual(["A", "B"]);
      for (const [tensorId, { region }] of frame.reach)
        expect(count(region)).toBe(count(cone.tensors.get(tensorId)!.region));
    }
  });

  /* The inspected task C(0, 1) reads A's first row band and B's second column
     band. A task in the same row shares that A band and none of B's. */
  it("marks what each task shares with the inspected one", () => {
    const sameRow = replay.frames.find((f) => f.task.coord.join(",") === "0,3")!;
    expect(sameRow.reach.get("A")!.shared!.boxes).toEqual([box([0, 64], [0, 128])]);
    expect(sameRow.reach.get("B")!.shared).toBeNull();
    const sameColumn = replay.frames.find((f) => f.task.coord.join(",") === "2,1")!;
    expect(sameColumn.reach.get("A")!.shared).toBeNull();
    expect(count(sameColumn.reach.get("B")!.shared!)).toBe(128 * 64);
    const elsewhere = replay.frames.find((f) => f.task.coord.join(",") === "3,0")!;
    expect(elsewhere.reach.get("A")!.shared).toBeNull();
    expect(elsewhere.reach.get("B")!.shared).toBeNull();
    // The overlap is the intersection itself, measured against the inspected task's cone.
    const mine = taskDemand(plan, { tensorId: "C", coord: [0, 1] }).tensors.get("A")!.region;
    expect(count(sameRow.reach.get("A")!.shared!)).toBe(count(intersect(fromBox(box([0, 64], [0, 128])), mine)));
  });

  it("samples a long family evenly, keeping listing order and real tasks", () => {
    const long = familyReplay(tilePlan(resolved, { C: [8, 8] }), { tensorId: "C", coord: [0, 0] });
    expect(long.tasks).toBe(1024);
    expect(long.frames).toHaveLength(48);
    const ordinals = long.frames.map((f) => f.task.coord[0] * 32 + f.task.coord[1]);
    expect(ordinals).toEqual(replayOrdinals(1024, 48));
    expect(ordinals.every((o, i) => i === 0 || o > ordinals[i - 1])).toBe(true);
  });
});

describe("replay positions", () => {
  it("takes every position up to the cap, and one per equal stretch beyond it", () => {
    expect(replayOrdinals(5, 48)).toEqual([0, 1, 2, 3, 4]);
    expect(replayOrdinals(10, 5)).toEqual([1, 3, 5, 7, 9]);
    for (const [total, cap] of [[1000, 48], [49, 48], [7, 3]]) {
      const picks = replayOrdinals(total, cap);
      expect(picks).toHaveLength(cap);
      picks.forEach((pick, i) => {
        expect(pick).toBeGreaterThanOrEqual(Math.floor((i * total) / cap));
        expect(pick).toBeLessThan(Math.floor(((i + 1) * total) / cap));
      });
    }
  });
});
