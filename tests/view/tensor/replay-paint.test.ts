import { describe, expect, it } from "vitest";
import { box, count, fromBox, type Region } from "../../../src/core/region";
import { buildReplayPaint, playbackDifference } from "../../../src/view/tensor/replay-paint";
import { tilePlan } from "../../../src/core/plan/plan";
import { familyReplay } from "../../../src/core/plan/replay";
import { compileDSL } from "../../../src/parse/compiler";

describe("bounded playback differences", () => {
  it("subtracts exact overlap and reuses geometry on subsequent paints", () => {
    const reach = fromBox(box([0, 8], [0, 8]));
    const shared = fromBox(box([0, 4], [0, 8]));
    const rest = playbackDifference(reach, shared);
    expect(count(rest)).toBe(32);
    expect(rest.exact).toBe(true);
    expect(playbackDifference(reach, shared)).toBe(rest);
  });

  it("retains all reach when the work budget is exceeded", () => {
    const reach = fromBox(box([0, 8], [0, 8]));
    const shared: Region = {
      boxes: Array.from({ length: 3000 }, (_, i) => box([20 + i, 21 + i], [0, 8])),
      exact: true, reasons: [],
    };
    const rest = playbackDifference(reach, shared);
    expect(rest.boxes).toBe(reach.boxes);
    expect(rest.exact).toBe(false);
    expect(rest.reasons).toContain("playback subtraction budget");
    expect(playbackDifference(reach, shared)).toBe(rest);
  });

  it("does not erase reach using an approximate overlap", () => {
    const reach = fromBox(box([0, 8], [0, 8]));
    const shared = { ...reach, exact: false, reasons: ["widened"] };
    const rest = playbackDifference(reach, shared);
    expect(count(rest)).toBe(64);
    expect(rest.exact).toBe(false);
    expect(rest.reasons).toContain("widened");
  });
});

/* Several steps of a deeper replay share one family task. Its tile is the
   trail once, however many steps walked it, and never under the active fill. */
describe("the family trail of a stepped replay", () => {
  const { resolved } = compileDSL(
    "A = Tensor(16, 8)\nB = Tensor(8, 16)\nW = Tensor(16, 8)\nC = matmul(A, B)\nY = matmul(C, W)\n"
  );
  const plan = tilePlan(resolved, { C: [4, 4], Y: [4, 4] });
  const replay = familyReplay(plan, { tensorId: "Y", coord: [0, 0] }, { depth: 2 });
  const trail = (visited: number) =>
    buildReplayPaint({ tensorId: "Y", dark: false, playback: { replay, plan, visited, paused: true, exiting: false, opacity: 1 } })
      .layers.filter((layer) => !layer.seed);
  // Y(0, 0) reads C's first row band through four C tasks: four steps, one tile.
  const perTask = replay.frames.filter((f) => f.first.coord.join(",") === "0,0").length;

  it("paints nothing behind the active task while its own steps play", () => {
    expect(perTask).toBe(4);
    for (let visited = 1; visited <= perTask; visited++) expect(trail(visited)).toEqual([]);
  });

  it("paints each walked task's tile once, at one strength", () => {
    const layers = trail(perTask * 2 + 1);
    const boxes = layers.flatMap((layer) => layer.region.boxes);
    expect(boxes).toEqual([box([0, 4], [0, 4]), box([0, 4], [4, 8])]);
    expect(new Set(layers.map((layer) => layer.alpha)).size).toBe(1);
  });
});
