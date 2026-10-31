import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Inspector } from "../../../src/components/inspector/Inspector";
import { replayExplanation } from "../../../src/components/inspector/PlanPanel";
import { useStore } from "../../../src/state/store";
import { familyReplay } from "../../../src/core/plan/replay";
import { EXAMPLES } from "../../../src/examples/index";

// Static-render tests read the live test store rather than Zustand's initial
// server snapshot. Actions and all derivation logic remain the real implementation.
vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const CHAIN = "A = Tensor(32, 32, dtype=float16)\nB = Tensor(32, 32, dtype=float16)\n" +
  "W = Tensor(32, 16, dtype=float16)\nC = matmul(A, B)\nY = matmul(C, W)\n";
const S = () => useStore.getState();
const render = () => renderToStaticMarkup(createElement(Inspector));

describe("operand reuse in the Plan view", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
    S().setInspectorTab("plan");
    S().setPlanTileAt("Y", [8, 4], [8, 4]);
    S().setPlanTile("C", [8, 8]);
  });

  it("counts what the inspected task computes, under its own heading", () => {
    const html = render();
    expect(html).toContain("Operand reuse in this task");
    // An 8 x 4 tile of Y reads a row of C with its 4 columns and a column of W with its 8 rows.
    expect(html).toContain("via matmul → Y");
    expect(html).toContain("4 uses per element");
    expect(html).toContain("8 uses per element");
    expect(html).not.toContain("via matmul → C");
  });

  it("shows the fused operand's own reuse once the tasks compute it", () => {
    S().setPlanTile("C", null);
    const html = render();
    expect(html).toContain("via matmul → C");
    // The task computes C's 8-row band: each of those rows of A is read by all 32 columns.
    expect(html).toContain("32 uses per element");
  });
});

describe("producer paging in the Plan view", () => {
  it("shows a bounded first page with the full producer count", () => {
    S().applyDSL(CHAIN);
    S().setInspectorTab("plan");
    S().setPlanTile("C", [1, 1]);
    S().setPlanTileAt("Y", [1, 1], [0, 0]);
    const html = render();
    expect(html).toContain("requires 32/1024 tasks");
    expect(html).toContain("1–8 of 32");
    expect(html).toContain("previous");
    expect(html).toContain("next");
    expect(html).not.toContain("show all");
  });
});

describe("family replay in the Plan view", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
    S().setInspectorTab("plan");
    S().setPlanTileAt("Y", [8, 4], [8, 4]);
  });

  /* What a section's figures mean is read once; it waits behind a "?" in the
     heading rather than standing between the reader and the figures. */
  it("offers each section's explanation behind a closed question mark", () => {
    const html = render();
    for (const topic of ["a plan", "upstream levels", "family demand and replay"])
      expect(html).toContain(`aria-expanded="false" aria-label="what ${topic} means"`);
    expect(html).not.toContain("A plan divides produced tensors into tiles");
    expect(html).not.toContain("Level 1 is what this task reads");
  });

  it("offers a replay beside the family figures", () => {
    const html = render();
    expect(html).toMatch(/Across all 16 Y tasks<span class="ins-title-actions"><button[^>]*>replay<\/button>/);
  });

  it("says where the replay is, and that listing order is not an execution order", () => {
    const replay = familyReplay(S().plan!, S().planTask!);
    S().setReplayPlayback({ replay, plan: S().plan!, visited: 3, paused: false, exiting: false, opacity: 1 });
    const html = render();
    expect(html).toContain("replay · Y tasks");
    expect(html).toContain("Task Y (0, 2) · 3 of 16 · listing order, not execution order");
    // The longer account sits behind the family's "?", closed by default.
    expect(html).not.toContain("All 16 tasks in");
    expect(replayExplanation(S().replayPlayback, "Y", (id) => id)).toBe(
      "All 16 tasks in listing order, which is not an execution order. Full strength marks what each task shares with this task; outlines are the producer tiles."
    );
    S().setReplayPlayback(null);
  });

  /* C is tiled, so Y's tasks have producers upstream, and the replay can follow
     them one level further. */
  it("offers a depth per family, and steps through producer tasks deeper than 1", () => {
    const html = render();
    expect(html).toContain('aria-label="replay depth"');
    expect(html).toMatch(/aria-pressed="true"[^>]*>1<\/button>/);
    S().setReplayDepth("Y", 2);
    expect(render()).toMatch(/aria-pressed="true"[^>]*>2<\/button>/);
    const replay = familyReplay(S().plan!, S().planTask!, { depth: 2 });
    S().setReplayPlayback({ replay, plan: S().plan!, visited: 1, paused: false, exiting: false, opacity: 1 });
    const stepped = render();
    // Deeper than 1 a step is a C task feeding a Y task, named as that path.
    expect(stepped).toContain("replay · C tasks feeding Y");
    // 16 Y tasks × 64 C tasks each: one frame per step, not runs of them.
    expect(stepped).toMatch(/Y \(0, 0\) ← C \(0, 0\) · 1 of 1024/);
    expect(replayExplanation(S().replayPlayback, "Y", (id) => id)).toMatch(
      /^All 1024 steps, each a C task feeding the Y tasks in listing order/
    );
    // A replay made at another depth is not this one.
    S().setReplayDepth("Y", 1);
    expect(render()).not.toContain("listing order");
    S().setReplayPlayback(null);
  });

  it("offers no depth where nothing upstream is tiled", () => {
    S().setPlanTile("C", null);
    expect(render()).not.toContain('aria-label="replay depth"');
  });

  it("steps a replay by hand: previous, play or pause, next, a scrubber, and clear", () => {
    const replay = familyReplay(S().plan!, S().planTask!);
    S().setReplayPlayback({ replay, plan: S().plan!, visited: 1, paused: false, exiting: false, opacity: 1 });
    let html = render();
    expect(html).toContain('aria-label="replay controls"');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*aria-label="previous task"/);
    expect(html).toContain('aria-label="pause"');
    expect(html).toMatch(/<input type="range" min="1" max="16"[^>]*value="1"/);
    expect(html).toContain('aria-label="clear replay"');
    S().setReplayPlayback({ replay, plan: S().plan!, visited: 16, paused: true, exiting: false, opacity: 1 });
    html = render();
    expect(html).toContain('aria-label="play again"');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*aria-label="next task"/);
    S().setReplayPlayback(null);
  });

  it("drops a replay that belongs to another task", () => {
    const replay = familyReplay(S().plan!, S().planTask!);
    S().setReplayPlayback({ replay, plan: S().plan!, visited: 3, paused: false, exiting: false, opacity: 1 });
    S().selectPlanTask({ tensorId: "Y", coord: [0, 0] });
    expect(render()).not.toContain("listing order");
    S().setReplayPlayback(null);
  });
});

describe("fusing and materializing in the Plan view", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
    S().setInspectorTab("plan");
    S().setPlanTileAt("Y", [8, 4], [8, 4]);
  });

  it("names untiling an intermediate as fusing it, and untiling an output as not writing it", () => {
    S().setPlanTile("C", [8, 8]);
    const html = render();
    expect(html).toMatch(/aria-label="fuse C into its consumers"[^>]*>fuse<\/button>/);
    expect(html).toMatch(/aria-label="stop writing Y"[^>]*>stop writing<\/button>/);
  });

  it("offers to materialize what a task computes", () => {
    S().setPlanTile("C", null);
    const html = render();
    expect(html).toContain("Computed in this task");
    expect(html).toMatch(/title="materialize C:[^"]*"[^>]*>materialize<\/button>/);
    S().tilePlanTensor("C");
    expect(S().planTiles.C).toBeDefined();
  });
});

describe("what the current plan changes from each kept one", () => {
  it("says it under each kept plan, decisions first", () => {
    S().applyDSL(EXAMPLES.find((example) => example.name === "Experiment: fusion and recomputation")!.dsl);
    const html = render();
    expect(html).toContain("From P1, the current plan fuses H into its consumers and retiles Y 64×64 → 64×128.");
    expect(html).toMatch(/From P2, the current plan retiles Y 64×64 → 64×128\. Lower: [^.<]*recomputed/);
  });
});
