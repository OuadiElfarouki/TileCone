import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { formatBoxIndices } from "../../../src/core/region";
import { tilePlan } from "../../../src/core/plan/plan";
import { taskLineage } from "../../../src/core/plan/replay";
import { compileDSL } from "../../../src/parse/compiler";
import { Inspector } from "../../../src/components/inspector/Inspector";
import { useStore } from "../../../src/state/store";
import { activePlanTrail } from "../../../src/state/plan";
import { readOrder } from "../../../src/view/plan-work";

vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const S = () => useStore.getState();
const RESHAPED = "X = Tensor(4, 4)\nF = reshape(X, shape=[16])\nY = reshape(F, shape=[2, 8])\nA = Tensor(8,4)\nZ = matmul(Y,A)\n";
const render = () => renderToStaticMarkup(createElement(Inspector));
const Z00 = { tensorId: "Z", coord: [0, 0] };

describe("the reads of a task, in operand order", () => {
  it("lists arg0 before arg1 whatever the tensors are called or where they come from", () => {
    const demand = [
      { tensorId: "A", node: "mm", slot: 1 },
      { tensorId: "Y", node: "mm", slot: 0 },
    ];
    expect(readOrder(demand, "mm", [{ id: "mm" }])).toEqual(["Y", "A"]);
  });

  it("puts the task's own operands first, then fused reads nearest first", () => {
    const demand = [
      { tensorId: "X", node: "far", slot: 0 },
      { tensorId: "W", node: "near", slot: 1 },
      { tensorId: "B", node: "own", slot: 1 },
      { tensorId: "W", node: "own", slot: 0 },
    ];
    expect(readOrder(demand, "own", [{ id: "far" }, { id: "near" }, { id: "own" }])).toEqual(["W", "B", "X"]);
  });
});

describe("a task's lineage, to the graph's inputs", () => {
  const lineage = (tiles: Record<string, number[]>) =>
    taskLineage(tilePlan(compileDSL(RESHAPED).resolved, tiles), Z00)
      .map((r) => `${r.level} ${r.tensorId} ${r.kind}${r.kind === "tasks" ? ` ${r.tasks}/${r.of}` : ""} ${r.region.boxes.map(formatBoxIndices).join(";")}`);

  it("names each upstream tensor at the nearest level, with tasks, inputs and fused work", () => {
    expect(lineage({ Y: [1, 1], Z: [1, 1] })).toEqual([
      "1 Y tasks 8/16 0, 0:8",
      "1 A input 0:8, 0",
      "2 X input 0:2, 0:4",
      "2 F computed 0:8",
    ]);
  });

  it("goes as deep as the plan does", () => {
    expect(lineage({ F: [4], Y: [1, 1], Z: [1, 1] })).toEqual([
      "1 Y tasks 8/16 0, 0:8",
      "1 A input 0:8, 0",
      "2 F tasks 2/4 0:8",
      "3 X input 0:2, 0:4",
    ]);
  });
});

describe("how certain a lineage row's tasks are", () => {
  it("keeps tasks reached only through a widened demand apart from definite ones", () => {
    const { resolved } = compileDSL(
      "W0 = Tensor(8, 4)\nW = relu(W0)\nIdx = Tensor(3, dtype=int32)\nE = gather(W, Idx, axis=0)\n"
    );
    const plan = tilePlan(resolved, { W: [2, 4], E: [3, 4] });
    const [w] = taskLineage(plan, { tensorId: "E", coord: [0, 0] });
    expect(w).toMatchObject({ tensorId: "W", kind: "tasks", tasks: 4, definiteTasks: 0, of: 4 });
    if (w.kind !== "tasks") throw new Error("Expected task lineage");
    expect(w.definiteCoordinates.boxes).toEqual([]);
  });

  it("counts every task definite where the demand is exact", () => {
    const [y] = taskLineage(tilePlan(compileDSL(RESHAPED).resolved, { Y: [1, 1], Z: [1, 1] }), Z00);
    if (y.kind !== "tasks") throw new Error("Expected task lineage");
    expect([y.tasks, y.definiteTasks]).toEqual([8, 8]);
  });
});

describe("following producers and stepping back", () => {
  beforeEach(() => {
    S().applyDSL(RESHAPED);
    S().setInspectorTab("plan");
    S().setPlanTile("Y", [1, 1]);
    S().setPlanTileAt("Z", [1, 1], [0, 0]);
  });

  it("keeps the way back through every task followed, and steps back to any of them", () => {
    S().followPlanTask({ tensorId: "Y", coord: [0, 3] });
    S().setPlanTile("F", [4]);
    S().followPlanTask({ tensorId: "F", coord: [0] });
    expect(activePlanTrail(S())!.from).toEqual([Z00, { tensorId: "Y", coord: [0, 3] }]);
    S().backPlanTask(1);
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 3] });
    expect(activePlanTrail(S())!.from).toEqual([Z00]);
    S().backPlanTask(0);
    expect(S().planTask).toEqual(Z00);
    expect(activePlanTrail(S())).toBeNull();
  });

  it("lets the trail go once the task is reached any other way", () => {
    S().followPlanTask({ tensorId: "Y", coord: [0, 3] });
    S().selectPlanTask({ tensorId: "Y", coord: [1, 0] });
    expect(activePlanTrail(S())).toBeNull();
    S().selectPlanTask(Z00);
    S().followPlanTask({ tensorId: "Y", coord: [0, 3] });
    S().undoWorkspace();
    expect(activePlanTrail(S())).toBeNull();
  });

  /* Equal coordinates are not the same arrival: landing on the trail's task
     again by a click must not bring back a trail that was left. */
  it("does not revive a trail when its task is reached again directly", () => {
    S().followPlanTask({ tensorId: "Y", coord: [0, 0] });
    S().selectPlanTask({ tensorId: "Y", coord: [1, 0] });
    S().selectPlanTask({ tensorId: "Y", coord: [0, 0] });
    expect(activePlanTrail(S())).toBeNull();
    S().backPlanTask(0);
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 0] });
  });

  it("brings the trail back with an undo to the task it reached", () => {
    S().followPlanTask({ tensorId: "Y", coord: [0, 0] });
    S().selectPlanTask({ tensorId: "Y", coord: [1, 0] });
    S().undoWorkspace();
    expect(activePlanTrail(S())!.from).toEqual([Z00]);
  });

  it("shows the reads in operand order, inputs as a badge, producers as k/n, and the trail", () => {
    let html = render();
    expect(html.indexOf("<code>Y</code>")).toBeLessThan(html.indexOf("<code>A</code>"));
    expect(html).toContain("graph input</span>");
    expect(html).not.toContain("produced by no task.");
    expect(html).toContain("requires 8/16 tasks");
    expect(html).toContain("Upstream, to the inputs");
    expect(html).toContain("level 2 · the tasks producing those tiles");
    S().followPlanTask({ tensorId: "Y", coord: [0, 3] });
    html = render();
    expect(html).toMatch(/<nav class="plan-trail" aria-label="followed from"><button[^>]*>Z \(0, 0\)<\/button>/);
  });
});
