/**
 * Step declarations in the store: declared and cleared as plan edits, undone
 * with them, dropped when a tile edit outdates them, and carried by a link.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "../../src/state/store";
import { decodeWorkspace, encodeWorkspace, linkToRestore, workspaceToLink } from "../../src/state/share";
import { stagingOf, type Staged } from "../../src/core/plan/steps";

const S = () => useStore.getState();

const MLP = `X = Tensor(16, 8, dtype=float16)
W1 = Parameter(8, 32, dtype=float16)
W2 = Parameter(32, 8, dtype=float16)
H = matmul(X, W1)
Y = matmul(H, W2)
`;

describe("step declarations", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().applyDSL(MLP);
    S().setPlanTile("H", [8, 16]);
    S().setPlanTile("Y", [8, 4]);
  });

  it("are part of the plan the store derives", () => {
    S().setPlanSteps("Y", { chunk: { k: 8 } });
    expect(S().planSteps).toEqual({ Y: { chunk: { k: 8 } } });
    expect((stagingOf(S().plan!, "Y") as Staged).steps).toBe(4);
    expect(S().plan!.steps.get("Y")).toEqual({ chunk: { k: 8 } });
  });

  it("change nothing when the graph does not support them", () => {
    const before = S();
    S().setPlanSteps("Y", { chunk: { k: 99 } });
    S().setPlanSteps("Y", { chunk: { m: 2 } });
    S().setPlanSteps("X", { chunk: { k: 2 } });
    S().setPlanSteps("Y", null);
    expect(S().planSteps).toBe(before.planSteps);
    expect(S().plan).toBe(before.plan);
    expect(S().workspaceHistory).toBe(before.workspaceHistory);
  });

  it("undo and clear as plan edits", () => {
    const history = S().workspaceHistory.length;
    S().setPlanSteps("Y", { chunk: { k: 8 } });
    S().setPlanSteps("Y", { chunk: { k: 16 }, accumulator: "f16" });
    expect(S().workspaceHistory.length).toBe(history + 2);
    S().undoWorkspace();
    expect(S().planSteps).toEqual({ Y: { chunk: { k: 8 } } });
    S().undoWorkspace();
    expect(S().planSteps).toEqual({});
    S().setPlanSteps("Y", { chunk: { k: 8 } });
    S().setPlanSteps("Y", null);
    expect(S().planSteps).toEqual({});
    S().undoWorkspace();
    expect(S().planSteps).toEqual({ Y: { chunk: { k: 8 } } });
  });

  it("keep their identity across edits that do not touch them", () => {
    S().setPlanSteps("Y", { chunk: { k: 8 } });
    const steps = S().planSteps;
    S().selectPlanTask({ tensorId: "Y", coord: [1, 1] });
    S().setPlanTile("H", [4, 16]);
    expect(S().planSteps).toBe(steps);
  });

  it("are dropped when their tensor is untiled, and come back with an undo", () => {
    S().setPlanSteps("H", { chunk: { k: 4 } });
    S().setPlanSteps("Y", { chunk: { k: 8 } });
    S().setPlanTile("H", null);
    // Y's own contraction is unchanged by fusing H; H has no tasks to divide.
    expect(S().planSteps).toEqual({ Y: { chunk: { k: 8 } } });
    S().undoWorkspace();
    expect(S().planSteps).toEqual({ H: { chunk: { k: 4 } }, Y: { chunk: { k: 8 } } });
  });

  it("travel in a link and restore with the plan", () => {
    S().setPlanSteps("Y", { chunk: { k: 8 }, order: ["k"], accumulator: "f16" });
    S().selectPlanTask({ tensorId: "Y", coord: [1, 0] });
    const link = workspaceToLink(S());
    expect(link.steps).toEqual({ Y: { c: { k: 8 }, o: ["k"], a: "f16" } });
    const decoded = decodeWorkspace(`#s=${encodeWorkspace(link)}`)!;
    expect(decoded.steps).toEqual(link.steps);

    S().loadExample(0);
    expect(S().planSteps).toEqual({});
    expect(S().restoreWorkspace(linkToRestore(decoded))).toBe(true);
    expect(S().planSteps).toEqual({ Y: { chunk: { k: 8 }, order: ["k"], accumulator: "f16" } });
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [1, 0] });
    expect((stagingOf(S().plan!, "Y") as Staged).accumulator).toBe("f16");
  });

  it("leave a link without them exactly as it was", () => {
    expect("steps" in workspaceToLink(S())).toBe(false);
  });

  it("refuse a link whose declaration does not fit, or is malformed", () => {
    const link = workspaceToLink(S());
    const withSteps = (steps: unknown) => decodeWorkspace(`#s=${encodeWorkspace({ ...link, steps } as never)}`);
    expect(withSteps({ Y: { c: { k: 0 } } })).toBeNull();
    expect(withSteps({ Y: { c: { k: 2.5 } } })).toBeNull();
    expect(withSteps({ Y: { o: [1] } })).toBeNull();
    expect(withSteps({ Y: { extra: 1 } })).toBeNull();
    expect(withSteps([])).toBeNull();
    // Well-formed, but not this graph's: refused when restored against it.
    for (const steps of [{ Y: { c: { k: 99 } } }, { Y: { c: { z: 2 } } }, { Y: { a: "float32" } }, { X: { c: { k: 2 } } }]) {
      const decoded = withSteps(steps)!;
      expect(decoded).not.toBeNull();
      const before = S();
      expect(S().restoreWorkspace(linkToRestore(decoded))).toBe(false);
      expect(S().plan).toBe(before.plan);
    }
  });

  it("are refused in a link that has no plan", () => {
    const { plan: _plan, task: _task, ...bare } = workspaceToLink(S());
    expect(decodeWorkspace(`#s=${encodeWorkspace({ ...bare, steps: { Y: { c: { k: 8 } } } } as never)}`)).toBeNull();
  });
});
