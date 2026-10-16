import { beforeEach, describe, expect, it } from "vitest";
import { box, fromBox } from "../../src/core/region";
import { useStore } from "../../src/state/store";

const S = () => useStore.getState();
const point = (b: number, r: number, c: number) => box([b, b + 1], [r, r + 1], [c, c + 1]);
beforeEach(() => {
  S().applyDSL("X = Tensor(2, 3, 4)\nY = relu(X)\nZ = Tensor(2, 3, 4)\n");
  S().setSnapToGrid(false);
  S().setInspectorTab("dependencies");
});

describe("selection carry", () => {
  it.each(["across", "down", "grid"] as const)("carries a pinned tile between panels when unfolded %s", (unfold) => {
    S().setViewCfg("Y", { projection: false, unfold });
    S().setSelection("Y", fromBox(point(0, 2, 3)), "replace");
    S().togglePinBox(0);
    const cfg = S().viewCfgs.Y;
    const depth = S().workspaceHistory.length;
    S().nudgeSelection(2, 1);
    expect(S().selection!.parts[0].box).toEqual(point(1, 0, 0));
    expect(S().backwardRes!.tensors.get("X")!.region.boxes).toEqual([point(1, 0, 0)]);
    expect(S().viewCfgs.Y).toBe(cfg);
    expect(S().pinnedBox).toBe(0);
    expect(S().focusedBox).toBe(0);
    expect(S().workspaceHistory.length).toBe(depth + 1);
    S().nudgeSelection(2, -1);
    expect(S().selection!.parts[0].box).toEqual(point(0, 2, 3));
  });

  it("shows the new slice on carry and the restored slice on undo", () => {
    S().setViewCfg("Y", { projection: false });
    S().setSelection("Y", fromBox(point(0, 2, 3)), "replace");
    S().nudgeSelection(2, 1);
    expect(S().viewCfgs.Y.sliders[0]).toBe(1);
    S().undoWorkspace();
    expect(S().selection!.parts[0].box).toEqual(point(0, 2, 3));
    expect(S().viewCfgs.Y.sliders[0]).toBe(0);
  });

  it("leaves a scrubbed slice alone when undo restores something other than tiles", () => {
    S().setViewCfg("Y", { projection: false });
    S().setSelection("Y", fromBox(point(0, 2, 3)), "replace");
    S().setViewCfg("Y", { sliders: [1, 0, 0] });
    S().setNodeOffset("t:Y", { dx: 40, dy: 0 });
    S().commitNodeMove("t:Y", { dx: 0, dy: 0 });
    S().undoWorkspace();
    expect(S().viewCfgs.Y.sliders[0]).toBe(1);
  });

  it("preserves peer parts, enabled state, and other tensor groups when one tile is focused", () => {
    S().setSelection("Y", fromBox(point(0, 2, 3)), "replace");
    S().setSelection("Y", fromBox(point(0, 0, 0)), "union");
    S().setSelection("Z", fromBox(point(1, 1, 1)), "union");
    S().toggleBoxHidden(1);
    S().togglePinBox(0);
    const peers = S().selection!.parts.slice(1);
    S().nudgeSelection(2, 1);
    expect(S().selection!.parts[0].box).toEqual(point(1, 0, 0));
    expect(S().selection!.parts[1]).toBe(peers[0]);
    expect(S().selection!.parts[2]).toBe(peers[1]);
    expect(S().hiddenBoxes).toEqual(new Set([1]));
  });

  it("carries only the tile just drawn when no tile is focused", () => {
    S().setSelection("Y", fromBox(point(0, 0, 2)), "replace");
    S().setSelection("Y", fromBox(point(0, 1, 3)), "union");
    const first = S().selection!.parts[0];
    S().nudgeSelection(2, 1);
    expect(S().selection!.parts[0]).toBe(first);
    expect(S().selection!.parts[1].box).not.toEqual(point(0, 1, 3));
  });

  it("keeps held and fast movement as one undo step and stops without history at the outer end", () => {
    S().setSelection("Y", fromBox(point(0, 0, 0)), "replace");
    const depth = S().workspaceHistory.length;
    S().nudgeSelection(2, 1, 8);
    expect(S().selection!.parts[0].box).toEqual(point(0, 2, 0));
    S().nudgeSelection(2, 1, 8, false);
    expect(S().selection!.parts[0].box).toEqual(point(1, 1, 0));
    expect(S().workspaceHistory.length).toBe(depth + 1);
    S().undoWorkspace();
    expect(S().selection!.parts[0].box).toEqual(point(0, 0, 0));
    S().setSelection("Y", fromBox(point(1, 2, 3)), "replace");
    const selection = S().selection, history = S().workspaceHistory;
    S().nudgeSelection(2, 1);
    expect(S().selection).toBe(selection);
    expect(S().workspaceHistory).toBe(history);
  });

  it("preserves a projected tile's full hidden span instead of inventing a batch position", () => {
    S().setSelection("Y", fromBox(box([0, 2], [2, 3], [3, 4])), "replace");
    const selection = S().selection;
    S().nudgeSelection(2, 1);
    expect(S().selection).toBe(selection);
  });
});

describe("plan task carry", () => {
  it("uses the family grid across tails, preserves plan identity, and follows the slice", () => {
    S().setViewCfg("Y", { projection: false });
    S().setInspectorTab("plan");
    S().setPlanTileAt("Y", [1, 2, 3], [0, 2, 3]);
    const plan = S().plan;
    expect(S().planTask!.coord).toEqual([0, 1, 1]);
    S().movePlanTask(2, 1);
    expect(S().planTask!.coord).toEqual([1, 0, 0]);
    expect(S().plan).toBe(plan);
    expect(S().viewCfgs.Y.sliders[0]).toBe(1);
    S().undoWorkspace();
    expect(S().planTask!.coord).toEqual([0, 1, 1]);
    expect(S().viewCfgs.Y.sliders[0]).toBe(0);
  });
});
