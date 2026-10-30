import { beforeEach, describe, expect, it } from "vitest";
import { fromBox, box } from "../../src/core/region";
import { useStore } from "../../src/state/store";
import { parseWorkspaceDocument, serializeWorkspace, workspaceDocument } from "../../src/state/workspace-document";
import { workspaceToLink } from "../../src/state/share";
import { planReport } from "../../src/core/plan/interfaces";

const SOURCE = "A = Tensor(8, 8, dtype=float32)\nB = relu(A)\n";
const S = () => useStore.getState();
beforeEach(() => { useStore.setState(useStore.getInitialState(), true); S().applyDSL(SOURCE); });

describe("portable workspace documents", () => {
  it("restores unfinished source, disabled tiles, scope, views, placement, and plans", () => {
    S().setSelection("B", fromBox(box([0, 2], [0, 2])), "replace");
    S().setSelection("A", fromBox(box([2, 4], [2, 4])));
    S().toggleBoxHidden(1);
    S().togglePinBox(0);
    S().setViewCfg("B", { projection: false, tile: [2, 2] });
    S().setNodeOffset("t:B", { dx: 27, dy: -16 });
    S().setPlanTile("B", [2, 2]);
    S().selectPlanTask({ tensorId: "B", coord: [1, 1] });
    S().keepPlan(planReport(S().plan!));
    S().setInspectorTab("plan");
    S().setDraftText("unfinished λ = (");
    const expected = workspaceDocument(S());
    const saved = serializeWorkspace(S());
    S().loadExample(0);
    expect(S().restoreWorkspace(parseWorkspaceDocument(saved))).toBe(true);
    expect(workspaceDocument(S())).toEqual(expected);
    expect(S().dslText).toBe(SOURCE);
    expect(S().draftText).toBe("unfinished λ = (");
    expect(S().backwardRes).not.toBeNull();
    expect(S().workspaceHistory).toEqual([]);
  });

  it("accepts raw Share JSON beyond the URL limit", () => {
    const raw = { ...workspaceToLink(S()), dsl: SOURCE + "#" + "x".repeat(12000) };
    expect(S().restoreWorkspace(parseWorkspaceDocument(JSON.stringify(raw)))).toBe(true);
  });

  it.each(["null", "[]", "{", JSON.stringify({ format: "tilecone-workspace", version: 2 })])(
    "refuses malformed or unsupported documents: %s", (text) => {
      expect(() => parseWorkspaceDocument(text)).toThrow();
    });

  it("validates tile metadata and graph-relative settings without changing the workspace", () => {
    const before = S();
    const doc = workspaceDocument(before);
    expect(() => parseWorkspaceDocument(JSON.stringify({ ...doc, scope: { hidden: [0], group: null, pinned: null } }))).toThrow();
    const badGraph = { ...doc, workspace: { ...doc.workspace, views: { absent: { projection: true, sliders: [0] } } } };
    expect(S().restoreWorkspace(parseWorkspaceDocument(JSON.stringify(badGraph)))).toBe(false);
    expect(S().graph).toBe(before.graph);
    expect(S().draftText).toBe(before.draftText);
  });
});
