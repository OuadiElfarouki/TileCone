import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "../../src/state/store";
import { serializeWorkspace, parseWorkspaceDocument } from "../../src/state/workspace-document";
import { workspaceToLink, encodeWorkspace, decodeWorkspace, linkToRestore } from "../../src/state/share";
import { layoutArtifact } from "../../src/worker/analysis-jobs";
import { buildGraphScene } from "../../src/view/graph/graph-scene";
import { cardShapeView, type CardShapeView } from "../../src/view/tensor/card-size";
import { displayedCards } from "../../src/view/tensor/resize";

const SOURCE = "X = Tensor(8, 16, dtype=fp32)\nY = relu(X)\n";
const S = () => useStore.getState();
beforeEach(() => { useStore.setState(useStore.getInitialState(), true); S().applyDSL(SOURCE); });
describe("card resize workspace edits", () => {
  it("enters Move and leaves it selected when resizing ends", () => {
    S().setResizingTensor("Y"); expect(S().canvasTool).toBe("move");
    expect(S().resizingTensor).toBe("Y");
    S().setResizingTensor(null); expect(S().canvasTool).toBe("move");
    S().setResizingTensor("X"); S().setCanvasTool("select"); expect(S().resizingTensor).toBeNull();
  });
  it("does not enter resizing for a missing tensor, stale draft, or fixed Plan layout", () => {
    S().setResizingTensor("absent"); expect(S().resizingTensor).toBeNull();
    S().setDraftText("unfinished"); S().setResizingTensor("Y"); expect(S().resizingTensor).toBeNull();
    S().setDraftText(SOURCE); S().setInspectorTab("plan");
    S().setResizingTensor("Y"); expect(S().resizingTensor).toBeNull();
  });
  it("records one drag, restoring both size and top-corner movement on undo", () => {
    const before = S();
    S().setCardScale("Y", 1.5, false); S().setCardScale("Y", 2, false);
    S().setNodeOffset("t:Y", { dx: 0, dy: -128 });
    expect(S().workspaceHistory).toHaveLength(0);
    S().commitCardResize("Y", 1, { dx: 0, dy: 0 });
    expect(S().workspaceHistory).toHaveLength(1);
    expect(S().resolved).toBe(before.resolved); expect(S().selection).toBe(before.selection);
    S().undoWorkspace(); expect(S().viewCfgs.Y.scale).toBeUndefined(); expect(S().nodeOffsets["t:Y"]).toBeUndefined();
  });
  it("restores size and position in a single store update", () => {
    S().setCardScale("Y", 2, false); S().setNodeOffset("t:Y", { dx: 0, dy: -64 });
    S().commitCardResize("Y", 1, { dx: 0, dy: 0 });
    const seen: [number | undefined, unknown][] = [];
    const stop = useStore.subscribe((s) => seen.push([s.viewCfgs.Y.scale, s.nodeOffsets["t:Y"]]));
    S().undoWorkspace(); stop();
    expect(seen).toEqual([[undefined, undefined]]);
  });
  it("keeps ordinary move undo independent of the card's size", () => {
    S().setCardScale("Y", 2);
    S().setNodeOffset("t:Y", { dx: 10, dy: 20 }); S().commitNodeMove("t:Y", { dx: 0, dy: 0 });
    S().undoWorkspace(); expect(S().viewCfgs.Y.scale).toBe(2);
    S().undoWorkspace(); expect(S().viewCfgs.Y.scale).toBeUndefined();
  });
  it("normalizes resets and records no step for no-op or invalid resizing", () => {
    S().setCardScale("Y", 1 + 1e-10); S().setCardScale("Y", Infinity); S().setCardScale("Y", 9);
    S().commitCardResize("Y", 1, { dx: 0, dy: 0 }); expect(S().workspaceHistory).toHaveLength(0);
    S().setCardScale("Y", 2); S().setCardScale("Y", 1); S().undoWorkspace(); expect(S().viewCfgs.Y.scale).toBe(2);
  });
  it("persists size in files and links, without persisting the active mode", () => {
    S().setCardScale("Y", 1.7); S().setResizingTensor("Y");
    const file = serializeWorkspace(S()), link = decodeWorkspace(`#s=${encodeWorkspace(workspaceToLink(S()))}`)!;
    S().loadExample(0); expect(S().restoreWorkspace(parseWorkspaceDocument(file))).toBe(true);
    expect(S().viewCfgs.Y.scale).toBe(1.7); expect(S().resizingTensor).toBeNull();
    S().loadExample(0); expect(S().restoreWorkspace(linkToRestore(link))).toBe(true);
    expect(S().viewCfgs.Y.scale).toBe(1.7);
  });
  it("carries sizing through parameter-only rebuilds and retires the mode", () => {
    S().setCardScale("Y", 2); S().setResizingTensor("Y");
    S().applyDSL(SOURCE.replace("8, 16", "16, 16"));
    expect(S().viewCfgs.Y.scale).toBe(2); expect(S().resizingTensor).toBeNull();
  });
  it("draws a resized card smaller, not over its neighbour, when a rebuild leaves it less room", () => {
    const scene = () => {
      const s = S(), resolved = s.resolved!;
      const views: Record<string, CardShapeView> = {};
      for (const [id, cfg] of Object.entries(s.viewCfgs)) {
        const view = cardShapeView(resolved.tensors[id].resolved!, cfg);
        if (view) views[id] = view;
      }
      const base = layoutArtifact(resolved, s.graphPx, views);
      const cards = displayedCards(base, s.nodeOffsets, s.viewCfgs, resolved, s.graphPx);
      return { cards, nodes: buildGraphScene(base, s.nodeOffsets, cards.sizes).nodes };
    };
    const overlapping = (nodes: { x: number; y: number; w: number; h: number }[]) => nodes.some((a, i) =>
      nodes.some((b, j) => i < j && Math.min(a.x + a.w, b.x + b.w) > Math.max(a.x, b.x) && Math.min(a.y + a.h, b.y + b.h) > Math.max(a.y, b.y)));
    // Store the largest X that fits beside its operation, as a resize stopping there would.
    S().setCardScale("X", 8);
    const largest = scene().cards.scales["t:X"];
    expect(largest).toBeGreaterThan(1);
    S().setCardScale("X", largest);
    expect(scene().cards.scales["t:X"]).toBe(largest);
    expect(overlapping(scene().nodes)).toBe(false);
    S().applyDSL(SOURCE.replace("8, 16", "8, 64"));
    const after = scene();
    expect(S().viewCfgs.X.scale).toBe(largest);
    expect(after.cards.scales["t:X"]).toBeLessThan(largest);
    expect(overlapping(after.nodes)).toBe(false);
  });
});
