import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "../../src/state/store";
import { serializeWorkspace, parseWorkspaceDocument } from "../../src/state/workspace-document";
import { workspaceToLink, encodeWorkspace, decodeWorkspace, linkToRestore } from "../../src/state/share";
import { layoutArtifact } from "../../src/worker/analysis-jobs";
import { buildGraphScene } from "../../src/view/graph/graph-scene";
import { cardShapeView, type CardShapeView } from "../../src/view/tensor/card-size";
import { displayedCards } from "../../src/view/tensor/resize";
import type { BaseGraphLayout } from "../../src/view/graph/graph-scene";
import { box, fromBox } from "../../src/core/region";

const SOURCE = "X = Tensor(8, 16, dtype=fp32)\nY = relu(X)\n";
const S = () => useStore.getState();
/** The placed graph as the canvas sees it: layout, offsets, and resized cards. */
const sceneOf = (base: BaseGraphLayout) =>
  buildGraphScene(base, S().nodeOffsets, displayedCards(S().viewCfgs, S().resolved!, S().graphPx).sizes).nodes;
const overlapping = (nodes: { x: number; y: number; w: number; h: number }[]) => nodes.some((a, i) =>
  nodes.some((b, j) => i < j && Math.min(a.x + a.w, b.x + b.w) > Math.max(a.x, b.x) && Math.min(a.y + a.h, b.y + b.h) > Math.max(a.y, b.y)));
/** The structural layout the canvas would place this build with. */
const layoutFor = () => {
  const s = S(), resolved = s.resolved!;
  const views: Record<string, CardShapeView> = {};
  for (const [id, cfg] of Object.entries(s.viewCfgs)) {
    const view = cardShapeView(resolved.tensors[id].resolved!, cfg);
    if (view) views[id] = view;
  }
  return layoutArtifact(resolved, s.graphPx, views);
};
beforeEach(() => { useStore.setState(useStore.getInitialState(), true); S().applyDSL(SOURCE); });
describe("card resize workspace edits", () => {
  it("updates and cancels live geometry atomically, without recording intermediate frames", () => {
    const seen: [number | undefined, unknown, number][] = [];
    const stop = useStore.subscribe((s) => seen.push([s.viewCfgs.Y.scale, s.nodeOffsets["t:Y"], s.workspaceHistory.length]));
    S().setCardGeometry("Y", 2, { dx: 0, dy: -128 }, false);
    S().setCardGeometry("Y", 1, { dx: 0, dy: 0 }, false);
    stop();
    expect(seen).toEqual([[2, { dx: 0, dy: -128 }, 0], [undefined, undefined, 0]]);
  });
  it("records precise size edits and their geometry in the same update", () => {
    const seen: [number | undefined, number][] = [];
    const stop = useStore.subscribe((s) => seen.push([s.viewCfgs.Y.scale, s.workspaceHistory.length]));
    S().setCardScale("Y", 1.251);
    stop(); expect(seen).toEqual([[1.251, 1]]);
    S().undoWorkspace(); expect(S().viewCfgs.Y.scale).toBeUndefined();
  });
  it("rejects invalid complete geometry without changing size, position or history", () => {
    const before = S();
    S().setCardGeometry("Y", 2, { dx: Infinity, dy: 0 });
    S().setCardGeometry("Y", 0, { dx: 5, dy: 0 });
    S().setCardGeometry("Y", 2, { dx: 1e9, dy: 0 });
    expect(S()).toBe(before);
  });
  it("disables ineffective singleton resizing but permits clearing a legacy stored size", () => {
    S().applyDSL("X = Tensor(1, 1, dtype=fp32)\n");
    const before = S();
    S().setResizingTensor("X"); S().setCardScale("X", 2);
    expect(S()).toBe(before);
    S().setViewCfg("X", { scale: 2 });
    S().setCardScale("X", 1);
    expect(S().viewCfgs.X.scale).toBeUndefined();
    expect(S().workspaceHistory).toHaveLength(1);
  });
  it("retires resize mode when the chosen plane becomes fixed", () => {
    S().applyDSL("X = Tensor(8, 1, 1, dtype=fp32)\n");
    S().setViewAxes("X", [0, 2]); S().setResizingTensor("X");
    expect(S().resizingTensor).toBe("X");
    S().setViewAxes("X", null); expect(S().resizingTensor).toBeNull();
    S().setViewAxes("X", [0, 2]); S().setResizingTensor("X");
    S().setViewCfg("X", { axes: undefined }); expect(S().resizingTensor).toBeNull();
    expect(S().canvasTool).toBe("move");
  });
  it("makes room for requested sizes, preserves analysis and the anchor, and undoes in one step", () => {
    S().setSelection("X", fromBox(box([0, 2], [0, 2])), "replace");
    S().setCardScale("X", 4); S().setCardScale("Y", 1.5);
    const base = layoutArtifact(S().resolved!, S().graphPx);
    S().setNodeOffset("t:X", { dx: 5, dy: 10 });
    const before = S();
    expect(overlapping(sceneOf(base))).toBe(true);
    const seen: unknown[] = [];
    const stop = useStore.subscribe((state) => seen.push(state));
    S().makeRoomForCard("X", base); stop();
    expect(seen).toHaveLength(1);
    expect(S().nodeOffsets["t:X"]).toEqual(before.nodeOffsets["t:X"]);
    expect(S().selection).toBe(before.selection); expect(S().backwardRes).toBe(before.backwardRes);
    expect(displayedCards(S().viewCfgs, S().resolved!, S().graphPx).scales).toMatchObject({ "t:X": 4, "t:Y": 1.5 });
    expect(overlapping(sceneOf(base))).toBe(false);
    expect(S().workspaceHistory).toHaveLength(before.workspaceHistory.length + 1);
    S().makeRoomForCard("X", base);
    expect(S().workspaceHistory).toHaveLength(before.workspaceHistory.length + 1);
    S().undoWorkspace(); expect(S().nodeOffsets).toEqual(before.nodeOffsets);
    expect(S().viewCfgs.X.scale).toBe(4);
  });
  it("does not arrange the fixed Plan view or an outdated workspace", () => {
    const base = layoutArtifact(S().resolved!, S().graphPx);
    S().setCardScale("X", 4); S().setInspectorTab("plan");
    const before = S(); S().makeRoomForCard("X", base); expect(S()).toBe(before);
    S().setInspectorTab("dependencies"); S().setDraftText("unfinished");
    const stale = S(); S().makeRoomForCard("X", base); expect(S()).toBe(stale);
  });
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
    S().commitCardResize("Y", 1, {});
    expect(S().workspaceHistory).toHaveLength(1);
    expect(S().resolved).toBe(before.resolved); expect(S().selection).toBe(before.selection);
    S().undoWorkspace(); expect(S().viewCfgs.Y.scale).toBeUndefined(); expect(S().nodeOffsets["t:Y"]).toBeUndefined();
  });
  it("restores size and position in a single store update", () => {
    S().setCardScale("Y", 2, false); S().setNodeOffset("t:Y", { dx: 0, dy: -64 });
    S().commitCardResize("Y", 1, {});
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
    S().commitCardResize("Y", 1, {}); expect(S().workspaceHistory).toHaveLength(0);
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
  it("keeps a resized card at its stored scale when a rebuild leaves it less room", () => {
    S().setCardScale("X", 3);
    const fits = layoutFor();
    const roomy = () => !overlapping(sceneOf(layoutFor()));
    S().makeRoomForCard("X", fits);
    expect(roomy()).toBe(true);
    S().applyDSL(SOURCE.replace("8, 16", "8, 64"));
    expect(S().viewCfgs.X.scale).toBe(3);
    expect(displayedCards(S().viewCfgs, S().resolved!, S().graphPx).scales["t:X"]).toBe(3);
    expect(roomy()).toBe(false);
    S().makeRoomForCard("X", layoutFor());
    expect(roomy()).toBe(true);
  });
});

describe("resizing makes its own room", () => {
  const offsetOf = (key: string) => S().nodeOffsets[key] ?? { dx: 0, dy: 0 };

  it("moves a neighbour aside as the card grows, and back as it shrinks", () => {
    const layout = layoutFor();
    const from = S().nodeOffsets;
    const grow = (scale: number) => S().setCardGeometry("X", scale, offsetOf("t:X"), false, { layout, from });
    grow(1.2);
    const start = sceneOf(layout);
    grow(4);
    expect(overlapping(sceneOf(layout))).toBe(false);
    const moved = sceneOf(layout).filter((node) => {
      const was = start.find((n) => n.kind === node.kind && n.id === node.id)!;
      return node.x !== was.x || node.y !== was.y;
    });
    expect(moved.length).toBeGreaterThan(0);
    expect(moved.some((node) => node.kind === "tensor" && node.id === "X")).toBe(false);
    grow(1);
    expect(S().nodeOffsets).toEqual(from);
    expect(S().workspaceHistory).toHaveLength(0);
  });

  it("restores every node when cancelled, and undoes the neighbours with the size", () => {
    const layout = layoutFor();
    const from = S().nodeOffsets;
    S().setCardGeometry("X", 4, offsetOf("t:X"), false, { layout, from });
    expect(S().nodeOffsets).not.toEqual(from);
    S().restoreCardGeometry("X", 1, from);
    expect(S().viewCfgs.X.scale).toBeUndefined();
    expect(S().nodeOffsets).toEqual(from);
    S().setCardGeometry("X", 4, offsetOf("t:X"), false, { layout, from });
    S().commitCardResize("X", 1, from);
    expect(S().workspaceHistory).toHaveLength(1);
    S().undoWorkspace();
    expect(S().viewCfgs.X.scale).toBeUndefined();
    expect(S().nodeOffsets).toEqual(from);
  });
});
