import { beforeEach, describe, expect, it, vi } from "vitest";
import React, { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { useGraphGestures } from "../../../src/components/graph/useGraphGestures";
import { useStore } from "../../../src/state/store";
import { layoutArtifact } from "../../../src/worker/analysis-jobs";
import { buildGraphScene } from "../../../src/view/graph/graph-scene";
import { displayedCards } from "../../../src/view/tensor/resize";
import { cardShapeView, type CardShapeView } from "../../../src/view/tensor/card-size";

vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});
const S = () => useStore.getState();
const event = (x: number, pointerId = 1) => ({
  pointerId, button: 0, clientX: x, clientY: x,
  currentTarget: { setPointerCapture: vi.fn(), hasPointerCapture: () => true, releasePointerCapture: vi.fn() },
  preventDefault: vi.fn(), stopPropagation: vi.fn(),
}) as unknown as React.PointerEvent<HTMLElement>;
function gestures() {
  const state = S(), graph = state.resolved!;
  const views: Record<string, CardShapeView> = {};
  for (const [id, cfg] of Object.entries(state.viewCfgs)) {
    const view = cardShapeView(graph.tensors[id].resolved, cfg);
    if (view) views[id] = view;
  }
  const base = layoutArtifact(graph, state.graphPx, views);
  const scene = buildGraphScene(base, state.nodeOffsets, displayedCards(state.viewCfgs, graph, state.graphPx).sizes);
  let result!: ReturnType<typeof useGraphGestures>;
  function Probe() {
    result = useGraphGestures({ sceneRef: { current: scene }, baseLayoutRef: { current: base },
      tfRef: { current: { k: 1 } }, movedRef: { current: false }, interactive: true });
    return null;
  }
  renderToStaticMarkup(createElement(Probe));
  return result;
}
beforeEach(() => {
  useStore.setState(useStore.getInitialState(), true);
  S().applyDSL("X = Tensor(8, 16)\nY = relu(X)\n");
  S().setResizingTensor("X");
});
describe("resize gesture transactions", () => {
  it("previews without history, ignores other pointers, and cancels all geometry", () => {
    const g = gestures(), before = S(), press = event(0);
    g.startResize(press, "X", "bottom");
    g.dragFollow.onPointerMove(event(80));
    expect(S().viewCfgs.X.scale).toBeGreaterThan(1);
    expect(S().workspaceHistory).toEqual(before.workspaceHistory);
    g.dragFollow.onPointerUp(event(90, 2));
    expect(g.isResizing()).toBe(true);
    g.dragFollow.onPointerCancel(event(80));
    expect(S().viewCfgs.X.scale).toBe(before.viewCfgs.X.scale);
    expect(S().nodeOffsets).toEqual(before.nodeOffsets);
    expect(S().workspaceHistory).toEqual(before.workspaceHistory);
    expect(press.currentTarget.releasePointerCapture).toHaveBeenCalledWith(1);
  });
  it("commits the release geometry as one undoable edit", () => {
    const g = gestures(), before = S();
    g.startResize(event(0), "X", "bottom");
    g.dragFollow.onPointerUp(event(80));
    expect(S().viewCfgs.X.scale).toBeGreaterThan(1);
    expect(S().workspaceHistory).toHaveLength(before.workspaceHistory.length + 1);
    S().undoWorkspace();
    expect(S().viewCfgs.X.scale).toBe(before.viewCfgs.X.scale);
    expect(S().nodeOffsets).toEqual(before.nodeOffsets);
  });
  it("drops stale updates after graph replacement", () => {
    const g = gestures();
    g.startResize(event(0), "X", "bottom");
    S().applyDSL("X = Tensor(16, 32)\nY = relu(X)\n");
    const before = S();
    g.dragFollow.onPointerMove(event(80));
    g.dragFollow.onPointerUp(event(90));
    expect(S().viewCfgs).toEqual(before.viewCfgs);
    expect(S().nodeOffsets).toEqual(before.nodeOffsets);
    expect(S().workspaceHistory).toEqual(before.workspaceHistory);
  });
});
