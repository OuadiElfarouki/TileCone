import React, { createElement, isValidElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GraphView } from "../../../src/components/graph/GraphView";
import { Inspector } from "../../../src/components/inspector/Inspector";
import { SidePanel } from "../../../src/components/chrome/SidePanel";
import { useStore } from "../../../src/state/store";
import { graphIsInteractive } from "../../../src/view/workspace";
import { fromBox } from "../../../src/core/region";

// Static rendering reads the current store rather than the server's initial snapshot.
vi.mock("../../../src/state/store", async (original) => {
  const actual = await original<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (select: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => select(actual.useStore.getState()),
    actual.useStore,
  ) };
});
const SOURCE = "M = 8\nX = Tensor(M)\nY = relu(X)\n";
const render = (component: () => React.ReactElement | null) => renderToStaticMarkup(createElement(component));
const findOperation = (node: ReactNode): React.HTMLAttributes<HTMLDivElement> | undefined => {
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findOperation(child);
      if (found) return found;
    }
  } else if (isValidElement<React.HTMLAttributes<HTMLDivElement>>(node)) {
    if (node.props.className?.split(" ").includes("op-node")) return node.props;
    return findOperation(node.props.children);
  }
};

beforeEach(() => {
  useStore.getState().applyDSL(SOURCE);
  useStore.getState().setSelection("Y", fromBox([{ lo: 0, hi: 2 }]), "replace");
  useStore.getState().setNodeOffset("t:X", { dx: 20, dy: 0 });
  useStore.getState().setInspectorTab("dependencies");
});

describe("inactive compiled graph", () => {
  it.each(["empty", "example", "parameter", "building", "failed"])(
    "keeps navigation and disables inspection for %s source", (kind) => {
      const { resolved, selection, nodeOffsets } = useStore.getState();
      if (kind === "empty") useStore.getState().setDraftText("");
      if (kind === "example") useStore.getState().stageExample(0);
      if (kind === "parameter") useStore.getState().setDraftText(SOURCE.replace("M = 8", "M = 16"));
      if (kind === "building") useStore.setState({ compiling: true });
      if (kind === "failed") useStore.getState().applyDSL("Z = relu(Missing)\n");
      expect(graphIsInteractive(useStore.getState())).toBe(false);
      const graph = render(GraphView);
      expect(graph).toContain('class="graph-canvas inactive');
      expect(graph).toContain('inert=""');
      expect(graph).toContain('aria-disabled="true"');
      expect(graph).toContain('title="zoom in"');
      expect(graph).toContain('title="fit to view (f)"');
      expect(graph).not.toContain('class="layout-controls"');
      expect(graph).not.toContain('class="grid-controls"');
      expect(render(SidePanel)).not.toContain('class="operation-row');
      const inspector = render(Inspector);
      expect(inspector).toContain("Run the source to inspect the graph.");
      expect(inspector).not.toContain("Cost to compute");
      const state = useStore.getState();
      expect(state.resolved).toBe(resolved);
      expect(state.selection).toBe(selection);
      expect(state.nodeOffsets).toBe(nodeOffsets);
      useStore.setState({ compiling: false });
    }
  );

  it("reactivates on reverting source or successfully installing a build", () => {
    useStore.getState().setDraftText("");
    expect(graphIsInteractive(useStore.getState())).toBe(false);
    useStore.getState().setDraftText(SOURCE);
    expect(graphIsInteractive(useStore.getState())).toBe(true);
    expect(render(GraphView)).not.toContain('inert=""');
    expect(render(SidePanel)).toContain('class="operation-row');
    expect(render(Inspector)).toContain("Cost to compute");
    useStore.getState().applyDSL("X = Tensor(4)\nZ = exp(X)\n");
    expect(graphIsInteractive(useStore.getState())).toBe(true);
    expect(render(GraphView)).not.toContain('class="graph-canvas inactive');
  });

  it("rejects a node event from before the source changed", () => {
    let operation: React.HTMLAttributes<HTMLDivElement> | undefined;
    function Probe() { operation = findOperation(GraphView()); return null; }
    render(Probe);
    expect(operation).toBeDefined();
    useStore.getState().setSelectedOp(null);
    useStore.getState().setDraftText("");
    const event = { target: { closest: () => null }, stopPropagation: vi.fn() };
    operation!.onClick!(event as unknown as React.MouseEvent<HTMLDivElement>);
    expect(useStore.getState().selectedOp).toBeNull();
  });
});
