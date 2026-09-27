import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React, { createElement, isValidElement, ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { GraphView } from "../../../src/components/graph/GraphView";
import { useStore } from "../../../src/state/store";

vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) =>
      selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

type OpProps = React.HTMLAttributes<HTMLDivElement>;

function findOp(node: ReactNode): OpProps | undefined {
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findOp(child);
      if (found) return found;
    }
  } else if (isValidElement<{ className?: string; children?: ReactNode }>(node)) {
    if (node.props.className?.split(" ").includes("op-node")) return node.props;
    return findOp(node.props.children);
  }
}

// Capture the actual JSX handlers with React providing the hooks. This tests
// handler wiring and store effects, not the browser's pointer-capture behavior.
function operationProps(): OpProps {
  let props: OpProps | undefined;
  function Probe() {
    props = findOp(GraphView());
    return null;
  }
  renderToStaticMarkup(createElement(Probe));
  expect(props).toBeDefined();
  return props!;
}

const pointer = (x: number) => ({
  pointerId: 1, clientX: x, clientY: 0, target: null,
  currentTarget: { setPointerCapture: vi.fn() },
  preventDefault: vi.fn(), stopPropagation: vi.fn(),
}) as unknown as React.PointerEvent<HTMLDivElement>;

beforeEach(() => {
  useStore.getState().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  useStore.getState().setMoveOps(false);
  // Layout effects do not run in this handler-only server-render harness.
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => { vi.restoreAllMocks(); });

describe("locking operations during a drag", () => {
  it("blocks initiation but retains movement and cleanup handlers while locked", () => {
    const props = operationProps();
    expect(props.onPointerDown).toBeUndefined();
    for (const name of ["onPointerMove", "onPointerUp", "onPointerCancel", "onLostPointerCapture"] as const)
      expect(props[name]).toBeTypeOf("function");
  });

  it.each(["onPointerUp", "onPointerCancel", "onLostPointerCapture"] as const)(
    "%s finishes an active gesture after locking",
    (finish) => {
      useStore.getState().setMoveOps(true);
      const props = operationProps();
      const depth = useStore.getState().workspaceHistory.length;
      props.onPointerDown!(pointer(0));
      props.onPointerMove!(pointer(5));
      expect(useStore.getState().dragging).toBe(true);
      expect(Object.keys(useStore.getState().nodeOffsets)).toHaveLength(1);

      useStore.getState().setMoveOps(false);
      props[finish]!(pointer(5));
      expect(useStore.getState().dragging).toBe(false);
      if (finish === "onPointerUp") {
        expect(useStore.getState().workspaceHistory).toHaveLength(depth + 1);
        useStore.getState().undoWorkspace();
      } else {
        expect(useStore.getState().workspaceHistory).toHaveLength(depth);
      }
      expect(useStore.getState().nodeOffsets).toEqual({});
    }
  );
});
