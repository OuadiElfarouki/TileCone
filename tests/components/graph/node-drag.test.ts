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

function findByClass(node: ReactNode, wanted: string): OpProps | undefined {
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findByClass(child, wanted);
      if (found) return found;
    }
  } else if (isValidElement<{ className?: string; children?: ReactNode }>(node)) {
    if (node.props.className?.split(" ").includes(wanted)) return node.props;
    return findByClass(node.props.children, wanted);
  }
}

const findOp = (node: ReactNode) => findByClass(node, "op-node");

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

/** The first tensor card's move handle, probed as `operationProps` probes the
 *  operation node. */
function cardHandleProps(): OpProps {
  let props: OpProps | undefined;
  function Probe() {
    props = findByClass(GraphView(), "tensor-grab");
    return null;
  }
  renderToStaticMarkup(createElement(Probe));
  expect(props).toBeDefined();
  return props!;
}

const arrow = (key: string, shiftKey = false) => ({
  key, shiftKey, ctrlKey: false, metaKey: false, altKey: false,
  preventDefault: vi.fn(), stopPropagation: vi.fn(),
}) as unknown as React.KeyboardEvent<HTMLDivElement>;

const pointer = (x: number) => ({
  pointerId: 1, button: 0, clientX: x, clientY: 0, target: null,
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

describe("moving a node by keyboard", () => {
  it("nudges a card one step per press, each its own undo entry", () => {
    const props = cardHandleProps();
    const depth = useStore.getState().workspaceHistory.length;

    props.onKeyDown!(arrow("ArrowRight"));
    props.onKeyDown!(arrow("ArrowDown"));
    const moved = Object.values(useStore.getState().nodeOffsets)[0];
    expect(moved.dx).toBeGreaterThan(0);
    expect(moved.dy).toBeGreaterThan(0);
    expect(useStore.getState().workspaceHistory).toHaveLength(depth + 2);

    useStore.getState().undoWorkspace();
    expect(Object.values(useStore.getState().nodeOffsets)[0].dy).toBe(0);
  });

  it("travels eight times as far with Shift", () => {
    // Upwards, where this row of nodes leaves the card open space to move into.
    const props = cardHandleProps();
    props.onKeyDown!(arrow("ArrowUp"));
    const step = Object.values(useStore.getState().nodeOffsets)[0].dy;
    expect(step).toBeLessThan(0);

    useStore.getState().resetNodeLayout();
    props.onKeyDown!(arrow("ArrowUp", true));
    expect(Object.values(useStore.getState().nodeOffsets)[0].dy).toBe(step * 8);
  });

  /* A nudge is the drag's motion sampled once, so it inherits the collision
     sweep: the operation node sits to this card's right in a left-to-right
     layout, and a long press lands against it rather than through it. */
  it("stops a nudge at a neighbour instead of tunnelling through it", () => {
    const props = cardHandleProps();
    props.onKeyDown!(arrow("ArrowRight", true));

    const dx = Object.values(useStore.getState().nodeOffsets)[0].dx;
    expect(dx).toBeGreaterThan(0);
    expect(dx).toBeLessThan(8 * 8);
  });

  it("leaves keys that are not a bare arrow to whoever else wants them", () => {
    const props = cardHandleProps();
    const event = arrow("ArrowRight");
    (event as { ctrlKey: boolean }).ctrlKey = true;
    props.onKeyDown!(event);
    props.onKeyDown!(arrow("Home"));

    expect(useStore.getState().nodeOffsets).toEqual({});
    expect(event.preventDefault).not.toHaveBeenCalled();
  });

  /* The same gate as the pointer: an operation is not focusable while locked,
     so the handler is absent rather than present and silently refusing. */
  it("gives an operation the keys only while the unlock is on", () => {
    expect(operationProps().onKeyDown).toBeUndefined();

    useStore.getState().setMoveOps(true);
    const unlocked = operationProps();
    expect(unlocked.tabIndex).toBe(0);
    unlocked.onKeyDown!(arrow("ArrowUp"));
    expect(Object.keys(useStore.getState().nodeOffsets)).toEqual(["n:elementwise_Y"]);
    expect(useStore.getState().nodeOffsets["n:elementwise_Y"].dy).toBeLessThan(0);
  });
});
