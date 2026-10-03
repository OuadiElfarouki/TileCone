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

/** The first tensor card's slot, which owns the card's move gesture, probed as
 *  `operationProps` probes the operation node. */
function cardProps(): OpProps {
  let props: OpProps | undefined;
  function Probe() {
    props = findByClass(GraphView(), "card-slot");
    return null;
  }
  renderToStaticMarkup(createElement(Probe));
  expect(props).toBeDefined();
  return props!;
}

/** The canvas container, which owns the pan and the background click. */
function canvasProps(): OpProps {
  let props: OpProps | undefined;
  function Probe() {
    props = findByClass(GraphView(), "graph-canvas");
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

const pointer = (x: number, held: { ctrlKey?: boolean; metaKey?: boolean } = {}) => ({
  pointerId: 1, button: 0, clientX: x, clientY: 0, target: null,
  ctrlKey: false, metaKey: false, ...held,
  currentTarget: { setPointerCapture: vi.fn(), contains: vi.fn(() => true), focus: vi.fn() },
  preventDefault: vi.fn(), stopPropagation: vi.fn(),
}) as unknown as React.PointerEvent<HTMLDivElement>;

beforeEach(() => {
  useStore.getState().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  useStore.getState().setCanvasTool("select");
  useStore.getState().setSelectedOp(null);
  // Layout effects do not run in this handler-only server-render harness.
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => { vi.restoreAllMocks(); });

describe("pressing a node under the select tool", () => {
  it("selects an operation without moving it, keeping the follow-up handlers", () => {
    const props = operationProps();
    props.onPointerDownCapture!(pointer(0));
    props.onPointerDown!(pointer(0));
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");
    expect(useStore.getState().dragging).toBe(false);
    expect(useStore.getState().nodeOffsets).toEqual({});
    for (const name of ["onPointerMove", "onPointerUp", "onPointerCancel", "onLostPointerCapture"] as const)
      expect(props[name]).toBeTypeOf("function");
  });

  /* The press has to reach the canvas inside, which draws the tile. */
  it("leaves a press on a card to the card", () => {
    const event = pointer(0);
    cardProps().onPointerDownCapture!(event);
    expect(useStore.getState().dragging).toBe(false);
    expect(event.stopPropagation).not.toHaveBeenCalled();
    expect(event.preventDefault).not.toHaveBeenCalled();
  });
});

describe("moving a node by pointer", () => {
  it.each(["card", "operation"])("leaves portaled menu presses outside the %s DOM to the menu", (node) => {
    useStore.getState().setCanvasTool("move");
    const props = node === "card" ? cardProps() : operationProps();
    const event = pointer(0);
    vi.mocked(event.currentTarget.contains).mockReturnValue(false);
    props.onPointerDownCapture!(event);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(event.stopPropagation).not.toHaveBeenCalled();
    expect(event.currentTarget.setPointerCapture).not.toHaveBeenCalled();
    expect(useStore.getState().dragging).toBe(false);
    expect(useStore.getState().selectedOp).toBeNull();
  });

  it.each(["card", "operation"])("focuses a moved %s for subsequent keyboard nudges", (node) => {
    useStore.getState().setCanvasTool("move");
    const props = node === "card" ? cardProps() : operationProps();
    const event = pointer(0);
    props.onPointerDownCapture!(event);
    expect(event.currentTarget.focus).toHaveBeenCalledWith({ preventScroll: true });
    props.onPointerUp!(event);
    const key = arrow("ArrowUp");
    Object.assign(key, { target: event.currentTarget, currentTarget: event.currentTarget });
    props.onKeyDown!(key);
    expect(Object.values(useStore.getState().nodeOffsets)[0].dy).toBeLessThan(0);
  });

  it.each([
    ["card", "ctrlKey"],
    ["card", "metaKey"],
    ["operation", "ctrlKey"],
    ["operation", "metaKey"],
  ] as const)("borrows the move tool for one %s drag while %s is held", (node, key) => {
    const props = node === "card" ? cardProps() : operationProps();
    const depth = useStore.getState().workspaceHistory.length;
    const press = pointer(0, { [key]: true });
    props.onPointerDownCapture!(press);
    expect(press.stopPropagation).toHaveBeenCalled();
    expect(useStore.getState().dragging).toBe(true);

    props.onPointerMove!(pointer(5, { [key]: true }));
    props.onPointerUp!(pointer(5, { [key]: true }));
    expect(useStore.getState().dragging).toBe(false);
    expect(Object.values(useStore.getState().nodeOffsets)[0].dx).toBeGreaterThan(0);
    expect(useStore.getState().workspaceHistory).toHaveLength(depth + 1);
    expect(useStore.getState().canvasTool).toBe("select");
    expect(press.currentTarget.focus).not.toHaveBeenCalled();
    if (node === "operation") expect(useStore.getState().selectedOp).toBe("elementwise_Y");
  });

  it("moves a card from a plain press under the move tool", () => {
    useStore.getState().setCanvasTool("move");
    const props = cardProps();
    const press = pointer(0);
    props.onPointerDownCapture!(press);
    expect(press.stopPropagation).toHaveBeenCalled();
    props.onPointerMove!(pointer(5));
    props.onPointerUp!(pointer(5));
    expect(Object.keys(useStore.getState().nodeOffsets)).toEqual(["t:X"]);
  });

  it.each(["onPointerUp", "onPointerCancel", "onLostPointerCapture"] as const)(
    "%s finishes an active gesture after switching back to select",
    (finish) => {
      useStore.getState().setCanvasTool("move");
      const props = operationProps();
      const depth = useStore.getState().workspaceHistory.length;
      props.onPointerDownCapture!(pointer(0));
      props.onPointerMove!(pointer(5));
      expect(useStore.getState().dragging).toBe(true);
      expect(Object.keys(useStore.getState().nodeOffsets)).toHaveLength(1);

      useStore.getState().setCanvasTool("select");
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
  beforeEach(() => useStore.getState().setCanvasTool("move"));

  it("nudges a card one step per press, each its own undo entry", () => {
    const props = cardProps();
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
    const props = cardProps();
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
    const props = cardProps();
    props.onKeyDown!(arrow("ArrowRight", true));

    const dx = Object.values(useStore.getState().nodeOffsets)[0].dx;
    expect(dx).toBeGreaterThan(0);
    expect(dx).toBeLessThan(8 * 8);
  });

  it("leaves keys that are not a bare arrow to whoever else wants them", () => {
    const props = cardProps();
    const event = arrow("ArrowRight");
    (event as { ctrlKey: boolean }).ctrlKey = true;
    props.onKeyDown!(event);
    props.onKeyDown!(arrow("Home"));

    expect(useStore.getState().nodeOffsets).toEqual({});
    expect(event.preventDefault).not.toHaveBeenCalled();
  });

  /* Selection stays keyboard-accessible under the select tool; only the move
     tool gives the arrows to a node. A card takes no focus at all there, so Tab
     goes on to its name. */
  it("gives a node the keys only under the move tool", () => {
    useStore.getState().setCanvasTool("select");
    const locked = operationProps();
    expect(locked.tabIndex).toBe(0);
    locked.onKeyDown!(arrow("ArrowUp"));
    const card = cardProps();
    expect(card.tabIndex).toBeUndefined();
    card.onKeyDown!(arrow("ArrowUp"));
    expect(useStore.getState().nodeOffsets).toEqual({});

    useStore.getState().setCanvasTool("move");
    const unlocked = operationProps();
    expect(unlocked.tabIndex).toBe(0);
    unlocked.onKeyDown!(arrow("ArrowUp"));
    expect(cardProps().tabIndex).toBe(0);
    cardProps().onKeyDown!(arrow("ArrowUp"));
    expect(Object.keys(useStore.getState().nodeOffsets).sort()).toEqual(["n:elementwise_Y", "t:X"]);
    expect(useStore.getState().nodeOffsets["n:elementwise_Y"].dy).toBeLessThan(0);
  });
});


describe("selecting operations on the canvas", () => {
  it.each(["select", "move"] as const)("selects without recentering or drawing a tile (tool: %s)", (tool) => {
    useStore.getState().setCanvasTool(tool);
    const before = useStore.getState().selection;
    const props = operationProps();
    props.onClick!({ target: null, stopPropagation: vi.fn() } as unknown as React.MouseEvent<HTMLDivElement>);
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");
    expect(useStore.getState().focusNode).toBeNull();
    expect(useStore.getState().selection).toBe(before);
    expect(operationProps().className).toContain("selected");
  });

  it.each(["Enter", " "])("selects with %s under the select tool", (key) => {
    operationProps().onKeyDown!(arrow(key));
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");
    expect(useStore.getState().nodeOffsets).toEqual({});
  });

  /* Moving the view is not choosing something else: a pan keeps the selected
     operation, and so does a cancelled press. A click on empty canvas clears it. */
  it("keeps the operation through a pan and clears it on a background click", () => {
    useStore.getState().setSelectedOp("elementwise_Y");
    const canvas = canvasProps();
    canvas.onPointerDown!(pointer(100));
    canvas.onPointerUp!(pointer(160));
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");

    canvas.onPointerDown!(pointer(100));
    canvas.onPointerCancel!(pointer(100));
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");

    canvas.onPointerDown!(pointer(100));
    canvas.onPointerUp!(pointer(101));
    expect(useStore.getState().selectedOp).toBeNull();
  });

  it("leaves the substitute button's press to its own handler", () => {
    const event = pointer(0);
    Object.assign(event, { target: { closest: () => ({}) } });
    operationProps().onPointerDown!(event);
    expect(useStore.getState().selectedOp).toBeNull();
    expect(event.preventDefault).not.toHaveBeenCalled();
  });
});
