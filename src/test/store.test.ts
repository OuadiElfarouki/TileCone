import { beforeEach, describe, expect, it } from "vitest";
import { canonicalize, count, fromBox, box } from "../core/region";
import { computeMetrics } from "../core/metrics";
import { EXAMPLES } from "../examples";
import {
  enabledPropResult,
  MAX_PER_BOX_PROPS, PANEL_COLLAPSE_AT, PANEL_MAX, PANEL_MIN,
  planesOf, startingTiles, useStore, viewAxes,
} from "../ui/store";
import { cardPx, graphScale, MAX_ELEM_PX, planeExtents, TILE_SCALE_NONE } from "../ui/tiling";
import { nudgeUnit, snapSpan, tileOf } from "../ui/grid";

const S = () => useStore.getState();
const sel = () => S().selection;
/** The drawn boxes, in order. Parts carry their own tensor now; these helpers
 * keep the single-tensor assertions reading the way they always did. */
const selBoxes = () => sel()!.parts.map((p) => p.box);
const selTensors = () => sel()!.parts.map((p) => p.tensorId);
/** Distinct elements across the parts -- overlapping parts counted once. */
const selCount = () =>
  count(canonicalize({ boxes: selBoxes(), exact: true, reasons: [] }));
const regionOf = (tensorId: string) => S().backwardRes?.tensors.get(tensorId)?.region;
const loadExampleNamed = (name: string) => {
  const index = EXAMPLES.findIndex((example) => example.name === name);
  if (index < 0) throw new Error(`missing built-in example "${name}"`);
  S().loadExample(index);
};

describe("independent cone toggles", () => {
  beforeEach(() => {
    S().setDirection("both");
    S().loadExample(0);
  });

  it("toggles both directions independently, including figures-only", () => {
    expect(S().direction).toBe("both");
    S().toggleDirection("backward");
    expect(S().direction).toBe("forward");
    S().toggleDirection("forward");
    expect(S().direction).toBe("none");
    S().toggleDirection("backward");
    expect(S().direction).toBe("backward");
    S().toggleDirection("forward");
    expect(S().direction).toBe("both");
  });

  it("keeps figures-only when set directly", () => {
    S().setDirection("none");
    expect(S().direction).toBe("none");
  });

  it("filters what is shown without gating what is analysed", () => {
    // A mode is a question, not a budget: switching to downstream must not
    // discard the upstream numbers the inspector still reports.
    for (const direction of ["both", "backward", "forward", "none"] as const) {
      S().setDirection(direction);
      expect(S().backwardRes, direction).not.toBeNull();
      expect(S().forwardRes, direction).not.toBeNull();
      expect(S().perBox![0].backward, direction).not.toBeNull();
      expect(S().perBox![0].forward, direction).not.toBeNull();
    }
  });
});

describe("DSL compiler integration", () => {
  beforeEach(() => S().loadExample(0));

  it("loads a compiled program through the UI boundary", () => {
    S().applyDSL(`input X [2, 3] f32
Y = softmax(X, axis=-1)
`);
    expect(S().loadError).toBeNull();
    expect(S().graph?.tensors.Y.resolved).toBeUndefined();
    expect(S().resolved?.tensors.Y.resolved).toEqual([2, 3]);
  });

  it("expands an intermediate composite and keeps the displayed DSL executable", () => {
    S().applyDSL(`input A [2, 3] f32
input B [3, 4] f32
S = matmul(A, B)
P = softmax(S, axis=-1)
`);

    S().expandNodeInPlace("softmax_P");
    expect(S().loadError).toBeNull();
    expect(S().resolved?.tensors.P.resolved).toEqual([2, 4]);
    expect(S().resolved?.nodes.some((node) => node.op === "softmax")).toBe(false);
    expect(S().dslText).not.toContain("softmax(");
    expect(S().dslText).toContain("softmax_P$max");

    const expandedSource = S().dslText;
    const expandedOps = S().resolved?.nodes.map((node) => node.op);
    S().applyDSL(expandedSource);
    expect(S().loadError).toBeNull();
    expect(S().resolved?.nodes.map((node) => node.op)).toEqual(expandedOps);
  });

  it("surfaces semantic failures with the originating DSL line", () => {
    S().applyDSL(`input A [2, 3] f32
input B [4, 5] f32
C = matmul(A, B)
`);
    expect(S().loadError).toMatch(/^line 3: node "matmul_C".*shape inference failed/);
  });

  it("starts a fresh undo history when the graph is replaced", () => {
    S().moveSelection(0, 8);
    expect(S().workspaceHistory.length).toBeGreaterThan(0);

    loadExampleNamed("Multi-head attention"); // different tensor IDs from Plain GEMM
    expect(S().workspaceHistory).toEqual([]);
    expect(() => S().undoWorkspace()).not.toThrow();
    expect((sel() ? selTensors()[0] : undefined)).toBe("Out");

    S().moveSelection(0, 1);
    expect(S().workspaceHistory.length).toBeGreaterThan(0);
    S().applyDSL("input X [2, 3] f32\nY = identity(X)\n");
    expect(S().workspaceHistory).toEqual([]);
    expect(S().selection).toBeNull();
    expect(() => S().undoWorkspace()).not.toThrow();
  });

  it("drops graph-specific tensor positions when the graph is replaced", () => {
    S().setTensorOffset("A", { dx: 80, dy: 25 });
    S().commitTensorMove("A", { dx: 0, dy: 0 });
    expect(S().tensorOffsets.A).toEqual({ dx: 80, dy: 25 });

    S().applyDSL("input X [2, 3] f32\nY = identity(X)\n");
    expect(S().tensorOffsets).toEqual({});
    expect(S().workspaceHistory).toEqual([]);
  });
});

describe("transactional workspace restore", () => {
  beforeEach(() => S().loadExample(0));

  it("installs source, settings, and an ordered selection together", () => {
    const restored = S().restoreWorkspace({
      dsl: "input X [4, 4] f32\nY = relu(X)\n",
      direction: "both",
      tileScale: 2,
      snapToGrid: false,
      axisMode: "symbolic",
      parts: [
        { tensorId: "Y", box: box([0, 2], [1, 3]) },
        { tensorId: "X", box: box([2, 4], [0, 1]) },
      ],
    });

    expect(restored).toBe(true);
    expect(S().dslText).toContain("Y = relu(X)");
    expect(S().direction).toBe("both");
    expect(S().tileScale).toBe(2);
    expect(S().snapToGrid).toBe(false);
    expect(selTensors()).toEqual(["Y", "X"]);
    expect(S().workspaceHistory).toEqual([]);
    expect(S().exampleIndex).toBe(-1);
  });

  it("leaves the current workspace untouched when compilation fails", () => {
    const before = S();
    const restored = S().restoreWorkspace({
      dsl: "this is not TileCone DSL",
      direction: "none",
      tileScale: -3,
      snapToGrid: false,
      axisMode: "symbolic",
      parts: null,
    });

    expect(restored).toBe(false);
    expect(S()).toBe(before);
  });

  it("restores a shared figures-only view", () => {
    const restored = S().restoreWorkspace({
      dsl: "input X [4, 4] f32\nY = relu(X)\n",
      direction: "none",
      tileScale: 0,
      snapToGrid: true,
      axisMode: "symbolic",
      parts: null,
    });

    expect(restored).toBe(true);
    expect(S().direction).toBe("none");
  });

  it("validates every part in a legacy none workspace", () => {
    const before = S();
    for (const parts of [
      [{ tensorId: "missing", box: box([0, 1]) }],
      [{ tensorId: "A", box: box([0, 999], [0, 1]) }],
    ]) {
      const restored = S().restoreWorkspace({
        dsl: "input A [4, 4] f32\nB = relu(A)\n",
        direction: "none",
        tileScale: 0,
        snapToGrid: true,
        axisMode: "symbolic",
        parts,
      });
      expect(restored).toBe(false);
      expect(S()).toBe(before);
    }
  });
});

describe("tensor layout transactions", () => {
  beforeEach(() => S().loadExample(0));

  it("records a complete drag as one undo step", () => {
    const depth = S().workspaceHistory.length;
    S().setTensorOffset("A", { dx: 25, dy: 10 });
    S().setTensorOffset("A", { dx: 60, dy: 30 });
    expect(S().workspaceHistory).toHaveLength(depth); // live drag previews are unrecorded

    S().commitTensorMove("A", { dx: 0, dy: 0 });
    expect(S().workspaceHistory).toHaveLength(depth + 1);
    S().undoWorkspace();
    expect(S().tensorOffsets.A).toBeUndefined();
  });

  it("does not record a pointer gesture that produced no displacement", () => {
    const depth = S().workspaceHistory.length;
    S().commitTensorMove("A", { dx: 0, dy: 0 });
    expect(S().workspaceHistory).toHaveLength(depth);
  });

  it("undoes layout and selection changes in chronological order", () => {
    const initial = selBoxes();
    S().moveSelection(0, 64);
    const afterSelectionMove = selBoxes();

    S().setTensorOffset("A", { dx: 90, dy: 35 });
    S().commitTensorMove("A", { dx: 0, dy: 0 });
    S().moveSelection(1, 64);

    S().undoWorkspace(); // selection edit after the tensor drag
    expect(selBoxes()).toEqual(afterSelectionMove);
    expect(S().tensorOffsets.A).toEqual({ dx: 90, dy: 35 });

    S().undoWorkspace(); // tensor drag itself
    expect(selBoxes()).toEqual(afterSelectionMove);
    expect(S().tensorOffsets.A).toBeUndefined();

    S().undoWorkspace(); // selection edit before the tensor drag
    expect(selBoxes()).toEqual(initial);
  });

  it("can undo the first selection without disturbing tensor placement", () => {
    S().applyDSL("input X [8, 8] f32\nY = identity(X)\n");
    S().setTensorOffset("X", { dx: 45, dy: 20 });
    S().commitTensorMove("X", { dx: 0, dy: 0 });
    S().setSelection("Y", fromBox(box([0, 2], [0, 2])), "replace");

    S().undoWorkspace();
    expect(S().selection).toBeNull();
    expect(S().tensorOffsets.X).toEqual({ dx: 45, dy: 20 });
  });
});

/** Drives the store exactly as the UI does, to cover the selection-editing actions. */
describe("snapping is a gesture setting, not an analysis one", () => {
  beforeEach(() => {
    S().loadExample(0);
    useStore.setState({ snapToGrid: true });
  });

  it("defaults to on", () => {
    expect(S().snapToGrid).toBe(true);
  });

  it("toggles without disturbing the selection or its cone", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    const before = regionOf("A")!.boxes;
    S().setSnapToGrid(false);
    expect(selBoxes()).toEqual([box([0, 64], [0, 64])]);
    expect(regionOf("A")!.boxes).toEqual(before);
  });

  it("a nudge steps by one element when snapping is off, one tile when on", () => {
    // The keyboard must not reach offsets the mouse cannot: with snapping off a
    // drag cuts an exact range, so a nudge has to step by one element too.
    const shape = S().resolved!.tensors.C.resolved!;
    const tile = tileOf(shape, S().tileScale, S().graphPx);
    expect(tile).toBeGreaterThan(1); // otherwise the two cases are the same
    expect(nudgeUnit(shape, S().tileScale, S().graphPx, false)).toBe(1);
    expect(nudgeUnit(shape, S().tileScale, S().graphPx, true)).toBe(tile);
  });

  it("applies that step verbatim, so an odd offset survives", () => {
    S().setSnapToGrid(false);
    S().setSelection("C", fromBox(box([64, 128], [0, 64])), "replace");
    S().moveSelection(0, nudgeUnit(S().resolved!.tensors.C.resolved!, S().tileScale, S().graphPx, false));
    expect(selBoxes()[0][0]).toEqual({ lo: 65, hi: 129 });
  });

  it("an unsnapped range propagates exactly like any other", () => {
    // Regions were always element-precise; only the drag gesture rounded, so an
    // odd range must be as ordinary to the engine as an aligned one.
    S().setSnapToGrid(false);
    S().setSelection("C", fromBox(box([3, 7], [1, 2])), "replace");
    expect(selBoxes()).toEqual([box([3, 7], [1, 2])]);
    expect(regionOf("A")!.boxes).toEqual([box([3, 7], [0, 512])]);
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [1, 2])]);
  });

  it("a text edit remains exact with snap on, before and after grid changes", () => {
    S().setSelection("C", fromBox(box([64, 128], [0, 64])), "replace");
    const typed = box([65, 129], [3, 67]);
    S().setSnapToGrid(true);
    S().replaceBox(0, typed);
    expect(selBoxes()).toEqual([typed]);

    S().setTileScale(1);
    expect(selBoxes()).toEqual([typed]);
    S().setSnapToGrid(false);
    S().setTileScale(TILE_SCALE_NONE);
    S().setSnapToGrid(true);
    expect(selBoxes()).toEqual([typed]);
  });

  it("changing snap or detail never creates an undo entry", () => {
    S().setSelection("C", fromBox(box([65, 129], [3, 67])), "replace");
    const depth = S().workspaceHistory.length;
    S().setSnapToGrid(false);
    S().setTileScale(2);
    S().setSnapToGrid(true);
    S().setTileScale(TILE_SCALE_NONE);
    expect(S().workspaceHistory).toHaveLength(depth);
    expect(selBoxes()).toEqual([box([65, 129], [3, 67])]);
  });

  it("a snapped nudge uses the current grid but preserves a typed extent and offset", () => {
    const typed = box([65, 129], [3, 67]);
    S().setSelection("C", fromBox(typed), "replace");
    S().setTileScale(2);
    S().setSnapToGrid(true);
    const shape = S().resolved!.tensors.C.resolved!;
    const step = nudgeUnit(shape, S().tileScale, S().graphPx, S().snapToGrid);
    expect(step).toBeGreaterThan(1);

    S().moveSelection(0, step);
    const moved = selBoxes()[0];
    expect(moved[0].hi - moved[0].lo).toBe(64);
    expect(moved[0].lo).toBe(65 + step);
    expect(moved[0].lo % step).toBe(65 % step);
    expect(moved[1]).toEqual(typed[1]);
  });

  it("changing detail changes only the next snapped stride", () => {
    const typed = box([65, 129], [3, 67]);
    S().setSelection("C", fromBox(typed), "replace");
    S().setSnapToGrid(true);
    const shape = S().resolved!.tensors.C.resolved!;

    S().setTileScale(TILE_SCALE_NONE);
    expect(nudgeUnit(shape, S().tileScale, S().graphPx, true)).toBe(1);
    expect(selBoxes()).toEqual([typed]);

    S().setTileScale(1);
    const coarse = nudgeUnit(shape, S().tileScale, S().graphPx, true);
    expect(coarse).toBeGreaterThan(1);
    expect(selBoxes()).toEqual([typed]);
    S().moveSelection(0, coarse);
    expect(selBoxes()[0][0]).toEqual({ lo: 65 + coarse, hi: 129 + coarse });
  });

  it("a coarse nudge clamps a typed range flush to an edge without resizing it", () => {
    const typed = box([189, 253], [3, 67]);
    S().setSelection("C", fromBox(typed), "replace");
    S().setTileScale(3);
    S().setSnapToGrid(true);
    const shape = S().resolved!.tensors.C.resolved!;
    const coarse = nudgeUnit(shape, S().tileScale, S().graphPx, true);
    S().moveSelection(0, coarse);
    expect(selBoxes()[0][0]).toEqual({ lo: 192, hi: 256 });
  });

  it("snap off always nudges by one even after changing grid detail", () => {
    const shape = S().resolved!.tensors.C.resolved!;
    S().setSnapToGrid(false);
    for (const scale of [3, TILE_SCALE_NONE, 0]) {
      S().setTileScale(scale);
      expect(nudgeUnit(shape, S().tileScale, S().graphPx, S().snapToGrid)).toBe(1);
    }
  });

  it("none makes snapped gestures and nudges element-sized", () => {
    const shape = S().resolved!.tensors.C.resolved!;
    S().setTileScale(TILE_SCALE_NONE);
    S().setSnapToGrid(true);
    const tile = tileOf(shape, S().tileScale, S().graphPx);
    expect(tile).toBe(1);
    expect(nudgeUnit(shape, S().tileScale, S().graphPx, true)).toBe(1);
    expect(snapSpan(7, 7, tile, shape[0])).toEqual([7, 8]);
  });

  it("undo reverts movement without reverting snap or grid settings", () => {
    const typed = box([65, 129], [3, 67]);
    S().setSelection("C", fromBox(typed), "replace");
    S().setTileScale(TILE_SCALE_NONE);
    S().setSnapToGrid(true);
    S().moveSelection(0, 1);
    expect(selBoxes()[0][0]).toEqual({ lo: 66, hi: 130 });

    S().undoWorkspace();
    expect(selBoxes()).toEqual([typed]);
    expect(S().tileScale).toBe(TILE_SCALE_NONE);
    expect(S().snapToGrid).toBe(true);
  });
});

describe("side panel geometry", () => {
  beforeEach(() => {
    useStore.setState({
      panelW: { left: 330, right: 300 },
      panelCollapsed: { left: false, right: false },
    });
  });

  it("clamps a drag to the allowed range", () => {
    S().setPanelWidth("left", 999);
    expect(S().panelW.left).toBe(PANEL_MAX);
    S().setPanelWidth("left", PANEL_COLLAPSE_AT + 1);
    expect(S().panelW.left).toBe(PANEL_MIN);
    expect(S().panelCollapsed.left).toBe(false);
  });

  it("keeps a sub-threshold resize preview open until release", () => {
    S().setPanelWidth("right", PANEL_COLLAPSE_AT - 1);
    expect(S().panelW.right).toBe(PANEL_MIN);
    expect(S().panelCollapsed.right).toBe(false);
  });

  it("collapses when a sub-threshold resize is committed", () => {
    S().setPanelWidth("right", PANEL_COLLAPSE_AT - 1);
    S().finishPanelResize("right", PANEL_COLLAPSE_AT - 1);
    expect(S().panelCollapsed.right).toBe(true);
  });

  it("remembers the open width across a collapse", () => {
    S().setPanelWidth("left", 420);
    S().togglePanel("left");
    expect(S().panelCollapsed.left).toBe(true);
    expect(S().panelW.left).toBe(420); // width survives being hidden
    S().togglePanel("left");
    expect(S().panelCollapsed.left).toBe(false);
    expect(S().panelW.left).toBe(420);
  });

  it("reopens at a usable width even if it was collapsed from a drag", () => {
    S().finishPanelResize("right", PANEL_COLLAPSE_AT - 1); // collapse by drag release
    S().togglePanel("right");
    expect(S().panelW.right).toBeGreaterThanOrEqual(PANEL_MIN);
  });

  it("the two panels are independent", () => {
    S().togglePanel("left");
    expect(S().panelCollapsed.left).toBe(true);
    expect(S().panelCollapsed.right).toBe(false);
    S().setPanelWidth("right", 400);
    expect(S().panelW.left).toBe(330);
    expect(S().panelCollapsed.left).toBe(true);
  });
});

describe("the graph's render scale", () => {
  /** Card width/height for a tensor at the store's current scale. */
  const cardOf = (name: string) => {
    const shape = S().resolved!.tensors[name].resolved!;
    const { rowAxis, colAxis } = viewAxes(shape);
    const { rows, cols } = planeExtents(shape, rowAxis, colAxis);
    return cardPx(rows, cols, S().graphPx);
  };

  it("is derived from the resolved graph at load", () => {
    S().loadExample(0);
    expect(S().graphPx).toBe(graphScale(planesOf(S().resolved!)));
  });

  it("draws a shared dimension at the same length in both tensors", () => {
    // Plain GEMM: A[256,512] @ B[512,256]. K=512 runs along A's columns and down
    // B's rows; if those differ, the contraction stops reading as one axis.
    S().loadExample(0);
    expect(cardOf("A").w).toBe(cardOf("B").h);
    expect(cardOf("A").h).toBe(cardOf("C").h);
    expect(cardOf("B").w).toBe(cardOf("C").w);
  });

  it("survives view settings: tile detail must not resize the graph", () => {
    S().loadExample(0);
    const before = S().graphPx;
    S().setTileScale(3);
    expect(S().graphPx).toBe(before);
    S().setTileScale(0);
  });

  it("is recomputed for a different graph", () => {
    S().applyDSL("input X [4, 4] f32\nY = reshape(X, shape=[16])\n");
    const small = S().graphPx;
    S().applyDSL("input X [1024, 1024] f16\ninput W [1024, 1024] f16\nZ = matmul(X, W)\n");
    expect(S().graphPx).toBeLessThan(small);
  });

  it("a rank-1 tensor does not peg the whole graph at the cap", () => {
    // W and Bb are [E]: their row extent is 1. Letting a degenerate axis count as
    // "the smallest side" would demand 14px for it and blow every card up.
    loadExampleNamed("Layernorm + residual");
    expect(S().resolved!.tensors.W.resolved).toEqual([64]);
    expect(S().graphPx).toBeLessThan(MAX_ELEM_PX);
    expect(cardOf("X").w).toBe(cardOf("W").w); // both span E = 64
  });
});

describe("selection editing through the store", () => {
  beforeEach(() => {
    // the store is a module singleton, so reset every axis the tests vary
    S().loadExample(0); // Plain GEMM: A[256,512] @ B[512,256] -> C[256,256]
    S().setDirection("backward");
    S().setSelection("C", fromBox(box([64, 128], [0, 64])), "replace");
  });

  it("loads with a propagated selection", () => {
    expect(selTensors()[0]).toBe("C");
    expect(regionOf("A")!.boxes).toEqual([box([64, 128], [0, 512])]);
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [0, 64])]);
  });

  it("moveSelection shifts the selection and repropagates", () => {
    S().moveSelection(0, 64); // down one tile on the row axis
    expect(selBoxes()).toEqual([box([128, 192], [0, 64])]);
    expect(regionOf("A")!.boxes).toEqual([box([128, 192], [0, 512])]);
    // B is unaffected by a pure row move
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [0, 64])]);
  });

  it("replaceBox edits one stable part and repropagates", () => {
    const depth = S().workspaceHistory.length;
    S().replaceBox(0, box([16, 48], [32, 96]));
    expect(selBoxes()).toEqual([box([16, 48], [32, 96])]);
    expect(regionOf("A")!.boxes).toEqual([box([16, 48], [0, 512])]);
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [32, 96])]);
    expect(S().workspaceHistory.length).toBe(depth + 1);
  });

  it("replaceBox preserves the edited part's pin", () => {
    S().setSelection("C", fromBox(box([160, 192], [96, 128])));
    S().togglePinBox(1);
    S().replaceBox(1, box([161, 193], [97, 129]));
    expect(S().pinnedBox).toBe(1);
    expect(S().focusedBox).toBe(1);
    expect(selBoxes()[1]).toEqual(box([161, 193], [97, 129]));
  });

  it("moveSelection clamps at the tensor edge without shrinking", () => {
    const before = selCount();
    for (let i = 0; i < 10; i++) S().moveSelection(0, 64);
    expect(selCount()).toBe(before);
    expect(selBoxes()).toEqual([box([192, 256], [0, 64])]);
  });

  it("direct drawing adds a multi-box selection by default", () => {
    S().setSelection("C", fromBox(box([0, 32], [200, 232])));
    expect(selBoxes()).toHaveLength(2);
    expect(selCount()).toBe(64 * 64 + 32 * 32);
    // both output tiles' row bands appear upstream in A
    expect(regionOf("A")!.boxes).toEqual([box([0, 32], [0, 512]), box([64, 128], [0, 512])]);
  });

  it("an explicit subtract gesture cuts a hole", () => {
    S().setSelection("C", fromBox(box([80, 96], [0, 64])), "subtract");
    expect(selCount()).toBe(64 * 64 - 16 * 64);
    expect(selBoxes()).toHaveLength(2);
  });

  it("an explicit union gesture adds a part", () => {
    S().setSelection("C", fromBox(box([0, 32], [0, 32])), "union");
    expect(selBoxes()).toHaveLength(2);
  });

  it("deleteBox removes exactly one box", () => {
    S().setSelection("C", fromBox(box([0, 32], [200, 232])));
    expect(selBoxes()).toHaveLength(2);
    S().deleteBox(0);
    expect(selBoxes()).toHaveLength(1);
  });

  it("workspace undo walks back through selection edits", () => {
    S().moveSelection(0, 64);
    S().moveSelection(0, 64);
    expect(selBoxes()).toEqual([box([192, 256], [0, 64])]);
    S().undoWorkspace();
    expect(selBoxes()).toEqual([box([128, 192], [0, 64])]);
    S().undoWorkspace();
    expect(selBoxes()).toEqual([box([64, 128], [0, 64])]);
    expect(regionOf("A")!.boxes).toEqual([box([64, 128], [0, 512])]);
  });

  it("clearing then undoing restores the selection", () => {
    S().clearSelection();
    expect(sel()).toBeNull();
    S().undoWorkspace();
    expect(selBoxes()).toEqual([box([64, 128], [0, 64])]);
  });

  it("editing actions are no-ops without a selection", () => {
    S().clearSelection();
    expect(() => {
      S().moveSelection(0, 1);
      S().deleteBox(0);
      S().hoverBox(1);
    }).not.toThrow();
    expect(sel()).toBeNull();
  });

  it("emptying the selection by subtraction clears it", () => {
    S().setSelection("C", fromBox(box([64, 128], [0, 64])), "subtract");
    expect(sel()).toBeNull();
    expect(S().backwardRes).toBeNull();
  });
});

/**
 * Per-box attribution: each selection box carries its own propagation, so a
 * highlighted upstream region can be traced to the box that caused it.
 */
describe("per-box dependency attribution", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().setDirection("backward");
    S().clearSelection(); // loadExample seeds a default selection
  });

  it("a single box reuses the aggregate result rather than propagating twice", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    const pb = S().perBox!;
    expect(pb).toHaveLength(1);
    expect(pb[0].backward).toBe(S().backwardRes); // same object, not a recomputation
  });

  it("each box gets its own cone, and they union to the aggregate", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([192, 256], [128, 192])), "union");
    const pb = S().perBox!;
    expect(pb).toHaveLength(2);

    const boxes = selBoxes();
    // part order matches selection.parts, so row bands line up per part
    boxes.forEach((b, i) => {
      const aRegion = pb[i].backward!.tensors.get("A")!.region;
      expect(aRegion.boxes).toEqual([box([b[0].lo, b[0].hi], [0, 512])]);
      const bRegion = pb[i].backward!.tensors.get("B")!.region;
      expect(bRegion.boxes).toEqual([box([0, 512], [b[1].lo, b[1].hi])]);
    });

    // the per-box cones together cover exactly the aggregate cone
    const aggA = count(regionOf("A")!);
    const perA = pb.reduce((acc, p) => acc + count(p.backward!.tensors.get("A")!.region), 0);
    expect(perA).toBe(aggA); // the two row bands are disjoint here
  });

  it("forward attribution is computed per box in downstream mode", () => {
    S().setDirection("forward");
    S().setSelection("A", fromBox(box([0, 8], [0, 512])), "replace");
    S().setSelection("A", fromBox(box([100, 108], [0, 512])), "union");
    const pb = S().perBox!;
    expect(pb).toHaveLength(2);
    expect(pb[0].forward!.tensors.get("C")!.region.boxes).toEqual([box([0, 8], [0, 256])]);
    expect(pb[1].forward!.tensors.get("C")!.region.boxes).toEqual([box([100, 108], [0, 256])]);
    // the upstream half is attributed too: downstream mode hides it, and the
    // per-tile footprint numbers are read from it either way
    expect(pb[0].backward!.tensors.get("A")!.region.boxes).toEqual([box([0, 8], [0, 512])]);
  });

  it("focusing a box scopes the readout to that box alone", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([192, 256], [128, 192])), "union");
    S().togglePinBox(1);
    const focused = S().perBox![S().focusedBox!].backward!;
    expect(focused.tensors.get("A")!.region.boxes).toEqual([box([192, 256], [0, 512])]);
    // the aggregate still holds both bands
    expect(regionOf("A")!.boxes).toHaveLength(2);
  });

  it("focus survives a move, so a part can be nudged repeatedly", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([192, 256], [128, 192])), "union");
    S().togglePinBox(1);
    S().moveSelection(0, -8);
    expect(S().focusedBox).toBe(1);
    S().moveSelection(0, -8);
    expect(S().focusedBox).toBe(1);
    expect(selBoxes()[1]).toEqual(box([176, 240], [128, 192]));
  });

  it("focus resets when parts are added or removed", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([192, 256], [128, 192])), "union");
    S().togglePinBox(1);
    S().deleteBox(0); // indices shift, so a stale focus must not survive
    expect(S().focusedBox).toBeNull();
  });

  it("beyond the propagation cap, per-box attribution is skipped", () => {
    for (let i = 0; i < MAX_PER_BOX_PROPS + 2; i++)
      S().setSelection("C", fromBox(box([i * 8, i * 8 + 4], [0, 4])), "union");
    expect(selBoxes().length).toBeGreaterThan(MAX_PER_BOX_PROPS);
    expect(S().perBox).toBeNull();
    // the aggregate cone is still available
    expect(regionOf("A")).toBeDefined();
  });
});


/**
 * Moving one part of a multi-part selection. The other parts must be untouched,
 * and the propagation must follow the moved part immediately — including when
 * the move puts two parts on top of each other.
 */
describe("moving a single selected part", () => {
  const twoParts = () => {
    S().loadExample(0); // A[256,512] @ B[512,256] -> C[256,256]
    S().setDirection("backward");
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([128, 192], [128, 192])), "union");
  };
  beforeEach(twoParts);

  it("moves only the focused part", () => {
    const before = selBoxes()[0];
    S().togglePinBox(1);
    S().moveSelection(0, 32);
    const after = selBoxes();
    expect(after).toHaveLength(2);
    expect(after[0]).toEqual(before); // untouched
    expect(after[1]).toEqual(box([160, 224], [128, 192]));
  });

  it("moves the whole selection when nothing is focused", () => {
    S().clearFocus();
    S().moveSelection(0, 32);
    expect(selBoxes()).toEqual([
      box([32, 96], [0, 64]),
      box([160, 224], [128, 192]),
    ]);
  });

  it("repropagates the moved part's cone and leaves the other's alone", () => {
    const before0 = S().perBox![0].backward!.tensors.get("A")!.region.boxes;
    S().togglePinBox(1);
    S().moveSelection(0, 32);
    const pb = S().perBox!;
    expect(pb[0].backward!.tensors.get("A")!.region.boxes).toEqual(before0);
    // the moved part's upstream rows follow it
    expect(pb[1].backward!.tensors.get("A")!.region.boxes).toEqual([box([160, 224], [0, 512])]);
  });

  it("parts may be moved to overlap, and are not merged away", () => {
    S().togglePinBox(1);
    S().moveSelection(0, -128); // slide part 1 onto part 0's rows
    S().moveSelection(1, -128);
    const boxes = selBoxes();
    expect(boxes).toHaveLength(2);
    expect(boxes[0]).toEqual(box([0, 64], [0, 64]));
    expect(boxes[1]).toEqual(box([0, 64], [0, 64])); // exactly coincident
  });

  it("overlapping parts are counted once, never doubled", () => {
    S().togglePinBox(1);
    S().moveSelection(0, -128);
    S().moveSelection(1, -128);
    // two coincident 64x64 parts denote 4096 elements, not 8192
    expect(selCount()).toBe(64 * 64);
    // and the upstream footprint is the single row band, not two
    expect(regionOf("A")!.boxes).toEqual([box([0, 64], [0, 512])]);
    expect(count(regionOf("A")!)).toBe(64 * 512);
  });

  it("partial overlap yields the union upstream, in real time", () => {
    S().togglePinBox(1);
    S().moveSelection(0, -96); // rows 32:96 -> overlaps part 0's rows 0:64
    S().moveSelection(1, -128); // cols 0:64, same as part 0
    expect(selBoxes()[1]).toEqual(box([32, 96], [0, 64]));
    // union of rows 0:64 and 32:96 is 0:96, merged into one band upstream
    expect(regionOf("A")!.boxes).toEqual([box([0, 96], [0, 512])]);
    expect(count(regionOf("A")!)).toBe(96 * 512);
  });

  it("a moved part is clamped to the tensor without shrinking", () => {
    S().togglePinBox(1);
    const vol = count(fromBox(selBoxes()[1]));
    for (let i = 0; i < 20; i++) S().moveSelection(0, 64);
    const b = selBoxes()[1];
    expect(b).toEqual(box([192, 256], [128, 192]));
    expect(count(fromBox(b))).toBe(vol);
  });

  it("a stale focus index is ignored rather than throwing", () => {
    S().togglePinBox(5);
    expect(() => S().moveSelection(0, 8)).not.toThrow();
    // falls back to moving everything
    expect(selBoxes()).toHaveLength(2);
  });
});

/**
 * Escape backs out of the innermost active mode. It must never destroy the
 * selection — clearing tiles is the right panel's explicit "clear all" button.
 */
describe("cone visibility is independent of focus", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().setDirection("backward");
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([128, 192], [128, 192])), "union");
    S().setSelection("C", fromBox(box([192, 256], [0, 64])), "union");
  });

  it("excludes a hidden box from merged analysis without rerunning its cached propagation", () => {
    const before = S().perBox![1].backward!.tensors.get("A")!.region;
    S().toggleBoxHidden(1);
    expect(S().hiddenBoxes.has(1)).toBe(true);
    expect(S().perBox![1].backward!.tensors.get("A")!.region).toEqual(before);
    const enabled = enabledPropResult(
      S().backwardRes,
      S().perBox,
      S().hiddenBoxes,
      null,
      "backward"
    )!;
    const enabledCount = [0, 2].reduce(
      (total, index) => total + count(S().perBox![index].backward!.tensors.get("A")!.region),
      0
    );
    expect(count(enabled.tensors.get("A")!.region)).toBe(enabledCount);
    expect(count(S().backwardRes!.tensors.get("A")!.region)).toBeGreaterThan(enabledCount);
    expect(selBoxes().length).toBe(3);
  });

  it("clears focus when the focused tile is hidden", () => {
    S().togglePinBox(1);
    expect(S().focusedBox).toBe(1);
    S().toggleBoxHidden(1);
    expect(S().focusedBox).toBeNull();
    expect(S().pinnedBox).toBeNull();
    S().hoverBox(1);
    expect(S().focusedBox).toBeNull();
  });

  it("keeps a text-edited hidden tile excluded until it is re-enabled", () => {
    S().toggleBoxHidden(1);
    const before = enabledPropResult(
      S().backwardRes,
      S().perBox,
      S().hiddenBoxes,
      null,
      "backward"
    )!;
    const beforeA = before.tensors.get("A")!.region;

    const edited = box([96, 160], [80, 144]);
    S().replaceBox(1, edited);
    expect(S().hiddenBoxes).toEqual(new Set([1]));
    expect(selBoxes()[1]).toEqual(edited);
    const whileHidden = enabledPropResult(
      S().backwardRes,
      S().perBox,
      S().hiddenBoxes,
      null,
      "backward"
    )!;
    expect(whileHidden.tensors.get("A")!.region).toEqual(beforeA);

    S().toggleBoxHidden(1);
    const enabled = enabledPropResult(
      S().backwardRes,
      S().perBox,
      S().hiddenBoxes,
      null,
      "backward"
    )!;
    expect(enabled.tensors.get("A")!.region).not.toEqual(beforeA);
  });

  it("toggles independently of focus, in both directions", () => {
    S().toggleBoxHidden(0);
    S().hoverBox(2);
    expect(S().focusedBox).toBe(2);
    expect(S().hiddenBoxes.has(0)).toBe(true); // focusing did not unhide
    S().toggleBoxHidden(0);
    expect(S().hiddenBoxes.has(0)).toBe(false);
    expect(S().focusedBox).toBe(2); // unhiding did not steal focus
  });

  it("more than one box can be hidden at once", () => {
    S().toggleBoxHidden(0);
    S().toggleBoxHidden(2);
    expect([...S().hiddenBoxes].sort()).toEqual([0, 2]);
  });

  it("survives a move, which preserves part order", () => {
    S().toggleBoxHidden(1);
    S().hoverBox(1);
    S().moveSelection(0, 8);
    expect(S().hiddenBoxes.has(1)).toBe(true);
  });

  it("is dropped when parts are renumbered", () => {
    // deleteBox shifts every higher index down by one, so a retained set would
    // silently point at a different box than the user hid.
    S().toggleBoxHidden(2);
    S().deleteBox(0);
    expect(selBoxes().length).toBe(2);
    expect(S().hiddenBoxes.size).toBe(0);
  });

  it("is dropped by a new selection, by clear and by undo", () => {
    S().toggleBoxHidden(1);
    S().setSelection("C", fromBox(box([0, 8], [0, 8])), "union");
    expect(S().hiddenBoxes.size).toBe(0);

    S().toggleBoxHidden(0);
    S().undoWorkspace();
    expect(S().hiddenBoxes.size).toBe(0);

    S().toggleBoxHidden(0);
    S().clearSelection();
    expect(S().hiddenBoxes.size).toBe(0);
  });
});

describe("an arrow-key burst is one undo step", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().setSelection("C", fromBox(box([64, 128], [0, 64])), "replace");
    // Loading the graph starts a fresh history, but replacing its seeded default
    // selection records that replacement. Reset it to isolate this key burst.
    useStore.setState({ workspaceHistory: [] });
  });

  it("repeats append no history, so the cap is not evicted", () => {
    const depth = S().workspaceHistory.length;
    S().moveSelection(0, 8); // first key of the burst
    expect(S().workspaceHistory.length).toBe(depth + 1);
    for (let i = 0; i < 30; i++) S().moveSelection(0, 8, false); // auto-repeat
    expect(S().workspaceHistory.length).toBe(depth + 1);
  });

  it("undo returns to where the burst began, not one tile back", () => {
    const start = selBoxes()[0][0].lo;
    S().moveSelection(0, 8);
    for (let i = 0; i < 5; i++) S().moveSelection(0, 8, false);
    expect(selBoxes()[0][0].lo).toBe(start + 48);
    S().undoWorkspace();
    expect(selBoxes()[0][0].lo).toBe(start);
  });

  it("the nudge buttons still record every press", () => {
    const depth = S().workspaceHistory.length;
    S().moveSelection(0, 8);
    S().moveSelection(0, 8);
    expect(S().workspaceHistory.length).toBe(depth + 2);
  });
});

describe("escape cancels the current mode, never the selection", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().setDirection("backward");
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
    S().setSelection("C", fromBox(box([128, 192], [128, 192])), "union");
  });

  it("clearFocus unpins without touching the selection", () => {
    S().togglePinBox(1);
    expect(S().pinnedBox).toBe(1);
    expect(S().focusedBox).toBe(1);

    S().clearFocus();
    expect(S().pinnedBox).toBeNull();
    expect(S().focusedBox).toBeNull();
    // the tiles survive, untouched
    expect(selBoxes()).toEqual([
      box([0, 64], [0, 64]),
      box([128, 192], [128, 192]),
    ]);
    expect(selCount()).toBe(2 * 64 * 64);
    expect(regionOf("A")).toBeDefined();
  });

  it("a pinned part outranks hovering, and survives the pointer leaving", () => {
    S().togglePinBox(0);
    S().hoverBox(1); // hovering a different row must not steal focus
    expect(S().focusedBox).toBe(0);
    S().hoverBox(null); // pointer leaves the list
    expect(S().focusedBox).toBe(0);
  });

  it("hovering works normally when nothing is pinned", () => {
    S().hoverBox(1);
    expect(S().focusedBox).toBe(1);
    S().hoverBox(null);
    expect(S().focusedBox).toBeNull();
    expect(S().pinnedBox).toBeNull();
  });

  it("clicking the pinned part again unpins it", () => {
    S().togglePinBox(1);
    S().togglePinBox(1);
    expect(S().pinnedBox).toBeNull();
    expect(S().focusedBox).toBeNull();
  });

  it("the pin survives a move, so a pinned part can be nudged repeatedly", () => {
    S().togglePinBox(1);
    S().moveSelection(0, -16);
    expect(S().pinnedBox).toBe(1);
    S().moveSelection(0, -16);
    expect(S().pinnedBox).toBe(1);
    expect(selBoxes()[1]).toEqual(box([96, 160], [128, 192]));
  });

  it("the pin is dropped when parts are removed", () => {
    S().togglePinBox(1);
    S().deleteBox(0);
    expect(S().pinnedBox).toBeNull();
    expect(S().focusedBox).toBeNull();
  });

  it("clear still exists as the explicit way to drop the selection", () => {
    S().clearSelection();
    expect(sel()).toBeNull();
    expect(S().pinnedBox).toBeNull();
  });
});

describe("tiles on different tensors coexist", () => {
  beforeEach(() => {
    // Plain GEMM: A[256,512] @ B[512,256] -> C[256,256]
    S().loadExample(0);
    S().setDirection("backward");
    S().setSelection("C", fromBox(box([0, 64], [0, 64])), "replace");
  });

  it("keeps the tile on C when a second is drawn on A", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    expect(selTensors()).toEqual(["C", "A"]);
    expect(selBoxes()).toHaveLength(2);
  });

  it("unions both cones into the aggregate", () => {
    // C[0:64, 0:64] alone pulls A[0:64, :]
    expect(regionOf("A")!.boxes).toEqual([box([0, 64], [0, 512])]);
    // adding a tile on A itself contributes A[128:192, 0:64] as its own seed
    S().setSelection("A", fromBox(box([128, 192], [0, 64])));
    expect(regionOf("A")!.boxes).toEqual([
      box([0, 64], [0, 512]),
      box([128, 192], [0, 64]),
    ]);
  });

  it("names every seeded tensor in the merged result's roots", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    expect(S().backwardRes!.roots.sort()).toEqual(["A", "C"]);
  });

  it("attributes one cone per part, across tensors", () => {
    S().setSelection("A", fromBox(box([128, 192], [0, 64])));
    const perBox = S().perBox!;
    expect(perBox).toHaveLength(2);
    // part 0 is the C tile: it reaches A one hop back
    expect(perBox[0].backward!.tensors.get("A")!.depth).toBe(1);
    // part 1 is the A tile: A is its own seed, and it reaches no further back
    expect(perBox[1].backward!.tensors.get("A")!.depth).toBe(0);
    expect(perBox[1].backward!.tensors.get("C")).toBeUndefined();
  });

  it("composes only within the tensor drawn on", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    // a subtract gesture on A must not reach the part on C
    S().setSelection("A", fromBox(box([0, 64], [0, 64])), "subtract");
    expect(selTensors()).toEqual(["C"]);
  });

  it("keeps untouched parts at their index, so hues do not shuffle", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    S().setSelection("B", fromBox(box([0, 64], [0, 64])));
    const before = selBoxes()[0];
    // redrawing on B leaves C at index 0 and A at index 1
    S().setSelection("B", fromBox(box([64, 128], [0, 64])));
    expect(selTensors()).toEqual(["C", "A", "B", "B"]);
    expect(selBoxes()[0]).toEqual(before);
  });

  it("preserves hidden state on untouched tensors", () => {
    S().toggleBoxHidden(0);
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));

    expect(S().hiddenBoxes).toEqual(new Set([0]));
    expect(S().pinnedBox).toBeNull();
    expect(S().focusedBox).toBeNull();
  });

  it("remaps untouched UI state when edits on another tensor shift indices", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    S().setSelection("B", fromBox(box([0, 64], [0, 64])));
    S().toggleBoxHidden(2); // B

    S().setSelection("A", fromBox(box([0, 64], [0, 64])), "subtract");
    expect(selTensors()).toEqual(["C", "B"]);
    expect(S().hiddenBoxes).toEqual(new Set([1]));
    expect(S().pinnedBox).toBeNull();
    expect(S().focusedBox).toBeNull();
  });

  it("moves only the anchor tensor's parts, leaving the others put", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    const cBefore = selBoxes()[0];
    S().moveSelection(0, 64); // anchor is A: the last tensor drawn on
    expect(selBoxes()[0]).toEqual(cBefore);
    expect(selBoxes()[1]).toEqual(box([64, 128], [0, 64]));
  });

  it("moves the focused part alone, whichever tensor it is on", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    S().togglePinBox(0); // focus the tile on C
    S().moveSelection(0, 64);
    expect(selBoxes()[0]).toEqual(box([64, 128], [0, 64]));
    expect(selBoxes()[1]).toEqual(box([0, 64], [0, 64]));
  });

  it("counts a selected tensor's own bytes as output, not intermediate", () => {
    S().setSelection("A", fromBox(box([0, 64], [0, 64])));
    const m = computeMetrics(S().resolved!, S().backwardRes!);
    // A is an input tensor, so its bytes land in inputBytes either way; C is
    // the producer output. What matters is that nothing seeded is counted as
    // an intermediate.
    expect(m.intermediateBytes).toBe(0);
    expect(m.outputBytes).toBeGreaterThan(0);
  });

  it("drops per-part attribution past the cap but still merges every tensor", () => {
    S().setSelection("C", fromBox(box([0, 8], [0, 8])), "replace");
    for (let i = 1; i <= MAX_PER_BOX_PROPS; i++)
      S().setSelection(i % 2 ? "A" : "C", fromBox(box([i * 8, i * 8 + 8], [0, 8])));
    expect(sel()!.parts.length).toBeGreaterThan(MAX_PER_BOX_PROPS);
    expect(S().perBox).toBeNull();
    expect(S().backwardRes!.roots.sort()).toEqual(["A", "C"]);
  });
});

describe("offered starting tiles", () => {
  beforeEach(() => {
    S().setDirection("both");
    loadExampleNamed("Plain GEMM");
  });

  it("offers the graph's result and the step before it", () => {
    const starts = startingTiles(S().resolved!, S().tileScale, S().graphPx);
    expect(starts.map((s) => s.tensorId)).toEqual(["C"]); // GEMM has no intermediate
    loadExampleNamed("Multi-head attention");
    const attention = startingTiles(S().resolved!, S().tileScale, S().graphPx);
    expect(attention).toHaveLength(2);
    const [output, previous] = attention;
    expect(S().resolved!.consumers[output.tensorId]).toHaveLength(0);
    expect(S().resolved!.tensors[previous.tensorId].producer).toBeDefined();
  });

  it("offers a tile the canvas would have snapped to, in bounds", () => {
    const [start] = startingTiles(S().resolved!, S().tileScale, S().graphPx);
    const shape = S().resolved!.tensors[start.tensorId].resolved!;
    const { rowAxis, colAxis } = viewAxes(shape);
    const tile = tileOf(shape, S().tileScale, S().graphPx);
    start.box.forEach((interval, axis) => {
      expect(interval.lo).toBe(0);
      expect(interval.hi).toBeLessThanOrEqual(shape[axis]);
      expect(interval.hi).toBe(
        axis === rowAxis || axis === colAxis ? Math.min(tile, shape[axis]) : 1
      );
    });
  });

  it("offers a single element when detail is none", () => {
    S().setTileScale(TILE_SCALE_NONE);
    const [start] = startingTiles(S().resolved!, S().tileScale, S().graphPx);
    start.box.forEach((interval) => {
      expect(interval.lo).toBe(0);
      expect(interval.hi).toBe(1);
    });
  });

  it("produces a selection the executor accepts", () => {
    const [start] = startingTiles(S().resolved!, S().tileScale, S().graphPx);
    S().setSelection(start.tensorId, fromBox(start.box), "replace");
    expect(selTensors()).toEqual([start.tensorId]);
    expect(S().backwardRes).not.toBeNull();
  });
});
