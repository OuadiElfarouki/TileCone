import { beforeEach, describe, expect, it } from "vitest";
import { count, fromBox, box } from "../core/region";
import { EXAMPLES } from "../examples";
import {
  MAX_PER_BOX_PROPS, PANEL_COLLAPSE_AT, PANEL_MAX, PANEL_MIN,
  planesOf, useStore, viewAxes,
} from "../ui/store";
import { cardPx, graphScale, MAX_ELEM_PX, planeExtents } from "../ui/tiling";
import { nudgeUnit, tileOf } from "../ui/grid";

const S = () => useStore.getState();
const sel = () => S().selection;
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

  it("represents all four upstream/downstream combinations", () => {
    expect(S().direction).toBe("both");
    expect(S().backwardRes).not.toBeNull();
    expect(S().forwardRes).not.toBeNull();

    S().toggleDirection("backward");
    expect(S().direction).toBe("forward");
    expect(S().backwardRes).toBeNull();
    expect(S().forwardRes).not.toBeNull();

    S().toggleDirection("forward");
    expect(S().direction).toBe("none");
    expect(S().backwardRes).toBeNull();
    expect(S().forwardRes).toBeNull();
    expect(S().perBox).toBeNull();

    S().toggleDirection("backward");
    expect(S().direction).toBe("backward");
    S().toggleDirection("forward");
    expect(S().direction).toBe("both");
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
    expect(S().selection?.tensorId).toBe("Out");

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
    const initial = S().selection!.region.boxes;
    S().moveSelection(0, 64);
    const afterSelectionMove = S().selection!.region.boxes;

    S().setTensorOffset("A", { dx: 90, dy: 35 });
    S().commitTensorMove("A", { dx: 0, dy: 0 });
    S().moveSelection(1, 64);

    S().undoWorkspace(); // selection edit after the tensor drag
    expect(S().selection!.region.boxes).toEqual(afterSelectionMove);
    expect(S().tensorOffsets.A).toEqual({ dx: 90, dy: 35 });

    S().undoWorkspace(); // tensor drag itself
    expect(S().selection!.region.boxes).toEqual(afterSelectionMove);
    expect(S().tensorOffsets.A).toBeUndefined();

    S().undoWorkspace(); // selection edit before the tensor drag
    expect(S().selection!.region.boxes).toEqual(initial);
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
    expect(S().selection!.region.boxes).toEqual([box([0, 64], [0, 64])]);
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
    expect(S().selection!.region.boxes[0][0]).toEqual({ lo: 65, hi: 129 });
  });

  it("an unsnapped range propagates exactly like any other", () => {
    // Regions were always element-precise; only the drag gesture rounded, so an
    // odd range must be as ordinary to the engine as an aligned one.
    S().setSnapToGrid(false);
    S().setSelection("C", fromBox(box([3, 7], [1, 2])), "replace");
    expect(S().selection!.region.boxes).toEqual([box([3, 7], [1, 2])]);
    expect(regionOf("A")!.boxes).toEqual([box([3, 7], [0, 512])]);
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [1, 2])]);
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
    expect(sel()!.tensorId).toBe("C");
    expect(regionOf("A")!.boxes).toEqual([box([64, 128], [0, 512])]);
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [0, 64])]);
  });

  it("moveSelection shifts the selection and repropagates", () => {
    S().moveSelection(0, 64); // down one tile on the row axis
    expect(sel()!.region.boxes).toEqual([box([128, 192], [0, 64])]);
    expect(regionOf("A")!.boxes).toEqual([box([128, 192], [0, 512])]);
    // B is unaffected by a pure row move
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [0, 64])]);
  });

  it("replaceBox edits one stable part and repropagates", () => {
    const depth = S().workspaceHistory.length;
    S().replaceBox(0, box([16, 48], [32, 96]));
    expect(sel()!.region.boxes).toEqual([box([16, 48], [32, 96])]);
    expect(regionOf("A")!.boxes).toEqual([box([16, 48], [0, 512])]);
    expect(regionOf("B")!.boxes).toEqual([box([0, 512], [32, 96])]);
    expect(S().workspaceHistory.length).toBe(depth + 1);
  });

  it("moveSelection clamps at the tensor edge without shrinking", () => {
    const before = count(sel()!.region);
    for (let i = 0; i < 10; i++) S().moveSelection(0, 64);
    expect(count(sel()!.region)).toBe(before);
    expect(sel()!.region.boxes).toEqual([box([192, 256], [0, 64])]);
  });

  it("direct drawing adds a multi-box selection by default", () => {
    S().setSelection("C", fromBox(box([0, 32], [200, 232])));
    expect(sel()!.region.boxes).toHaveLength(2);
    expect(count(sel()!.region)).toBe(64 * 64 + 32 * 32);
    // both output tiles' row bands appear upstream in A
    expect(regionOf("A")!.boxes).toEqual([box([0, 32], [0, 512]), box([64, 128], [0, 512])]);
  });

  it("an explicit subtract gesture cuts a hole", () => {
    S().setSelection("C", fromBox(box([80, 96], [0, 64])), "subtract");
    expect(count(sel()!.region)).toBe(64 * 64 - 16 * 64);
    expect(sel()!.region.boxes).toHaveLength(2);
  });

  it("an explicit union gesture adds a part", () => {
    S().setSelection("C", fromBox(box([0, 32], [0, 32])), "union");
    expect(sel()!.region.boxes).toHaveLength(2);
  });

  it("deleteBox removes exactly one box", () => {
    S().setSelection("C", fromBox(box([0, 32], [200, 232])));
    expect(sel()!.region.boxes).toHaveLength(2);
    S().deleteBox(0);
    expect(sel()!.region.boxes).toHaveLength(1);
  });

  it("workspace undo walks back through selection edits", () => {
    S().moveSelection(0, 64);
    S().moveSelection(0, 64);
    expect(sel()!.region.boxes).toEqual([box([192, 256], [0, 64])]);
    S().undoWorkspace();
    expect(sel()!.region.boxes).toEqual([box([128, 192], [0, 64])]);
    S().undoWorkspace();
    expect(sel()!.region.boxes).toEqual([box([64, 128], [0, 64])]);
    expect(regionOf("A")!.boxes).toEqual([box([64, 128], [0, 512])]);
  });

  it("clearing then undoing restores the selection", () => {
    S().clearSelection();
    expect(sel()).toBeNull();
    S().undoWorkspace();
    expect(sel()!.region.boxes).toEqual([box([64, 128], [0, 64])]);
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

    const boxes = sel()!.region.boxes;
    // box order matches selection.region.boxes, so row bands line up per box
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
    expect(pb[0].backward).toBeNull(); // backward is not computed in forward-only mode
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
    expect(sel()!.region.boxes[1]).toEqual(box([176, 240], [128, 192]));
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
    expect(sel()!.region.boxes.length).toBeGreaterThan(MAX_PER_BOX_PROPS);
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
    const before = sel()!.region.boxes[0];
    S().togglePinBox(1);
    S().moveSelection(0, 32);
    const after = sel()!.region.boxes;
    expect(after).toHaveLength(2);
    expect(after[0]).toEqual(before); // untouched
    expect(after[1]).toEqual(box([160, 224], [128, 192]));
  });

  it("moves the whole selection when nothing is focused", () => {
    S().clearFocus();
    S().moveSelection(0, 32);
    expect(sel()!.region.boxes).toEqual([
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
    const boxes = sel()!.region.boxes;
    expect(boxes).toHaveLength(2);
    expect(boxes[0]).toEqual(box([0, 64], [0, 64]));
    expect(boxes[1]).toEqual(box([0, 64], [0, 64])); // exactly coincident
  });

  it("overlapping parts are counted once, never doubled", () => {
    S().togglePinBox(1);
    S().moveSelection(0, -128);
    S().moveSelection(1, -128);
    // two coincident 64x64 parts denote 4096 elements, not 8192
    expect(count(sel()!.region)).toBe(64 * 64);
    // and the upstream footprint is the single row band, not two
    expect(regionOf("A")!.boxes).toEqual([box([0, 64], [0, 512])]);
    expect(count(regionOf("A")!)).toBe(64 * 512);
  });

  it("partial overlap yields the union upstream, in real time", () => {
    S().togglePinBox(1);
    S().moveSelection(0, -96); // rows 32:96 -> overlaps part 0's rows 0:64
    S().moveSelection(1, -128); // cols 0:64, same as part 0
    expect(sel()!.region.boxes[1]).toEqual(box([32, 96], [0, 64]));
    // union of rows 0:64 and 32:96 is 0:96, merged into one band upstream
    expect(regionOf("A")!.boxes).toEqual([box([0, 96], [0, 512])]);
    expect(count(regionOf("A")!)).toBe(96 * 512);
  });

  it("a moved part is clamped to the tensor without shrinking", () => {
    S().togglePinBox(1);
    const vol = count(fromBox(sel()!.region.boxes[1]));
    for (let i = 0; i < 20; i++) S().moveSelection(0, 64);
    const b = sel()!.region.boxes[1];
    expect(b).toEqual(box([192, 256], [128, 192]));
    expect(count(fromBox(b))).toBe(vol);
  });

  it("a stale focus index is ignored rather than throwing", () => {
    S().togglePinBox(5);
    expect(() => S().moveSelection(0, 8)).not.toThrow();
    // falls back to moving everything
    expect(sel()!.region.boxes).toHaveLength(2);
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

  it("hiding a box leaves its metrics live", () => {
    // The whole point of parking a probe: it stops painting, it does not stop
    // being measured.
    const before = S().perBox![1].backward!.tensors.get("A")!.region;
    S().toggleBoxHidden(1);
    expect(S().hiddenBoxes.has(1)).toBe(true);
    expect(S().perBox![1].backward!.tensors.get("A")!.region).toEqual(before);
    expect(S().selection!.region.boxes.length).toBe(3);
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
    expect(S().selection!.region.boxes.length).toBe(2);
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
    const start = S().selection!.region.boxes[0][0].lo;
    S().moveSelection(0, 8);
    for (let i = 0; i < 5; i++) S().moveSelection(0, 8, false);
    expect(S().selection!.region.boxes[0][0].lo).toBe(start + 48);
    S().undoWorkspace();
    expect(S().selection!.region.boxes[0][0].lo).toBe(start);
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
    expect(sel()!.region.boxes).toEqual([
      box([0, 64], [0, 64]),
      box([128, 192], [128, 192]),
    ]);
    expect(count(sel()!.region)).toBe(2 * 64 * 64);
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
    expect(sel()!.region.boxes[1]).toEqual(box([96, 160], [128, 192]));
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
