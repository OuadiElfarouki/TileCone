import { beforeEach, describe, expect, it } from "vitest";
import { count, fromBox, box } from "../core/region";
import { MAX_PER_BOX_PROPS, useStore } from "../ui/store";

const S = () => useStore.getState();
const sel = () => S().selection;
const regionOf = (tensorId: string) => S().backwardRes?.tensors.get(tensorId)?.region;

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
});

/** Drives the store exactly as the UI does, to cover the selection-editing actions. */
describe("selection editing through the store", () => {
  beforeEach(() => {
    // the store is a module singleton, so reset every axis the tests vary
    S().loadExample(0); // Plain GEMM: A[256,512] @ B[512,256] -> C[256,256]
    S().setDirection("backward");
    S().setComposeMode("union");
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

  it("moveSelection clamps at the tensor edge without shrinking", () => {
    const before = count(sel()!.region);
    for (let i = 0; i < 10; i++) S().moveSelection(0, 64);
    expect(count(sel()!.region)).toBe(before);
    expect(sel()!.region.boxes).toEqual([box([192, 256], [0, 64])]);
  });

  it("union compose mode builds a multi-box selection", () => {
    S().setComposeMode("union");
    S().setSelection("C", fromBox(box([0, 32], [200, 232])));
    expect(sel()!.region.boxes).toHaveLength(2);
    expect(count(sel()!.region)).toBe(64 * 64 + 32 * 32);
    // both output tiles' row bands appear upstream in A
    expect(regionOf("A")!.boxes).toEqual([box([0, 32], [0, 512]), box([64, 128], [0, 512])]);
  });

  it("subtract compose mode cuts a hole", () => {
    S().setComposeMode("subtract");
    S().setSelection("C", fromBox(box([80, 96], [0, 64])));
    expect(count(sel()!.region)).toBe(64 * 64 - 16 * 64);
    expect(sel()!.region.boxes).toHaveLength(2);
  });

  it("explicit compose argument overrides the mode (modifier keys)", () => {
    S().setComposeMode("subtract"); // toolbar says cut...
    S().setSelection("C", fromBox(box([0, 32], [0, 32])), "union"); // ...Shift says add
    expect(sel()!.region.boxes).toHaveLength(2);
  });

  it("deleteBox removes exactly one box", () => {
    S().setComposeMode("union");
    S().setSelection("C", fromBox(box([0, 32], [200, 232])));
    expect(sel()!.region.boxes).toHaveLength(2);
    S().deleteBox(0);
    expect(sel()!.region.boxes).toHaveLength(1);
  });

  it("undoSelection walks back through edits", () => {
    S().moveSelection(0, 64);
    S().moveSelection(0, 64);
    expect(sel()!.region.boxes).toEqual([box([192, 256], [0, 64])]);
    S().undoSelection();
    expect(sel()!.region.boxes).toEqual([box([128, 192], [0, 64])]);
    S().undoSelection();
    expect(sel()!.region.boxes).toEqual([box([64, 128], [0, 64])]);
    expect(regionOf("A")!.boxes).toEqual([box([64, 128], [0, 512])]);
  });

  it("clearing then undoing restores the selection", () => {
    S().clearSelection();
    expect(sel()).toBeNull();
    S().undoSelection();
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
    S().setComposeMode("union");
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
    S().setComposeMode("union");
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
 * selection — clearing tiles is the toolbar's explicit "clear" button.
 */
describe("escape cancels the current mode, never the selection", () => {
  beforeEach(() => {
    S().loadExample(0);
    S().setDirection("backward");
    S().setComposeMode("union");
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
