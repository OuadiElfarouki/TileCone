import { beforeEach, describe, expect, it } from "vitest";
import { count, fromBox, box } from "../core/region";
import { MAX_PER_BOX_PROPS, useStore } from "../ui/store";

const S = () => useStore.getState();
const sel = () => S().selection;
const regionOf = (tensorId: string) => S().backwardRes?.tensors.get(tensorId)?.region;

/** Drives the store exactly as the UI does, to cover the selection-editing actions. */
describe("selection editing through the store", () => {
  beforeEach(() => {
    // the store is a module singleton, so reset every axis the tests vary
    S().loadExample(0); // Plain GEMM: A[256,512] @ B[512,256] -> C[256,256]
    S().setDirection("backward");
    S().setComposeMode("replace");
    S().setSelection("C", fromBox(box([64, 128], [0, 64])));
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
    S().setComposeMode("replace");
    S().setSelection("C", fromBox(box([0, 32], [0, 32])), "union");
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
      S().setFocusedBox(1);
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
    S().setComposeMode("replace");
    S().clearSelection(); // loadExample seeds a default selection
  });

  it("a single box reuses the aggregate result rather than propagating twice", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])));
    const pb = S().perBox!;
    expect(pb).toHaveLength(1);
    expect(pb[0].backward).toBe(S().backwardRes); // same object, not a recomputation
  });

  it("each box gets its own cone, and they union to the aggregate", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])));
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
    S().setSelection("A", fromBox(box([0, 8], [0, 512])));
    S().setSelection("A", fromBox(box([100, 108], [0, 512])), "union");
    const pb = S().perBox!;
    expect(pb).toHaveLength(2);
    expect(pb[0].forward!.tensors.get("C")!.region.boxes).toEqual([box([0, 8], [0, 256])]);
    expect(pb[1].forward!.tensors.get("C")!.region.boxes).toEqual([box([100, 108], [0, 256])]);
    expect(pb[0].backward).toBeNull(); // backward is not computed in forward-only mode
  });

  it("focusing a box scopes the readout to that box alone", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])));
    S().setSelection("C", fromBox(box([192, 256], [128, 192])), "union");
    S().setFocusedBox(1);
    const focused = S().perBox![S().focusedBox!].backward!;
    expect(focused.tensors.get("A")!.region.boxes).toEqual([box([192, 256], [0, 512])]);
    // the aggregate still holds both bands
    expect(regionOf("A")!.boxes).toHaveLength(2);
  });

  it("focus resets when the selection changes", () => {
    S().setSelection("C", fromBox(box([0, 64], [0, 64])));
    S().setSelection("C", fromBox(box([192, 256], [128, 192])), "union");
    S().setFocusedBox(1);
    expect(S().focusedBox).toBe(1);
    S().moveSelection(0, 8);
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
