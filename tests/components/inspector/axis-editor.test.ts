import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AxisEditor } from "../../../src/components/inspector/AxisEditor";
import { box, fromBox } from "../../../src/core/region";
import { useStore } from "../../../src/state/store";
import { tileOf } from "../../../src/view/tensor/grid";
import { gestureTile } from "../../../src/view/tensor/tile-spec";

// Static-render tests read the live test store rather than Zustand's initial
// server snapshot. Actions and all derivation logic remain the real implementation.
vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const S = () => useStore.getState();
const render = () => renderToStaticMarkup(createElement(AxisEditor, { index: 0 }));
/** The tile a gesture on X would take right now. */
const currentTile = () => {
  const shape = S().resolved!.tensors.X.resolved!;
  return gestureTile(shape, S().viewCfgs.X, tileOf(shape, S().tileScale, S().graphPx, S().viewCfgs.X));
};

describe("keeping the canvas tile as a tensor's own", () => {
  beforeEach(() => {
    S().applyDSL("X = Tensor(256, 256, dtype=fp32)\nY = relu(X)\n");
    const [rows, cols] = currentTile();
    // Exactly one tile of the canvas lattice, so nothing distinguishes it from
    // the tile a click would take.
    S().setSelection("X", fromBox(box([0, rows], [0, cols])), "replace");
  });

  it("offers 'use as tile' on a lattice tile the tensor does not own yet", () => {
    expect(S().viewCfgs.X.tile).toBeUndefined();
    expect(render()).toContain("use as tile");
    expect(render()).toContain("so later detail changes no longer resize it");
  });

  it("keeps that tile when detail changes afterwards", () => {
    const frozen = currentTile();
    // Unowned, the same detail change resizes it; otherwise this proves nothing.
    S().setTileScale(S().tileScale + 2);
    expect(currentTile()).not.toEqual(frozen);
    S().setTileScale(S().tileScale - 2);
    expect(currentTile()).toEqual(frozen);
    // What the button does: the tile's own extents, refitting nothing.
    S().setTensorTile("X", frozen, 0);
    expect(S().viewCfgs.X.tile).toEqual(frozen);
    S().setTileScale(S().tileScale + 2);
    expect(currentTile()).toEqual(frozen);
    // Owned and already matching: nothing left to offer.
    expect(render()).not.toContain("use as tile");
  });
});
