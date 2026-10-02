import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AxisEditor, axisOptions } from "../../../src/components/inspector/AxisEditor";
import { box, fromBox } from "../../../src/core/region";
import { useStore } from "../../../src/state/store";
import { tileOf } from "../../../src/view/tensor/grid";
import { gestureTile, isLatticeTile } from "../../../src/view/tensor/tile-spec";

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

/** The menu's options for the inspected part of X, derived as the panel derives them. */
const optionsNow = () => {
  const shape = S().resolved!.tensors.X.resolved!;
  return axisOptions({
    own: !!S().viewCfgs.X.tile,
    matchesTile: isLatticeTile(S().selection!.parts[0].box, currentTile(), shape),
    planeRemapped: false,
    input: true,
  });
};
const option = (id: string) => optionsNow().find((o) => o.id === id)!;

describe("the axis table at rest", () => {
  beforeEach(() => {
    S().applyDSL("X = Tensor(256, 256, dtype=fp32)\nY = relu(X)\n");
    const [rows, cols] = currentTile();
    S().setSelection("X", fromBox(box([0, rows], [0, cols])), "replace");
  });

  it("states axis, size, tile and range, and keeps its actions behind the options menu", () => {
    const html = render();
    expect(html).toContain('aria-label="axis options"');
    expect(html).toContain('aria-haspopup="menu"');
    for (const hidden of ["tile #", "use as tile", "reset tile", "draw default plane", "plan with this tile"])
      expect(html).not.toContain(hidden);
  });
});

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
    expect(option("own")).toMatchObject({ disabled: false });
    expect(option("own").title).toContain("so later detail changes no longer resize it");
    expect(option("reset").disabled).toBe(true);
  });

  it("keeps that tile when detail changes afterwards", () => {
    const frozen = currentTile();
    // Unowned, the same detail change resizes it; otherwise this proves nothing.
    S().setTileScale(S().tileScale + 2);
    expect(currentTile()).not.toEqual(frozen);
    S().setTileScale(S().tileScale - 2);
    expect(currentTile()).toEqual(frozen);
    // What the option does: the tile's own extents, refitting nothing.
    S().setTensorTile("X", frozen, 0);
    expect(S().viewCfgs.X.tile).toEqual(frozen);
    S().setTileScale(S().tileScale + 2);
    expect(currentTile()).toEqual(frozen);
    // Owned and already matching: nothing left to use, and something to reset.
    expect(option("own").disabled).toBe(true);
    expect(option("reset").disabled).toBe(false);
  });
});

describe("the options that do not apply", () => {
  it("says why, rather than disappearing", () => {
    const options = axisOptions({ own: false, matchesTile: false, planeRemapped: false, input: true });
    const byId = Object.fromEntries(options.map((o) => [o.id, o]));
    expect(options.map((o) => o.id)).toEqual(["own", "reset", "plane", "plan"]);
    expect(byId.plane).toMatchObject({ disabled: true, title: "the card already draws the last two axes" });
    expect(byId.plan).toMatchObject({ disabled: true, title: "a graph input has no tasks: nothing computes it" });
    expect(axisOptions({ own: false, matchesTile: false, planeRemapped: true, input: false })
      .filter((o) => o.disabled).map((o) => o.id)).toEqual(["reset"]);
  });
});
