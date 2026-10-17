import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AxisEditor } from "../../../src/components/inspector/AxisEditor";
import { box, fromBox } from "../../../src/core/region";
import { useStore } from "../../../src/state/store";
import { displayTile } from "../../../src/view/tensor/tiling";
import { gestureTile, isLatticeTile } from "../../../src/view/tensor/tile-spec";
import { axisTableMenu, planWithTile, type AxisTableAction } from "../../../src/view/tensor/menus";
import type { MenuAction } from "../../../src/view/menu";

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
  return gestureTile(shape, S().viewCfgs.X, displayTile(shape, S().tileScale));
};

/** The menu's actions for the inspected part of X, derived as the panel derives them. */
const actionsNow = () => {
  const shape = S().resolved!.tensors.X.resolved!;
  return axisTableMenu({
    own: !!S().viewCfgs.X.tile,
    matchesTile: isLatticeTile(S().selection!.parts[0].box, currentTile(), shape),
  }).filter((entry): entry is MenuAction<AxisTableAction> => entry.kind === "action");
};
const option = (id: AxisTableAction) => actionsNow().find((o) => o.id === id)!;

describe("the axis table at rest", () => {
  beforeEach(() => {
    S().applyDSL("X = Tensor(256, 256, dtype=fp32)\nY = relu(X)\n");
    const [rows, cols] = currentTile();
    S().setSelection("X", fromBox(box([0, rows], [0, cols])), "replace");
  });

  it("states axis, size, grid and range, and keeps the grid actions behind the options menu", () => {
    const html = render();
    expect(html).toContain('aria-label="axis options"');
    expect(html).toContain('aria-haspopup="menu"');
    expect(html).toMatch(/<th[^>]*>grid<\/th>/);
    for (const hidden of ["tile #", "use as card grid", "reset card grid"])
      expect(html).not.toContain(hidden);
  });

  /* Planning is the way from Dependencies into Plan, so it is on the panel
     rather than in the menu; on a graph input it says why it does nothing. */
  it("offers planning with the tile beside the heading", () => {
    expect(render()).toMatch(/<button class="mini" disabled=""[^>]*>plan with this tile<\/button>/);
    S().setSelection("Y", fromBox(box([0, 16], [0, 16])), "replace");
    expect(render()).toMatch(/<button class="mini" title="divide this tensor[^"]*">plan with this tile<\/button>/);
  });

  it("does not choose the card's axes: that is the tensor's setting, made on the card", () => {
    const html = render();
    for (const gone of ["↕", "↔", "draw ax0 as", "draw default plane"]) expect(html).not.toContain(gone);
  });
});

describe("keeping the canvas grid as a tensor's own", () => {
  beforeEach(() => {
    S().applyDSL("X = Tensor(256, 256, dtype=fp32)\nY = relu(X)\n");
    const [rows, cols] = currentTile();
    // Exactly one tile of the canvas lattice, so nothing distinguishes it from
    // the tile a click would take.
    S().setSelection("X", fromBox(box([0, rows], [0, cols])), "replace");
  });

  it("offers 'use as card grid' on a lattice tile the tensor does not own yet", () => {
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

describe("the actions that do not apply", () => {
  it("says why, rather than disappearing", () => {
    const actions = axisTableMenu({ own: false, matchesTile: false })
      .filter((entry): entry is MenuAction<AxisTableAction> => entry.kind === "action");
    expect(actions.map((o) => o.id)).toEqual(["own", "reset"]);
    expect(actions.filter((o) => o.disabled).map((o) => o.id)).toEqual(["reset"]);
    expect(planWithTile(true)).toEqual({
      disabled: true,
      title: "a graph input has no tasks: nothing computes it",
    });
    expect(planWithTile(false).disabled).toBe(false);
  });
});
