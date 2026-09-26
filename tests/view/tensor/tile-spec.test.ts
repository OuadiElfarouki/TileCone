import { beforeEach, describe, expect, it } from "vitest";
import { box } from "../../../src/core/region";
import { gridGeometry, nudgeUnit } from "../../../src/view/tensor/grid";
import { decodeWorkspace, encodeWorkspace, type WorkspaceLink } from "../../../src/state/share";
import { defaultPlanTile, startingTiles, useStore } from "../../../src/state/store";
import { planGesture, selectionBoxFromDrag } from "../../../src/components/card/TensorCard";
import { selectionReading } from "../../../src/components/inspector/AxisEditor";
import {
  gestureTile,
  lastTileExtent,
  seedTile,
  tileCount,
  tilePosition,
  tileSpanAt,
} from "../../../src/view/tensor/tile-spec";

const S = () => useStore.getState();

/* [B, H, S, D] with a sequence length the tile does not divide, so boundary
   tiles are exercised as well as ordinary ones. */
const DSL = "B = 2\nH = 8\nL = 200\nD = 128\nX = Tensor(B, H, L, D, dtype=fp16)\nY = relu(X)\n";
const SHAPE = [2, 8, 200, 128];
const TILE = [1, 2, 64, 128];

describe("tile arithmetic", () => {
  it("clips the last tile of an axis the tile does not divide", () => {
    expect(tileCount(200, 64)).toBe(4);
    expect(lastTileExtent(200, 64)).toBe(8);
    expect(lastTileExtent(256, 64)).toBe(64);
    expect(tileSpanAt(199, 64, 200)).toEqual({ lo: 192, hi: 200 });
  });

  it("recognises a shortened last tile as a tile of the lattice", () => {
    expect(tilePosition({ lo: 192, hi: 200 }, 64, 200)).toEqual({ coord: 3, last: 3, count: 4, aligned: true, whole: true });
    expect(tilePosition({ lo: 100, hi: 164 }, 64, 200)).toMatchObject({ aligned: false, whole: false });
    // Several whole tiles: a run, not a tile.
    expect(tilePosition({ lo: 64, hi: 200 }, 64, 200)).toMatchObject({ coord: 1, last: 3, aligned: false, whole: true });
  });
});

describe("the tile a gesture takes", () => {
  it("falls back to the canvas behaviour without a tile of the tensor's own", () => {
    expect(gestureTile(SHAPE, { projection: true }, 16)).toEqual([2, 8, 16, 16]);
    expect(gestureTile(SHAPE, { projection: false }, 16)).toEqual([1, 1, 16, 16]);
    expect(seedTile(SHAPE, undefined, 16)).toEqual([1, 1, 16, 16]);
  });

  it("uses the tensor's own tile on every axis, whatever the view mode", () => {
    for (const projection of [true, false]) {
      expect(gestureTile(SHAPE, { projection, tile: TILE }, 16)).toEqual(TILE);
      expect(seedTile(SHAPE, { tile: TILE }, 16)).toEqual(TILE);
    }
  });

  it("draws a rectangular lattice and snaps a drag to it", () => {
    const cfg = { projection: true, sliders: [0, 5, 0, 0], tile: TILE };
    const geom = gridGeometry(SHAPE, cfg, 0, 1);
    expect([geom.rowTile, geom.colTile]).toEqual([64, 128]);
    expect([geom.tileRows, geom.tileCols]).toEqual([4, 1]);
    // One cell in the third row band: the hidden axes take the tile holding
    // the position, not the whole axis, because the tile names that extent.
    const picked = selectionBoxFromDrag(SHAPE, cfg, geom, { r0: 130, c0: 3, r1: 130, c1: 3 }, true);
    expect(picked).toEqual(box([0, 1], [4, 6], [128, 192], [0, 128]));
    expect(selectionReading(picked, (axis) => "BHSD"[axis])).toBe("B[0:1] H[4:6] S[128:192] D[0:128]");
  });

  it("gives the boundary tile its actual extent", () => {
    const cfg = { projection: false, sliders: [1, 7, 0, 0], tile: TILE };
    const geom = gridGeometry(SHAPE, cfg, 0, 1);
    const picked = selectionBoxFromDrag(SHAPE, cfg, geom, { r0: 199, c0: 0, r1: 199, c1: 0 }, true);
    expect(picked).toEqual(box([1, 2], [6, 8], [192, 200], [0, 128]));
  });

  it("steps the keyboard by the tile on the axis it moves", () => {
    const cfg = { projection: true, sliders: [0, 0, 0, 0], tile: TILE };
    expect(nudgeUnit(SHAPE, cfg, 0, 1, true, 2)).toBe(64);
    expect(nudgeUnit(SHAPE, cfg, 0, 1, true, 3)).toBe(128);
    expect(nudgeUnit(SHAPE, cfg, 0, 1, false, 2)).toBe(1);
  });

  it("divides a plan at the drawn visible extents and the tensor's tile on the rest", () => {
    const cfg = { projection: true, sliders: [0, 0, 0, 0], tile: TILE };
    const geom = gridGeometry(SHAPE, cfg, 0, 1);
    const gesture = planGesture(SHAPE, cfg, geom, { r0: 0, c0: 0, r1: 100, c1: 5 }, false);
    if (gesture.kind !== "divide") throw new Error(gesture.kind);
    expect(gesture.tile).toEqual([1, 2, 128, 128]);
  });
});

describe("a tensor's tile in the workspace", () => {
  beforeEach(() => {
    S().applyDSL(DSL);
    S().setSelection("X", { boxes: [box([0, 1], [0, 8], [0, 16], [0, 16])], exact: true, reasons: [] }, "replace");
  });

  it("refits the inspected tile and leaves the tile out of undo", () => {
    S().setTensorTile("X", TILE, 0);
    expect(S().viewCfgs.X.tile).toEqual(TILE);
    expect(S().selection!.parts[0].box).toEqual(box([0, 1], [0, 2], [0, 64], [0, 128]));
    S().undoWorkspace();
    expect(S().selection!.parts[0].box).toEqual(box([0, 1], [0, 8], [0, 16], [0, 16]));
    // A gesture setting, like snap and detail.
    expect(S().viewCfgs.X.tile).toEqual(TILE);
  });

  it("refuses a tile that does not fit the tensor", () => {
    S().setTensorTile("X", [1, 2, 64], 0);
    S().setTensorTile("X", [1, 9, 64, 128], 0);
    S().setTensorTile("X", [0, 2, 64, 128], 0);
    expect(S().viewCfgs.X.tile).toBeUndefined();
  });

  it("steps along a hidden axis and keeps the slice on the tile", () => {
    S().setViewCfg("X", { projection: false });
    S().setTensorTile("X", TILE, 0);
    S().stepTile(0, 1, 1);
    expect(S().selection!.parts[0].box[1]).toEqual({ lo: 2, hi: 4 });
    expect(S().viewCfgs.X.sliders[1]).toBe(2);
    // To the shortened last tile, and back onto the full one before it.
    S().stepTile(0, 2, 5);
    expect(S().selection!.parts[0].box[2]).toEqual({ lo: 192, hi: 200 });
    S().stepTile(0, 2, -1);
    expect(S().selection!.parts[0].box[2]).toEqual({ lo: 128, hi: 192 });
  });

  it("seeds starter tiles and plan divisions from the tensor's tile", () => {
    S().setTensorTile("Y", TILE);
    expect(defaultPlanTile(S().resolved!, "Y", S().tileScale, S().graphPx, S().viewCfgs)).toEqual(TILE);
    const [output] = startingTiles(S().resolved!, S().tileScale, S().graphPx, S().viewCfgs);
    expect(output.box).toEqual(box([0, 1], [0, 2], [0, 64], [0, 128]));
  });

  it("clears on reset and survives a share link", () => {
    S().setTensorTile("X", TILE);
    const link: WorkspaceLink = {
      dsl: DSL, dir: "both", tile: 0, sel: null, views: { X: S().viewCfgs.X },
    };
    const decoded = decodeWorkspace(`#s=${encodeWorkspace(link)}`)!;
    S().setTensorTile("X", null);
    expect(S().viewCfgs.X.tile).toBeUndefined();
    expect(S().restoreWorkspace({
      dsl: DSL, direction: "both", tileScale: 0, snapToGrid: true, axisMode: "symbolic",
      parts: null, viewCfgs: decoded.views,
    })).toBe(true);
    expect(S().viewCfgs.X.tile).toEqual(TILE);
    expect(decodeWorkspace(`#s=${encodeWorkspace({ ...link, views: { X: { ...link.views!.X, tile: [1.5] } } })}`)).toBeNull();
  });
});
