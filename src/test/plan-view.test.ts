import { beforeEach, describe, expect, it } from "vitest";
import { tileFamily } from "../core/plan/tile-family";
import { box, fromBox } from "../core/region";
import { defaultPlanTile, useStore } from "../ui/store";
import { buildPlanPaint, planElementFromCell, planGesture } from "../ui/TensorCard";
import { formatTileExtents, parseTileExtents } from "../ui/PlanPanel";
import { gridGeometry } from "../ui/grid";
import { TILE_SCALE_NONE } from "../ui/tiling";
import { defaultViewCfg } from "../ui/tensor-view";

const S = () => useStore.getState();

/* The chain of NEXT_FEATS §8, small enough to keep the tile arithmetic obvious. */
const CHAIN = `A = Tensor(16, 16)
B = Tensor(16, 16)
W = Tensor(16, 8)
C = matmul(A, B)
Y = matmul(C, W)
`;

describe("planning from the canvas", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
  });

  it("tiles a produced tensor on first click, at the size the canvas is drawing", () => {
    const expected = defaultPlanTile(S().resolved!, "Y", S().tileScale, S().graphPx);
    S().planTaskAt("Y", [9, 2]);
    expect(S().planTiles.Y).toEqual(expected);
    expect(S().planTask).toEqual({
      tensorId: "Y",
      coord: [Math.floor(9 / expected[0]), Math.floor(2 / expected[1])],
    });
  });

  it("tiles the produced tensors a task reads, so its producers are named at once", () => {
    S().setPlanTileAt("Y", [4, 4], [9, 2]);
    // C is produced and read by Y's operation, so it is tiled with it. W is a
    // graph input and stays one.
    expect(Object.keys(S().planTiles).sort()).toEqual(["C", "Y"]);
    expect(S().planSupply!.demand.map((d) => [d.tensorId, d.supplier])).toEqual([
      ["C", "tasks"],
      ["W", "input"],
    ]);
    expect(S().planSupply!.complete).toBe(true);
    expect(S().planSupply!.producers.length).toBeGreaterThan(0);
  });

  it("sets the tile a drag draws, and inspects the task it started in", () => {
    S().setPlanTileAt("Y", [4, 4], [9, 2]);
    expect(S().planTiles.Y).toEqual([4, 4]);
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [2, 0] });
    expect(S().plan!.families.get("Y")!.count).toBe(8);
  });

  it("divides a tensor once: a later drag inspects instead of redividing", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    const tiles = { ...S().planTiles };

    // The same drag that divided an untiled tensor now only names a task.
    S().planTaskAt("Y", [9, 6]);
    expect(S().planTiles).toEqual(tiles);
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [2, 1] });

    // Clearing the tiling is what allows a different one.
    S().setPlanTile("Y", null);
    S().setPlanTileAt("Y", [8, 8], [0, 0]);
    expect(S().planTiles.Y).toEqual([8, 8]);
  });

  it("keeps the plan when the canvas grid changes", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    const before = S().planSupply!.demand.map((d) => d.region.boxes);
    for (const scale of [TILE_SCALE_NONE, 0, 3, -2]) {
      S().setTileScale(scale);
      expect(S().planTiles.Y).toEqual([4, 4]);
      expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 0] });
      expect(S().planSupply!.demand.map((d) => d.region.boxes)).toEqual(before);
    }
  });

  it("ignores a graph input, which no task produces", () => {
    S().planTaskAt("A", [0, 0]);
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
  });

  it("names producer tasks once both sides are tiled", () => {
    S().setPlanTileAt("Y", [4, 4], [4, 0]);
    S().setPlanTile("C", [4, 4]);
    const supply = S().planSupply!;
    expect(supply.complete).toBe(true);
    expect(supply.producers.map((p) => p.task.coord.join(","))).toEqual(["1,0", "1,1", "1,2", "1,3"]);
    expect(supply.producers.every((p) => p.definite)).toBe(true);
  });

  it("keeps the task on the same elements when its tensor is retiled", () => {
    S().setPlanTileAt("Y", [4, 4], [9, 2]); // tile (2, 0) covers rows 8-12
    S().setPlanTile("Y", [8, 8]);
    // Rows 8-12 now lie in tile (1, 0), which covers rows 8-16.
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [1, 0] });
  });

  it("drops the task when its tensor stops being tiled", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    S().setPlanTile("Y", null);
    expect(S().planTiles.Y).toBeUndefined();
    expect(S().planTask).toBeNull();
    expect(S().planSupply).toBeNull();
  });

  it("steps the task by tiles and stops at the edge of the grid", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    S().movePlanTask(0, 1);
    expect(S().planTask!.coord).toEqual([1, 0]);
    S().movePlanTask(0, 8);
    expect(S().planTask!.coord).toEqual([3, 0]); // Y has four row tiles
    S().movePlanTask(0, -8);
    expect(S().planTask!.coord).toEqual([0, 0]);
  });

  it("refuses a task the plan does not contain", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    S().setPlanTile("C", null); // C is tiled with Y; remove it to leave no tasks there
    S().selectPlanTask({ tensorId: "C", coord: [0, 0] });
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 0] });
    S().selectPlanTask({ tensorId: "Y", coord: [9, 9] }); // outside the grid
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 0] });
  });

  it("recomputes what a task reads when the tiling changes", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    const before = S().planSupply!.demand.find((d) => d.tensorId === "C")!.region.boxes[0];
    expect(before).toEqual(box([0, 4], [0, 16]));
    S().setPlanTile("Y", [8, 8]);
    expect(S().planSupply!.demand.find((d) => d.tensorId === "C")!.region.boxes[0]).toEqual(
      box([0, 8], [0, 16])
    );
  });

  it("does not let display-lattice snapping change the plan task under the pointer", () => {
    const shape = [16, 16];
    const cfg = defaultViewCfg(shape);
    const geom = gridGeometry(shape, cfg, 1, 8);
    expect(geom.tile).toBeGreaterThan(1);

    // The exact element is required here: a finer plan may put it in a
    // different task than the display tile's snapped lower edge.
    expect(planElementFromCell(shape, cfg, geom, { row: 6, col: 3 })).toEqual([6, 3]);
  });
});

describe("plan edits and the workspace", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
  });

  it("undoes a plan edit like any other workspace edit", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]); // tiles Y, and C with it
    const seeded = { ...S().planTiles };
    S().setPlanTile("C", [8, 8]);
    expect(S().planTiles.C).toEqual([8, 8]);
    S().undoWorkspace();
    expect(S().planTiles).toEqual(seeded);
    S().undoWorkspace();
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
  });

  it("brings the Plan view forward when an undo restores a plan edit", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    S().setInspectorTab("dependencies");
    S().undoWorkspace();
    // The step back emptied the plan, which the Dependencies view cannot show.
    expect(S().inspectorTab).toBe("plan");
    expect(S().planTiles).toEqual({});
  });

  it("leaves the view alone when an undo restores only a tile edit", () => {
    S().setSelection("C", fromBox(box([0, 2], [0, 2])));
    S().setSelection("C", fromBox(box([4, 6], [0, 2])));
    S().setInspectorTab("dependencies");
    S().undoWorkspace();
    expect(S().inspectorTab).toBe("dependencies");
  });

  it("backs out of the inspected task without clearing the tilings", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    const tiles = { ...S().planTiles };
    S().selectPlanTask(null);
    expect(S().planTask).toBeNull();
    expect(S().planTiles).toEqual(tiles);
  });

  it("clears the plan when the graph is replaced, as it clears the selection", () => {
    S().planTaskAt("Y", [0, 0]);
    S().applyDSL(CHAIN.replace("W = Tensor(16, 8)", "W = Tensor(16, 4)"));
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
    expect(S().plan).toBeNull();
  });

  it("lights the operation that computes the inspected task", () => {
    S().planTaskAt("Y", [0, 0]);
    expect(S().selectedOp).toBe(S().resolved!.tensors.Y.producer!.nodeId);
  });

  it("does not record a no-op when the current task is inspected again", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    const task = S().planTask!;
    const history = S().workspaceHistory.length;

    S().selectPlanTask({ tensorId: task.tensorId, coord: [...task.coord] });
    S().planTaskAt("Y", [1, 1]);

    expect(S().workspaceHistory).toHaveLength(history);
    expect(S().planTask).toEqual(task);
  });

  it("restores the task's operation highlight without recording an edit", () => {
    S().setPlanTileAt("Y", [4, 4], [0, 0]);
    const history = S().workspaceHistory.length;
    S().setSelectedOp(null);

    S().planTaskAt("Y", [1, 1]);

    expect(S().workspaceHistory).toHaveLength(history);
    expect(S().selectedOp).toBe(S().resolved!.tensors.Y.producer!.nodeId);
  });
});

describe("the plan view's paint", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
    S().setPlanTileAt("Y", [4, 4], [4, 0]);
    S().setPlanTile("C", [4, 4]);
  });

  const paintOf = (tensorId: string) =>
    buildPlanPaint({
      tensorId,
      rowAxis: 0,
      colAxis: 1,
      dark: false,
      plan: S().plan,
      supply: S().planSupply,
    });

  it("draws the inspected task like a placed tile", () => {
    const { layers } = paintOf("Y");
    expect(layers).toHaveLength(1);
    expect(layers[0]).toMatchObject({ outline: true, seed: true, hatch: false });
    expect(layers[0].region.boxes).toEqual([box([4, 8], [0, 4])]);
  });

  it("fills what the task reads and outlines the producer tiles it needs", () => {
    const { layers, paint } = paintOf("C");
    expect(layers).toHaveLength(1); // the demand, filled as needs
    expect(layers[0].region.boxes).toEqual([box([4, 8], [0, 16])]);
    expect(layers[0].outline).toBeUndefined();
    expect(paint.tiles.map((t) => t.box)).toEqual([
      box([4, 8], [0, 4]),
      box([4, 8], [4, 8]),
      box([4, 8], [8, 12]),
      box([4, 8], [12, 16]),
    ]);
    expect(paint.tiles.every((t) => t.definite)).toBe(true);
    expect(paint.lattice).toEqual({ rows: 4, cols: 4 });
  });

  it("gives a tensor the plan does not tile no lattice", () => {
    const { paint } = paintOf("W"); // a graph input: read, never tiled
    expect(paint.lattice).toBeNull();
    expect(paint.tiles).toEqual([]);
  });

  it("paints one fill for a tensor two slots read, so the overlap is not doubled", () => {
    S().applyDSL(`A = Tensor(8, 8)
X = relu(A)
C = matmul(X, X)
`);
    S().setPlanTileAt("C", [4, 4], [0, 0]);
    S().setPlanTile("X", [4, 4]);
    expect(S().planSupply!.demand).toHaveLength(2); // one per slot
    const { layers } = buildPlanPaint({
      tensorId: "X",
      rowAxis: 0,
      colAxis: 1,
      dark: false,
      plan: S().plan,
      supply: S().planSupply,
    });
    expect(layers).toHaveLength(1);
    expect(layers[0].region.boxes).toEqual([box([0, 4], [0, 8]), box([0, 8], [0, 4])]);
  });
});

describe("what a gesture in the plan view does", () => {
  const setup = (shape: number[]) => {
    const cfg = defaultViewCfg(shape);
    return { cfg, geom: gridGeometry(shape, cfg, 0, 6) };
  };

  it("divides an untiled tensor at the extents drawn, snapped to the lattice", () => {
    const shape = [64, 64];
    const { cfg, geom } = setup(shape);
    const drag = { r0: 1, c0: 1, r1: 20, c1: 40 };
    const gesture = planGesture(shape, cfg, geom, drag, false);
    if (gesture.kind !== "divide") throw new Error(gesture.kind);
    // Whole cells of the drawn lattice, never the raw pointer extents.
    expect(gesture.tile.every((extent) => extent % geom.tile === 0)).toBe(true);
    expect(gesture.element).toEqual([0, 0]);
  });

  it("ignores the snap toggle: a tile is a quantity", () => {
    const shape = [64, 64];
    const { cfg, geom } = setup(shape);
    const drag = { r0: 0, c0: 0, r1: 3, c1: 3 };
    const snapped = planGesture(shape, cfg, geom, drag, false);
    // Same result whatever the workspace toggle says, because it is not read.
    S().setSnapToGrid(false);
    expect(planGesture(shape, cfg, geom, drag, false)).toEqual(snapped);
    S().setSnapToGrid(true);
  });

  it("inspects rather than divides once the tensor has a tiling", () => {
    const shape = [64, 64];
    const { cfg, geom } = setup(shape);
    const drag = { r0: 1, c0: 1, r1: 20, c1: 40 };
    expect(planGesture(shape, cfg, geom, drag, true)).toEqual({
      kind: "inspect",
      element: planElementFromCell(shape, cfg, geom, { row: 1, col: 1 }),
    });
  });

  it("treats a press that did not move as inspecting the tile under it", () => {
    const shape = [64, 64];
    const { cfg, geom } = setup(shape);
    expect(planGesture(shape, cfg, geom, { r0: 5, c0: 7, r1: 5, c1: 7 }, false).kind).toBe("inspect");
  });
});

describe("tile extents as text", () => {
  it("accepts the separators the panel prints and rejects the rest", () => {
    expect(parseTileExtents("64 × 64", 2)).toEqual([64, 64]);
    expect(parseTileExtents("64x64", 2)).toEqual([64, 64]);
    expect(parseTileExtents("64, 32", 2)).toEqual([64, 32]);
    expect(parseTileExtents(" 8 4 ", 2)).toEqual([8, 4]);
    expect(parseTileExtents("64", 2)).toBeNull();
    expect(parseTileExtents("64 × 0", 2)).toBeNull();
    expect(parseTileExtents("64 × -4", 2)).toBeNull();
    expect(parseTileExtents("64 × 1.5", 2)).toBeNull();
    expect(parseTileExtents("", 0)).toEqual([]);
    expect(parseTileExtents("scalar", 0)).toEqual([]);
    expect(parseTileExtents("SCALAR", 0)).toEqual([]);
  });

  it("round-trips the displayed scalar extent", () => {
    expect(parseTileExtents(formatTileExtents([]), 0)).toEqual([]);
  });

  it("defaults a tile to the displayed size on the visible axes and one elsewhere", () => {
    S().applyDSL(CHAIN);
    const resolved = S().resolved!;
    const tile = defaultPlanTile(resolved, "Y", S().tileScale, S().graphPx);
    // Y is 16x8, so the displayed tile clips to the axis rather than exceeding it.
    expect(tile).toEqual([Math.min(16, tile[0]), Math.min(8, tile[1])]);
    expect(tileFamily("Y", resolved.tensors.Y.resolved!, tile).count).toBeGreaterThan(0);

    S().applyDSL(`Q = Tensor(2, 4, 8, 8)
R = relu(Q)
`);
    // Hidden axes get one element each: one task per batch and head index.
    expect(defaultPlanTile(S().resolved!, "R", S().tileScale, S().graphPx).slice(0, 2)).toEqual([1, 1]);
  });
});
