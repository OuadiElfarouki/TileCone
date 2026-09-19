import { beforeEach, describe, expect, it } from "vitest";
import { tileFamily } from "../core/plan/tile-family";
import { box } from "../core/region";
import { useStore } from "../ui/store";
import { buildPlanPaint, planElementFromCell } from "../ui/TensorCard";
import { defaultPlanTile, formatTileExtents, parseTileExtents } from "../ui/PlanPanel";
import { gridGeometry } from "../ui/grid";
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

  it("tiles a produced tensor on first click and inspects the task under the pointer", () => {
    S().planTaskAt("Y", [9, 2], [4, 4]);
    expect(S().planTiles).toEqual({ Y: [4, 4] });
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [2, 0] });
    expect(S().plan!.families.get("Y")!.count).toBe(8);
    // The task's producers are named even though C is not tiled yet.
    expect(S().planSupply!.demand.map((d) => [d.tensorId, d.supplier])).toEqual([
      ["C", "unplanned"],
      ["W", "input"],
    ]);
    expect(S().planSupply!.complete).toBe(false);
  });

  it("ignores a graph input, which no task produces", () => {
    S().planTaskAt("A", [0, 0], [4, 4]);
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
  });

  it("names producer tasks once both sides are tiled", () => {
    S().planTaskAt("Y", [4, 0], [4, 4]);
    S().setPlanTile("C", [4, 4]);
    const supply = S().planSupply!;
    expect(supply.complete).toBe(true);
    expect(supply.producers.map((p) => p.task.coord.join(","))).toEqual(["1,0", "1,1", "1,2", "1,3"]);
    expect(supply.producers.every((p) => p.definite)).toBe(true);
  });

  it("keeps the task on the same elements when its tensor is retiled", () => {
    S().planTaskAt("Y", [9, 2], [4, 4]); // tile (2, 0) covers rows 8-12
    S().setPlanTile("Y", [8, 8]);
    // Rows 8-12 now lie in tile (1, 0), which covers rows 8-16.
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [1, 0] });
  });

  it("drops the task when its tensor stops being tiled", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    S().setPlanTile("Y", null);
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
    expect(S().planSupply).toBeNull();
  });

  it("steps the task by tiles and stops at the edge of the grid", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    S().movePlanTask(0, 1);
    expect(S().planTask!.coord).toEqual([1, 0]);
    S().movePlanTask(0, 8);
    expect(S().planTask!.coord).toEqual([3, 0]); // Y has four row tiles
    S().movePlanTask(0, -8);
    expect(S().planTask!.coord).toEqual([0, 0]);
  });

  it("refuses a task the plan does not contain", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    S().selectPlanTask({ tensorId: "C", coord: [0, 0] }); // C is not tiled
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 0] });
    S().selectPlanTask({ tensorId: "Y", coord: [9, 9] }); // outside the grid
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [0, 0] });
  });

  it("recomputes what a task reads when the tiling changes", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
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
    S().planTaskAt("Y", [0, 0], [4, 4]);
    S().setPlanTile("C", [4, 4]);
    expect(Object.keys(S().planTiles).sort()).toEqual(["C", "Y"]);
    S().undoWorkspace();
    expect(Object.keys(S().planTiles)).toEqual(["Y"]);
    S().undoWorkspace();
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
  });

  it("clears the plan when the graph is replaced, as it clears the selection", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    S().applyDSL(CHAIN.replace("W = Tensor(16, 8)", "W = Tensor(16, 4)"));
    expect(S().planTiles).toEqual({});
    expect(S().planTask).toBeNull();
    expect(S().plan).toBeNull();
  });

  it("lights the operation that computes the inspected task", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    expect(S().selectedOp).toBe(S().resolved!.tensors.Y.producer!.nodeId);
  });

  it("does not record a no-op when the current task is inspected again", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    const task = S().planTask!;
    const history = S().workspaceHistory.length;

    S().selectPlanTask({ tensorId: task.tensorId, coord: [...task.coord] });
    S().planTaskAt("Y", [1, 1], [4, 4]);

    expect(S().workspaceHistory).toHaveLength(history);
    expect(S().planTask).toEqual(task);
  });

  it("restores the task's operation highlight without recording an edit", () => {
    S().planTaskAt("Y", [0, 0], [4, 4]);
    const history = S().workspaceHistory.length;
    S().setSelectedOp(null);

    S().planTaskAt("Y", [1, 1], [4, 4]);

    expect(S().workspaceHistory).toHaveLength(history);
    expect(S().selectedOp).toBe(S().resolved!.tensors.Y.producer!.nodeId);
  });
});

describe("the plan view's paint", () => {
  beforeEach(() => {
    S().applyDSL(CHAIN);
    S().planTaskAt("Y", [4, 0], [4, 4]);
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
    S().planTaskAt("C", [0, 0], [4, 4]);
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
    expect(defaultPlanTile([2, 64, 100], 1, 2, 32)).toEqual([1, 32, 32]);
    // Never wider than the axis: a short tensor gets its own extent.
    expect(defaultPlanTile([4, 4], 0, 1, 32)).toEqual([4, 4]);
    expect(tileFamily("T", [4, 4], defaultPlanTile([4, 4], 0, 1, 32)).count).toBe(1);
  });
});
