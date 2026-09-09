import { describe, expect, it } from "vitest";
import { box, type Region } from "../core/region";
import { gridGeometry, regionFillRects } from "../ui/grid";
import { decodeWorkspace, encodeWorkspace, type WorkspaceLink } from "../ui/share";
import { useStore } from "../ui/store";
import type { ViewCfg } from "../ui/tensor-view";

describe("hidden-axis coverage", () => {
  it("aggregates projected coverage once, including overlapping boxes and two hidden axes", () => {
    const shape = [2, 3, 4, 4];
    const region: Region = {
      boxes: [
        box([0, 1], [0, 3], [0, 4], [0, 2]),
        box([1, 2], [0, 3], [0, 2], [0, 4]),
        box([0, 2], [1, 2], [1, 4], [1, 4]),
      ], exact: true, reasons: [],
    };
    for (const projection of [true, false]) {
      const cfg = { sliders: [1, 1, 0, 0], projection };
      const geom = gridGeometry(shape, cfg, 0, 20);
      const rects = regionFillRects(region, shape, cfg, geom);
      for (let row = 0; row < 4; row++) for (let col = 0; col < 4; col++) {
        let hits = 0;
        for (let batch = 0; batch < 2; batch++) for (let head = 0; head < 3; head++) {
          if (!projection && (batch !== 1 || head !== 1)) continue;
          const point = [batch, head, row, col];
          if (region.boxes.some((b) => b.every((v, axis) =>
            point[axis] >= v.lo && point[axis] < v.hi))) hits++;
        }
        const x = (col + 0.5) / 4 * geom.canvasW;
        const y = (row + 0.5) / 4 * geom.canvasH;
        const covering = rects.filter((q) => x >= q.x && x < q.x + q.w && y >= q.y && y < q.y + q.h);
        expect(covering.length).toBeLessThanOrEqual(1);
        expect(covering[0]?.alpha ?? 0).toBeCloseTo(hits / (projection ? 6 : 1));
      }
    }
  });
});

const workspace = {
  dsl: "X = Tensor(4, 3, 8, 8)\nY = relu(X)\n",
  direction: "both" as const, tileScale: 0, snapToGrid: true,
  axisMode: "symbolic" as const, parts: null,
};
const view = { projection: false, sliders: [3, 2, 0, 0] };

describe("batch/head view persistence and preview", () => {
  it("round trips and restores the exact slice while retaining defaults for other tensors", () => {
    const link: WorkspaceLink = {
      dsl: workspace.dsl, dir: "both", tile: 0, sel: null, views: { X: view },
    };
    const decoded = decodeWorkspace(`#s=${encodeWorkspace(link)}`)!;
    expect(decoded.views).toEqual({ X: view });
    expect(useStore.getState().restoreWorkspace({ ...workspace, viewCfgs: decoded.views })).toBe(true);
    expect(useStore.getState().viewCfgs.X).toEqual(view);
    expect(useStore.getState().viewCfgs.Y).toEqual({ projection: true, sliders: [0, 0, 0, 0] });
  });

  it("rejects out-of-range, wrong-rank, and unknown-tensor views transactionally", () => {
    useStore.getState().restoreWorkspace(workspace);
    const before = useStore.getState();
    const invalidViews: Record<string, ViewCfg>[] = [
      { X: { projection: false, sliders: [4, 0, 0, 0] } },
      { X: { projection: false, sliders: [0] } },
      { missing: view },
    ];
    for (const viewCfgs of invalidViews) {
      expect(useStore.getState().restoreWorkspace({ ...workspace, viewCfgs })).toBe(false);
      expect(useStore.getState()).toBe(before);
    }
  });

  it("rejects malformed serialized view settings", () => {
    for (const cfg of [null, { projection: "slice", sliders: [0] },
      { projection: true, sliders: [-1] }, { projection: true, sliders: [0.5] }]) {
      const hash = btoa(JSON.stringify({ dsl: workspace.dsl, views: { X: cfg } }));
      expect(decodeWorkspace(`#s=${hash}`)).toBeNull();
    }
  });

  it("clears an old preview and rejects a queued probe from the previous slice", () => {
    useStore.getState().restoreWorkspace(workspace);
    useStore.getState().setViewCfg("X", view);
    const oldView = useStore.getState().viewCfgs.X;
    const tile = box([3, 4], [2, 3], [0, 1], [0, 1]);
    useStore.getState().setPreviewBox("X", tile, oldView);
    expect(useStore.getState().preview).not.toBeNull();
    useStore.getState().setViewCfg("X", { sliders: [2, 2, 0, 0] });
    expect(useStore.getState().preview).toBeNull();
    useStore.getState().setPreviewBox("X", tile, oldView);
    expect(useStore.getState().preview).toBeNull();
    const current = useStore.getState().viewCfgs.X;
    useStore.getState().setPreviewBox("X", box([2, 3], [2, 3], [0, 1], [0, 1]), current);
    expect(useStore.getState().preview).not.toBeNull();
  });
});
