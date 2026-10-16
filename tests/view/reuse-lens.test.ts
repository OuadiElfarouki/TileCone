import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { box, fromBox } from "../../src/core/region";
import { operandReads, operandTallyOf } from "../../src/core/multiplicity";
import { measureTally } from "../../src/core/tally";
import { compileDSL } from "../../src/parse/compiler";
import { executeQuery } from "../../src/core/executor";
import { drawGrid } from "../../src/components/card/draw-grid";
import { Inspector } from "../../src/components/inspector/Inspector";
import { useStore } from "../../src/state/store";
import { gridGeometry } from "../../src/view/tensor/grid";
import { buildLayers, composeCardLayers } from "../../src/view/tensor/layers";
import {
  lensColor,
  lensPaint,
  lensRamp,
  lensReading,
  lensReadingText,
  lensRegions,
  type ReuseLens,
} from "../../src/view/reuse-lens";

// Static renders read the live test store, as the other inspector contracts do.
vi.mock("../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const S = () => useStore.getState();
const GEMM = "A = Tensor(16, 32, dtype=fp16)\nB = Tensor(32, 16, dtype=fp16)\nC = matmul(A, B)\n";
const SHARED = "A = Tensor(4, 4)\nC = matmul(A, A)\n";
const coneOf = (dsl: string, tensorId: string, b: ReturnType<typeof box>) => {
  const { resolved } = compileDSL(dsl);
  const cone = executeQuery(resolved, { tensorId, region: fromBox(b), direction: "backward" }).backward!;
  return { resolved, regions: cone.tensors };
};

describe("the lens and its row count the same thing", () => {
  it("draws the tally the row's figures measure", () => {
    const { resolved, regions } = coneOf(SHARED, "C", box([0, 2], [0, 2]));
    const [row] = operandReads(resolved, regions);
    const tally = operandTallyOf(resolved, regions, row.nodeId, "A")!;
    const m = measureTally(tally);
    expect([m.min, m.max, m.total]).toEqual([row.min.value, row.max.value, row.reads.value]);
  });

  it("is null for an operation or operand the scope does not have", () => {
    const { resolved, regions } = coneOf(GEMM, "C", box([0, 4], [0, 4]));
    expect(operandTallyOf(resolved, regions, "nope", "A")).toBeNull();
    expect(operandTallyOf(resolved, regions, resolved.topo[0].id, "C")).toBeNull();
  });
});

describe("what the lens draws", () => {
  it("is one uniform band for a GEMM tile: every element of A used by each column", () => {
    const { resolved, regions } = coneOf(GEMM, "C", box([0, 4], [0, 8]));
    const paint = lensPaint(resolved, regions, { view: "dependencies", nodeId: resolved.topo[0].id, tensorId: "A" })!;
    expect(paint).toMatchObject({ complete: true, min: 8, max: 8, exact: true });
    expect(new Set(paint.pieces.map((p) => p.count))).toEqual(new Set([8]));
  });

  it("shows where counts differ: the shared corner of A @ A is used twice as often", () => {
    const { resolved, regions } = coneOf(SHARED, "C", box([0, 2], [0, 2]));
    const paint = lensPaint(resolved, regions, { view: "dependencies", nodeId: resolved.topo[0].id, tensorId: "A" })!;
    expect([paint.min, paint.max]).toEqual([2, 4]);
    expect(lensReading(paint, box([0, 1], [0, 1]))).toEqual({ lo: 4, hi: 4 });
    expect(lensReading(paint, box([0, 1], [3, 4]))).toEqual({ lo: 2, hi: 2 });
    // A window reaching unread elements has a low end of zero.
    expect(lensReading(paint, box([3, 4], [3, 4]))).toBeNull();
    expect(lensReading(paint, box([0, 4], [3, 4]))).toEqual({ lo: 0, hi: 2 });
  });

  it("states a reading as uses, a range, or not read", () => {
    expect(lensReadingText({ lo: 1, hi: 1 })).toBe("1 use");
    expect(lensReadingText({ lo: 4, hi: 4 })).toBe("4 uses");
    expect(lensReadingText({ lo: 0, hi: 2 })).toBe("0–2 uses");
    expect(lensReadingText(null)).toBe("not read");
  });
});

describe("the lens colour", () => {
  it("runs from zero to the largest count along the validated ramp, in both themes", () => {
    for (const dark of [false, true]) {
      const ramp = lensRamp(dark);
      expect(lensColor(8, 8, dark)).toEqual(ramp[ramp.length - 1]);
      expect(lensColor(0, 8, dark)).toEqual(ramp[0]);
      // Half the count sits at the middle step, not at the bottom.
      expect(lensColor(4, 8, dark)).toEqual(ramp[2]);
    }
  });
});

/** A canvas context that records the colour of each filled rectangle. */
function heatCanvas() {
  const fills: { style: string; rect: number[] }[] = [];
  const ctx = new Proxy({ fillStyle: "", strokeStyle: "", lineWidth: 1, globalAlpha: 1 } as Record<string, unknown>, {
    get(target, key: string) {
      if (key in target) return target[key];
      if (key === "fillRect") return (...rect: number[]) => fills.push({ style: String(target.fillStyle), rect });
      return () => {};
    },
    set(target, key: string, value) { target[key] = value; return true; },
  });
  const canvas = { width: 0, height: 0, getContext: () => ctx } as unknown as HTMLCanvasElement;
  return { canvas, fills };
}

describe("the heat pass", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("paints each piece opaque in its colour, lowest count first", () => {
    vi.stubGlobal("window", { devicePixelRatio: 1 });
    const shape = [4, 4];
    const cfg = { sliders: [], projection: true };
    const geom = gridGeometry(shape, cfg, 0, 4);
    const { canvas, fills } = heatCanvas();
    drawGrid(canvas, shape, cfg, geom, [], false, 1, 1, undefined, undefined, {
      pieces: [
        { box: box([0, 4], [0, 4]), color: [1, 2, 3] },
        { box: box([0, 1], [0, 1]), color: [4, 5, 6] },
      ],
      hatch: false,
    });
    const heat = fills.filter((f) => f.style.startsWith("rgb("));
    expect(heat.map((f) => f.style)).toEqual(["rgb(1,2,3)", "rgb(4,5,6)"]);
  });
});

/* Counts are colour, so nothing may composite over them on the lens card. */
describe("a card's layers under the lens", () => {
  const parts = [{ index: 0, box: box([0, 2], [0, 2]) }, { index: 1, box: box([2, 4], [2, 4]) }];
  const dragRegion = fromBox(box([1, 3], [1, 3]));
  const cone = { region: fromBox(box([0, 4], [0, 4])), depth: 1 };
  const layers = () => buildLayers({
    tensorId: "A", dark: false, direction: "backward", isSelected: false, parts, partCount: 2,
    perBox: null, hiddenBoxes: new Set(), focusedBox: null, keyTile: 1, back: cone, fwd: undefined,
    prev: undefined, prevForward: undefined, dragRegion, entangled: undefined, showEntangled: false,
  });
  const replay = [{ region: fromBox(box([0, 4], [0, 4])), color: [1, 2, 3] as [number, number, number], alpha: 0.6, hatch: false }];

  it("keeps everything, replay included, when the lens is not on the card", () => {
    const base = layers();
    expect(composeCardLayers({ layers: base, replay, lens: false, gesture: dragRegion })).toEqual([...base, ...replay]);
  });

  it("keeps only the tiles' and the gesture's outlines and corner marks, and no replay", () => {
    const out = composeCardLayers({ layers: layers(), replay, lens: true, gesture: dragRegion });
    expect(out.map((layer) => layer.region)).toEqual([fromBox(parts[0].box), fromBox(parts[1].box), dragRegion]);
    expect(out.every((layer) => layer.alpha === 0 && !layer.hatch && !layer.pattern)).toBe(true);
    expect(out.every((layer) => layer.outline)).toBe(true);
    expect(out.filter((layer) => layer.seed)).toHaveLength(2);
    expect(out[1].lineWidth).toBe(2.7);
  });
});

describe("the lens in the workspace", () => {
  beforeEach(() => {
    S().applyDSL(SHARED);
    S().setInspectorTab("dependencies");
    S().setSelection("C", fromBox(box([0, 2], [0, 2])), "replace");
  });
  const lensOnA = (): ReuseLens => ({ view: "dependencies", nodeId: S().resolved!.topo[0].id, tensorId: "A" });
  const render = () => renderToStaticMarkup(createElement(Inspector));

  it("scopes its regions by the inspector's rules", () => {
    const regions = lensRegions("dependencies", S())!;
    expect(regions.get("A")).toBeDefined();
    expect(lensRegions("plan", S())).toBeNull();
  });

  it("scopes a Plan lens to what the inspected task computes", () => {
    S().applyDSL(GEMM);
    S().setInspectorTab("plan");
    S().setPlanTileAt("C", [4, 8], [0, 0]);
    const regions = lensRegions("plan", S())!;
    expect(regions).toBe(S().planSupply!.work);
    const paint = lensPaint(S().resolved!, regions, { view: "plan", nodeId: S().resolved!.topo[0].id, tensorId: "A" })!;
    // A task of 4 x 8 outputs uses each element of its A band once per column.
    expect([paint.min, paint.max]).toEqual([8, 8]);
  });

  it("offers each reused row as a switch, and shows the legend for the one that is on", () => {
    expect(render()).toMatch(/<button class="reads-figures lens-toggle" aria-pressed="false"/);
    S().setReuseLens(lensOnA());
    const html = render();
    expect(html).toMatch(/<li class="lens-on">/);
    expect(html).toMatch(/aria-pressed="true"/);
    expect(html).toContain("0 → 4 uses per element");
    expect(html).toContain("Shaded on A&#x27;s card");
  });

  it("keeps the shaded row listed, with its legend, once its count falls to one", () => {
    S().applyDSL(GEMM);
    S().setSelection("C", fromBox(box([0, 4], [0, 8])), "replace");
    S().setReuseLens({ view: "dependencies", nodeId: S().resolved!.topo[0].id, tensorId: "A" });
    S().setSelection("C", fromBox(box([0, 1], [0, 1])), "replace");
    const html = render();
    expect(html).toMatch(/<li class="lens-on"><code>A<\/code>/);
    expect(html).toContain("1 use per element, wherever it is read");
    // The other operand, not shaded, still joins the single-use line.
    expect(html).toContain("Used once: B → C");
  });

  it("is taken down by leaving its view and by replacing the graph", () => {
    S().setReuseLens(lensOnA());
    S().setInspectorTab("dependencies");
    expect(S().reuseLens).not.toBeNull();
    S().setInspectorTab("plan");
    expect(S().reuseLens).toBeNull();
    S().setInspectorTab("dependencies");
    S().setReuseLens(lensOnA());
    S().applyDSL(GEMM);
    expect(S().reuseLens).toBeNull();
  });
});
