import { describe, expect, it } from "vitest";
import { cardSize, selectionBoxFromDrag, visibleApproximation } from "../ui/TensorCard";
import { gridGeometry } from "../ui/grid";
import { graphScale } from "../ui/tiling";

describe("frameless tensor footprint", () => {
  it("reserves collision width for the visible name and numeric shape", () => {
    const shape = [4, 4];
    const px = graphScale([{ rows: 4, cols: 4 }]);
    const short = cardSize(shape, px, "X");
    const long = cardSize(shape, px, "attention_output_projection");
    expect(long.w).toBeGreaterThan(short.w);
    expect(long.h).toBe(short.h);
  });

  it("keeps the label and grid inside the solid collision height", () => {
    const shape = [16, 16];
    const px = graphScale([{ rows: 16, cols: 16 }]);
    const size = cardSize(shape, px, "X");
    expect(size.h).toBeGreaterThan(16 * px); // numeric label sits above the grid
  });
});

describe("drawing through higher-rank tensor views", () => {
  const shape = [3, 5, 16, 24];
  const drag = { r0: 2, c0: 3, r1: 6, c1: 9 };

  it("selects the full hidden-axis union in projection mode", () => {
    const cfg = { sliders: [1, 2, 0, 0], projection: true };
    const geom = gridGeometry(shape, cfg, 0, 8);

    expect(selectionBoxFromDrag(shape, cfg, geom, drag, false)).toEqual([
      [0, 3],
      [0, 5],
      [2, 7],
      [3, 10],
    ]);
  });

  it("selects only the displayed hidden-axis slice in slice mode", () => {
    const cfg = { sliders: [1, 2, 0, 0], projection: false };
    const geom = gridGeometry(shape, cfg, 0, 8);

    expect(selectionBoxFromDrag(shape, cfg, geom, drag, false)).toEqual([
      [1, 2],
      [2, 3],
      [2, 7],
      [3, 10],
    ]);
  });
});

/* ---------------- what the canvas paints ---------------- */

import { buildLayers, LayerInputs } from "../ui/TensorCard";
import { box, fromBox } from "../core/region";
import type { BoxProp } from "../ui/store";

const region = (...iv: [number, number][]) => fromBox(iv.map(([lo, hi]) => ({ lo, hi })));

/** A per-box result placing `id` in both cones at depth 1. */
const bothCones = (id: string): BoxProp => ({
  backward: { tensors: new Map([[id, { region: region([0, 4]), depth: 1 }]]), depth: 1 } as never,
  forward: { tensors: new Map([[id, { region: region([4, 8]), depth: 1 }]]), depth: 1 } as never,
});

const inputs = (over: Partial<LayerInputs> = {}): LayerInputs => ({
  tensorId: "T",
  dark: true,
  direction: "both",
  isSelected: false,
  parts: [],
  partCount: 0,
  perBox: [bothCones("T")],
  hiddenBoxes: new Set<number>(),
  focusedBox: null,
  dragRegion: null,
  ...over,
});

describe("direction stays readable once hue means box identity", () => {
  it("fills upstream and outlines downstream when both cones are shown", () => {
    const layers = buildLayers(inputs({ direction: "both" }));
    expect(layers).toHaveLength(2);
    expect(layers[0].strokeOnly).toBeFalsy(); // upstream: filled
    expect(layers[1].strokeOnly).toBe(true); // downstream: outlined
  });

  it("fills both when only one direction is on, since nothing needs telling apart", () => {
    for (const direction of ["backward", "forward"] as const) {
      const layers = buildLayers(inputs({ direction }));
      expect(layers.every((l) => !l.strokeOnly), direction).toBe(true);
    }
  });

  it("paints only the cone the direction asks for, though both were analysed", () => {
    // upstream sits at [0,4) and downstream at [4,8) in this fixture, so which
    // cone was painted is readable from the layer itself
    const upstream = buildLayers(inputs({ direction: "backward" }));
    expect(upstream).toHaveLength(1);
    expect(upstream[0].region.boxes).toEqual(region([0, 4]).boxes);

    const downstream = buildLayers(inputs({ direction: "forward" }));
    expect(downstream).toHaveLength(1);
    expect(downstream[0].region.boxes).toEqual(region([4, 8]).boxes);

    expect(buildLayers(inputs({ direction: "none" }))).toEqual([]);
  });

  it("paints nothing in either mode once attribution is capped", () => {
    const back = { region: region([0, 4]), depth: 1 };
    const fwd = { region: region([4, 8]), depth: 1 };
    expect(buildLayers(inputs({ perBox: null, back, fwd, direction: "none" }))).toEqual([]);
    expect(buildLayers(inputs({ perBox: null, back, fwd, direction: "backward" }))).toHaveLength(1);
  });

  it("gives the two cones the same hue, so hue is left to mean the box", () => {
    const layers = buildLayers(inputs({ direction: "both" }));
    expect(layers[0].color).toEqual(layers[1].color);
  });

  it("keeps fill versus outline encoding after per-box attribution is capped", () => {
    const back = { region: region([0, 4]), depth: 1 };
    const fwd = { region: region([4, 8]), depth: 1 };
    const layers = buildLayers(inputs({ perBox: null, back, fwd, direction: "both" }));
    expect(layers[0].strokeOnly).toBeFalsy();
    expect(layers[1].strokeOnly).toBe(true);
  });
});

describe("focus fades peers and never removes them", () => {
  const twoBoxes = () => inputs({ perBox: [bothCones("T"), bothCones("T")], direction: "backward" });

  it("keeps every box painted when one is focused", () => {
    const all = buildLayers(twoBoxes());
    const focused = buildLayers({ ...twoBoxes(), focusedBox: 0 });
    expect(focused).toHaveLength(all.length);
  });

  it("dims the unfocused box rather than dropping it", () => {
    // the emphasised layer is appended last so it paints over its peers
    const layers = buildLayers({ ...twoBoxes(), focusedBox: 0 });
    const emphasised = layers[layers.length - 1];
    const peer = layers[0];
    expect(peer.alpha).toBeLessThan(emphasised.alpha);
    expect(peer.alpha).toBeGreaterThan(0); // faded, still legible
  });

  it("emphasises the focused box and draws it last", () => {
    const layers = buildLayers({ ...twoBoxes(), focusedBox: 1 });
    const emphasised = layers[layers.length - 1];
    expect(emphasised.outline).toBe(true);
    expect(emphasised.lineWidth).toBeGreaterThan(1.5);
  });
});

describe("hiding removes paint but nothing else", () => {
  it("drops a hidden box's cone", () => {
    const two = inputs({ perBox: [bothCones("T"), bothCones("T")], direction: "backward" });
    const visible = buildLayers(two);
    const hidden = buildLayers({ ...two, hiddenBoxes: new Set([0]) });
    expect(hidden.length).toBeLessThan(visible.length);
  });

  it("keeps a hidden box's own rectangle so it can still be found", () => {
    const layers = buildLayers(
      inputs({
        isSelected: true,
        parts: [{ index: 0, box: box([0, 4]) }],
        partCount: 1,
        perBox: null,
        hiddenBoxes: new Set([0]),
      })
    );
    expect(layers.length).toBeGreaterThan(0);
    expect(layers[layers.length - 1].alpha).toBeGreaterThan(0);
  });
});

describe("inexact regions are hatched", () => {
  it("marks an over-approximation so it cannot read as ground truth", () => {
    const inexact: BoxProp = {
      backward: {
        tensors: new Map([
          ["T", { region: { boxes: [box([0, 4])], exact: false, reasons: ["strided conv"] }, depth: 1 }],
        ]),
        depth: 1,
      } as never,
      forward: null,
    };
    const layers = buildLayers(inputs({ perBox: [inexact], direction: "backward" }));
    expect(layers[0].hatch).toBe(true);
  });

  it("reports an inexact forward cone even when the backward cone is exact", () => {
    const result = visibleApproximation(
      region([0, 4]),
      { boxes: [box([4, 8])], exact: false, reasons: ["conservative forward map"] }
    );
    expect(result).toEqual({
      approximate: true,
      reasons: ["conservative forward map"],
    });
  });
});
