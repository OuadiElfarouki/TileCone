import { describe, expect, it } from "vitest";
import {
  cardSize,
  DEFAULT_DOWNSTREAM_DENSITY,
  selectionBoxFromDrag,
  visibleApproximation,
} from "../ui/TensorCard";
import { box, fromBox } from "../core/region";
import { gridGeometry, stripeAngleDeg } from "../ui/grid";
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

  it("reserves the widest shape reading so changing modes cannot overlap cards", () => {
    const shape = [4, 4];
    const px = graphScale([{ rows: 4, cols: 4 }]);
    const numeric = cardSize(shape, px, "X");
    const labelled = cardSize(shape, px, "X", ["[long_sequence_axis × embedding_projection]"]);
    expect(labelled.w).toBeGreaterThan(numeric.w);
    expect(labelled.h).toBe(numeric.h);
  });
});

describe("drawing through higher-rank tensor views", () => {
  const shape = [3, 5, 16, 24];
  const drag = { r0: 2, c0: 3, r1: 6, c1: 9 };

  it("selects the full hidden-axis union in projection mode", () => {
    const cfg = { sliders: [1, 2, 0, 0], projection: true };
    const geom = gridGeometry(shape, cfg, 0, 8);

    expect(selectionBoxFromDrag(shape, cfg, geom, drag, false)).toEqual(
      box([0, 3], [0, 5], [2, 7], [3, 10])
    );
  });

  it("selects only the displayed hidden-axis slice in slice mode", () => {
    const cfg = { sliders: [1, 2, 0, 0], projection: false };
    const geom = gridGeometry(shape, cfg, 0, 8);

    expect(selectionBoxFromDrag(shape, cfg, geom, drag, false)).toEqual(
      box([1, 2], [2, 3], [2, 7], [3, 10])
    );
  });
});

/* A hover readout and its preview cone are built from the box a click would
   commit, so they are the degenerate case of the same gesture. These pin what
   that box is; the card has no other definition of it to drift towards. */
describe("the box a single-cell gesture commits", () => {
  const shape = [3, 5, 16, 24];
  const hover = (row: number, col: number) => ({ r0: row, c0: col, r1: row, c1: col });

  it("covers whole hidden axes in projection mode, not the slider slice", () => {
    const cfg = { sliders: [1, 2, 0, 0], projection: true };
    const geom = gridGeometry(shape, cfg, 0, 8);

    expect(selectionBoxFromDrag(shape, cfg, geom, hover(6, 9), false)).toEqual(
      box([0, 3], [0, 5], [6, 7], [9, 10])
    );
  });

  it("covers the whole tile under the pointer while snapping", () => {
    const cfg = { sliders: [1, 2, 0, 0], projection: false };
    const geom = gridGeometry(shape, cfg, 0, 8);
    const selected = selectionBoxFromDrag(shape, cfg, geom, hover(6, 9), true);

    expect(selected[2].hi - selected[2].lo).toBe(geom.tile);
    expect(selected[3].hi - selected[3].lo).toBe(geom.tile);
    expect(selected[0]).toEqual({ lo: 1, hi: 2 });
  });
});

/* ---------------- what the canvas paints ---------------- */

import { buildLayers, LayerInputs } from "../ui/TensorCard";
import type { BoxProp } from "../ui/store";

const region = (...pairs: [number, number][]) => fromBox(box(...pairs));

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
  it("fills upstream uniformly and rules downstream", () => {
    const layers = buildLayers(inputs({ direction: "both" }));
    expect(layers).toHaveLength(2);
    expect(layers[0].pattern).toBeUndefined();
    expect(layers[1].pattern).toEqual({
      kind: "stripe",
      density: DEFAULT_DOWNSTREAM_DENSITY,
      angle: stripeAngleDeg(0),
    });
  });

  it("keeps direction encoded even when only one cone is shown", () => {
    expect(buildLayers(inputs({ direction: "backward" }))[0].pattern).toBeUndefined();
    expect(buildLayers(inputs({ direction: "forward" }))[0].pattern?.kind).toBe("stripe");
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

  it("keeps uniform versus ruled encoding after per-box attribution is capped", () => {
    const back = { region: region([0, 4]), depth: 1 };
    const fwd = { region: region([4, 8]), depth: 1 };
    const layers = buildLayers(inputs({ perBox: null, back, fwd, direction: "both" }));
    expect(layers[0].pattern).toBeUndefined();
    expect(layers[1].pattern?.kind).toBe("stripe");
  });

  it("gives each box its own ruling angle, so overlapping cones cross", () => {
    const layers = buildLayers(
      inputs({ perBox: [bothCones("T"), bothCones("T"), bothCones("T")], direction: "forward" })
    );
    const angles = layers.map((l) => l.pattern!.angle);
    expect(new Set(angles).size).toBe(angles.length);
    // Hue runs out before the angles do: every box past the third shares one
    // neutral colour, so the slope has to keep separating them.
    expect(stripeAngleDeg(3)).not.toBe(stripeAngleDeg(2));
  });

  it("keeps every ruling clear of the lattice and of the approximation hatch", () => {
    for (let i = 0; i < 8; i++) {
      const angle = stripeAngleDeg(i);
      // Mod 180: a line at 15 degrees and one at 195 are the same ruling.
      const from = (ref: number) => {
        const d = Math.abs(((angle - ref) % 180 + 180) % 180);
        return Math.min(d, 180 - d);
      };
      expect(from(0), `box ${i} vs the lattice`).toBeGreaterThanOrEqual(15);
      expect(from(90), `box ${i} vs the lattice`).toBeGreaterThanOrEqual(15);
      expect(from(45), `box ${i} vs the hatch`).toBeGreaterThanOrEqual(30);
    }
  });

  it("does not overload downstream alpha with graph depth", () => {
    const shallow = bothCones("T");
    const deep = bothCones("T");
    deep.forward!.tensors.get("T")!.depth = 6;
    const a = buildLayers(inputs({ perBox: [shallow], direction: "forward" }))[0].alpha;
    const b = buildLayers(inputs({ perBox: [deep], direction: "forward" }))[0].alpha;
    expect(b).toBe(a);
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

  it("never brings a downstream perimeter back for focus", () => {
    const layers = buildLayers(inputs({ focusedBox: 0, direction: "forward" }));
    expect(layers[0].pattern?.kind).toBe("stripe");
    expect(layers[0].outline).toBeFalsy();
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

  it("composes the downstream ruling with the approximation hatch", () => {
    const inexact: BoxProp = {
      backward: null,
      forward: {
        tensors: new Map([
          ["T", { region: { boxes: [box([4, 8])], exact: false, reasons: ["conservative"] }, depth: 2 }],
        ]),
        depth: 2,
      } as never,
    };
    const [layer] = buildLayers(inputs({ perBox: [inexact], direction: "forward" }));
    expect(layer.pattern?.kind).toBe("stripe");
    expect(layer.hatch).toBe(true);
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
