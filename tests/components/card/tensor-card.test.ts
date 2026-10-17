import { DECK_ROOM_PX } from "../../../src/view/tensor/deck";
import { describe, expect, it } from "vitest";
import { cardSize } from "../../../src/components/card/TensorCard";
import { box, count, fromBox, intersect, unionOf, type Box } from "../../../src/core/region";
import { gridGeometry } from "../../../src/view/tensor/grid";
import { graphScale } from "../../../src/view/tensor/tiling";

describe("plan replay paint", () => {
  /* Three runs of Y. Each reads a band of X wider than the part the inspected
     task also reads, so `region` and `shared` are genuinely different sets.
     The last run reads a different band, so an earlier run's reach left on the
     card would be visible. */
  const reach = (region: Box, shared: Box | null, level = 1) => ({
    region: fromBox(region),
    shared: shared ? fromBox(shared) : null,
    level,
  });
  const frame = (coord: number[], tile: Box, read: Box) => ({
    first: { tensorId: "Y", coord },
    last: { tensorId: "Y", coord },
    tasks: 1,
    firstPath: [{ tensorId: "Y", coord }],
    lastPath: [{ tensorId: "Y", coord }],
    steps: 1,
    region: fromBox(tile),
    walked: new Map(),
    reach: new Map([["X", reach(read, box([0, 2], [0, 4]))]]),
  });
  const playback = (visited = 3): ReplayPlayback => ({
    replay: {
      task: { tensorId: "Y", coord: [0, 0] },
      depth: 1,
      frames: [
        frame([0, 0], box([0, 2], [0, 2]), box([0, 4], [0, 4])),
        frame([0, 1], box([0, 2], [2, 4]), box([0, 4], [0, 4])),
        frame([1, 0], box([2, 4], [0, 2]), box([0, 2], [4, 8])),
      ],
      tasks: 3,
      steps: 3,
      stepTensors: ["Y"],
    },
    plan: {} as ReplayPlayback["plan"],
    visited,
    paused: false,
    exiting: false,
    opacity: 1,
  });

  it("keeps every walked run as a trail on the family's tensor, the last one active", () => {
    const paint = buildReplayPaint({ tensorId: "Y", dark: false, playback: playback() });
    // The walked runs are one trail, each tile once; the active run on top.
    expect(paint.layers.map((layer) => layer.region.boxes)).toEqual([
      [box([0, 2], [0, 2]), box([0, 2], [2, 4])],
      [box([2, 4], [0, 2])],
    ]);
    expect(paint.layers.map((layer) => layer.alpha)).toEqual([0.1, 0.68]);
    expect(paint.layers[1].seed).toBe(true);
    // Not yet walked is not painted.
    expect(buildReplayPaint({ tensorId: "Y", dark: false, playback: playback(1) }).layers).toHaveLength(1);
  });

  /* A run paints where its demand lands as well as where it meets the
     inspected task's: a family whose runs mostly share nothing would otherwise
     paint nothing. */
  it("paints the active run's whole reach, with the part the inspected task shares on top", () => {
    const active = buildReplayPaint({ tensorId: "X", dark: false, playback: playback(2) });
    const [rest, shared] = active.layers;
    expect(rest.region.boxes).toEqual([box([2, 4], [0, 4])]); // reach minus the meeting
    expect(rest.alpha).toBe(0.26);
    // Outlined, so a read too thin for its fill to show still does.
    expect(rest).toMatchObject({ outline: true, lineWidth: 1 });
    expect(shared.region.boxes).toEqual([box([0, 2], [0, 4])]);
    expect(shared.alpha).toBe(0.62);
    expect(shared.seed).toBe(true);
  });

  /* What earlier runs read is not left behind on the other cards: only the
     active run is drawn there, so nothing accumulates into a stale summary. */
  it("leaves no trace of earlier runs on the cards they read", () => {
    const paint = buildReplayPaint({ tensorId: "X", dark: false, playback: playback(3) });
    const painted = unionOf(paint.layers.map((layer) => layer.region));
    expect(count(painted)).toBe(count(fromBox(box([0, 2], [4, 8]))) + count(fromBox(box([0, 2], [0, 4]))));
    expect(count(intersect(painted, fromBox(box([2, 4], [0, 4]))))).toBe(0);
  });

  it("fades the trail as it stands", () => {
    const fading = { ...playback(), exiting: true, opacity: 0.5 };
    const paint = buildReplayPaint({ tensorId: "Y", dark: false, playback: fading });
    expect(paint.layers.map((layer) => layer.alpha)).toEqual([0.05, 0.34]);
  });

  /* What a run reads is the Plan view's solid fill: a ruling or a stipple
     would read as a cone the plan does not draw. */
  it("paints what a run reads in the plan's solid fill", () => {
    const needs = buildReplayPaint({ tensorId: "X", dark: false, playback: playback() });
    expect(needs.layers.every((layer) => layer.pattern === undefined)).toBe(true);
  });

  /* The plan's own lattice is already on every tiled card. */
  it("draws no lattice of its own", () => {
    const studied = buildReplayPaint({ tensorId: "Y", dark: false, playback: playback() });
    const input = buildReplayPaint({ tensorId: "X", dark: false, playback: playback() });
    expect(studied).not.toHaveProperty("lattice");
    expect(input).not.toHaveProperty("lattice");
  });

  /* Deeper than 1 a step is a producer task, and it walks its own tensor the
     way depth 1 walks the family's: earlier steps as a trail, this one strong. */
  it("draws the producer tasks a deeper replay steps through as a walk", () => {
    const stepped = playback(3);
    stepped.replay.frames.forEach((f, i) =>
      Object.assign(f, { walked: new Map([["X", fromBox(box([i * 2, i * 2 + 2], [0, 2]))]]) }));
    const paint = buildReplayPaint({ tensorId: "X", dark: false, playback: stepped });
    expect(paint.layers[0]).toMatchObject({ alpha: 0.1 });
    expect(paint.layers[0].region.boxes).toEqual([box([0, 2], [0, 2]), box([2, 4], [0, 2])]);
    const step = paint.layers[paint.layers.length - 1];
    expect(step).toMatchObject({ alpha: 0.68, seed: true });
    expect(step.region.boxes).toEqual([box([4, 6], [0, 2])]);
  });

  /* A level reached through the plan's producers is drawn fainter, one step
     per level, and a tiled tensor outlines the producer tiles the run needs. */
  it("draws deeper levels fainter, and outlines the producer tiles a run needs", () => {
    const deep = playback(2);
    const reached = deep.replay.frames[1].reach.get("X")!;
    Object.assign(reached, { level: 2, producers: fromBox(box([0, 4], [0, 4])) });
    const paint = buildReplayPaint({ tensorId: "X", dark: false, playback: deep });
    expect(paint.layers.map((layer) => layer.alpha)).toEqual([0.26 * levelStrength(2), 0.62 * levelStrength(2)]);
    expect(paint.tiles).toEqual([{ box: box([0, 4], [0, 4]), definite: true }]);
    // Leaving, the run's outlines go first so the plan's own can return.
    expect(buildReplayPaint({ tensorId: "X", dark: false, playback: { ...deep, exiting: true } }).tiles).toEqual([]);
    expect(levelStrength(1)).toBe(1);
    expect(levelStrength(9)).toBe(0.35);
  });

  it("hatches a run whose demand was widened", () => {
    const widened = playback(2);
    const reached = widened.replay.frames[1].reach.get("X")!;
    reached.shared = { ...reached.shared!, exact: false, reasons: ["strided fallback"] };
    const paint = buildReplayPaint({ tensorId: "X", dark: false, playback: widened });
    expect(paint.layers[1].hatch).toBe(true);
    expect(paint.layers[1].region.reasons).toEqual(["strided fallback"]);
  });
});

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

    expect(selected[2].hi - selected[2].lo).toBe(geom.rowTile);
    expect(selected[3].hi - selected[3].lo).toBe(geom.colTile);
    expect(selected[0]).toEqual({ lo: 1, hi: 2 });
  });
});

/* ---------------- what the canvas paints ---------------- */

import { ReplayPlayback, MAX_PER_BOX_PROPS, BoxProp } from "../../../src/view/workspace";
import { buildReplayPaint, levelStrength } from "../../../src/view/tensor/replay-paint";
import { DEFAULT_DOWNSTREAM_DENSITY, stripeAngleDeg, buildLayers, LayerInputs } from "../../../src/view/tensor/layers";
import { selectionBoxFromDrag, visibleApproximation } from "../../../src/view/tensor/gesture";

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
    const perBox = Array.from({ length: MAX_PER_BOX_PROPS }, () => bothCones("T"));
    const layers = buildLayers(inputs({ perBox, direction: "forward" }));
    const angles = layers.map((l) =>
      l.pattern?.kind === "stripe" ? l.pattern.angle : undefined
    );
    expect(new Set(angles).size).toBe(angles.length);
  });

  it("keeps every ruling clear of the lattice and of the approximation hatch", () => {
    for (let i = 0; i < MAX_PER_BOX_PROPS; i++) {
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

  it("does not overload persistent cone alpha with graph depth", () => {
    const shallow = bothCones("T");
    const deep = bothCones("T");
    deep.backward!.tensors.get("T")!.depth = 6;
    deep.forward!.tensors.get("T")!.depth = 6;
    for (const direction of ["backward", "forward"] as const) {
      const a = buildLayers(inputs({ perBox: [shallow], direction }))[0].alpha;
      const b = buildLayers(inputs({ perBox: [deep], direction }))[0].alpha;
      expect(b).toBe(a);
    }
  });

  it("keeps depth fading only on the transient hover preview", () => {
    const shallow = { region: region([0, 4]), depth: 1 };
    const deep = { ...shallow, depth: 6 };
    const a = buildLayers(inputs({ perBox: null, prev: shallow, direction: "backward" }))[0].alpha;
    const b = buildLayers(inputs({ perBox: null, prev: deep, direction: "backward" }))[0].alpha;
    expect(b).toBeLessThan(a);
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

  it("reports an inexact entanglement region when it is the only highlight", () => {
    const result = visibleApproximation(undefined, undefined, {
      boxes: [box([0, 4])],
      exact: false,
      reasons: ["composed from forward and backward"],
    });
    expect(result).toEqual({
      approximate: true,
      reasons: ["composed from forward and backward"],
    });
  });
});


describe("hover direction filtering", () => {
  it("shows both preview treatments, each direction alone, and neither in figures-only", () => {
    const preview = { region: region([0, 4]), depth: 1 };
    const args = inputs({ perBox: null, prev: preview, prevForward: preview });
    const both = buildLayers(args);
    expect(both).toHaveLength(2);
    expect(both[0].pattern).toBeUndefined();
    expect(both[1].pattern?.kind).toBe("stripe");
    expect(buildLayers({ ...args, direction: "backward" })).toEqual([both[0]]);
    expect(buildLayers({ ...args, direction: "forward" })).toEqual([both[1]]);
    expect(buildLayers({ ...args, direction: "none" })).toEqual([]);
  });
});

describe("where a card's grid sits", () => {
  it("is centred under the header on the card's bottom edge, beside a slice deck's room", () => {
    const px = 2;
    const wide = cardSize([8], px, "normalize_H$mean", ["[S × 1]"]);
    expect(wide.port.w).toBeLessThan(wide.w);
    expect(wide.port.x).toBeCloseTo((wide.w - wide.port.w) / 2, 6);
    expect(wide.port.y + wide.port.h).toBe(wide.h);
    const sliced = cardSize([4, 16, 16], px, "X", [], { projection: false, sliders: [0, 0, 0] } as never);
    expect(sliced.port.x).toBeCloseTo((sliced.w - sliced.port.w - DECK_ROOM_PX) / 2, 6);
    expect(sliced.port.y + sliced.port.h).toBe(sliced.h);
  });
});
