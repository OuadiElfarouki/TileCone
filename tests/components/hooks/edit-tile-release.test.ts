import { beforeEach, describe, expect, it } from "vitest";
import { releaseEditTileOnPointer } from "../../../src/components/hooks/useEditTileRelease";
import { box, fromBox } from "../../../src/core/region";
import { useStore } from "../../../src/state/store";
import { buildLayers } from "../../../src/view/tensor/layers";

const S = () => useStore.getState();

/** A press target inside the given surfaces; a card names its tensor. */
const pressOn = (surfaces: string[], cardTensor?: string) => {
  const card = { getAttribute: (name: string) => (name === "data-tensor-id" ? cardTensor ?? null : null) };
  return {
    target: {
      closest: (selectors: string) => {
        const wanted = selectors.split(", ");
        if (wanted.includes(".card-slot")) return cardTensor !== undefined ? card : null;
        return wanted.some((s) => surfaces.includes(s)) ? {} : null;
      },
    } as unknown as EventTarget,
  };
};

beforeEach(() => {
  S().applyDSL("X = Tensor(8, 8)\nY = relu(X)\n");
  S().setSelection("Y", fromBox(box([0, 2], [0, 2])), "replace");
  S().setSelection("Y", fromBox(box([4, 6], [4, 6])), "union");
});

describe("letting go of the keyboard's tile", () => {
  it("takes the tile just drawn", () => {
    expect(S().editBox).toBe(1);
  });

  it("keeps it for presses on its own card and in the inspector", () => {
    releaseEditTileOnPointer(pressOn([".graph-canvas"], "Y"));
    releaseEditTileOnPointer(pressOn([".inspector"]));
    expect(S().editBox).toBe(1);
  });

  it("leaves the canvas background to the canvas, which releases on a click", () => {
    releaseEditTileOnPointer(pressOn([".graph-canvas"]));
    expect(S().editBox).toBe(1);
  });

  it.each([
    ["another card", pressOn([".graph-canvas"], "X")],
    ["an operation", pressOn([".graph-canvas", ".op-node"])],
    ["the source panel", pressOn([".side-panel"])],
  ])("releases it for a press on %s, leaving the tiles and the pin", (_, press) => {
    S().togglePinBox(0);
    S().setSelection("Y", fromBox(box([4, 6], [4, 6])), "union");
    const selection = S().selection;
    releaseEditTileOnPointer(press);
    expect(S().editBox).toBeNull();
    expect(S().selection).toBe(selection);
  });
});

describe("the keyboard's tile on its card", () => {
  it("is outlined heavier than the other tiles", () => {
    const parts = S().selection!.parts.map((part, index) => ({ index, box: part.box }));
    const layers = buildLayers({
      tensorId: "Y", dark: false, direction: "backward", isSelected: true, parts, partCount: 2,
      perBox: null, hiddenBoxes: new Set(), focusedBox: null, keyTile: 1, back: undefined, fwd: undefined,
      prev: undefined, prevForward: undefined, dragRegion: null, entangled: undefined, showEntangled: false,
    });
    const seeds = layers.filter((layer) => layer.seed);
    expect(seeds.map((layer) => layer.lineWidth)).toEqual([undefined, 2.7]);
    expect(seeds.every((layer) => layer.outline)).toBe(true);
  });
});
