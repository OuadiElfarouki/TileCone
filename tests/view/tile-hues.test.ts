import { beforeEach, describe, expect, it } from "vitest";
import { box, fromBox } from "../../src/core/region";
import { useStore } from "../../src/state/store";
import { decodeWorkspace, encodeWorkspace, linkToRestore, workspaceToLink } from "../../src/state/share";
import { boxColor, MAX_DISTINCT_HUES } from "../../src/view/palette";
import { nextHue, partHue, withHues, type SelPart } from "../../src/view/workspace";

const S = () => useStore.getState();
const hues = () => S().selection!.parts.map((part) => part.hue);
const part = (lo: number, hue?: number): SelPart => ({
  tensorId: "C",
  box: box([lo, lo + 1], [0, 1]),
  ...(hue !== undefined ? { hue } : {}),
});

describe("tile colour slots", () => {
  it("has six distinct hues per theme, wrapping past them", () => {
    expect(MAX_DISTINCT_HUES).toBe(6);
    for (const dark of [false, true]) {
      const all = Array.from({ length: 6 }, (_, hue) => boxColor(hue, dark).join());
      expect(new Set(all).size).toBe(6);
      expect(boxColor(6, dark)).toEqual(boxColor(0, dark));
    }
  });

  it("gives a new tile the lowest free slot, then the least used", () => {
    expect(nextHue([])).toBe(0);
    expect(nextHue([part(0, 0), part(1, 2)])).toBe(1);
    const full = Array.from({ length: 6 }, (_, hue) => part(hue, hue));
    expect(nextHue(full)).toBe(0);
    expect(nextHue([...full, part(6, 0)])).toBe(1);
  });

  it("fills in missing slots without touching parts that have one", () => {
    const kept = part(0, 3);
    const filled = withHues([kept, part(1), part(2)]);
    expect(filled[0]).toBe(kept);
    expect(filled.map((p) => p.hue)).toEqual([3, 0, 1]);
    const settled = withHues(filled);
    expect(settled).toBe(filled);
    expect(partHue([part(0)], 0)).toBe(0);
  });
});

describe("colours across tile edits", () => {
  const GEMM = "A = Tensor(16, 32, dtype=float16)\nB = Tensor(32, 16, dtype=float16)\nC = matmul(A, B)\n";
  const draw = (lo: number) => S().setSelection("C", fromBox(box([lo, lo + 2], [0, 2])), "union");

  beforeEach(() => {
    S().applyDSL(GEMM);
    S().setSelection("C", fromBox(box([0, 2], [0, 2])), "replace");
    draw(4);
    draw(8);
  });

  it("keeps every other tile's colour when one is removed, and reuses the gap", () => {
    expect(hues()).toEqual([0, 1, 2]);
    S().deleteBox(1);
    expect(hues()).toEqual([0, 2]);
    draw(12);
    expect(hues()).toEqual([0, 2, 1]);
  });

  it("keeps colours through a move, a grow and an undo", () => {
    S().togglePinBox(2);
    S().moveSelection(1, 1);
    S().growSelection(1, true, true, 2);
    expect(hues()).toEqual([0, 1, 2]);
    S().deleteBox(0);
    S().undoWorkspace();
    expect(hues()).toEqual([0, 1, 2]);
  });

  it("treats what is left of a cut tile as a new tile, and keeps the untouched ones", () => {
    S().setSelection("C", fromBox(box([0, 2], [0, 1])), "subtract");
    const hueAt = (lo: number) => S().selection!.parts.find((p) => p.box[0].lo === lo)!.hue;
    expect(hueAt(4)).toBe(1);
    expect(hueAt(8)).toBe(2);
    // The piece takes the lowest free slot, which the cut tile just gave up.
    expect(S().selection!.parts.find((p) => p.box[0].lo === 0)!.box).toEqual(box([0, 2], [1, 2]));
    expect(hueAt(0)).toBe(0);
  });

  it("travels in a share link, and an older link takes slots in order", () => {
    S().deleteBox(0);
    const link = decodeWorkspace(`#s=${encodeWorkspace(workspaceToLink(S()))}`)!;
    expect(link.sel!.map((p) => p.h)).toEqual([1, 2]);
    expect(S().restoreWorkspace(linkToRestore(link))).toBe(true);
    expect(hues()).toEqual([1, 2]);
    const legacy = { ...link, sel: link.sel!.map(({ t, box: b }) => ({ t, box: b })) };
    expect(S().restoreWorkspace(linkToRestore(legacy))).toBe(true);
    expect(hues()).toEqual([0, 1]);
  });
});
