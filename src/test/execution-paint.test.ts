import { describe, expect, it } from "vitest";
import { box, count, fromBox, type Region } from "../core/region";
import { playbackDifference } from "../ui/execution-paint";

describe("bounded playback differences", () => {
  it("subtracts exact overlap and reuses geometry on subsequent paints", () => {
    const reach = fromBox(box([0, 8], [0, 8]));
    const shared = fromBox(box([0, 4], [0, 8]));
    const rest = playbackDifference(reach, shared);
    expect(count(rest)).toBe(32);
    expect(rest.exact).toBe(true);
    expect(playbackDifference(reach, shared)).toBe(rest);
  });

  it("retains all reach when the work budget is exceeded", () => {
    const reach = fromBox(box([0, 8], [0, 8]));
    const shared: Region = {
      boxes: Array.from({ length: 3000 }, (_, i) => box([20 + i, 21 + i], [0, 8])),
      exact: true, reasons: [],
    };
    const rest = playbackDifference(reach, shared);
    expect(rest.boxes).toBe(reach.boxes);
    expect(rest.exact).toBe(false);
    expect(rest.reasons).toContain("playback subtraction budget");
    expect(playbackDifference(reach, shared)).toBe(rest);
  });

  it("does not erase reach using an approximate overlap", () => {
    const reach = fromBox(box([0, 8], [0, 8]));
    const shared = { ...reach, exact: false, reasons: ["widened"] };
    const rest = playbackDifference(reach, shared);
    expect(count(rest)).toBe(64);
    expect(rest.exact).toBe(false);
    expect(rest.reasons).toContain("widened");
  });
});
