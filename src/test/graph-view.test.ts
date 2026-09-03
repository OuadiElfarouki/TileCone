import { describe, expect, it } from "vitest";
import {
  canStartCardDrag,
  canStartGraphPan,
  edgePresentation,
  fittedTransform,
  graphZoomBounds,
} from "../ui/GraphView";

const target = (blocked: boolean) => ({
  closest: (selector: string) => {
    expect(selector).toContain(".card-slot");
    expect(selector).toContain(".op-node");
    return blocked ? {} : null;
  },
});

describe("graph viewport pan hit-testing", () => {
  it("accepts the canvas and inner layout background", () => {
    expect(canStartGraphPan(null)).toBe(true);
    expect(canStartGraphPan(target(false))).toBe(true);
  });

  it("leaves tensor cards, operation nodes, and controls in charge of their gestures", () => {
    expect(canStartGraphPan(target(true))).toBe(false);
  });
});

describe("card drag hit-testing", () => {
  const headerTarget = (onName: boolean) => ({
    closest: (selector: string) => {
      expect(selector).toContain(".tc-name-wrap");
      return onName ? {} : null;
    },
  });

  it("drags from anywhere on the header chrome", () => {
    expect(canStartCardDrag(null)).toBe(true);
    expect(canStartCardDrag(headerTarget(false))).toBe(true);
  });

  it("leaves the tensor name its click, so the shape popover stays reachable", () => {
    expect(canStartCardDrag(headerTarget(true))).toBe(false);
  });
});

describe("connector stacking", () => {
  it("puts inactive context behind cards and active connectors in front", () => {
    expect(edgePresentation(true, false)).toEqual({ className: "edge dim", layer: "behind" });
    expect(edgePresentation(true, true)).toEqual({ className: "edge hot", layer: "front" });
    expect(edgePresentation(false, false)).toEqual({ className: "edge", layer: "front" });
  });
});

describe("graph viewport fit", () => {
  it("translates negative scene origins into the fitted viewport", () => {
    const tf = fittedTransform(
      { left: -100, top: -50, width: 400, height: 200 },
      { width: 440, height: 300 }
    );

    expect(tf.k).toBe(1);
    expect(tf.x + -100 * tf.k).toBe(20);
    expect(tf.y + -50 * tf.k).toBe(20);
    expect(tf.x + (-100 + 400) * tf.k).toBe(420);
  });

  it("keeps the smallest tensor legible and stops where supersampling stops", () => {
    const bounds = graphZoomBounds([
      { kind: "tensor", w: 120, h: 35 },
      { kind: "tensor", w: 300, h: 200 },
      { kind: "op", w: 64, h: 20 },
    ]);
    expect(bounds.min).toBe(14 / 35);
    expect(bounds.max).toBe(4);
  });

  it("does not make fit shrink cards below the legibility floor", () => {
    const tf = fittedTransform(
      { left: 0, top: 0, width: 10_000, height: 10_000 },
      { width: 400, height: 300 },
      { min: 0.4, max: 4 }
    );
    expect(tf.k).toBe(0.4);
  });
});
