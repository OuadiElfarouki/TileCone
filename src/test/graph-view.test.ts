import { describe, expect, it } from "vitest";
import {
  canStartCardDrag,
  canStartGraphPan,
  edgePresentation,
  fittedTransform,
  centredOn,
  graphZoomBounds,
  lowZoomBound,
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
  /* The axis that limited the scale keeps the 20px margin exactly; the slack on
     the other is split, rather than collecting below and to the right. */
  it("centres the fitted scene, translating negative scene origins", () => {
    const tf = fittedTransform(
      { left: -100, top: -50, width: 400, height: 200 },
      { width: 440, height: 300 }
    );

    expect(tf.k).toBe(1);
    const left = tf.x + -100 * tf.k;
    const right = 440 - (tf.x + (-100 + 400) * tf.k);
    const top = tf.y + -50 * tf.k;
    const bottom = 300 - (tf.y + (-50 + 200) * tf.k);

    expect(left).toBe(20);
    expect(right).toBe(20);
    expect(top).toBe(bottom);
    expect(top).toBe(50);
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

  it("fits the entire scene even below the manual legibility floor", () => {
    const tf = fittedTransform(
      { left: 0, top: 0, width: 10_000, height: 10_000 },
      { width: 400, height: 300 },
      { min: 0.4, max: 4 }
    );
    expect(tf.k).toBeLessThan(0.4);
    // Height limited the scale here, so the vertical margin is the 20px one and
    // the horizontal slack is shared.
    expect(tf.y).toBe(20);
    expect(300 - (tf.y + 10_000 * tf.k)).toBeCloseTo(20);
    expect(tf.x).toBeCloseTo(400 - (tf.x + 10_000 * tf.k));
    expect(tf.x + 10_000 * tf.k).toBeLessThanOrEqual(380);
    expect(tf.y + 10_000 * tf.k).toBeLessThanOrEqual(280);
  });

  /* The companion to the rule above. Since a fitted view may sit below the
     manual floor, zooming out has to be allowed back down to it, from wherever
     the user has zoomed to. The clamp therefore takes no current scale: keyed
     to one, zooming in past the floor raised the floor and left the overview
     reachable only through `fit`, with the zoom-out button still live. */
  it("lets zoom-out return to a fitted view below the legibility floor", () => {
    const scene = { left: 0, top: 0, width: 10_000, height: 10_000 };
    const viewport = { width: 400, height: 300 };
    const bounds = { min: 0.4, max: 4 };
    const fitted = fittedTransform(scene, viewport, bounds).k;

    expect(lowZoomBound(scene, viewport, bounds)).toBeCloseTo(fitted);
    expect(lowZoomBound(scene, viewport, bounds)).toBeLessThan(bounds.min);
  });

  /* An operation row centres its operator, so the seam takes any node and puts
     its middle in the middle - a tensor card and a 70x30 op node alike. */
  it("puts a node's centre in the middle of the viewport", () => {
    const viewport = { width: 1000, height: 600 };
    for (const node of [
      { x: 0, y: 0, w: 70, h: 30 },
      { x: 400, y: 250, w: 300, h: 200 },
    ]) {
      for (const k of [0.2, 1, 2.5]) {
        const { x, y } = centredOn(node, viewport, k);
        expect(x + (node.x + node.w / 2) * k).toBeCloseTo(viewport.width / 2);
        expect(y + (node.y + node.h / 2) * k).toBeCloseTo(viewport.height / 2);
      }
    }
  });

  it("keeps the ordinary floor when the scene fits above it", () => {
    const scene = { left: 0, top: 0, width: 200, height: 150 };
    const viewport = { width: 1200, height: 800 };
    const bounds = { min: 0.4, max: 4 };
    expect(lowZoomBound(scene, viewport, bounds)).toBe(0.4);
  });
});
