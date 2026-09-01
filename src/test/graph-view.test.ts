import { describe, expect, it } from "vitest";
import { canStartGraphPan, fittedTransform } from "../ui/GraphView";

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
});
