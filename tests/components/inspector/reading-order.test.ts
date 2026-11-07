import { describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Inspector } from "../../../src/components/inspector/Inspector";
import { coneGroups } from "../../../src/components/inspector/cones";
import { EXAMPLES } from "../../../src/examples/index";
import { useStore } from "../../../src/state/store";
import type { TensorReadout } from "../../../src/core/metrics";

// Static-render tests read the live test store rather than Zustand's initial
// server snapshot. Actions and all derivation logic remain the real implementation.
vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const attention = () => {
  useStore.getState().applyDSL(EXAMPLES.find((example) => example.name === "Multi-head attention")!.dsl);
  return renderToStaticMarkup(createElement(Inspector));
};

describe("the Dependencies panel's reading order", () => {
  it("states cost and constraints before the cone rows", () => {
    const html = attention();
    const at = (text: string) => {
      const index = html.indexOf(text);
      expect(index, text).toBeGreaterThanOrEqual(0);
      return index;
    };
    expect(at("Cost to compute")).toBeLessThan(at("Dependency notes"));
    expect(at("Dependency notes")).toBeLessThan(at("Operand reuse"));
    expect(at("Operand reuse")).toBeLessThan(at("Backward Cone"));
    expect(at("Backward Cone")).toBeLessThan(at("Forward Cone"));
  });

  it("lists the backward cone's graph inputs before its intermediates", () => {
    const html = attention();
    expect(html.indexOf("Graph inputs")).toBeGreaterThan(html.indexOf("Backward Cone"));
    expect(html.indexOf("Graph inputs")).toBeLessThan(html.indexOf("Intermediates"));
    expect(html.indexOf("<b>Wo</b>")).toBeLessThan(html.indexOf("Intermediates"));
    expect(html.indexOf("<b>Scores</b>")).toBeGreaterThan(html.indexOf("Intermediates"));
  });

  it("offers the constraint the summary leaves out, softmax among them", () => {
    const html = attention();
    expect(html).toContain("4 of 5 constraints");
    expect(html).toContain("show 1 more");
    // The summary leaves it out; the flag on its tensor row still names it.
    expect(html).not.toContain("softmax normalises Scores");
  });
});

describe("cone row groups", () => {
  const row = (tensorId: string, isInput: boolean) => ({ tensorId, isInput }) as TensorReadout;

  it("splits a backward cone into graph inputs, then intermediates, keeping order", () => {
    const groups = coneGroups("backward", [row("T1", false), row("X", true), row("T2", false), row("W", true)]);
    expect(groups.map((group) => [group.title, group.rows.map((r) => r.tensorId)])).toEqual([
      ["Graph inputs", ["X", "W"]],
      ["Intermediates", ["T1", "T2"]],
    ]);
  });

  it("leaves a cone with one kind of row, and the forward cone, ungrouped", () => {
    expect(coneGroups("backward", [row("X", true), row("W", true)])).toEqual([
      { title: null, rows: [row("X", true), row("W", true)] },
    ]);
    const forward = [row("Y", false), row("Z", false)];
    expect(coneGroups("forward", forward)).toEqual([{ title: null, rows: forward }]);
  });
});
