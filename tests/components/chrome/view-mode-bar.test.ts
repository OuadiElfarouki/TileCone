import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ViewModeBar } from "../../../src/components/chrome/ViewModeBar";
import { canStartGraphPan } from "../../../src/components/graph/GraphView";
import { useStore } from "../../../src/state/store";
import { canvasHint } from "../../../src/view/workspace";

// Static renders observe the live test store, as the other chrome contracts do.
vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) =>
      selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const S = () => useStore.getState();
const render = () => renderToStaticMarkup(createElement(ViewModeBar));

beforeEach(() => {
  S().applyDSL("A = Tensor(8, 8)\nB = Tensor(8, 8)\nC = matmul(A, B)\n");
  S().setInspectorTab("dependencies");
  S().setCanvasTool("select");
});

/* One gesture, two meanings: the canvas has to say which is in force. */
describe("what a press on the canvas does", () => {
  const hint = (state: Partial<Parameters<typeof canvasHint>[0]>) =>
    canvasHint({ inspectorTab: "dependencies", canvasTool: "select", planTiled: false, ...state });

  it("draws tiles in the Dependencies view, and moves nodes under the move tool", () => {
    expect(hint({})).toBe("drag on a tensor to draw a tile · Shift adds one, Alt subtracts");
    expect(hint({ canvasTool: "move" })).toBe("drag a card or operation to move it · Esc returns to drawing tiles");
  });

  it("divides tensors in the Plan view, and inspects tasks once there are any", () => {
    expect(hint({ inspectorTab: "plan" })).toBe(
      "drag on a produced tensor to divide it into tasks, or click it to use the drawn grid"
    );
    expect(hint({ inspectorTab: "plan", planTiled: true })).toBe(
      "click a tile to inspect its task · drag on an untiled tensor to divide it"
    );
  });

  it("ignores a remembered move tool in the Plan view, which always selects", () => {
    expect(hint({ inspectorTab: "plan", canvasTool: "move" })).toBe(hint({ inspectorTab: "plan" }));
  });
});

describe("the view switch on the canvas", () => {
  it("presses the view in force and states its gesture", () => {
    const html = render();
    expect(html).toContain('id="view-dependencies" aria-pressed="true"');
    expect(html).toContain('id="view-plan" aria-pressed="false"');
    expect(html).toContain("drag on a tensor to draw a tile");
  });

  it("follows the store into the Plan view and its tiling", () => {
    S().setInspectorTab("plan");
    expect(render()).toContain('id="view-plan" aria-pressed="true"');
    expect(render()).toContain("drag on a produced tensor to divide it into tasks");
    S().setPlanTile("C", [4, 4]);
    expect(render()).toContain("click a tile to inspect its task");
  });

  it("is buttons in a labelled group, not a tab list claiming arrow navigation", () => {
    const html = render();
    expect(html).toContain('role="group" aria-label="view (P)"');
    expect(html).not.toContain('role="tab"');
    expect(html).not.toContain('role="tablist"');
  });

  /* The switch takes presses on the canvas, so the canvas must not start a pan
     under it; the hint beside it is text over the graph and takes none. */
  it("takes its own presses rather than panning the canvas", () => {
    const pressed = (selectorMatch: string) => ({
      closest: (selector: string) => (selector.includes(selectorMatch) ? {} : null),
    });
    expect(canStartGraphPan(pressed(".view-mode-switch"))).toBe(false);
    expect(render()).toContain('class="view-mode-switch"');
  });
});
