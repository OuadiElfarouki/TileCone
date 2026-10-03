import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { LayoutControls } from "../../../src/components/chrome/LayoutControls";
import { useStore } from "../../../src/state/store";

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
const onResetLayout = vi.fn();
const render = () => renderToStaticMarkup(createElement(LayoutControls, { onResetLayout }));

beforeEach(() => {
  S().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  S().setCanvasTool("select");
});

describe("the layout group", () => {
  it("reads as an unpressed toggle under the select tool", () => {
    const html = render();
    expect(html).toContain('aria-pressed="false"');
    expect(html).not.toContain("mini toggle on");
    expect(html).toContain(">move<");
  });

  it("reads as pressed under the move tool", () => {
    S().setCanvasTool("move");
    const html = render();
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain("mini toggle on");
  });

  /* The canvas owns this control, so it has to keep taking the pointer while the
     graph behind it pans: `.layout-controls` is a pan blocker by class name. */
  it("carries the class the canvas hit-test excludes from panning", () => {
    expect(render()).toContain('class="layout-controls"');
  });
});

describe("restoring generated placement", () => {
  it("offers reset only once something has been moved", () => {
    expect(render()).toContain("disabled");

    S().setNodeOffset("t:X", { dx: 30, dy: 0 });
    S().commitNodeMove("t:X", { dx: 0, dy: 0 });
    expect(render()).not.toContain("disabled");
  });
});
