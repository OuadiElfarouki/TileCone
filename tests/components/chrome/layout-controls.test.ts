import { beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { LayoutControls } from "../../../src/components/chrome/LayoutControls";
import { moveModifierKey } from "../../../src/components/hooks/useMoveModifier";
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

/* The Plan view divides and inspects tasks on the arrangement as it stands.
   Both controls stay in place, unavailable, with the reason on each. */
describe("the layout group in the Plan view", () => {
  it("is unavailable, reads unpressed, and keeps the remembered tool", () => {
    S().setCanvasTool("move");
    S().setNodeOffset("t:X", { dx: 30, dy: 0 });
    S().commitNodeMove("t:X", { dx: 0, dy: 0 });
    S().setInspectorTab("plan");
    const html = render();
    expect(html).toContain('aria-pressed="false"');
    expect(html).not.toContain("mini toggle on");
    expect(html.match(/disabled=""/g)).toHaveLength(2);
    expect(html).toContain("switch to Dependencies to move them");

    S().setInspectorTab("dependencies");
    const back = render();
    expect(back).toContain('aria-pressed="true"');
    expect(back).not.toContain("disabled");
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

/* Borrowing the move tool for one drag is the common way to move a node, so
   the group says how, in the key that works on this platform. */
describe("the move modifier hint", () => {
  it("names the modifier beside the toggle under the select tool", () => {
    expect(render()).toMatch(/<span class="layout-hint"[^>]*>or (Ctrl|⌘)-drag<\/span>/);
  });

  it("is gone under the move tool and in the Plan view, where it does not apply", () => {
    S().setCanvasTool("move");
    expect(render()).not.toContain("layout-hint");
    S().setCanvasTool("select");
    S().setInspectorTab("plan");
    expect(render()).not.toContain("layout-hint");
    S().setInspectorTab("dependencies");
  });

  it("says ⌘ on Apple platforms and Ctrl elsewhere", () => {
    expect(moveModifierKey({ platform: "MacIntel" })).toBe("⌘");
    expect(moveModifierKey({ userAgent: "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X)" })).toBe("⌘");
    expect(moveModifierKey({ platform: "Linux x86_64" })).toBe("Ctrl");
    expect(moveModifierKey({ platform: "Win32" })).toBe("Ctrl");
    expect(moveModifierKey(undefined)).toBe("Ctrl");
  });
});
