import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { PanelFrame } from "../ui/PanelFrame";
import { ShortcutsDialog } from "../ui/ShortcutsDialog";
import { SidePanel } from "../ui/SidePanel";
import { useStore } from "../ui/store";

// Static renders should observe the live test store, matching the inspector's
// render-contract tests. Actions remain the real Zustand implementation.
vi.mock("../ui/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ui/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) =>
      selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

const S = () => useStore.getState();
let restoreConsoleError = () => {};

beforeEach(() => {
  S().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  useStore.setState({
    panelCollapsed: { left: false, right: false },
    panelW: { left: 330, right: 300 },
  });
  // React warns about useLayoutEffect during server rendering. These tests
  // inspect markup only; focus transitions require browser-level coverage.
  const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  restoreConsoleError = () => consoleError.mockRestore();
});

afterEach(() => restoreConsoleError());

describe("panel accessibility contracts", () => {
  it("connects source errors to the editor and announces them", () => {
    S().applyDSL("Y = relu(Missing)\n");
    const html = renderToStaticMarkup(createElement(SidePanel));

    expect(html).toContain('<aside class="side-panel" aria-label="Graph source and operations">');
    expect(html).toContain('aria-invalid="true"');
    expect(html).toContain('aria-describedby="source-status"');
    expect(html).toContain('aria-errormessage="source-status"');
    expect(html).toContain('id="source-status" role="alert" aria-live="assertive"');
  });

  it("unmounts collapsed contents and exposes an adjustable open separator", () => {
    useStore.setState({ panelCollapsed: { left: true, right: false } });
    const collapsed = renderToStaticMarkup(createElement(
      PanelFrame,
      {
        side: "left",
        label: "source",
        children: createElement("span", null, "transient panel content"),
      }
    ));
    expect(collapsed).not.toContain("transient panel content");
    expect(collapsed).toContain("show source");

    useStore.setState({ panelCollapsed: { left: false, right: false } });
    const open = renderToStaticMarkup(createElement(
      PanelFrame,
      {
        side: "left",
        label: "source",
        children: createElement("span", null, "transient panel content"),
      }
    ));
    expect(open).toContain("transient panel content");
    expect(open).toContain('role="separator" tabindex="0" aria-label="source panel width"');
    expect(open).toContain('aria-valuenow="330"');
  });

  it("marks the shortcut sheet as modal", () => {
    const html = renderToStaticMarkup(createElement(ShortcutsDialog, {
      open: true,
      onClose: () => {},
    }));
    expect(html).toContain('role="dialog" aria-modal="true"');
    expect(html).toContain('aria-labelledby="shortcut-title"');
  });
});
