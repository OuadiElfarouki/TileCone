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

  it("shows an imported model its report rather than an editor it has no text for", () => {
    S().importJSON(
      JSON.stringify({
        graph: {
          nodes: [
            {
              id: "/head/Resize",
              op: "opaque",
              inputs: ["input"],
              outputs: ["/head/Resize_output_0"],
              attrs: { op: "Resize", shapes: [[4, 12]], dtypes: ["f32"] },
            },
          ],
          tensors: {
            input: { id: "input", name: "input", shape: [4, 6], dtype: "f32" },
            "/head/Resize_output_0": {
              id: "/head/Resize_output_0",
              name: "/head/Resize_output_0",
              shape: [],
              dtype: "f32",
            },
          },
          params: {},
        },
        report: {
          origin: { fileName: "tiny.onnx", format: "onnx", opset: 17 },
          entries: [
            { kind: "barrier", node: "/head/Resize", sourceOp: "Resize", reason: "no mapping" },
          ],
        },
      }),
      { fileName: "tiny.onnx", format: "onnx" }
    );
    const html = renderToStaticMarkup(createElement(SidePanel));

    expect(html).toContain("Imported model");
    expect(html).toContain("tiny.onnx · onnx · opset 17");
    expect(html).toContain("regions through them are bounds");
    expect(html).toContain("replace with example");
    expect(html).toContain("choose replacement");
    // No editor, because there is no text behind this graph. A textarea here
    // would offer a Run that replaces the imported model with whatever it held.
    expect(html).not.toContain('aria-label="graph source"');
    // Sharing is refused in the open, not by producing a link that restores
    // nothing: a link carries DSL source and this workspace has none.
    expect(html).toMatch(/<button class="mini share-btn" disabled/);

    // A failed replacement preserves the installed model, so its error must be
    // rendered by the import summary rather than by the unmounted DSL editor.
    expect(S().importJSON("{ not json")).toBe(false);
    const failed = renderToStaticMarkup(createElement(SidePanel));
    expect(failed).toContain('class="import-errors error" role="alert"');
    expect(failed).toContain("invalid JSON");
    expect(failed).toContain("tiny.onnx · onnx · opset 17");
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
