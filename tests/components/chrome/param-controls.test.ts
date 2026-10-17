import { describe, expect, it, vi } from "vitest";
import React, { createElement, isValidElement, ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SidePanel } from "../../../src/components/chrome/SidePanel";
import { EXAMPLES } from "../../../src/examples";
import { ParamControls } from "../../../src/components/chrome/ParamControls";
import { useStore } from "../../../src/state/store";
import { graphIsInteractive } from "../../../src/view/workspace";

vi.mock("../../../src/state/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) =>
      selector(actual.useStore.getState()),
    actual.useStore
  ) };
});

function findIncrease(node: ReactNode, name: string): React.ButtonHTMLAttributes<HTMLButtonElement> | undefined {
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findIncrease(child, name);
      if (found) return found;
    }
  } else if (isValidElement<React.ButtonHTMLAttributes<HTMLButtonElement>>(node)) {
    if (node.props["aria-label"] === `increase ${name}`) return node.props;
    return findIncrease(node.props.children, name);
  }
}

describe("parameter controls with user-authored names", () => {
  it.each(["constructor", "toString", "__proto__"])("renders and steps %s", (name) => {
    const source = `${name} = 4\nX = Tensor(${name})\n`;
    useStore.getState().applyDSL(source);
    const html = renderToStaticMarkup(createElement(ParamControls));
    expect(html).toContain('value="4"');
    expect(html).toContain("×2");

    let increase: React.ButtonHTMLAttributes<HTMLButtonElement> | undefined;
    function Probe() {
      increase = findIncrease(ParamControls(), name);
      return null;
    }
    renderToStaticMarkup(createElement(Probe));
    expect(increase).toBeDefined();
    increase!.onClick!({} as React.MouseEvent<HTMLButtonElement>);
    expect(useStore.getState().draftText).toBe(source.replace("= 4", "= 8"));
    expect(useStore.getState().graph!.params[name]).toBe(4);
    expect(graphIsInteractive(useStore.getState())).toBe(true);
  });
});


describe("parameter controls follow the source being edited", () => {
  const source = "M = 4\nX = Tensor(M)\nY = relu(X)\n";
  const renderPanel = () => renderToStaticMarkup(createElement(SidePanel));

  it("hides controls and editor hints when switching examples", () => {
    useStore.getState().applyDSL(source);
    expect(renderPanel()).toContain('aria-label="size parameters"');
    useStore.getState().stageExample(0);
    expect(useStore.getState().draftText).toBe(EXAMPLES[0].dsl);
    const html = renderPanel();
    expect(html).not.toContain('aria-label="size parameters"');
    expect(html).not.toContain('class="syntax-hint"');
    expect(useStore.getState().dslText).toBe(source);
  });

  it.each(["", "M = 4\nX = Tensor(M)\nY = exp(X)\n", "M = 4\nX = Tensor(\n"])(
    "hides stale controls while editing another or incomplete source: %s", (draft) => {
      useStore.getState().applyDSL(source);
      useStore.getState().setDraftText(draft);
      expect(renderPanel()).not.toContain('aria-label="size parameters"');
      expect(renderPanel()).not.toContain('class="syntax-hint"');
    }
  );

  it("keeps controls active for parameter-only changes and restores them after a new build", () => {
    useStore.getState().applyDSL(source);
    useStore.getState().setDraftText(source.replace("M = 4", "M = 8"));
    const resized = renderPanel();
    expect(resized).toContain('aria-label="size parameters"');
    expect(resized).toContain('class="syntax-hint"');
    expect(resized).toContain('id="param-M"');
    useStore.getState().setDraftText("");
    expect(renderPanel()).not.toContain('aria-label="size parameters"');
    useStore.getState().applyDSL("N = 16\nA = Tensor(N)\n");
    const rebuilt = renderPanel();
    expect(rebuilt).toContain('id="param-N"');
    expect(rebuilt).toContain('value="16"');
    expect(rebuilt).not.toContain('id="param-M"');
  });

  it("keeps controls hidden after a failed build and restores them when reverting the source", () => {
    useStore.getState().applyDSL(source);
    useStore.getState().applyDSL("N = 4\nY = relu(Missing)\n");
    expect(useStore.getState().loadError).not.toBeNull();
    expect(renderPanel()).not.toContain('aria-label="size parameters"');
    useStore.getState().setDraftText(source);
    expect(renderPanel()).toContain('id="param-M"');
  });
});


describe("controls for parameters used by the graph", () => {
  it("shows M, N and K but no control or hint for the unused L declaration", () => {
    useStore.getState().applyDSL("M = 256\nN = 128\nK = 512\nL = 32\n" +
      "A = Tensor(M, K, dtype=fp16)\nB = Tensor(K, N, dtype=fp16)\nC = matmul(A, B)\n");
    const html = renderToStaticMarkup(createElement(SidePanel));
    for (const name of ["M", "N", "K"]) expect(html).toContain(`id="param-${name}"`);
    expect(html).not.toContain('id="param-L"');
    expect(html.match(/class="syntax-hint"/g)).toHaveLength(3);
  });

  it("hides the parameter section when all declarations are unused", () => {
    useStore.getState().applyDSL("L = 32\n# L does not size anything\nA = Tensor(32)\nB = relu(A)\n");
    const html = renderToStaticMarkup(createElement(SidePanel));
    expect(html).not.toContain('aria-label="size parameters"');
    expect(html).not.toContain('class="syntax-hint"');
  });

  it("offers controls for symbolic split sizes over an input of unknown extent", () => {
    useStore.getState().applyDSL('N = 2\nL = 4\nUnused = 16\nX = opaque(op="Src", shapes=[[6]], dtypes=[fp32])\n' +
      "Y, Z = split(X, axis=0, sizes=[N, L])\n");
    const html = renderToStaticMarkup(createElement(SidePanel));
    expect(html).toContain('id="param-N"');
    expect(html).toContain('id="param-L"');
    expect(html).not.toContain('id="param-Unused"');
    expect(html.match(/class="syntax-hint"/g)).toHaveLength(2);
  });
});
