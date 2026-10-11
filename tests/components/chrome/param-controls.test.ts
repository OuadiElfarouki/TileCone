import { describe, expect, it, vi } from "vitest";
import React, { createElement, isValidElement, ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SidePanel } from "../../../src/components/chrome/SidePanel";
import { EXAMPLES } from "../../../src/examples";
import { ParamControls } from "../../../src/components/chrome/ParamControls";
import { useStore } from "../../../src/state/store";

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
