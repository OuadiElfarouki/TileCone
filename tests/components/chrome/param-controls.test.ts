import { describe, expect, it, vi } from "vitest";
import React, { createElement, isValidElement, ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
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
