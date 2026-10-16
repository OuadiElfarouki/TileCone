import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import type { Tensor } from "../../../src/core/graph";
import { TensorCard } from "../../../src/components/card/TensorCard";
import { useStore } from "../../../src/state/store";

const probe = vi.hoisted(() => {
  const canvas = () => ({ width: 100, height: 100, getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }) });
  return { front: canvas(), out: canvas(), refs: 0 };
});
vi.mock("react", async (original) => {
  const actual = await original<typeof import("react")>();
  return {
    ...actual,
    useState: (value: unknown) => [value, () => {}],
    useMemo: (factory: () => unknown) => factory(),
    useRef: (value: unknown) => ({ current: value === null && probe.refs < 2
      ? probe.refs++ === 0 ? probe.front : probe.out : value }),
    useEffect: () => {},
  };
});
vi.mock("../../../src/components/hooks/useDebounced", () => ({ useDebounced: (fn: unknown) => fn }));
vi.mock("../../../src/components/hooks/useFrameThrottle", () => ({ useFrameThrottle: (fn: unknown) => fn }));
vi.mock("../../../src/components/chrome/OptionsMenu", () => ({
  useOptionsMenu: () => ({ anchor: null, onContextMenu: () => {} }), OptionsMenu: () => null,
}));
vi.mock("../../../src/state/store", async (original) => {
  const actual = await original<typeof import("../../../src/state/store")>();
  return { ...actual, useDark: () => actual.useStore.getState().theme === "dark", useStore: Object.assign(
    (selector: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => selector(actual.useStore.getState()), actual.useStore,
  ) };
});

const S = () => useStore.getState();
function canvasOf(displayScale?: number): ReactElement {
  probe.refs = 0;
  const view = TensorCard as unknown as { type: (props: { tensor: Tensor; displayScale?: number }) => ReactElement };
  const find = (element: ReactElement): ReactElement | undefined => element.type === "canvas" ? element
    : [element.props?.children].flat(Infinity)
      .filter((child) => child && typeof child === "object" && "props" in child)
      .map((child) => find(child as ReactElement)).find(Boolean);
  return find(view.type({ tensor: S().resolved!.tensors.X, displayScale }))!;
}
beforeEach(() => {
  useStore.setState(useStore.getInitialState(), true);
  S().applyDSL("X = Tensor(8, 16, dtype=fp32)\nY = relu(X)\n");
});

describe("hover preview on a resized card", () => {
  it("previews a card drawn smaller than its stored scale", () => {
    S().setCardScale("X", 2);
    canvasOf(1.5).props.onPointerMove({ clientX: 10, clientY: 10 });
    expect(S().preview).not.toBeNull();
  });
});
