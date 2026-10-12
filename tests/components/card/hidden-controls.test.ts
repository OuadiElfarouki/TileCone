import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import type { Tensor } from "../../../src/core/graph";
import { TensorCard } from "../../../src/components/card/TensorCard";
import { useStore } from "../../../src/state/store";

const probe = vi.hoisted(() => ({
  front: { width: 100, height: 100 },
  out: { width: 100, height: 100 },
  refs: 0,
  effects: [] as { callback: () => void; deps: unknown[] }[],
}));
vi.mock("react", async (original) => {
  const actual = await original<typeof import("react")>();
  return {
    ...actual,
    useState: (value: unknown) => [value, () => {}],
    useMemo: (factory: () => unknown) => factory(),
    useRef: (value: unknown) => ({ current: value === null && probe.refs < 2
      ? probe.refs++ === 0 ? probe.front : probe.out : value }),
    useEffect: (callback: () => void, deps: unknown[]) => { probe.effects.push({ callback, deps }); },
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
function render(rasterize = true) {
  const tensor = S().resolved!.tensors.X;
  const view = TensorCard as unknown as { type: (props: { tensor: Tensor; rasterize: boolean }) => ReactElement };
  return view.type({ tensor, rasterize });
}
function inputs(element: ReactElement): ReactElement[] {
  const children = [element.props.children].flat(Infinity).filter((child) => child && typeof child === "object" && "props" in child) as ReactElement[];
  return [...(element.type === "input" ? [element] : []), ...children.flatMap(inputs)];
}
beforeEach(() => {
  S().applyDSL("X = Tensor(4, 5, 8, 8)\nY = relu(X)\n");
  probe.refs = 0;
  probe.effects.length = 0;
  Object.assign(probe.front, { width: 100, height: 100 });
  Object.assign(probe.out, { width: 100, height: 100 });
});

describe("hidden-axis controls", () => {
  it("offers every narrow projected axis, including a shorter last tile", () => {
    S().setViewCfg("X", { tile: [1, 2, 2, 2] });
    const controls = inputs(render());
    expect(controls.map((input) => [input.props.type, input.props["aria-label"], input.props.max])).toEqual([
      ["number", "ax0 tile position", 3], ["number", "ax1 tile position", 2],
    ]);
    controls[1].props.onChange({ target: { valueAsNumber: 2 } });
    expect(S().viewCfgs.X.sliders).toEqual([0, 4, 0, 0]);
    expect(S().viewCfgs.X.projection).toBe(true);
    const updated = inputs(render());
    expect(updated[1].props.value).toBe(2);
    updated[1].props.onChange({ target: { valueAsNumber: NaN } });
    expect(S().viewCfgs.X.sliders[1]).toBe(4);
    updated[0].props.onChange({ target: { valueAsNumber: -1 } });
    expect(S().viewCfgs.X.sliders[0]).toBe(0);
  });

  it("keeps range sliders exclusive to slice mode and removes positions when unfolded", () => {
    expect(inputs(render())).toEqual([]);
    S().setViewCfg("X", { projection: false });
    expect(inputs(render()).map((input) => input.props.type)).toEqual(["range", "range"]);
    S().setViewCfg("X", { unfold: "grid" });
    expect(inputs(render())).toEqual([]);
  });

  it("releases both backing stores when a slice card moves offscreen", () => {
    S().setViewCfg("X", { projection: false });
    render(false);
    probe.effects.find((effect) => effect.deps?.includes(S().resolved!.tensors.X.resolved))!.callback();
    expect(probe.front).toEqual({ width: 0, height: 0 });
    expect(probe.out).toEqual({ width: 0, height: 0 });
  });
});
