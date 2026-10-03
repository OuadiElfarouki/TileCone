import { beforeEach, describe, expect, it, vi } from "vitest";
import { dismissOperationOnKey, dismissOperationOnPointer } from "../../../src/components/hooks/useOperationSelection";
import { useStore } from "../../../src/state/store";

beforeEach(() => {
  useStore.getState().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  useStore.getState().setSelectedOp("elementwise_Y");
});

describe("dismissing the operation selection", () => {
  it("Escape clears the highlight without changing tiles or consuming the key", () => {
    const selection = useStore.getState().selection;
    const event = { key: "Escape", preventDefault: vi.fn(), stopPropagation: vi.fn() };
    dismissOperationOnKey(event);
    expect(useStore.getState().selectedOp).toBeNull();
    expect(useStore.getState().selection).toBe(selection);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(event.stopPropagation).not.toHaveBeenCalled();
  });

  it("other keys preserve selection", () => {
    dismissOperationOnKey({ key: "Enter" });
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");
  });

  it.each([".op-node", ".operation-row"])("keeps presses inside %s", (surface) => {
    const target = { closest: (selectors: string) => selectors.split(", ").includes(surface) ? {} : null };
    dismissOperationOnPointer({ target: target as unknown as EventTarget });
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");
  });

  /* A press on the canvas background may be a pan, and one on its controls a
     zoom; the canvas decides those at release. A card is a choice of its own. */
  const onCanvas = (inCard: boolean) => ({
    closest: (selectors: string) =>
      selectors === ".graph-canvas" || (inCard && selectors === ".card-slot") ? {} : null,
  });

  it("leaves presses on the canvas background and its controls to the canvas", () => {
    dismissOperationOnPointer({ target: onCanvas(false) as unknown as EventTarget });
    expect(useStore.getState().selectedOp).toBe("elementwise_Y");
  });

  it("clears on a press on a card", () => {
    dismissOperationOnPointer({ target: onCanvas(true) as unknown as EventTarget });
    expect(useStore.getState().selectedOp).toBeNull();
  });

  it("outside presses clear selection", () => {
    const target = { closest: () => null };
    dismissOperationOnPointer({ target: target as unknown as EventTarget });
    expect(useStore.getState().selectedOp).toBeNull();
  });
});
