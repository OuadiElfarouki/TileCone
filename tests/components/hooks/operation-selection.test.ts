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

  it("outside presses clear selection", () => {
    const target = { closest: () => null };
    dismissOperationOnPointer({ target: target as unknown as EventTarget });
    expect(useStore.getState().selectedOp).toBeNull();
  });
});
