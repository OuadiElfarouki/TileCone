import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useKeyboard, FIT_GRAPH_EVENT } from "../../../src/components/hooks/useKeyboard";
import { useStore } from "../../../src/state/store";
import { fromBox } from "../../../src/core/region";

vi.mock("react", async (importOriginal) => ({
  ...await importOriginal<typeof import("react")>(),
  useEffect: (effect: () => unknown) => { effect(); },
}));

let onKey: (event: KeyboardEvent) => void;
const showShortcuts = vi.fn();
const dispatchEvent = vi.fn();

beforeEach(() => {
  vi.stubGlobal("window", {
    addEventListener: (type: string, listener: typeof onKey) => {
      if (type === "keydown") onKey = listener;
    },
    removeEventListener: vi.fn(),
    dispatchEvent,
  });
  useStore.getState().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  useStore.getState().setCanvasTool("move");
  useKeyboard({ shortcutsOpen: false, showShortcuts, closeShortcuts: vi.fn() });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

const target = (owner: "node" | "menu" | "field") => ({
  tagName: owner === "field" ? "TEXTAREA" : "DIV",
  closest: (selector: string) => {
    if (owner === "node" && selector.includes("[data-node-move]")) return {};
    if (owner === "menu" && selector.includes('[role="menu"]')) return {};
    return null;
  },
}) as unknown as HTMLElement;

const key = (name: string, owner: "node" | "menu" | "field" = "node", primary = false) => ({
  key: name, target: target(owner), ctrlKey: primary, metaKey: false,
  altKey: false, shiftKey: false, repeat: false,
  preventDefault: vi.fn(),
}) as unknown as KeyboardEvent;

describe("global shortcuts with movable node focus", () => {
  it("undoes a node move without requiring focus to leave the node", () => {
    const s = useStore.getState();
    s.setNodeOffset("t:X", { dx: 8, dy: 0 });
    s.commitNodeMove("t:X", { dx: 0, dy: 0 });
    const event = key("z", "node", true);
    onKey(event);
    expect(useStore.getState().nodeOffsets).toEqual({});
    expect(event.preventDefault).toHaveBeenCalled();
  });

  it("keeps Fit and Help available", () => {
    onKey(key("f"));
    expect(dispatchEvent.mock.calls[0][0].type).toBe(FIT_GRAPH_EVENT);
    onKey(key("?"));
    expect(showShortcuts).toHaveBeenCalledOnce();
  });

  it("leaves arrow keys to the node without moving a tile", () => {
    useStore.getState().setSelection("X", fromBox([{ lo: 0, hi: 1 }]), "replace");
    const selection = useStore.getState().selection;
    const move = vi.spyOn(useStore.getState(), "moveSelection");
    const event = key("ArrowRight");
    onKey(event);
    expect(move).not.toHaveBeenCalled();
    expect(useStore.getState().selection).toBe(selection);
    expect(event.preventDefault).not.toHaveBeenCalled();
  });

  it.each(["menu", "field"] as const)("preserves shortcut ownership for a %s", (owner) => {
    const undo = vi.spyOn(useStore.getState(), "undoWorkspace");
    onKey(key("z", owner, true));
    onKey(key("f", owner));
    onKey(key("?", owner));
    expect(undo).not.toHaveBeenCalled();
    expect(dispatchEvent).not.toHaveBeenCalled();
    expect(showShortcuts).not.toHaveBeenCalled();
  });
});
