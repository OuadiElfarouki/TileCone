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

const target = (owner: "node" | "menu" | "field" | "canvas") => ({
  tagName: owner === "field" ? "TEXTAREA" : "DIV",
  closest: (selector: string) => {
    if (owner === "node" && selector.includes("[data-node-move]")) return {};
    if (owner === "menu" && selector.includes('[role="menu"]')) return {};
    return null;
  },
}) as unknown as HTMLElement;

const key = (name: string, owner: "node" | "menu" | "field" | "canvas" = "node", primary = false) => ({
  key: name, target: target(owner), ctrlKey: primary, metaKey: false,
  altKey: false, shiftKey: false, repeat: false, defaultPrevented: false,
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

/* Escape backs out of the innermost mode, and under the move tool the tool is
   that mode - unless a gesture is in flight, which cancels itself and is the
   whole of the press. */
describe("Escape and the move tool", () => {
  it("returns to select, including from a focused node", () => {
    const event = key("Escape");
    onKey(event);
    expect(useStore.getState().canvasTool).toBe("select");
    expect(event.preventDefault).toHaveBeenCalled();
  });

  it("only cancels a drag in flight, keeping the tool", () => {
    useStore.getState().setDragging(true);
    onKey(key("Escape"));
    expect(useStore.getState().canvasTool).toBe("move");
    useStore.getState().setDragging(false);

    // A node drag cancels itself on the same press and says so by preventing it.
    const handled = { ...key("Escape"), defaultPrevented: true } as KeyboardEvent;
    onKey(handled);
    expect(useStore.getState().canvasTool).toBe("move");

    onKey(key("Escape"));
    expect(useStore.getState().canvasTool).toBe("select");
  });

  it("leaves an open menu to cancel itself", () => {
    onKey(key("Escape", "menu"));
    expect(useStore.getState().canvasTool).toBe("move");
  });

  /* The Plan view has no move tool in effect, so Escape releases the task and
     the remembered tool is left for the Dependencies view. */
  it("releases the inspected task in the Plan view and keeps the tool for later", () => {
    const s = useStore.getState();
    s.setInspectorTab("plan");
    s.planTaskAt("Y", [0]);
    expect(useStore.getState().planTask).not.toBeNull();
    onKey(key("Escape"));
    expect(useStore.getState().planTask).toBeNull();
    expect(useStore.getState().canvasTool).toBe("move");
    useStore.getState().setInspectorTab("dependencies");
  });
});


describe("a graph whose source is pending", () => {
  it.each(["draft", "building"])("blocks workspace shortcuts while %s", (reason) => {
    const s = useStore.getState();
    s.setSelection("Y", fromBox([{ lo: 0, hi: 2 }]), "replace");
    const selection = useStore.getState().selection;
    const history = useStore.getState().workspaceHistory;
    const direction = s.direction;
    if (reason === "draft") s.setDraftText("");
    else useStore.setState({ compiling: true });
    const undo = vi.spyOn(useStore.getState(), "undoWorkspace");
    for (const name of ["ArrowRight", "Delete", "h", "u", "d", "e", "Escape"])
      onKey(key(name, "canvas"));
    onKey(key("z", "canvas", true));
    expect(undo).not.toHaveBeenCalled();
    expect(useStore.getState().selection).toBe(selection);
    expect(useStore.getState().workspaceHistory).toBe(history);
    expect(useStore.getState().direction).toBe(direction);
    expect(useStore.getState().canvasTool).toBe("move");
    onKey(key("f", "canvas"));
    onKey(key("?", "canvas"));
    expect(dispatchEvent.mock.calls[0][0].type).toBe(FIT_GRAPH_EVENT);
    expect(showShortcuts).toHaveBeenCalledOnce();
    useStore.setState({ compiling: false });
  });
});


it.each([false, true])("keeps tile shortcuts active for parameter edits while compiling=%s", (compiling) => {
  const source = "M = 8\nX = Tensor(M)\nY = relu(X)\n";
  const s = useStore.getState();
  s.applyDSL(source);
  s.setCanvasTool("select");
  s.setSelection("Y", fromBox([{ lo: 0, hi: 2 }]), "replace");
  s.setDraftText(source.replace("M = 8", "M = 16"));
  useStore.setState({ compiling });
  const move = vi.spyOn(useStore.getState(), "moveSelection");
  const event = key("ArrowRight", "canvas");
  onKey(event);
  expect(move).toHaveBeenCalledOnce();
  expect(event.preventDefault).toHaveBeenCalled();
  useStore.setState({ compiling: false });
});
