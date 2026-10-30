import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStore } from "../../src/state/store";
import { compile } from "../../src/state/compile-epoch";
import { box, fromBox } from "../../src/core/region";
import { encodeWorkspace, workspaceToLink } from "../../src/state/share";
import { serializeWorkspace } from "../../src/state/workspace-document";
import { AUTOSAVE_KEY, RECOVERY_KEY, startWorkspaceSession } from "../../src/state/workspace-session";

const SOURCE = "A = Tensor(8, 8, dtype=float32)\nB = relu(A)\n";
const S = () => useStore.getState();
let sessions: ReturnType<typeof startWorkspaceSession>[];
let data: Map<string, string>;
const storage = () => ({ getItem: (key: string) => data.get(key) ?? null,
  setItem: vi.fn((key: string, value: string) => { data.set(key, value); }) });
const start = (hash = "", access: Pick<Storage, "getItem" | "setItem"> = storage()) => {
  const session = startWorkspaceSession(useStore, hash, () => access);
  sessions.push(session);
  return session;
};
beforeEach(() => {
  vi.useFakeTimers(); sessions = []; data = new Map();
  useStore.setState(useStore.getInitialState(), true);
});
afterEach(() => { sessions.forEach((session) => session.stop()); vi.useRealTimers(); });

describe("local workspace recovery", () => {
  it("loads the local graph and an invalid unrun draft without saving startup defaults", async () => {
    S().applyDSL(SOURCE); S().setDraftText("unrun (");
    const saved = serializeWorkspace(S()); data.set(AUTOSAVE_KEY, saved);
    useStore.setState(useStore.getInitialState(), true);
    const access = storage(); await start("", access).initialized;
    expect(S().dslText).toBe(SOURCE); expect(S().draftText).toBe("unrun (");
    expect(access.setItem).not.toHaveBeenCalled();
    expect(S().autosaveStatus).toBe("saved");
  });

  it("gives share links precedence and keeps the previous local workspace", async () => {
    S().applyDSL(SOURCE); S().setDraftText("local work");
    const local = serializeWorkspace(S()); data.set(AUTOSAVE_KEY, local);
    const link = { ...workspaceToLink(S()), dsl: "X = Tensor(4, dtype=float32)\n", sel: null };
    expect(await start(`#s=${encodeWorkspace(link)}`).initialized).toBe(true);
    expect(S().dslText).toBe(link.dsl); expect(S().draftText).toBe(link.dsl);
    expect(data.get(RECOVERY_KEY)).toBe(local);
    expect(JSON.parse(data.get(AUTOSAVE_KEY)!).workspace.dsl).toBe(link.dsl);
  });

  it("falls back from an invalid link to local recovery", async () => {
    S().applyDSL(SOURCE); data.set(AUTOSAVE_KEY, serializeWorkspace(S()));
    useStore.setState(useStore.getInitialState(), true);
    await start("#s=bad").initialized;
    expect(S().dslText).toBe(SOURCE);
  });

  it("preserves an unreadable save before saving a new edit", async () => {
    data.set(AUTOSAVE_KEY, "future or corrupt document");
    const session = start(); await session.initialized;
    expect(data.get(AUTOSAVE_KEY)).toBe("future or corrupt document");
    expect(S().workspaceNotice).toMatch(/could not be restored/);
    S().setDraftText("new draft"); session.flush();
    expect(data.get(RECOVERY_KEY)).toBe("future or corrupt document");
    expect(JSON.parse(data.get(AUTOSAVE_KEY)!).draftText).toBe("new draft");
  });

  it("debounces edits, ignores previews, and flushes the latest draft synchronously", async () => {
    const access = storage(); const session = start("", access); await session.initialized;
    S().hoverBox(0); S().setFocusNode({ kind: "tensor", id: "A" });
    vi.advanceTimersByTime(400); expect(access.setItem).not.toHaveBeenCalled();
    S().setDraftText("one"); vi.advanceTimersByTime(100); S().setDraftText("two");
    expect(access.setItem).not.toHaveBeenCalled();
    vi.advanceTimersByTime(300); expect(JSON.parse(data.get(AUTOSAVE_KEY)!).draftText).toBe("two");
    S().setDraftText("last keystroke"); session.flush();
    expect(JSON.parse(data.get(AUTOSAVE_KEY)!).draftText).toBe("last keystroke");
  });

  it("reports blocked storage and quota failure while leaving edits usable", async () => {
    const access = { getItem: () => null, setItem: () => { throw new Error("quota"); } };
    const session = start("", access); await session.initialized;
    S().setDraftText("still usable"); session.flush();
    expect(S().autosaveStatus).toBe("unavailable"); expect(S().draftText).toBe("still usable");
  });

  it("does not overwrite a workspace saved by another tab", async () => {
    const session = start(); await session.initialized;
    data.set(AUTOSAVE_KEY, "another tab");
    S().setDraftText("my tab"); session.flush();
    expect(data.get(AUTOSAVE_KEY)).toBe("another tab");
    expect(S().workspaceNotice).toMatch(/Another tab/);
  });

  const restoreInFlight = () => {
    S().applyDSL(SOURCE); S().setSelection("B", fromBox(box([0, 2], [0, 2])), "replace");
    const saved = serializeWorkspace(S()); data.set(AUTOSAVE_KEY, saved);
    useStore.setState(useStore.getInitialState(), true);
    let finish!: (restored: boolean) => void;
    useStore.setState({ restoreWorkspaceAsync: () => {
      compile.epoch++; useStore.setState({ compiling: true });
      return new Promise<boolean>((resolve) => { finish = resolve; });
    } });
    return { saved, finish: (restored: boolean) => finish(restored) };
  };

  it("never replaces a draft entered while restoration was in flight", async () => {
    const { saved, finish } = restoreInFlight();
    const session = start(); S().setDraftText("typed during startup"); finish(false);
    await session.initialized;
    expect(S().draftText).toBe("typed during startup");
    expect(JSON.parse(data.get(AUTOSAVE_KEY)!).draftText).toBe("typed during startup");
    expect(data.get(RECOVERY_KEY)).toBe(saved);
  });

  it("keeps the unrestored save when a setting changes during restoration", async () => {
    const { saved, finish } = restoreInFlight();
    const session = start(); useStore.setState({ snapToGrid: !S().snapToGrid }); finish(false);
    await session.initialized;
    expect(JSON.parse(data.get(AUTOSAVE_KEY)!).workspace.sel).toBeNull();
    expect(data.get(RECOVERY_KEY)).toBe(saved);
  });
});
