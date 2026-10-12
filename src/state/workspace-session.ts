import type { StoreApi } from "zustand";
import { compile } from "./compile-epoch";
import { decodeWorkspace, linkToRestore } from "./share";
import type { State, WorkspaceRestore } from "./types";
import { parseWorkspaceDocument, sameSavedWorkspace, serializeWorkspace } from "./workspace-document";

export const AUTOSAVE_KEY = "tilecone.workspace.v1";
export const RECOVERY_KEY = "tilecone.workspace.recovery";
type StorageAccess = () => Pick<Storage, "getItem" | "setItem">;

/** Called once outside React's StrictMode effects. A link or failed recovery
 * preserves the previous save in a second slot before replacing it. */
export function startWorkspaceSession(store: StoreApi<State>, hash: string, storage: StorageAccess) {
  let stopped = false;
  let ready = false;
  let dirty = false;
  let preserve = false;
  let expected: string | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const unavailable = (message?: string) => store.setState({ autosaveStatus: "unavailable",
    ...(message ? { workspaceNotice: message } : {}) });
  try { expected = storage().getItem(AUTOSAVE_KEY); } catch { unavailable(); }

  const flush = () => {
    clearTimeout(timer);
    if (!ready || stopped || !dirty) return;
    try {
      const target = storage();
      // A second tab must not silently overwrite this tab's recovery slot.
      if (target.getItem(AUTOSAVE_KEY) !== expected) {
        unavailable("Another tab changed the saved workspace. Export this tab to keep its work.");
        return;
      }
      const text = serializeWorkspace(store.getState());
      if (preserve && expected !== null) target.setItem(RECOVERY_KEY, expected);
      target.setItem(AUTOSAVE_KEY, text);
      expected = text;
      preserve = false;
      dirty = false;
      store.setState({ autosaveStatus: "saved" });
    } catch {
      unavailable();
    }
  };
  const unsubscribe = store.subscribe((state, previous) => {
    if (!ready || sameSavedWorkspace(state, previous)) return;
    if (state.graph !== previous.graph && expected !== null) preserve = true;
    dirty = true;
    store.setState({ autosaveStatus: "saving" });
    clearTimeout(timer);
    timer = setTimeout(flush, 300);
  });

  const initialized = (async () => {
    const attempt = async (workspace: WorkspaceRestore): Promise<"restored" | "failed" | "cancelled"> => {
      const before = store.getState();
      const pending = store.getState().restoreWorkspaceAsync(workspace);
      const epoch = compile.epoch;
      const restored = await pending;
      return stopped || epoch !== compile.epoch || (!restored && !sameSavedWorkspace(before, store.getState()))
        ? "cancelled" : restored ? "restored" : "failed";
    };
    let restored = false;
    let cancelled = false;
    let fromLink = false;
    const link = decodeWorkspace(hash);
    if (link) {
      const result = await attempt(linkToRestore(link));
      restored = result === "restored";
      fromLink = restored;
      cancelled = result === "cancelled";
      preserve = expected !== null;
    }
    if (!restored && !cancelled && expected !== null) {
      try {
        const result = await attempt(parseWorkspaceDocument(expected));
        restored = result === "restored";
        cancelled = result === "cancelled";
        if (!restored && !cancelled) throw new Error("Graph restoration failed");
      } catch {
        preserve = true;
        store.setState({ workspaceNotice: "The saved workspace could not be restored. Its recovery copy has been kept." });
      }
    }
    if (!restored && !cancelled && !stopped) {
      const pending = store.getState().loadExampleAsync(0);
      const epoch = compile.epoch;
      await pending;
      cancelled = epoch !== compile.epoch;
    }
    if (stopped) return;
    ready = true;
    if (restored && !fromLink) store.setState({ autosaveStatus: "saved" });
    // Once a link has been saved locally its hash can be consumed, so refresh
    // recovers subsequent edits rather than reopening the original link.
    if (fromLink) { dirty = true; flush(); }
    // Editing during compilation cancels that response. Keep that new draft
    // without letting the startup fallback replace it, and keep the unrestored
    // save as the recovery copy.
    if (cancelled) { preserve ||= expected !== null; dirty = true; flush(); }
    return fromLink && store.getState().autosaveStatus === "saved";
  })();
  return { initialized, flush, stop: () => { flush(); stopped = true; clearTimeout(timer); unsubscribe(); } };
}
