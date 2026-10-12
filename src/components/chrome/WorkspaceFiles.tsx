import React, { useRef, useState } from "react";
import { useStore } from "../../state/store";
import { MAX_WORKSPACE_BYTES, parseWorkspaceDocument, sameSavedWorkspace, serializeWorkspace } from "../../state/workspace-document";
import { RECOVERY_KEY } from "../../state/workspace-session";
import { WorkspaceStatus } from "./WorkspaceStatus";

function download(text: string, name: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function recoveryText(): string | null {
  try { return localStorage.getItem(RECOVERY_KEY); } catch { return null; }
}

/** Files include the unrun draft; Share continues to describe the built graph. */
export function WorkspaceFiles(): React.ReactElement {
  const input = useRef<HTMLInputElement>(null);
  const [opening, setOpening] = useState(false);
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null);
  const compiling = useStore((s) => s.compiling);
  const status = useStore((s) => s.autosaveStatus);
  const recovery = recoveryText();
  const open = async (read: () => Promise<string>, label: string) => {
    const before = useStore.getState();
    setOpening(true); setMessage(null);
    try {
      const workspace = parseWorkspaceDocument(await read());
      if (!sameSavedWorkspace(before, useStore.getState()))
        throw new Error("The workspace changed while reading. Open the file again to replace it.");
      const restored = await useStore.getState().restoreWorkspaceAsync(workspace);
      if (!restored) throw new Error(sameSavedWorkspace(before, useStore.getState())
        ? "This workspace could not be restored. Your current work is unchanged."
        : "Opening was cancelled because the workspace changed.");
      if (location.hash.startsWith("#s=")) history.replaceState(history.state, "", location.pathname + location.search);
      setMessage({ text: `${label} opened.`, error: false });
    } catch (error) {
      setMessage({ text: error instanceof Error ? error.message : "Could not open this workspace.", error: true });
    } finally {
      setOpening(false);
      if (input.current) input.current.value = "";
    }
  };
  const exportFile = (text: () => string, name: string) => {
    try { download(text(), name); setMessage(null); }
    catch (error) { setMessage({ text: error instanceof Error ? error.message : "Could not export this workspace.", error: true }); }
  };

  return <div className="workspace-files">
    <div className="workspace-file-actions">
      <input ref={input} type="file" accept=".json,application/json" hidden aria-label="open workspace file"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          void open(() => {
            if (file.size > MAX_WORKSPACE_BYTES) throw new Error("Workspace files must be smaller than 4 MB.");
            return file.text();
          }, "Workspace");
        }} />
      <button className="mini" disabled={opening || compiling} onClick={() => input.current?.click()}
        title="open a workspace JSON file">{opening ? "opening…" : "open"}</button>
      <button className="mini" onClick={() => exportFile(() => serializeWorkspace(useStore.getState()), "tilecone-workspace.json")}
        title="download this workspace, including unrun source changes">export</button>
      {recovery && <button className="mini" disabled={opening || compiling}
        onClick={() => void open(async () => recovery, "Previous save")}
        title="restore the previous local workspace saved before opening a link, file, or running new source">recover</button>}
      {recovery && (status === "unavailable" || message?.error) && <button className="mini"
        onClick={() => exportFile(() => recovery, "tilecone-recovery.json")}
        title="download the previous save even if this app cannot restore it">export recovery</button>}
    </div>
    <WorkspaceStatus />
    {message && <div className={`workspace-file-message ${message.error ? "error" : "muted"}`}
      role={message.error ? "alert" : "status"}>{message.text}</div>}
  </div>;
}
