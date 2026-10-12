import React from "react";
import { useStore } from "../../state/store";

export function WorkspaceStatus(): React.ReactElement {
  const status = useStore((s) => s.autosaveStatus);
  const notice = useStore((s) => s.workspaceNotice);
  return <div className="workspace-status">
    {status && <span role="status" className={status === "unavailable" ? "error" : "muted"}
      title="Autosave stores this workspace and its unrun draft in this browser.">
      {status === "saved" ? "saved locally" : status === "saving" ? "saving…" : "autosave unavailable · export to keep your work"}
    </span>}
    {notice && <div className="workspace-notice" role="status">
      <span>{notice}</span>
      <button className="mini" aria-label="dismiss workspace notice"
        onClick={() => useStore.setState({ workspaceNotice: null })}>×</button>
    </div>}
  </div>;
}
