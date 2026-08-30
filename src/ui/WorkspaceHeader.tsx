import React from "react";
import { useStore } from "./store";

/** Product identity only. Workspace controls belong to the side panels. */
export function WorkspaceHeader(): React.ReactElement {
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  const dark = theme === "dark";

  return (
    <header className="workspace-header">
      <span className="brand">tilecone<span>/</span></span>
      <span className="product-tag">tile dependency explorer</span>
      <span className="product-copy">
        Draw a tile on any tensor and read exactly what it needs upstream and what it feeds downstream.
      </span>
      <button
        className="theme-toggle"
        onClick={() => setTheme(dark ? "light" : "dark")}
        title="switch between light and dark"
        aria-label={`switch to ${dark ? "light" : "dark"} theme`}
      >
        {dark ? "◑ light" : "◐ dark"}
      </button>
    </header>
  );
}
