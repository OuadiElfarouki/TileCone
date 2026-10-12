import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { useStore } from "./state/store";
import { startWorkspaceSession } from "./state/workspace-session";
import "./styles.css";

// Apply the persisted/system theme before React paints so CSS and canvas colors
// start in agreement rather than flashing through the light palette.
document.documentElement.dataset.theme = useStore.getState().theme;

const startupHash = location.hash;
const session = startWorkspaceSession(useStore, startupHash, () => localStorage);
void session.initialized.then((savedLink) => {
  if (savedLink && location.hash === startupHash)
    history.replaceState(history.state, "", location.pathname + location.search);
});
window.addEventListener("pagehide", session.flush);
const saveWhenHidden = () => { if (document.visibilityState === "hidden") session.flush(); };
document.addEventListener("visibilitychange", saveWhenHidden);
if (import.meta.hot) import.meta.hot.dispose(() => {
  session.stop();
  window.removeEventListener("pagehide", session.flush);
  document.removeEventListener("visibilitychange", saveWhenHidden);
});

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
