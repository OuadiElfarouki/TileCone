import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { useStore } from "./ui/store";
import "./styles.css";

// Apply the persisted/system theme before React paints so CSS and canvas colors
// start in agreement rather than flashing through the light palette.
document.documentElement.dataset.theme = useStore.getState().theme;

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
