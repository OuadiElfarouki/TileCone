import React, { useEffect, useRef, useState } from "react";
import { copyText } from "./clipboard";

const FEEDBACK_MS = 1600;

/**
 * A button that copies and then says whether it did.
 *
 * Three places needed the same three states and the same timer, and each kept
 * its own copy of them - one of them lifting the state up through two component
 * layers to tell a row apart from its siblings. The feedback belongs to the
 * button that was pressed, so it lives here.
 */
export function CopyButton({
  text,
  label,
  title,
  className = "mini",
}: {
  /** Deferred, so a row builds its payload only when it is actually copied. */
  text: () => string;
  label: string;
  title: string;
  className?: string;
}): React.ReactElement {
  const [state, setState] = useState<"copied" | "failed" | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(timer.current), []);

  return (
    <button
      className={`${className}${state === "failed" ? " copy-failed" : ""}`}
      title={title}
      aria-live="polite"
      onClick={async (event) => {
        event.stopPropagation();
        const ok = await copyText(text());
        setState(ok ? "copied" : "failed");
        clearTimeout(timer.current);
        timer.current = setTimeout(() => setState(null), FEEDBACK_MS);
      }}
    >
      {state === "copied" ? "copied ✓" : state === "failed" ? "copy failed" : label}
    </button>
  );
}
