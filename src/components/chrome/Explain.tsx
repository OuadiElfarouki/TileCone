import React, { useState } from "react";

/**
 * A section's standing explanation, behind a "?" in its heading.
 *
 * What a section's figures mean is read once and then known, so it stays
 * closed until asked for rather than sitting between the reader and the
 * figures on every visit. What is true only now - a bound, a missing total, a
 * failure - is not an explanation and stays in the section as it is.
 *
 * Returns the button, for the heading's actions, and the text to place under
 * the heading, which is null while closed.
 */
export function useExplanation(
  topic: string,
  text: React.ReactNode
): { button: React.ReactElement; panel: React.ReactElement | null } {
  const [open, setOpen] = useState(false);
  return {
    button: (
      <button
        className="mini explain-toggle"
        aria-expanded={open}
        aria-label={`what ${topic} means`}
        title={open ? "hide the explanation" : `what ${topic} means`}
        onClick={() => setOpen(!open)}
      >
        ?
      </button>
    ),
    panel: open ? <p className="hint explain">{text}</p> : null,
  };
}
