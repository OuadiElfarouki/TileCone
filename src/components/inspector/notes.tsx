/** Dependency notes: why the footprint has the shape it has. */

import React from "react";
import { ConeFindings } from "../../core/notes";

/**
 * Why the footprint has the shape it has. The numbers above say how much; this
 * says what constrains it, which is the part that transfers to writing a kernel.
 */
export const NOTE_SEVERITY = {
  1: { word: "advisory" },
  2: { word: "moderate" },
  3: { word: "strong" },
} as const;

export function DependencyNotes({
  findings,
  hasSelection,
  focusedBox,
  attributed,
}: {
  findings: ConeFindings | null;
  hasSelection: boolean;
  focusedBox: number | null;
  attributed: boolean;
}): React.ReactElement | null {
  // The directional sections already explain an absent/disabled selection.
  // Repeating that state here would make three parts of one panel teach the
  // same two questions in different words.
  if (!hasSelection || !findings) return null;
  const capped = findings.constraintCount > findings.notes.length;
  return (
    <section className="ins-section notes-section">
      <div className="ins-title">
        Dependency notes
        {focusedBox !== null && attributed && <span className="muted"> · tile {focusedBox + 1}</span>}
        {capped && (
          <span className="muted"> · {findings.notes.length} of {findings.constraintCount} constraints</span>
        )}
      </div>
      {findings.notes.length ? (
        <ul className="notes-list">
          {findings.notes.map((note) => (
            <li
              key={`${note.nodeId}:${note.text}`}
              className={`severity-${note.severity}`}
              title={`${NOTE_SEVERITY[note.severity].word} constraint · severity ${note.severity} of 3`}
            >
              <span className="sr-only">
                {NOTE_SEVERITY[note.severity].word} constraint, severity {note.severity} of 3. {" "}
              </span>
              <b>{note.op}</b>
              {note.text}
            </li>
          ))}
        </ul>
      ) : findings.elementwise ? (
        <p className="hint">
          Everything this tile needs is elementwise: any tiling of the selection fuses
          without cross-tile traffic.
        </p>
      ) : (
        <p className="hint">
          Nothing this tile needs reaches an operation, so no dependency constrains it.
        </p>
      )}
    </section>
  );
}
