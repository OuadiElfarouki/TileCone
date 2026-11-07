/** Dependency notes: why the footprint has the shape it has. */

import React, { useState } from "react";
import { ConeFindings, DependencyNote } from "../../core/notes";

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
  /* The cap keeps the section a summary; the notes it leaves out stay one
     press away, since the one left out can be a different kind of constraint
     from the four shown. */
  const [showAll, setShowAll] = useState(false);
  // The directional sections already explain an absent/disabled selection.
  // Repeating that state here would make three parts of one panel teach the
  // same two questions in different words.
  if (!hasSelection || !findings) return null;
  const omitted = findings.omitted.length;
  return (
    <section className="ins-section notes-section">
      <div className="ins-title with-action">
        <span>
          Dependency notes
          {focusedBox !== null && attributed && <span className="muted"> · tile {focusedBox + 1}</span>}
          {omitted > 0 && !showAll && (
            <span className="muted"> · {findings.notes.length} of {findings.constraintCount} constraints</span>
          )}
        </span>
        {omitted > 0 && (
          <span className="ins-title-actions">
            <button
              className="mini"
              aria-expanded={showAll}
              onClick={() => setShowAll(!showAll)}
              title={showAll
                ? `show the ${findings.notes.length} strongest constraints only`
                : `show the ${omitted} constraint${omitted === 1 ? "" : "s"} left out of the summary`}
            >
              {showAll ? "show fewer" : `show ${omitted} more`}
            </button>
          </span>
        )}
      </div>
      {findings.notes.length ? (
        <>
          <NoteList notes={findings.notes} />
          {showAll && omitted > 0 && <NoteList notes={findings.omitted} />}
        </>
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

function NoteList({ notes }: { notes: DependencyNote[] }): React.ReactElement {
  return (
    <ul className="notes-list">
      {notes.map((note) => (
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
  );
}
