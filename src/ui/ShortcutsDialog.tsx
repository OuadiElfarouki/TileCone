import React, { useEffect, useRef } from "react";

const GROUPS = [
  {
    title: "Selection",
    items: [
      ["Arrow keys", "move the focused tile"],
      ["Shift + arrow", "move it eight steps"],
      ["H", "show or hide the focused tile's needs and feeds"],
      ["Esc", "cancel a gesture, unpin a tile, or leave a field"],
      ["Ctrl/Cmd + Z", "undo the last tile or tensor move"],
    ],
  },
  {
    title: "View",
    items: [
      ["U", "toggle What it needs"],
      ["D", "toggle What it feeds"],
      ["F", "fit the graph to the viewport"],
      ["[ / ]", "scrub the first hidden tensor axis"],
      ["Scroll", "zoom around the pointer"],
      ["?", "open this shortcut sheet"],
    ],
  },
  {
    title: "Panels",
    items: [
      ["Alt + 1", "toggle the source panel"],
      ["Alt + 2", "toggle the tile inspector"],
      ["Ctrl/Cmd + Enter", "run edited graph source"],
    ],
  },
] as const;

export function ShortcutsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}): React.ReactElement | null {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);

  if (!open) return null;
  return (
    <div className="shortcut-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section
        className="shortcut-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcut-title"
      >
        <header>
          <h2 id="shortcut-title" className="panel-title">Keyboard shortcuts</h2>
          <button ref={closeRef} className="mini" onClick={onClose} aria-label="close shortcuts">×</button>
        </header>
        <div className="shortcut-groups">
          {GROUPS.map((group) => (
            <section key={group.title}>
              <h3>{group.title}</h3>
              <dl>
                {group.items.map(([keys, action]) => (
                  <React.Fragment key={keys}>
                    <dt><kbd>{keys}</kbd></dt>
                    <dd>{action}</dd>
                  </React.Fragment>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </section>
    </div>
  );
}
