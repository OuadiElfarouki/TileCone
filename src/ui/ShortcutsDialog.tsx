import React, { useEffect, useRef } from "react";
import { SHORTCUT_GROUPS } from "./shortcuts";

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
          {SHORTCUT_GROUPS.map((group) => (
            <section key={group.title}>
              <h3>{group.title}</h3>
              <dl>
                {group.items.map((item) => (
                  <React.Fragment key={item.id}>
                    <dt><kbd>{item.label}</kbd></dt>
                    <dd>{item.action}</dd>
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
