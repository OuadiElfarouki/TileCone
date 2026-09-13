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
  const dialogRef = useRef<HTMLElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const backdrop = backdropRef.current;
    const siblings = backdrop?.parentElement
      ? [...backdrop.parentElement.children].filter((element) => element !== backdrop)
      : [];
    const previousInert = siblings.map((element) => (element as HTMLElement).inert);
    siblings.forEach((element) => { (element as HTMLElement).inert = true; });
    closeRef.current?.focus();
    return () => {
      siblings.forEach((element, index) => {
        (element as HTMLElement).inert = previousInert[index];
      });
      returnFocusRef.current?.focus();
      returnFocusRef.current = null;
    };
  }, [open]);

  if (!open) return null;
  return (
    <div
      className="shortcut-backdrop"
      ref={backdropRef}
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <section
        ref={dialogRef}
        className="shortcut-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcut-title"
        onKeyDown={(event) => {
          if (event.key !== "Tab") return;
          const focusable = [...(dialogRef.current?.querySelectorAll<HTMLElement>(
            'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
          ) ?? [])];
          if (!focusable.length) {
            event.preventDefault();
            return;
          }
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          if (focusable.length === 1 || (event.shiftKey && document.activeElement === first)) {
            event.preventDefault();
            last.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
          }
        }}
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
              {"note" in group && <p className="hint">{group.note}</p>}
            </section>
          ))}
        </div>
      </section>
    </div>
  );
}
