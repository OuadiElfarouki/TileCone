import React, { useEffect, useRef, useState } from "react";

/**
 * A text field that edits a value through a draft: typing changes only the
 * draft, Enter or leaving the field applies it, and Escape abandons it and
 * leaves the field - one Escape backs out of editing, as it backs out of
 * everything else. A draft that does not parse is marked and kept for
 * correction, never applied and never silently discarded.
 *
 * Only the keys the field consumes stop at it. The panel shortcuts still reach
 * the app from inside it, as they do from the source editor, and every other
 * global binding already stands down while focus is in a text field.
 */
export function DraftField<T>({
  value,
  parse,
  apply,
  label,
  title,
  invalidTitle,
  className,
  inputMode,
}: {
  /** The value as it stands, formatted: what the field shows when it is not being edited. */
  value: string;
  /** The draft read as a value, or null when it does not parse. */
  parse: (text: string) => T | null;
  /**
   * Apply a parsed draft. `explicit` is true for Enter and false for leaving
   * the field, so an owner can treat a deliberate Enter as an act even when
   * the value is unchanged; otherwise the owner ignores an unchanged value.
   */
  apply: (parsed: T, explicit: boolean) => void;
  label: string;
  title: string;
  invalidTitle: string;
  className: string;
  inputMode?: "numeric" | "text";
}): React.ReactElement {
  const [draft, setDraft] = useState(value);
  const [invalid, setInvalid] = useState(false);
  // Escape blurs the field, and the blur must not apply the draft it abandons.
  const abandoning = useRef(false);

  useEffect(() => {
    setDraft(value);
    setInvalid(false);
  }, [value]);

  const commit = (explicit: boolean) => {
    const parsed = parse(draft);
    if (parsed === null) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    apply(parsed, explicit);
  };

  return (
    <input
      className={`${className}${invalid ? " invalid" : ""}`}
      value={draft}
      inputMode={inputMode}
      aria-label={label}
      aria-invalid={invalid}
      title={invalid ? invalidTitle : title}
      spellCheck={false}
      // A click into the field is for editing it, never for what contains it.
      onClick={(event) => event.stopPropagation()}
      onChange={(event) => {
        setDraft(event.target.value);
        setInvalid(false);
      }}
      onBlur={() => {
        if (abandoning.current) {
          abandoning.current = false;
          return;
        }
        commit(false);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.stopPropagation();
          commit(true);
        } else if (event.key === "Escape") {
          event.stopPropagation();
          abandoning.current = true;
          setDraft(value);
          setInvalid(false);
          event.currentTarget.blur();
        }
      }}
    />
  );
}
