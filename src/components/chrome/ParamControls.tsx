import React, { useEffect, useMemo, useState } from "react";
import { useStore } from "../../state/store";
import { idRecord } from "../../view/workspace";
import { STEP_LABEL, StepMode, stepModeOf, stepValue, withParamNamed } from "../../view/params";

/** A field's text as a parameter value, or `null` while it is not one. */
const parsed = (text: string): number | null => {
  const value = Number(text.trim());
  return /^\d+$/.test(text.trim()) && Number.isSafeInteger(value) && value >= 1 ? value : null;
};

/**
 * One value box per size parameter used by the built graph, under the editor.
 *
 * The boxes belong to the build, not to the draft: they appear once a program
 * has compiled, show its values, and change only when the next run does - the
 * panel remounts this per build, which also drops anything entered here. An
 * edit rewrites that parameter's number in the draft, found by name, which
 * enables Run exactly as typing it would; text typed in the editor meanwhile
 * does not reach the boxes. The source panel hides them when the draft changes
 * anything beyond parameter values, so a new graph cannot use the old build's
 * controls. They reappear after a successful build or restoring that source.
 *
 * The ▴/▾ steppers and the arrow keys move by the value's step mode (see
 * `stepModeOf`), read when a run of steps begins and held until it ends: the
 * pointer leaves the row, focus leaves the field, or a value is typed.
 */
export function ParamControls(): React.ReactElement | null {
  const built = useStore((s) => s.resolved?.params);
  const shapeParameters = useStore((s) => s.resolved?.shapeParameters);
  const used = useMemo(() => new Set(shapeParameters), [shapeParameters]);
  const draft = useStore((s) => s.draftText);
  const setDraft = useStore((s) => s.setDraftText);
  const [fields, setFields] = useState(() => idRecord<string>());
  const [runs, setRuns] = useState(() => idRecord<StepMode>());

  // Running the same text also installs a new build, without changing the
  // parent's key. Drop local edits and stepping modes for that build too.
  useEffect(() => {
    setFields(idRecord<string>());
    setRuns(idRecord<StepMode>());
  }, [built]);

  const params = Object.entries(built ?? {}).filter(
    ([name, value]) => used.has(name) && Number.isSafeInteger(value) && value >= 1
  );
  if (!params.length) return null;

  const endRun = (name: string) =>
    setRuns((current) => {
      if (!(name in current)) return current;
      const next = idRecord(current);
      delete next[name];
      return next;
    });

  const write = (name: string, value: number) => {
    const text = withParamNamed(draft, name, value);
    if (text !== null && text !== draft) setDraft(text);
  };

  const step = (name: string, current: number, direction: 1 | -1) => {
    const mode = runs[name] ?? stepModeOf(current);
    const next = stepValue(current, mode, direction);
    setRuns((r) => idRecord({ ...r, [name]: mode }));
    setFields((f) => idRecord({ ...f, [name]: String(next) }));
    write(name, next);
  };

  return (
    <section className="param-controls" aria-label="size parameters">
      <span className="setup-kicker">parameters</span>
      {params.map(([name, builtValue]) => {
        const text = fields[name] ?? String(builtValue);
        const value = parsed(text);
        const mode = runs[name] ?? (value === null ? null : stepModeOf(value));
        const pending = value !== null && value !== builtValue;
        const inDraft = withParamNamed(draft, name, builtValue) !== null;
        const id = `param-${name}`;
        return (
          <div
            key={name}
            className={`param-row${pending ? " pending" : ""}`}
            onPointerLeave={() => endRun(name)}
            onBlur={(e) => {
              // Moving between the field and either button continues the run.
              if (!e.currentTarget.contains(e.relatedTarget as Node | null)) endRun(name);
            }}
          >
            <label className="param-name" htmlFor={id}>{name}</label>
            <input
              id={id}
              className="param-field"
              type="text"
              inputMode="numeric"
              spellCheck={false}
              value={text}
              aria-invalid={value === null}
              title={!inDraft
                ? `${name} is no longer in the source; this edit has nowhere to go`
                : pending
                  ? `${name} = ${value} in the source, ${builtValue} in the built graph; Run builds it`
                  : `sets ${name} in the source; Run builds it`}
              onChange={(e) => {
                const typed = e.target.value;
                setFields((f) => idRecord({ ...f, [name]: typed }));
                endRun(name);
                const next = parsed(typed);
                if (next !== null) write(name, next);
              }}
              onKeyDown={(e) => {
                if (value === null || (e.key !== "ArrowUp" && e.key !== "ArrowDown")) return;
                e.preventDefault();
                step(name, value, e.key === "ArrowUp" ? 1 : -1);
              }}
            />
            <span className="param-stepper">
              <button
                type="button"
                tabIndex={-1}
                aria-label={`increase ${name}`}
                disabled={value === null}
                onClick={() => value !== null && step(name, value, 1)}
              >
                ▴
              </button>
              <button
                type="button"
                tabIndex={-1}
                aria-label={`decrease ${name}`}
                disabled={value === null}
                onClick={() => value !== null && step(name, value, -1)}
              >
                ▾
              </button>
            </span>
            <span
              className="param-step"
              title="step for ▴/▾ and the arrow keys: doubling for a power of two, 32 for other even values, 1 for odd ones; kept while you keep stepping"
            >
              {mode ? STEP_LABEL[mode] : ""}
            </span>
          </div>
        );
      })}
    </section>
  );
}
