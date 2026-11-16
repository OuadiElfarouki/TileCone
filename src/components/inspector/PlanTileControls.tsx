import React, { useEffect, useMemo, useRef, useState } from "react";
import type { ResolvedGraph, ResolvedTensor } from "../../core/graph";
import { tilePlan } from "../../core/plan/plan";
import { useStore } from "../../state/store";
import { axisName } from "../../view/tensor/shape-label";
import { defaultPlanTile } from "../../view/tensor/seeds";
import { DraftField } from "../chrome/DraftField";
import "./PlanTileControls.css";

/** Names distinguish axes while preferring what the author wrote. */
export function planAxisNames(tensor: ResolvedTensor): string[] {
  const preferred = tensor.resolved.map((_, axis) => {
    const named = tensor.axisNames?.[axis];
    if (named) return named;
    const symbolic = tensor.symShape[axis];
    return tensor.symProven[axis] && typeof symbolic === "string" ? symbolic : axisName(tensor, axis);
  });
  const names = preferred.map((name, axis) =>
    preferred.filter((other) => other === name).length === 1 ? name : `ax${axis}`
  );
  // An author's name can itself be another axis's positional fallback.
  return new Set(names).size === names.length ? names : names.map((_, axis) => `ax${axis}`);
}

export function parseTileExtent(text: string): number | null {
  if (!/^\s*\d+\s*$/.test(text)) return null;
  const value = Number(text);
  return Number.isSafeInteger(value) && value >= 1 ? value : null;
}

/** Validation includes the task count, which must also fit a safe integer. */
function validTile(graph: ResolvedGraph, id: string, tile: number[]): boolean {
  try {
    tilePlan(graph, { [id]: tile });
    return true;
  } catch {
    return false;
  }
}

/** One field per axis of an existing family's tile, applied as a plan edit. */
export function PlanTileAxes({ tensor, tile }: { tensor: ResolvedTensor; tile: readonly number[] }): React.ReactElement {
  const graph = useStore((s) => s.resolved)!;
  const setPlanTile = useStore((s) => s.setPlanTile);
  const names = planAxisNames(tensor);
  return (
    <div className="plan-tile-axes" role="group" aria-label={`${tensor.name} tile`}>
      <span className="muted">tile</span>
      {tile.length === 0 ? <span className="muted">scalar</span> : tile.map((extent, axis) => (
        <label className="plan-tile-axis" key={axis}>
          <span>{names[axis]}</span>
          <DraftField
            value={String(extent)}
            parse={(text) => {
              const value = parseTileExtent(text);
              if (value === null) return null;
              const next = [...tile];
              next[axis] = value;
              return validTile(graph, tensor.id, next) ? value : null;
            }}
            apply={(value) => {
              if (value === extent) return;
              const next = [...tile];
              next[axis] = value;
              setPlanTile(tensor.id, next);
            }}
            label={`${tensor.name} tile ${names[axis]}`}
            title={`tile extent on ${names[axis]}; Enter or leaving the field applies, Escape abandons`}
            invalidTitle="expected a positive whole number and a safe task count"
            className="box-range plan-tile-axis-field"
            inputMode="numeric"
          />
        </label>
      ))}
    </div>
  );
}

/** Drafts here are local until the complete tile is applied in one action. */
function NewTileForm({ graph, tensor, onAdded }: {
  graph: ResolvedGraph;
  tensor: ResolvedTensor;
  onAdded: () => void;
}): React.ReactElement {
  const tileScale = useStore((s) => s.tileScale);
  const viewCfgs = useStore((s) => s.viewCfgs);
  const setPlanTile = useStore((s) => s.setPlanTile);
  const seed = useMemo(() => defaultPlanTile(graph, tensor.id, tileScale, viewCfgs), [graph, tensor.id, tileScale, viewCfgs]);
  const [draft, setDraft] = useState(() => seed.map(String));
  // A graph replacement can reuse the same tensor id with a different shape.
  useEffect(() => setDraft(defaultPlanTile(graph, tensor.id, tileScale, viewCfgs).map(String)), [graph, tensor.id]);
  const names = planAxisNames(tensor);
  const parsed = draft.map(parseTileExtent);
  const complete = parsed.length === tensor.resolved.length && parsed.every((n) => n !== null);
  const tile = complete ? parsed as number[] : null;
  const valid = tile !== null && validTile(graph, tensor.id, tile);
  return (
    <form className="plan-new-tile-form" onSubmit={(event) => {
      event.preventDefault();
      if (!valid || !tile) return;
      onAdded();
      setPlanTile(tensor.id, tile);
    }}>
      <div className="plan-tile-axes" role="group" aria-label={`new ${tensor.name} tile`}>
        <span className="muted">tile</span>
        {draft.length === 0 ? <span className="muted">scalar</span> : draft.map((value, axis) => (
          <label className="plan-tile-axis" key={axis}>
            <span>{names[axis]}</span>
            <input
              className={`box-range plan-tile-axis-field${parseTileExtent(value) === null ? " invalid" : ""}`}
              value={value}
              inputMode="numeric"
              aria-label={`new ${tensor.name} tile ${names[axis]}`}
              aria-invalid={parseTileExtent(value) === null}
              title="positive whole-number tile extent"
              onChange={(event) => {
                const value = event.target.value;
                setDraft((old) => old.map((v, i) => i === axis ? value : v));
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.stopPropagation();
                  setDraft((old) => old.map((v, i) => i === axis ? String(seed[axis]) : v));
                  event.currentTarget.blur();
                }
              }}
            />
          </label>
        ))}
      </div>
      {complete && !valid && <span className="hint">The task count is too large.</span>}
      <button className="mini" type="submit" disabled={!valid}>tile</button>
    </form>
  );
}

/** Choose a produced tensor and its tile before making any plan gesture. */
export function AddPlanTensor(): React.ReactElement {
  const graph = useStore((s) => s.resolved)!;
  const planTiles = useStore((s) => s.planTiles);
  const available = Object.values(graph.tensors).filter((tensor) => tensor.producer && !planTiles[tensor.id]);
  const [selected, setSelected] = useState("");
  const chosen = available.find((tensor) => tensor.id === selected) ?? available[0];
  const details = useRef<HTMLDetailsElement>(null);
  return (
    <details className="plan-add-tensor" ref={details}>
      <summary>tile tensor</summary>
      {chosen ? (
        <>
          <label className="plan-add-tensor-select">
            <span className="muted">tensor</span>
            <select aria-label="tensor to tile" value={chosen.id} onChange={(event) => setSelected(event.target.value)}>
              {available.map((tensor) => <option value={tensor.id} key={tensor.id}>{tensor.name}</option>)}
            </select>
          </label>
          <NewTileForm key={chosen.id} graph={graph} tensor={chosen} onAdded={() => {
            // If keyboard focus was inside the form it is about to disappear.
            // Pointer-used buttons have already been released by the focus policy.
            if (details.current?.contains(document.activeElement)) details.current.querySelector("summary")?.focus();
          }} />
        </>
      ) : <p className="hint">Every produced tensor is tiled.</p>}
    </details>
  );
}
