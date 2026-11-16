import React, { useMemo } from "react";
import type { ResolvedGraph } from "../../core/graph";
import { tilePlan } from "../../core/plan/plan";
import { stagingOf } from "../../core/plan/steps";
import type { SweepRow } from "../../core/plan/sweep";
import { fmt, formatBytes, formatFigure } from "../../view/format";
import { reducedAxisNames } from "../../view/plan-steps";
import { rowIsCurrent, sweepCaveats, sweepSummary } from "../../view/plan-sweep";
import { useStore } from "../../state/store";
import { useExplanation } from "../chrome/Explain";

/** A row's plan in words: each tiled tensor with its extents, and its chunks
 * by the names the source gave the reduced axes. */
function describeRow(graph: ResolvedGraph, row: SweepRow): { name: string; tile: string; chunks: string }[] {
  const plan = tilePlan(graph, row.tiles, row.steps);
  return Object.keys(row.tiles)
    .map((id) => {
      const staging = stagingOf(plan, id);
      const names = staging.staged ? reducedAxisNames(plan, staging) : [];
      const chunks = staging.staged
        ? staging.reduced.flatMap((axis, i) => (axis.chunk === axis.extent ? [] : [`${names[i]} ${axis.chunk}`]))
        : [];
      return {
        name: graph.tensors[id].name,
        tile: row.tiles[id].length ? row.tiles[id].join("×") : "scalar",
        chunks: chunks.join(", "),
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * The plans within the stated budget that no other beats, on request. A row is
 * a whole plan - tiling and chunks - with the figures plans are compared by,
 * and installing one is an ordinary plan edit.
 */
export function PlanSweep(): React.ReactElement {
  const resolved = useStore((s) => s.resolved)!;
  const sweep = useStore((s) => s.sweep);
  const planTiles = useStore((s) => s.planTiles);
  const planSteps = useStore((s) => s.planSteps);
  const budget = useStore((s) => s.planBudget);
  const runSweep = useStore((s) => s.runSweep);
  const clearSweep = useStore((s) => s.clearSweep);
  const installSweepRow = useStore((s) => s.installSweepRow);
  const tiled = Object.keys(planTiles).length;

  const rows = useMemo(
    () => (sweep.status === "done" ? sweep.result.rows.map((row) => ({ row, parts: describeRow(resolved, row) })) : []),
    [sweep, resolved]
  );
  const explain = useExplanation(
    "the sweep",
    <>
      The sweep tries power-of-two tile extents for the tiled tensors, takes for each the largest
      chunks whose step is within the budget, and lists the plans no other beats on recomputed
      FLOPs, bytes read and written, task count and largest step. "These tensors" keeps which
      tensors are tiled; "every fusion choice" also tries fusing and materializing intermediates.
      Every row is the plan's own report. What the sweep did not look at is listed under the rows.
    </>
  );

  return (
    <div className="ins-section">
      <div className="ins-title with-action">
        Plans within the budget
        <span className="ins-title-actions">{explain.button}</span>
      </div>
      {explain.panel}
      {sweep.status === "idle" && (
        <p className="hint">
          The plans within {formatBytes(budget.stepBytes)} per step
          {budget.accumulatorBytes === undefined ? "" : ` and ${formatBytes(budget.accumulatorBytes)} per accumulator`} that
          no other plan beats.
        </p>
      )}
      <div className="plan-sweep-actions">
        {sweep.status === "running" ? (
          <button className="mini" onClick={clearSweep} title="stop the sweep">stop</button>
        ) : (
          <>
            <span className="muted">sweep</span>
            {tiled > 0 && (
              <button
                className="mini"
                onClick={() => void runSweep("current")}
                title="sweep tile extents and chunks, keeping which tensors are tiled"
              >
                these tensors
              </button>
            )}
            <button
              className="mini"
              onClick={() => void runSweep("all")}
              title="also try fusing and materializing intermediates"
            >
              every fusion choice
            </button>
            {sweep.status !== "idle" && (
              <button className="mini" onClick={clearSweep} title="dismiss these rows">dismiss</button>
            )}
          </>
        )}
      </div>
      {sweep.status === "running" && <p className="hint">Measuring tile extents…</p>}
      {sweep.status === "failed" && <p className="hint overlap">The sweep failed: {sweep.message}</p>}
      {sweep.status === "done" && (
        <>
          {rows.length === 0 ? (
            // "None within the budget" is a finding about the budget only when
            // something was measured; when nothing was, it is the sweep that
            // fell short, and the lines below say how.
            <p className="hint">
              {sweep.result.counts.measured === 0
                ? "The sweep measured nothing, so it says nothing about the budget."
                : `No plan is within the budget among the ${sweep.result.counts.measured.toLocaleString("en-US")} tile extents measured.`}
            </p>
          ) : (
            <div className="plan-sweep-wrap">
              <div className="plan-sweep" role="table" aria-label="plans within the budget">
                <div className="plan-sweep-row head" role="row">
                  <span role="columnheader">tasks</span>
                  <span role="columnheader" title="FLOPs more than one task computes">recomp.</span>
                  <span role="columnheader" title="bytes read by the tasks, summed, plus bytes written">moved</span>
                  <span role="columnheader" title="the largest step over every task">step</span>
                  <span role="columnheader" aria-label="use" />
                </div>
                {rows.map(({ row, parts }, index) => {
                  const current = rowIsCurrent(row, planTiles, planSteps);
                  const moved = row.total.read.value! + row.total.written.value!;
                  return (
                    <React.Fragment key={index}>
                      <div className={`plan-sweep-plan${current ? " current" : ""}`}>
                        {parts.map((part) => (
                          <span key={part.name} className="plan-sweep-part">
                            <code>{part.name}</code> {part.tile}
                            {part.chunks && <span className="muted"> · {part.chunks}</span>}
                          </span>
                        ))}
                        {row.accumulatorBytes !== null && (
                          <span className="plan-sweep-acc muted" title="the largest accumulator among the plan's tasks">
                            acc {formatBytes(row.accumulatorBytes)}
                          </span>
                        )}
                      </div>
                      <div className={`plan-sweep-row${current ? " current" : ""}`} role="row">
                        <span role="cell">{row.total.tasks}</span>
                        <span role="cell">{formatFigure(row.total.recomputed, fmt)}</span>
                        <span role="cell">{formatBytes(moved)}</span>
                        <span role="cell">{formatBytes(row.stepBytes)}</span>
                        <span role="cell">
                          {current ? (
                            <span className="muted">current</span>
                          ) : (
                            <button
                              className="mini"
                              onClick={() => installSweepRow(index)}
                              title="make this the plan; undo brings the present one back"
                            >
                              use
                            </button>
                          )}
                        </span>
                      </div>
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          )}
          {rows.length > 0 && tiled > 0 && !rows.some(({ row }) => rowIsCurrent(row, planTiles, planSteps)) && (
            <p className="hint">
              The current plan is not among them: it is over the budget, outside the extents the sweep
              tries, or another plan is no worse on every figure.
            </p>
          )}
          <p className="hint">{sweepSummary(sweep.result)}.</p>
          {sweepCaveats(sweep.result.counts).map((caveat) => (
            <p className="hint plan-sweep-caveat" key={caveat}>{caveat}</p>
          ))}
        </>
      )}
    </div>
  );
}
