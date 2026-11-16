import React, { useEffect, useMemo } from "react";
import { regionSliceExprs, type Figure } from "../../core/metrics";
import type { TaskRef, TilePlan } from "../../core/plan/plan";
import {
  familyPayload,
  stagingOf,
  stepOf,
  stepPayload,
  taskSteps,
  type Staging,
} from "../../core/plan/steps";
import { formatBytes, formatFigure, fmt } from "../../view/format";
import { budgetVerdict, formatBudgetBytes, parseBudgetBytes, parseBytes, parseChunk, reducedAxisNames, withChunk, type Verdict } from "../../view/plan-steps";
import { useStore } from "../../state/store";
import { DraftField } from "../chrome/DraftField";
import { useExplanation } from "../chrome/Explain";

/** Steps of a task above this are not all measured in the panel; the step
 * shown still is. */
const TASK_STEPS_MEASURED = 512;

const bytesText = (f: Figure) => formatFigure(f, formatBytes);

/** Where a payload stands against the budget, as a word beside the figure. The
 * figure carries the number; this says which side of the stated limit it is on. */
function VerdictMark({ verdict, limit, what }: { verdict: Verdict | null; limit: number | undefined; what: string }): React.ReactElement | null {
  if (!verdict || limit === undefined || verdict === "unknown") return null;
  return (
    <span
      className={`verdict ${verdict}`}
      title={
        verdict === "within"
          ? `${what} is within the stated ${formatBytes(limit)}, by payload: padding and alignment are not counted`
          : `${what} is over the stated ${formatBytes(limit)}`
      }
    >
      {verdict === "within" ? "within" : "over"}
    </span>
  );
}

/** Slices joined by a union, each its own code span, so a line breaks between
 * slices rather than inside one. */
function Slices({ exprs }: { exprs: string[] }): React.ReactElement {
  return (
    <>
      {exprs.map((expr, i) => (
        <React.Fragment key={i}>
          {i > 0 && " ∪ "}
          <code>{expr}</code>
        </React.Fragment>
      ))}
    </>
  );
}

/** The budget the engineer states: two numbers, typed as byte counts. */
export function PlanBudget(): React.ReactElement {
  const budget = useStore((s) => s.planBudget);
  const setPlanBudget = useStore((s) => s.setPlanBudget);
  return (
    <div className="plan-budget" role="group" aria-label="budget">
      <span className="plan-budget-key">budget</span>
      <label>
        <span className="muted">step</span>
        <DraftField
          value={formatBudgetBytes(budget.stepBytes)}
          parse={(text) => parseBudgetBytes(text, budget.stepBytes)}
          apply={(stepBytes) => setPlanBudget({ ...budget, stepBytes })}
          label="bytes a task may hold of its operands per step"
          title="bytes a task may hold of its operands at a time: what one step of its contraction reads. Typed as 99 KB, 1.5 MB or a byte count"
          invalidTitle="expected a byte count such as 99 KB, 1.5 MB or 4096"
          className="box-range plan-budget-field"
        />
      </label>
      <label>
        <span className="muted">accumulator</span>
        <DraftField
          value={budget.accumulatorBytes === undefined ? "no limit" : formatBudgetBytes(budget.accumulatorBytes)}
          parse={(text) => (/^\s*(|none|no limit)\s*$/i.test(text) ? ("none" as const) :
            budget.accumulatorBytes === undefined ? parseBytes(text) : parseBudgetBytes(text, budget.accumulatorBytes))}
          apply={(parsed) =>
            setPlanBudget(parsed === "none" ? { stepBytes: budget.stepBytes } : { ...budget, accumulatorBytes: parsed })
          }
          label="bytes an accumulator may take"
          title="bytes one task's running sum may take; empty for no limit. Without a limit the sweep is led by one task computing a whole output"
          invalidTitle="expected a byte count such as 16 KB, or nothing for no limit"
          className="box-range plan-budget-field"
        />
      </label>
    </div>
  );
}

/**
 * One tiled tensor's steps, under its row: a chunk per reduced axis, the steps
 * per task that gives, and the family's largest step and accumulator against
 * the budget. A tensor without steps says why in a word, and in full on hover.
 */
export function TensorSteps({ plan, tensorId }: { plan: TilePlan; tensorId: string }): React.ReactElement {
  const budget = useStore((s) => s.planBudget);
  const setPlanSteps = useStore((s) => s.setPlanSteps);
  const staging: Staging = useMemo(() => stagingOf(plan, tensorId), [plan, tensorId]);
  const payload = useMemo(() => familyPayload(plan, tensorId), [plan, tensorId]);
  const names = useMemo(() => (staging.staged ? reducedAxisNames(plan, staging) : []), [plan, staging]);
  const name = plan.graph.tensors[tensorId].name;
  const declared = plan.steps.get(tensorId);

  const measured = payload.status === "measured" ? payload : null;
  const stepVerdict = measured ? budgetVerdict(measured.stepBytes, budget.stepBytes) : null;
  const accVerdict = measured?.accumulatorBytes ? budgetVerdict(measured.accumulatorBytes, budget.accumulatorBytes) : null;

  return (
    <div className="plan-steps">
      {staging.staged ? (
        <>
          {staging.reduced.map((axis, i) => (
            <label className="plan-chunk" key={axis.label}>
              <span className="muted">chunk</span>
              {/* One reduced axis the source never named needs no name here: the
                  loop label would be the operation's own, not the author's. */}
              {!(staging.reduced.length === 1 && names[i] === axis.label) && <code>{names[i]}</code>}
              <DraftField
                value={axis.chunk === axis.extent ? "all" : String(axis.chunk)}
                parse={(text) => parseChunk(text, axis.extent)}
                apply={(chunk) => setPlanSteps(tensorId, withChunk(declared, axis.label, chunk))}
                label={`${name} chunk along ${names[i]}`}
                title={`elements of ${names[i]} one step of a ${name} task reduces, of ${axis.extent}; "all" takes the axis in one step`}
                invalidTitle={`expected a whole number from 1 to ${axis.extent}, or "all"`}
                className="box-range plan-chunk-field"
                inputMode="numeric"
              />
              <span className="muted">of {axis.extent}</span>
            </label>
          ))}
          <span className="muted">
            {staging.steps} step{staging.steps === 1 ? "" : "s"} per task
          </span>
        </>
      ) : (
        <span className="muted" title={`${staging.reason}, so a task has no reduction to divide`}>
          one step per task
        </span>
      )}
      {measured ? (
        <span className="plan-steps-figures">
          <span title={measured.stepBytes.reasons.join("; ") || "the largest step over every task of this tensor"}>
            <span className="muted">largest step</span> {bytesText(measured.stepBytes)}
            <VerdictMark verdict={stepVerdict} limit={budget.stepBytes} what={`the largest step of ${name}`} />
          </span>
          {measured.accumulatorBytes && staging.staged && (
            <span title={`the running sum of the largest tile, held as ${staging.accumulator}`}>
              <span className="muted">accumulator</span> {bytesText(measured.accumulatorBytes)}
              <VerdictMark verdict={accVerdict} limit={budget.accumulatorBytes} what={`the accumulator of ${name}`} />
            </span>
          )}
        </span>
      ) : (
        <span className="plan-steps-figures muted" title="the largest step is taken over every task and step, and is not sampled; inspect a task to see its own steps">
          <span>largest step not measured: {payload.status === "unmeasured" ? payload.reason : ""}</span>
        </span>
      )}
    </div>
  );
}

/**
 * The inspected task's steps: how many, what one of them reads and needs, the
 * accumulator they build, and what the epilogue reads once at the end.
 */
export function TaskStepsSection({ plan, task }: { plan: TilePlan; task: TaskRef }): React.ReactElement {
  const budget = useStore((s) => s.planBudget);
  const steps = useMemo(() => taskSteps(plan, task), [plan, task]);
  const staged = "staging" in steps ? steps : null;
  const count = staged?.staging.steps ?? 1;
  // The step shown is the store's, so the cards paint the one the panel
  // describes. It belongs to this task of this plan; any other starts at the
  // first step, and a task taken in one step shows none.
  const planStep = useStore((s) => s.planStep);
  const setPlanStep = useStore((s) => s.setPlanStep);
  const current = planStep && planStep.plan === plan && planStep.task === task ? planStep.index : null;
  const index = current === null ? 0 : Math.min(current, count - 1);
  const setIndex = (next: number) => setPlanStep(Math.max(0, Math.min(count - 1, next)));
  useEffect(() => {
    if (staged && count > 1) {
      if (current === null) setPlanStep(0);
    } else if (planStep) setPlanStep(null);
  }, [staged, count, current, planStep, setPlanStep]);
  // Leaving the panel leaves no step on the cards.
  useEffect(() => () => setPlanStep(null), [setPlanStep]);

  const names = useMemo(() => (staged ? reducedAxisNames(plan, staged.staging) : []), [plan, staged]);
  const step = useMemo(() => (staged ? stepOf(plan, staged, index) : null), [plan, staged, index]);
  const payload = useMemo(() => (step ? stepPayload(plan, step) : null), [plan, step]);
  /** The largest step of this task, over all of them, where there are few
   * enough to measure here. */
  const largest = useMemo(() => {
    if (!staged || count > TASK_STEPS_MEASURED) return null;
    let best = { index: 0, bytes: -1 };
    let smallest = Infinity;
    let exact = true;
    for (let i = 0; i < count; i++) {
      const bytes = stepPayload(plan, stepOf(plan, staged, i)).bytes;
      if (bytes.value === null) return null;
      if (bytes.status !== "exact") exact = false;
      if (bytes.value > best.bytes) best = { index: i, bytes: bytes.value };
      smallest = Math.min(smallest, bytes.value);
    }
    // Where every step reads the same there is no largest one to go to.
    return { ...best, exact, uniform: smallest === best.bytes };
  }, [plan, staged, count]);

  const explain = useExplanation(
    "steps",
    <>
      A task's contraction sums over its reduced axes. A chunk divides such an axis, and a step is
      the task's tile with one chunk of each: what a step reads is what any kernel running it
      touches at a time, whatever it does with it. The accumulator is the running sum over the
      tile, complete after the last step. Work after the contraction - a bias, an activation, a
      cast - applies to the finished sum, and what it reads is read once per task. Set chunks
      beside the tensor's tile above.
    </>
  );
  const nameOf = (tensorId: string) => plan.graph.tensors[tensorId].name;

  return (
    <section className="ins-section">
      <div className="ins-title with-action">
        Steps
        <span className="ins-title-actions">{explain.button}</span>
      </div>
      {explain.panel}
      {!staged || !step || !payload ? (
        <p className="hint">
          One step: {"reason" in steps ? steps.reason : ""}. What this task reads, above, is what it
          holds.
        </p>
      ) : (
        <>
          <dl className="plan-facts">
            <dt>steps</dt>
            <dd>
              {count}
              <span className="muted">
                {staged.staging.reduced.map((axis, i) =>
                  axis.chunk === axis.extent ? "" : ` · ${axis.chunk} of ${names[i]} ${axis.extent}`
                )}
              </span>
            </dd>
            <dt>accumulator</dt>
            <dd>
              {bytesText(staged.accumulator.bytes)}{" "}
              <VerdictMark
                verdict={budgetVerdict(staged.accumulator.bytes, budget.accumulatorBytes)}
                limit={budget.accumulatorBytes}
                what="this task's accumulator"
              />
              <span className="muted"> · {staged.accumulator.dtype}</span>
              <pre className="plan-slice">
                {nameOf(staged.staging.output)}[{staged.outBox.map((I) => `${I.lo}:${I.hi}`).join(", ")}]
              </pre>
            </dd>
          </dl>
          {count > 1 && (
            <div className="plan-step-nav" role="group" aria-label="step">
              <button className="mini" onClick={() => setIndex(index - 1)} disabled={index === 0} aria-label="previous step" title="previous step">‹</button>
              <input
                type="range"
                min={0}
                max={count - 1}
                value={index}
                onChange={(e) => setIndex(Number(e.target.value))}
                aria-label="step of this task"
              />
              <button className="mini" onClick={() => setIndex(index + 1)} disabled={index === count - 1} aria-label="next step" title="next step">›</button>
            </div>
          )}
          <p className="plan-step-where">
            step {index + 1} of {count}
            <span className="muted">
              {staged.staging.reduced.map((axis, i) => {
                const I = step.box[axis.axis];
                return ` · ${names[i]} ${I.lo}:${I.hi}`;
              })}
            </span>
          </p>
          <p className="plan-op">
            <span className="muted">this step reads</span> {bytesText(payload.bytes)}{" "}
            <VerdictMark verdict={budgetVerdict(payload.bytes, budget.stepBytes)} limit={budget.stepBytes} what="this step" />
            <span className="muted"> · {formatFigure(step.flops, fmt)} FLOP</span>
          </p>
          {payload.tensors.map((t) => (
            <div className="plan-demand" key={t.tensorId}>
              <div className="plan-demand-head">
                <code>{nameOf(t.tensorId)}</code>
                <span className="muted">{bytesText(t.bytes)}</span>
                {!t.region.exact && <span className="badge approx" title={t.region.reasons.join("; ")}>≈</span>}
              </div>
              <pre className="plan-slice">{regionSliceExprs(nameOf(t.tensorId), t.region).join("\n")}</pre>
            </div>
          ))}
          {step.needs.length > 0 && (
            <p className="hint plan-steps-slices" title="untiled tensors between what the step reads and the contraction; whether a kernel keeps them or recomputes them is not modelled">
              needs{" "}
              {step.needs.map((need, i) => (
                <React.Fragment key={need.tensorId}>
                  {i > 0 && ", "}
                  <Slices exprs={regionSliceExprs(nameOf(need.tensorId), need.region)} />
                </React.Fragment>
              ))}
              , computed in the task
            </p>
          )}
          {count > 1 && (
            <p className="hint">
              {largest ? (
                <>
                  {largest.uniform ? "Every step of this task reads" : "Largest step of this task:"}{" "}
                  {largest.exact ? "" : "≤ "}{formatBytes(largest.bytes)}
                  {!largest.uniform && largest.index !== index && (
                    <>
                      {" "}
                      <button className="mini" onClick={() => setIndex(largest.index)} title="show that step">
                        step {largest.index + 1}
                      </button>
                    </>
                  )}
                </>
              ) : (
                <>More than {TASK_STEPS_MEASURED} steps: only the step shown is measured here.</>
              )}
            </p>
          )}
          {staged.epilogue.length > 0 && (
            <p className="hint plan-steps-slices" title="read by the work after the contraction, once per task and in no step">
              After the last step, read once:{" "}
              {staged.epilogue.map((read, i) => (
                <React.Fragment key={`${read.node}/${read.slot}`}>
                  {i > 0 && ", "}
                  <Slices exprs={regionSliceExprs(nameOf(read.tensorId), read.region)} />
                </React.Fragment>
              ))}
            </p>
          )}
        </>
      )}
    </section>
  );
}
