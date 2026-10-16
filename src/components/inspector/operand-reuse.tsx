import React from "react";
import type { OperandReads } from "../../core/multiplicity";
import { formatFigure, fmt } from "../../view/format";
import { readOnceLine, readsDetail, readsOperation, readsSummary } from "../../view/multiplicity";
import { lensRamp, sameLens, type ReuseLens } from "../../view/reuse-lens";
import type { InspectorTab } from "../../view/workspace";
import { rgbCss } from "../../view/palette";
import { useDark, useStore } from "../../state/store";
import { useLensPaint } from "../card/useLensPaint";

/**
 * Operand reuse: how many of the outputs each operation computes read each
 * element of its operands (`core/multiplicity.ts`). Shown for the drawn tiles
 * under Dependencies and for the inspected task under Plan, which count the
 * same thing over different regions.
 *
 * Per operation rather than per tensor. What is read once and what is reused
 * is the finding, so the operands some element of which is read more than once
 * are listed, and the single-use ones are named together on one line.
 *
 * A reused row's figures also switch the reuse lens: the same count drawn per
 * element on the operand's card (`view/reuse-lens.ts`), with its legend here.
 */
export function OperandReuse({
  reads,
  title,
  empty,
  nameOf,
  view,
}: {
  reads: readonly OperandReads[];
  /** The heading, which names the scope the counts are over. */
  title: string;
  /** Why there is nothing to count, for when no operation reads anything. */
  empty: string;
  nameOf: (tensorId: string) => string;
  /** The view the rows belong to, which a lens opened from them is scoped by. */
  view: InspectorTab;
}): React.ReactElement {
  const lens = useStore((s) => s.reuseLens);
  const setReuseLens = useStore((s) => s.setReuseLens);
  const keyOf = (row: OperandReads): ReuseLens => ({ view, nodeId: row.nodeId, tensorId: row.tensorId });
  // The row the lens draws stays a row with its switch and legend even once
  // its count falls to one, which a smaller tile can do: the card is still
  // shaded, and the reader needs the legend and the way to turn it off.
  const listed = (row: OperandReads) => row.reused.value !== 0 || sameLens(lens, keyOf(row));
  const reused = reads.filter(listed);
  const single = reads.filter((row) => !listed(row));
  const once = single.length && reused.length ? readOnceLine(single, nameOf) : null;
  return (
    <div className="ins-section">
      <div className="ins-title">{title}</div>
      {reads.length === 0 ? (
        <p className="hint">{empty}</p>
      ) : reused.length > 0 ? (
        <ul className="reuse-list">
          {reused.map((row) => {
            const on = sameLens(lens, keyOf(row));
            return (
              <li key={`${row.nodeId}:${row.tensorId}`} className={on ? "lens-on" : undefined}>
                <code>{nameOf(row.tensorId)}</code>
                <span className="muted">via {readsOperation(row, nameOf)}</span>
                <button
                  className="reads-figures lens-toggle"
                  aria-pressed={on}
                  title={`${readsDetail(row)}\n${on
                    ? "click again or press Esc to clear the shading"
                    : `shade ${nameOf(row.tensorId)}'s card by how many of these outputs use each element`}`}
                  onClick={() => setReuseLens(on ? null : keyOf(row))}
                >
                  {readsSummary(row)}
                </button>
                {!row.exact && (
                  <span className="badge approx" title={row.reasons.join("; ")}>≈</span>
                )}
                {on && <LensLegend row={row} nameOf={nameOf} />}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="hint">Every operand element here is used by one output.</p>
      )}
      {once && (
        <p className="hint" title={once.detail}>
          Used once: {once.text}
        </p>
      )}
    </div>
  );
}

/**
 * What the shading on the card means: the ramp with its two ends, the row it
 * draws, and the readings that qualify it. The ends are the row's own minimum
 * and maximum, so the legend and the figures beside it cannot disagree.
 */
function LensLegend({ row, nameOf }: { row: OperandReads; nameOf: (tensorId: string) => string }): React.ReactElement {
  const dark = useDark();
  const paint = useLensPaint(row.tensorId);
  const ramp = lensRamp(dark).map((c) => rgbCss(c)).join(", ");
  const uniform = row.min.value === row.max.value;
  const name = nameOf(row.tensorId);
  return (
    <div className="lens-legend">
      <div className="lens-scale">
        <span className="lens-ramp" aria-hidden style={{ background: `linear-gradient(to right, ${ramp})` }} />
        <span className="muted">
          {uniform
            ? `${formatFigure(row.max, fmt)} use${row.max.value === 1 ? "" : "s"} per element, wherever it is read`
            : `0 → ${formatFigure(row.max, fmt)} uses per element`}
        </span>
      </div>
      <p className="hint">
        Shaded on {name}&apos;s card: how many outputs of {readsOperation(row, nameOf)} use each
        element. A cell standing for several hidden indices shows the largest; hover the card to read
        a count.
      </p>
      {!row.exact && (
        <p className="hint overlap">Upper bounds, hatched on the card: {row.reasons.join("; ")}.</p>
      )}
      {paint && !paint.complete && (
        <p className="hint overlap">
          The count changes too often across {name} to draw at this size; the figures above still hold.
        </p>
      )}
    </div>
  );
}
