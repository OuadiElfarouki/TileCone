import React from "react";
import type { OperandReads } from "../../core/multiplicity";
import { readOnceLine, readsDetail, readsOperation, readsSummary } from "../../view/multiplicity";

/**
 * Operand reuse: how many of the outputs each operation computes read each
 * element of its operands (`core/multiplicity.ts`). Shown for the drawn tiles
 * under Dependencies and for the inspected task under Plan, which count the
 * same thing over different regions.
 *
 * Per operation rather than per tensor. What is read once and what is reused
 * is the finding, so the operands some element of which is read more than once
 * are listed, and the single-use ones are named together on one line.
 */
export function OperandReuse({
  reads,
  title,
  empty,
  nameOf,
}: {
  reads: readonly OperandReads[];
  /** The heading, which names the scope the counts are over. */
  title: string;
  /** Why there is nothing to count, for when no operation reads anything. */
  empty: string;
  nameOf: (tensorId: string) => string;
}): React.ReactElement {
  const reused = reads.filter((row) => row.reused.value !== 0);
  const single = reads.filter((row) => row.reused.value === 0);
  const once = single.length && reused.length ? readOnceLine(single, nameOf) : null;
  return (
    <div className="ins-section">
      <div className="ins-title">{title}</div>
      {reads.length === 0 ? (
        <p className="hint">{empty}</p>
      ) : reused.length > 0 ? (
        <ul className="reuse-list">
          {reused.map((row) => (
            <li key={`${row.nodeId}:${row.tensorId}`}>
              <code>{nameOf(row.tensorId)}</code>
              <span className="muted">via {readsOperation(row, nameOf)}</span>
              <span className="reads-figures" title={readsDetail(row)}>
                {readsSummary(row)}
              </span>
              {!row.exact && (
                <span className="badge approx" title={row.reasons.join("; ")}>≈</span>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="hint">Every operand element here is read by one output.</p>
      )}
      {once && (
        <p className="hint" title={once.detail}>
          Read once: {once.text}
        </p>
      )}
    </div>
  );
}
