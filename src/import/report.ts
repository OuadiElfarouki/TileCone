import {
  ImportEntry,
  ImportReport,
  barrierNodes,
  countByKind,
} from "./types";

/**
 * The import report, rendered.
 *
 * Kept apart from the types on purpose. The app holds a structured report, and
 * a structured report is what lets a finding be clicked back to the node it is
 * about; this module is the one place that turns it into words, so the panel,
 * a copied summary and a test assertion all read the same sentence.
 *
 * The order below is the order the conversion happened in: what was mapped,
 * what was evaluated to make a mapping possible, what was rewritten, what was
 * assumed, and what was left as a bound. `unknown` comes last because it is not
 * a category of its own - it is the consequence of the barriers above it, and
 * stating it separately is what stops a reader from taking a FLOP total through
 * a barrier as a count.
 */

export type ReportLine = {
  /** The category, as the left column. */
  label: string;
  text: string;
  /**
   * Converted-graph nodes this line is about, for highlighting on canvas.
   *
   * This is the import path's answer to source spans: a DSL diagnostic
   * underlines the text it is about, and an imported graph has no text, so a
   * line points at nodes instead.
   */
  nodes: string[];
};

/** `a`, `a and b`, `a, b and c` - a list a person would read aloud. */
function listOf(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** The distinct source operations an entry family names, most frequent first. */
function tally(entries: ImportEntry[], nameOf: (entry: ImportEntry) => string): string[] {
  const counts = new Map<string, number>();
  for (const entry of entries) {
    const name = nameOf(entry);
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name, n]) => (n > 1 ? `${name} ×${n}` : name));
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/**
 * The one-line header: what was opened, and how many operations it became.
 *
 * Source nodes and operations are both stated even when they are equal. A
 * rewrite inserts nodes the model does not contain, and a header that collapsed
 * the two whenever they happened to agree would make the insertion visible only
 * sometimes, which is worse than never.
 */
export function reportHeadline(report: ImportReport): string {
  const { origin } = report;
  const parts = [origin.fileName];
  if (origin.format !== "json") parts.push(origin.format);
  if (origin.opset !== undefined) parts.push(`opset ${origin.opset}`);
  parts.push(
    `${plural(report.sourceNodes, "source node")} → ${plural(report.operations, "operation")}`
  );
  return parts.join(" · ");
}

export function reportLines(report: ImportReport): ReportLine[] {
  const lines: ReportLine[] = [];
  const counts = countByKind(report);
  const of = <K extends ImportEntry["kind"]>(kind: K) =>
    report.entries.filter((entry): entry is Extract<ImportEntry, { kind: K }> => entry.kind === kind);

  if (counts.mapped) {
    const mapped = of("mapped");
    const checked = mapped.filter((entry) => entry.shapeChecked).length;
    lines.push({
      label: "mapped",
      // The cross-check is stated as a count rather than a yes: it is
      // unavailable wherever the model carried no shape to compare, and
      // "shapes cross-checked" over a model that mostly had none would claim a
      // guardrail that did not run.
      text:
        `${plural(mapped.length, "source node")}, ` +
        (checked === mapped.length
          ? "shapes cross-checked against the model"
          : checked === 0
            ? "no shapes in the model to cross-check"
            : `${checked} with shapes cross-checked against the model`),
      nodes: mapped.map((entry) => entry.node),
    });
  }

  if (counts.evaluated) {
    const evaluated = of("evaluated");
    lines.push({
      label: "evaluated",
      text:
        `${plural(evaluated.length, "initializer")} into attributes ` +
        `(${listOf(tally(evaluated, (entry) => (entry as { attribute: string }).attribute))})`,
      nodes: evaluated.map((entry) => entry.node),
    });
  }

  if (counts.rewritten) {
    const rewritten = of("rewritten");
    lines.push({
      label: "rewritten",
      text:
        `${plural(rewritten.length, "source node")} · ` +
        listOf(rewritten.map((entry) => `${entry.sourceOp} → ${entry.ops.join(", ")}`)),
      nodes: rewritten.flatMap((entry) => entry.nodes),
    });
  }

  if (counts.bound) {
    const bound = of("bound");
    lines.push({
      label: "bound",
      text:
        `${plural(bound.length, "dimension")} · ` +
        listOf(
          bound.map(
            (entry) =>
              `${entry.dimension} = ${entry.value}${entry.assumed ? " (assumption)" : ""}`
          )
        ),
      nodes: [],
    });
  }

  if (counts.barrier) {
    const barriers = of("barrier");
    lines.push({
      label: "barriers",
      text:
        `${plural(barriers.length, "source node")} · ` +
        `${listOf(tally(barriers, (entry) => (entry as { sourceOp: string }).sourceOp))}` +
        " · regions through them are bounds",
      nodes: barriers.map((entry) => entry.node),
    });
  }

  if (counts.renamed)
    lines.push({
      label: "renamed",
      text: `${plural(counts.renamed, "tensor")} · originals kept for display`,
      nodes: [],
    });

  const unknown = barrierNodes(report);
  if (unknown.length)
    lines.push({
      label: "unknown",
      // A barrier contributes zero FLOPs while every mapped operation around it
      // contributes an upper bound, so their sum is neither a ceiling nor a
      // floor. The figures themselves now say so - a FLOP total spanning one
      // reports `unknown` rather than a number - and this line is where a
      // reader finds out how many nodes are responsible before drawing a tile.
      text: `work in ${plural(unknown.length, "node")} · totals report status, not a number`,
      nodes: unknown,
    });

  return lines;
}

/** The whole report as text, for copying. Same sentences as the panel shows. */
export function formatReport(report: ImportReport): string {
  const width = Math.max(0, ...reportLines(report).map((line) => line.label.length)) + 2;
  return [
    `# ${reportHeadline(report)}`,
    ...reportLines(report).map((line) => `${line.label.padEnd(width)}${line.text}`),
  ].join("\n");
}
