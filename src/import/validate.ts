import type { ImportDiagnostic, ImportResult } from "./types";
import { opLabel } from "../core/ops/index";

/**
 * Cross-check conversion claims against the graph they describe.
 *
 * This is shared by the JSON door and preflight because JSON is not the trust
 * boundary: a protobuf decoder will eventually construct an `ImportResult` in
 * memory and call the same install action. Every producer is held to the same
 * contract before its report can be shown as an accuracy guarantee.
 */
export function validateImportReport(result: ImportResult): ImportDiagnostic[] {
  const { graph, report } = result;
  const diagnostics: ImportDiagnostic[] = [];
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const tensors = new Set(Object.keys(graph.tensors));
  const reportedBarriers = new Set<string>();

  const error = (message: string, subject?: ImportDiagnostic["subject"]): void => {
    diagnostics.push({ severity: "error", message, ...(subject ? { subject } : {}) });
  };
  const nodeNamed = (id: string, kind: string) => {
    const node = nodes.get(id);
    if (!node)
      error(`report: ${kind} entry references missing node "${id}"`, { kind: "node", id });
    return node;
  };

  if (report.operations !== graph.nodes.length)
    error(
      `report: operations says ${report.operations}, but the converted graph contains` +
        ` ${graph.nodes.length}`
    );

  for (const entry of report.entries) {
    if (entry.kind === "mapped") {
      const node = nodeNamed(entry.node, "mapped");
      if (node && node.op !== entry.op)
        error(
          `report: mapped node "${entry.node}" claims operation "${entry.op}",` +
            ` but the graph contains "${node.op}"`,
          { kind: "node", id: entry.node }
        );
    } else if (entry.kind === "evaluated") {
      nodeNamed(entry.node, "evaluated");
    } else if (entry.kind === "rewritten") {
      if (entry.nodes.length !== entry.ops.length)
        error(
          `report: rewrite of "${entry.sourceOp}" lists ${entry.nodes.length} node(s)` +
            ` for ${entry.ops.length} operation(s); they are parallel`
        );
      entry.nodes.forEach((id, slot) => {
        const node = nodeNamed(id, "rewritten");
        const claimed = entry.ops[slot];
        const actual = node ? opLabel(node) : undefined;
        if (node && claimed !== undefined && actual !== claimed)
          error(
            `report: rewritten node "${id}" claims operation "${claimed}",` +
              ` but the graph contains "${actual}"`,
            { kind: "node", id }
          );
      });
    } else if (entry.kind === "barrier") {
      const node = nodeNamed(entry.node, "barrier");
      if (node && node.op !== "opaque")
        error(
          `report: barrier entry names node "${entry.node}", whose operation is "${node.op}"`,
          { kind: "node", id: entry.node }
        );
      if (node && node.op === "opaque" && node.attrs.op !== entry.sourceOp)
        error(
          `report: barrier node "${entry.node}" names source operation "${entry.sourceOp}",` +
            ` but its opaque metadata names "${String(node.attrs.op)}"`,
          { kind: "node", id: entry.node, attribute: "op" }
        );
      if (reportedBarriers.has(entry.node))
        error(`report: barrier node "${entry.node}" is reported more than once`, {
          kind: "node",
          id: entry.node,
        });
      reportedBarriers.add(entry.node);
    } else if (entry.kind === "renamed" && !tensors.has(entry.tensor)) {
      error(`report: renamed entry references missing tensor "${entry.tensor}"`, {
        kind: "tensor",
        id: entry.tensor,
      });
    }
  }

  for (const node of graph.nodes)
    if (node.op === "opaque" && !reportedBarriers.has(node.id))
      error(`report: opaque node "${node.id}" is missing its barrier entry`, {
        kind: "node",
        id: node.id,
      });

  return diagnostics;
}
