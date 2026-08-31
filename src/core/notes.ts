import { ResolvedGraph } from "./graph";
import { getOp } from "./ops/index";
import { DependencyNoteDraft, NoteCtx } from "./ops/types";
import { PropResult } from "./propagate";

export type DependencyNote = {
  nodeId: string;
  op: string;
  /** The tensor this note is primarily about. */
  subject: string;
  text: string;
  /** Other tensors the same statement covers, when notes were merged. */
  alsoApplies?: string[];
};

/** Above this, the panel stops being a summary and becomes a wall. */
/** @internal Exported so note-cap behavior is tested at its exact boundary. */
export const MAX_NOTES = 4;

/**
 * Plain-language notes about what constrains the current cone.
 *
 * These answer the question the numbers do not: *why* is the footprint this
 * shape, and what does that mean for anyone trying to tile or fuse across it.
 * Each note is produced by the operation that causes the constraint — the only
 * place that knows the reason — and only when the cone actually exhibits it.
 *
 * Derived from a backward (upstream) result: a note describes what an output
 * tile had to pull in, which is a statement about dependencies, not about
 * influence. Nodes outside the cone are skipped entirely.
 */
export function dependencyNotes(
  graph: ResolvedGraph,
  back: PropResult,
  limit = MAX_NOTES
): DependencyNote[] {
  type Group = { note: DependencyNote; others: string[] };
  const groups = new Map<string, Group>();

  for (const node of graph.topo) {
    const spec = getOp(node.op);
    if (!spec?.dependencyNote) continue;
    // The node contributed only if one of its outputs is in the cone.
    if (!node.outputs.some((id) => back.tensors.has(id))) continue;

    let draft: DependencyNoteDraft | null;
    try {
      draft = spec.dependencyNote(noteCtx(graph, back, node));
    } catch {
      continue; // a note is never worth failing an analysis over
    }
    if (!draft) continue;

    // Three QKV projections state one fact about three tensors. Merging them
    // keeps a repeated constraint from pushing a different one past the cap.
    const group = groups.get(draft.key);
    if (group) {
      if (draft.subject !== group.note.subject && !group.others.includes(draft.subject))
        group.others.push(draft.subject);
      continue;
    }
    groups.set(draft.key, {
      note: { nodeId: node.id, op: node.op, subject: draft.subject, text: draft.text },
      others: [],
    });
  }

  return [...groups.values()].slice(0, limit).map(({ note, others }) => {
    if (!others.length) return note;
    const list =
      others.length === 1
        ? others[0]
        : `${others.slice(0, -1).join(", ")} and ${others[others.length - 1]}`;
    return { ...note, text: `${note.text} The same holds for ${list}.`, alsoApplies: others };
  });
}

/**
 * True when the cone reached at least one operation and none of them constrain
 * it — every step was elementwise, so any tiling of the selection fuses. This is
 * a real finding rather than an absence, so it is distinguished from "no cone".
 */
export function coneIsFullyElementwise(graph: ResolvedGraph, back: PropResult): boolean {
  let touchedAnOp = false;
  for (const node of graph.topo) {
    if (!node.outputs.some((id) => back.tensors.has(id))) continue;
    touchedAnOp = true;
    const spec = getOp(node.op);
    if (!spec?.dependencyNote) continue;
    try {
      // the op *can* constrain; only a produced note proves that it did
      if (spec.dependencyNote(noteCtx(graph, back, node))) return false;
    } catch {
      /* an op that cannot explain itself does not disprove elementwise-ness */
    }
  }
  return touchedAnOp;
}

/** Everything an operation needs to describe its own constraint. */
function noteCtx(
  graph: ResolvedGraph,
  back: PropResult,
  node: ResolvedGraph["topo"][number]
): NoteCtx {
  return {
    inShapes: graph.shapesOf(node.inputs),
    outShapes: graph.shapesOf(node.outputs),
    attrs: node.attrs,
    inNames: node.inputs.map((id) => graph.tensors[id].name),
    outNames: node.outputs.map((id) => graph.tensors[id].name),
    inDims: node.inputs.map((id) => graph.tensors[id].shape),
    inRegions: node.inputs.map((id) => back.tensors.get(id)?.region),
    outRegions: node.outputs.map((id) => back.tensors.get(id)?.region),
  };
}
