import { ResolvedGraph } from "./graph";
import { getOp } from "./ops/index";
import { DependencyNoteDraft, NoteCtx, NoteFlag } from "./ops/types";
import { PropResult } from "./propagate";

export type DependencyNote = {
  nodeId: string;
  op: string;
  /** The tensor this note is primarily about. */
  subject: string;
  text: string;
  /** How hard this constraint binds a fused kernel; see `DependencyNoteDraft`. */
  severity: 1 | 2 | 3;
};

/** Above this, the panel stops being a summary and becomes a wall. */
/** @internal Exported so note-cap behavior is tested at its exact boundary. */
export const MAX_NOTES = 4;

/** Everything the panel reads about *why* a cone has the shape it has. */
export type ConeFindings = {
  /** The surviving notes, in graph order. */
  notes: DependencyNote[];
  /** All distinct constraints before the display cap is applied. */
  constraintCount: number;
  elementwise: boolean;
  /** Tensor id -> its own short reasons, for annotating footprint rows. */
  flags: Map<string, string[]>;
};

type Group = {
  note: DependencyNote;
  others: string[];
  flags: NoteFlag[];
  /** Steps from the tile to the nearest output this note is about. */
  depth: number;
};

/**
 * Ask every operation the cone actually passed through to describe its own
 * constraint, and merge the ones that state the same fact.
 *
 * Three QKV projections state one fact about three tensors. Merging them keeps
 * a repeated constraint from pushing a different one past the cap.
 */
function collectGroups(
  graph: ResolvedGraph,
  back: PropResult
): { groups: Group[]; touchedAnOp: boolean } {
  const groups = new Map<string, Group>();
  let touchedAnOp = false;

  for (const node of graph.topo) {
    // The node contributed only if one of its outputs is in the cone.
    if (!node.outputs.some((id) => back.tensors.has(id))) continue;
    touchedAnOp = true;
    const spec = getOp(node.op);
    if (!spec?.dependencyNote) continue;

    let draft: DependencyNoteDraft | null;
    try {
      draft = spec.dependencyNote(noteCtx(graph, back, node));
    } catch {
      continue; // a note is never worth failing an analysis over
    }
    if (!draft) continue;

    const depth = Math.min(
      ...node.outputs.map((id) => back.tensors.get(id)?.depth ?? Infinity)
    );
    const group = groups.get(draft.key);
    if (group) {
      group.depth = Math.min(group.depth, depth);
      if (draft.subject !== group.note.subject && !group.others.includes(draft.subject))
        group.others.push(draft.subject);
      for (const flag of draft.flags ?? [])
        if (!group.flags.some((f) => f.tensorId === flag.tensorId && f.text === flag.text))
          group.flags.push(flag);
      continue;
    }
    groups.set(draft.key, {
      note: {
        nodeId: node.id,
        op: node.op,
        subject: draft.subject,
        text: draft.text,
        severity: draft.severity,
      },
      others: [],
      flags: [...(draft.flags ?? [])],
      depth,
    });
  }

  return { groups: [...groups.values()], touchedAnOp };
}

/** Fold merged subjects into the prose. */
function finalize({ note, others }: Group): DependencyNote {
  if (!others.length) return note;
  const list =
    others.length === 1
      ? others[0]
      : `${others.slice(0, -1).join(", ")} and ${others[others.length - 1]}`;
  return { ...note, text: `${note.text} The same holds for ${list}.` };
}

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
  return selectNotes(collectGroups(graph, back).groups, limit);
}

/**
 * Which constraints survive the cap, and in what order they are read.
 *
 * The cap has to drop something; dropping by graph order would drop whichever
 * the topological walk reached last, which can be the hardest one. Selection is
 * therefore by severity, then by nearness to the tile — the constraint a reader
 * hits first among equals — while the surviving notes are *displayed* in graph
 * order, because that is the order the computation happens in.
 */
function selectNotes(groups: Group[], limit: number): DependencyNote[] {
  return groups
    .map((group, index) => ({ group, index }))
    .sort(
      (a, b) =>
        b.group.note.severity - a.group.note.severity ||
        a.group.depth - b.group.depth ||
        a.index - b.index
    )
    .slice(0, limit)
    .sort((a, b) => a.index - b.index)
    .map(({ group }) => finalize(group));
}

/**
 * True when the cone reached at least one operation and none of them constrain
 * it — every step was elementwise, so any tiling of the selection fuses. This is
 * a real finding rather than an absence, so it is distinguished from "no cone".
 */
export function coneIsFullyElementwise(graph: ResolvedGraph, back: PropResult): boolean {
  const { groups, touchedAnOp } = collectGroups(graph, back);
  return touchedAnOp && groups.length === 0;
}

/**
 * Notes and per-tensor flags from one pass over the cone.
 *
 * Flags are collected over *every* constraint found rather than the capped
 * list: a flag is true of its tensor whether or not its note had room to be
 * shown, and the row it annotates is on screen either way.
 */
export function coneFindings(
  graph: ResolvedGraph,
  back: PropResult,
  limit = MAX_NOTES
): ConeFindings {
  const { groups, touchedAnOp } = collectGroups(graph, back);
  const elementwise = touchedAnOp && groups.length === 0;

  const flags = new Map<string, string[]>();
  for (const group of groups)
    for (const flag of group.flags) {
      const existing = flags.get(flag.tensorId);
      if (existing) {
        if (!existing.includes(flag.text)) existing.push(flag.text);
      } else flags.set(flag.tensorId, [flag.text]);
    }

  return {
    notes: selectNotes(groups, limit),
    constraintCount: groups.length,
    elementwise,
    flags,
  };
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
    inIds: node.inputs,
    outIds: node.outputs,
    inNames: node.inputs.map((id) => graph.tensors[id].name),
    outNames: node.outputs.map((id) => graph.tensors[id].name),
    inDims: node.inputs.map((id) => {
      const tensor = graph.tensors[id];
      return tensor.symShape ?? tensor.shape;
    }),
    inAxisNames: node.inputs.map((id) => {
      const tensor = graph.tensors[id];
      const names = tensor.axisNames ?? [];
      return Array.from({ length: tensor.resolved!.length }, (_, axis) => names[axis]);
    }),
    inRegions: node.inputs.map((id) => back.tensors.get(id)?.region),
    outRegions: node.outputs.map((id) => back.tensors.get(id)?.region),
  };
}
