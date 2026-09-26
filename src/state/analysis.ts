import { entangledWith, Entanglement } from "../core/entangle";
import { executeQuery } from "../core/executor";
import { ResolvedGraph } from "../core/graph";
import { mergeProps, PropResult } from "../core/propagate";
import { Box, fromBox } from "../core/region";
import { BoxProp, MAX_PER_BOX_PROPS, type Selection, SelPart } from "../view/workspace";
import { State } from "./types";

/**
 * One propagation per part, merged into the aggregate the panels read.
 *
 * The executor stays a single-root primitive : a cone is defined from one
 * tensor : and multiplicity lives here, where it belongs: the workspace is what
 * holds several probes at once. Merging is a per-tensor union, which is also
 * what the propagator already does internally when two paths reconverge.
 *
 * Both cones are always computed. `direction` is a view filter over the result,
 * not a gate on producing it: the panel answers "what does this tile need" and
 * "what does it feed" from the same analysis, and a global mode should not
 * decide whether a number exists. Painting applies the filter instead.
 *
 * Above `MAX_PER_BOX_PROPS` parts, per-part attribution is dropped and the
 * queries are grouped by tensor instead, so the cost is bounded by the number
 * of tensors drawn on rather than the number of tiles. Note that the grouped
 * result can only be equal or coarser than the per-part one: propagating a
 * union through an over-approximating op is never tighter than unioning the
 * separate propagations. Neither can under-approximate.
 */
export function recompute(
  resolved: ResolvedGraph | null,
  selection: Selection,
  previous?: {
    selection: Selection;
    perBox: BoxProp[] | null;
    entangled: Entanglement[][] | null;
  }
): Pick<State, "backwardRes" | "forwardRes" | "perBox" | "entangled" | "byTensorRes"> {
  const none = {
    backwardRes: null, forwardRes: null, perBox: null, entangled: null, byTensorRes: null,
  };
  if (!resolved || !selection || selection.parts.length === 0) return none;

  const parts = selection.parts;
  const backs: PropResult[] = [];
  const fwds: PropResult[] = [];
  let perBox: BoxProp[] | null = null;
  const byTensorRes = Object.create(null) as NonNullable<State["byTensorRes"]>;

  if (parts.length <= MAX_PER_BOX_PROPS) {
    // Geometry is the cache key rather than array position: deleting a part
    // renumbers its peers, and composition may recreate an equal SelPart
    // object. Reuse every unchanged cone and execute only new/edited parts.
    const cached = new Map<string, BoxProp[]>();
    const keyOf = (part: SelPart) =>
      `${part.tensorId}|${part.box.map((interval) => `${interval.lo}:${interval.hi}`).join(",")}`;
    if (previous?.selection && previous.perBox &&
        previous.selection.parts.length === previous.perBox.length) {
      previous.selection.parts.forEach((part, index) => {
        // Execution above the attribution cap only populates its anchor.
        if (!previous.perBox![index].backward && !previous.perBox![index].forward) return;
        const key = keyOf(part);
        const entries = cached.get(key);
        if (entries) entries.push(previous.perBox![index]);
        else cached.set(key, [previous.perBox![index]]);
      });
    }
    perBox = parts.map((p) => {
      const hit = cached.get(keyOf(p))?.shift();
      if (hit) {
        if (hit.backward) backs.push(hit.backward);
        if (hit.forward) fwds.push(hit.forward);
        return hit;
      }
      const r = executeQuery(resolved, {
        tensorId: p.tensorId,
        region: fromBox(p.box),
        direction: "both",
      });
      if (r.backward) backs.push(r.backward);
      if (r.forward) fwds.push(r.forward);
      return { backward: r.backward, forward: r.forward };
    });
    // The same grouping the branch below gets for free. Merging per-part cones
    // that already exist is cheaper than re-querying, and both regimes have to
    // offer the inspector the same shape.
    const grouped = new Map<string, { backs: PropResult[]; fwds: PropResult[] }>();
    parts.forEach((part, index) => {
      const entry = grouped.get(part.tensorId) ?? { backs: [], fwds: [] };
      const prop = perBox![index];
      if (prop.backward) entry.backs.push(prop.backward);
      if (prop.forward) entry.fwds.push(prop.forward);
      grouped.set(part.tensorId, entry);
    });
    for (const [tensorId, entry] of grouped)
      byTensorRes[tensorId] = {
        backward: mergeProps(entry.backs),
        forward: mergeProps(entry.fwds),
      };
  } else {
    const byTensor = new Map<string, Box[]>();
    for (const p of parts) {
      const cur = byTensor.get(p.tensorId);
      if (cur) cur.push(p.box);
      else byTensor.set(p.tensorId, [p.box]);
    }
    for (const [tensorId, boxes] of byTensor) {
      const r = executeQuery(resolved, {
        tensorId,
        region: { boxes, exact: true, reasons: [] },
        direction: "both",
      });
      if (r.backward) backs.push(r.backward);
      if (r.forward) fwds.push(r.forward);
      byTensorRes[tensorId] = { backward: r.backward, forward: r.forward };
    }
  }
  // Entanglement is attributed per part for the same reason as the cones: hue
  // identifies the tile that produced a region. Respect the same cap, rather
  // than reintroducing unbounded synchronous work after cone attribution has
  // deliberately switched to a grouped query. Geometry is also cached so an
  // edit only recomputes the part that changed.
  let entangled: Entanglement[][] | null = null;
  if (parts.length <= MAX_PER_BOX_PROPS) {
    const keyOf = (part: SelPart) =>
      `${part.tensorId}|${part.box.map((interval) => `${interval.lo}:${interval.hi}`).join(",")}`;
    const cached = new Map<string, Entanglement[][]>();
    if (
      previous?.selection &&
      previous.entangled &&
      previous.selection.parts.length === previous.entangled.length
    ) {
      previous.selection.parts.forEach((part, index) => {
        const key = keyOf(part);
        const entries = cached.get(key);
        if (entries) entries.push(previous.entangled![index]);
        else cached.set(key, [previous.entangled![index]]);
      });
    }
    entangled = parts.map(
      (part) =>
        cached.get(keyOf(part))?.shift() ??
        entangledWith(resolved, part.tensorId, fromBox(part.box))
    );
  }
  return {
    backwardRes: mergeProps(backs),
    forwardRes: mergeProps(fwds),
    perBox,
    entangled,
    byTensorRes,
  };
}
