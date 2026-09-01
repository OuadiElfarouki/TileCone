import { z } from "zod";
import { Graph } from "../core/graph";
import { DTYPES } from "../core/dtypes";

const symSchema = z.union([z.string(), z.number().int().min(0)]);

const tensorSchema = z.object({
  id: z.string(),
  name: z.string(),
  shape: z.array(symSchema),
  dtype: z.enum(DTYPES),
  // A hole is an axis with no name; JSON.stringify already writes `undefined`
  // array entries as null, so null is the wire form of a hole in both directions.
  axisNames: z
    .array(z.string().nullable())
    .optional()
    .transform((names) => names?.map((name) => name ?? undefined)),
  role: z.enum(["activation", "weight"]).optional(),
});

const nodeSchema = z.object({
  id: z.string(),
  op: z.string(),
  inputs: z.array(z.string()),
  outputs: z.array(z.string()),
  attrs: z.record(z.unknown()).default({}),
  label: z.string().optional(),
});

const graphSchema = z.object({
  nodes: z.array(nodeSchema),
  tensors: z.record(tensorSchema),
  params: z.record(z.number().int().min(1)).default({}),
});

/** Parse + structurally validate graph JSON. Deep validation (DAG, shapes, op
 * attrs) happens in resolveGraph. */
export function parseGraphJSON(text: string): Graph {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new Error(`invalid JSON: ${(e as Error).message}`);
  }
  const res = graphSchema.safeParse(raw);
  if (!res.success)
    throw new Error(
      "schema errors:\n" +
        res.error.issues.map((i) => `  at ${i.path.join(".") || "<root>"}: ${i.message}`).join("\n")
    );
  return res.data as Graph;
}

export function graphToJSON(g: Graph): string {
  const tensors: Record<string, unknown> = {};
  for (const [id, t] of Object.entries(g.tensors))
    tensors[id] = {
      id: t.id,
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
      ...(t.axisNames ? { axisNames: t.axisNames } : {}),
      ...(t.role ? { role: t.role } : {}),
    };
  return JSON.stringify(
    { nodes: g.nodes.map((n) => ({ id: n.id, op: n.op, inputs: n.inputs, outputs: n.outputs, attrs: n.attrs, ...(n.label ? { label: n.label } : {}) })), tensors, params: g.params },
    null,
    2
  );
}
