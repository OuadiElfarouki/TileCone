import { z } from "zod";
import { Graph } from "../core/graph";
import { DTYPES } from "../core/dtypes";

const symSchema = z.union([z.string(), z.number().int().min(0)]);

/** Zod's ordinary record reconstruction uses a prototype-bearing object, so
 * an own `__proto__` key is lost. Graph identifiers are allowed to have that
 * spelling; validate entries individually into a null-prototype dictionary. */
function safeRecord<S extends z.ZodTypeAny>(valueSchema: S) {
  return z.unknown().transform((value, ctx): Record<string, z.output<S>> => {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "Expected object" });
      return z.NEVER;
    }
    const out = Object.create(null) as Record<string, z.output<S>>;
    for (const [key, item] of Object.entries(value)) {
      const parsed = valueSchema.safeParse(item);
      if (parsed.success) out[key] = parsed.data;
      else
        for (const issue of parsed.error.issues)
          ctx.addIssue({ ...issue, path: [key, ...issue.path] });
    }
    return out;
  });
}

const dtypeWideningSchema = z
  .object({
    from: z.string().min(1),
    note: z.string().min(1),
  })
  .strict();

const tensorSchema = z.object({
  id: z.string(),
  name: z.string(),
  shape: z.array(symSchema),
  dtype: z.enum(DTYPES),
  dtypeWidening: dtypeWideningSchema.optional(),
  // A hole is an axis with no name; JSON.stringify already writes `undefined`
  // array entries as null, so null is the wire form of a hole in both directions.
  axisNames: z
    .array(z.string().nullable())
    .optional()
    .transform((names) => names?.map((name) => name ?? undefined)),
  role: z.enum(["activation", "weight"]).optional(),
}).strict();

const nodeSchema = z.object({
  id: z.string(),
  op: z.string(),
  inputs: z.array(z.string()),
  outputs: z.array(z.string()),
  attrs: safeRecord(z.unknown()).default(Object.create(null)),
  label: z.string().optional(),
}).strict();

const graphSchema = z.object({
  nodes: z.array(nodeSchema),
  tensors: safeRecord(tensorSchema),
  params: safeRecord(z.number().int().min(1)).default(Object.create(null)),
}).strict();

/**
 * Structurally validate an already-parsed graph value.
 *
 * Separate from `parseGraphJSON` so a document that *contains* a graph - an
 * import envelope carrying a graph beside its report - validates the graph half
 * through exactly this schema rather than a second copy of it that would be
 * free to drift. Deep validation (DAG, shapes, op attrs) happens in
 * `resolveGraph` either way.
 *
 * `where` names the position in the containing document, so an envelope's
 * errors read as `at graph.nodes.0.id` rather than losing their path.
 */
export function parseGraphValue(raw: unknown, where = ""): Graph {
  const res = graphSchema.safeParse(raw);
  if (!res.success)
    throw new Error(
      "schema errors:\n" +
        res.error.issues
          .map((i) => `  at ${[where, ...i.path].filter(Boolean).join(".") || "<root>"}: ${i.message}`)
          .join("\n")
    );
  return res.data as Graph;
}

/** Parse + structurally validate graph JSON. Deep validation (DAG, shapes, op
 * attrs) happens in resolveGraph. */
export function parseGraphJSON(text: string): Graph {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new Error(`invalid JSON: ${(e as Error).message}`);
  }
  return parseGraphValue(raw);
}

export function graphToJSON(g: Graph): string {
  const tensors = Object.create(null) as Record<string, unknown>;
  for (const [id, t] of Object.entries(g.tensors))
    tensors[id] = {
      id: t.id,
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
      ...(t.dtypeWidening ? { dtypeWidening: t.dtypeWidening } : {}),
      ...(t.axisNames ? { axisNames: t.axisNames } : {}),
      ...(t.role ? { role: t.role } : {}),
    };
  return JSON.stringify(
    { nodes: g.nodes.map((n) => ({ id: n.id, op: n.op, inputs: n.inputs, outputs: n.outputs, attrs: n.attrs, ...(n.label ? { label: n.label } : {}) })), tensors, params: g.params },
    null,
    2
  );
}
