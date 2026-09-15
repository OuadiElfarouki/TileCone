import { z } from "zod";
import { parseGraphValue } from "../parse/json";
import {
  IMPORT_FORMATS,
  ImportError,
  ImportReport,
  ImportResult,
  emptyReport,
  type ImportFormat,
} from "./types";

/**
 * The JSON door: a converted model arriving as a document rather than as bytes
 * this app decodes itself.
 *
 * It exists so a converter is testable against the app before any protobuf is,
 * and it is not a stopgap. A front end that runs shape inference where the
 * inference actually exists - Python, for ONNX - has to hand its result over
 * somehow, and this is the boundary it hands it over at. Whichever decoder goes
 * first, it emits this.
 *
 * Two documents are accepted. A bare graph is what `graphToJSON` already
 * writes, and it makes no claims about its own conversion, so it gets an empty
 * report: nothing was mapped, nothing was assumed, because nothing said. An
 * envelope carries the report beside the graph, which is what a real converter
 * emits - the graph alone cannot say which of its nodes is a barrier standing
 * in for a `Resize`, and a reader who cannot see that is reading bounds as
 * counts.
 */

const originSchema = z.object({
  fileName: z.string().min(1).optional(),
  format: z.enum(IMPORT_FORMATS as [ImportFormat, ...ImportFormat[]]).optional(),
  opset: z.number().int().min(1).optional(),
  producer: z.string().optional(),
});

const entrySchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("mapped"),
    node: z.string().min(1),
    sourceOp: z.string().min(1),
    op: z.string().min(1),
    shapeChecked: z.boolean().default(false),
  }),
  z.object({
    kind: z.literal("evaluated"),
    node: z.string().min(1),
    sourceName: z.string().min(1),
    attribute: z.string().min(1),
  }),
  z.object({
    kind: z.literal("rewritten"),
    nodes: z.array(z.string().min(1)).min(1),
    sourceOp: z.string().min(1),
    ops: z.array(z.string().min(1)).min(1),
  }),
  z.object({
    kind: z.literal("bound"),
    dimension: z.string().min(1),
    value: z.number().int().min(1),
    assumed: z.boolean().default(true),
  }),
  z.object({
    kind: z.literal("barrier"),
    node: z.string().min(1),
    sourceOp: z.string().min(1),
    reason: z.string().default(""),
  }),
  z.object({
    kind: z.literal("renamed"),
    tensor: z.string().min(1),
    original: z.string().min(1),
  }),
]);

const reportSchema = z.object({
  origin: originSchema.default({}),
  sourceNodes: z.number().int().min(0).optional(),
  operations: z.number().int().min(0).optional(),
  entries: z.array(entrySchema).default([]),
});

const envelopeSchema = z.object({
  graph: z.unknown(),
  report: reportSchema.optional(),
});

/** A document is an envelope when it has a `graph` member; a bare graph has `nodes`. */
function isEnvelope(raw: unknown): boolean {
  return typeof raw === "object" && raw !== null && "graph" in (raw as Record<string, unknown>);
}

function fail(message: string): never {
  throw new ImportError([{ severity: "error", message }]);
}

export type ImportJSONOptions = {
  /** What to call this model when the document does not name itself. */
  fileName?: string;
  /** What it was converted from, when the document does not say. */
  format?: ImportFormat;
};

/**
 * Read an import document into the value the install boundary takes.
 *
 * Structural only, deliberately. Whether the graph resolves - shapes agree,
 * operations exist, the DAG is acyclic - is `resolveGraph`'s question, asked
 * once at install for every source, so that a graph arriving through this door
 * is held to exactly the standard a compiled one is.
 *
 * A `rewritten` entry whose `nodes` and `ops` disagree in length is rejected
 * here rather than carried: the two are parallel by definition, and a report
 * that misreports what an insertion produced is worse than one that is absent.
 */
export function parseImportJSON(text: string, options: ImportJSONOptions = {}): ImportResult {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    fail(`invalid JSON: ${(e as Error).message}`);
  }

  const fileName = options.fileName ?? "imported.json";
  const format = options.format ?? "json";

  if (!isEnvelope(raw)) {
    let graph;
    try {
      graph = parseGraphValue(raw);
    } catch (e) {
      fail((e as Error).message);
    }
    return { graph, report: emptyReport({ fileName, format }, graph) };
  }

  const envelope = envelopeSchema.safeParse(raw);
  if (!envelope.success)
    fail(
      "schema errors:\n" +
        envelope.error.issues
          .map((i) => `  at ${i.path.join(".") || "<root>"}: ${i.message}`)
          .join("\n")
    );

  let graph;
  try {
    graph = parseGraphValue(envelope.data.graph, "graph");
  } catch (e) {
    fail((e as Error).message);
  }

  const parsed = envelope.data.report;
  if (!parsed) return { graph, report: emptyReport({ fileName, format }, graph) };

  for (const entry of parsed.entries)
    if (entry.kind === "rewritten" && entry.nodes.length !== entry.ops.length)
      fail(
        `report: rewrite of "${entry.sourceOp}" lists ${entry.nodes.length} node(s)` +
          ` for ${entry.ops.length} operation(s); they are parallel`
      );

  // Counts default to what the graph has rather than to zero. A document that
  // omits them is saying "I did not count", and a header reading "0 source
  // nodes -> 0 operations" over a graph with twelve would be a claim, not a
  // gap. Where they are stated they are kept as stated: an importer that
  // rewrote one source node into two is the only thing that knows the first
  // number, and the graph can no longer tell us.
  const report: ImportReport = {
    origin: {
      fileName: parsed.origin.fileName ?? fileName,
      format: parsed.origin.format ?? format,
      ...(parsed.origin.opset !== undefined ? { opset: parsed.origin.opset } : {}),
      ...(parsed.origin.producer !== undefined ? { producer: parsed.origin.producer } : {}),
    },
    sourceNodes: parsed.sourceNodes ?? graph.nodes.length,
    operations: parsed.operations ?? graph.nodes.length,
    entries: parsed.entries,
  };
  return { graph, report };
}
