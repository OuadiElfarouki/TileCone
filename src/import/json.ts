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
import { validateImportReport } from "./validate";

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
 * writes, and it makes no claims about its own conversion, so it gets a minimal
 * report: graph-visible barriers are derived, while nothing is called mapped or
 * assumed because nothing said so. An
 * envelope carries the report beside the graph, which is what a real converter
 * emits - the graph alone cannot say which of its nodes is a barrier standing
 * in for a `Resize`, and a reader who cannot see that is reading bounds as
 * counts.
 */

const originSchema = z
  .object({
    fileName: z.string().min(1).optional(),
    format: z.enum(IMPORT_FORMATS as [ImportFormat, ...ImportFormat[]]).optional(),
    opset: z.number().int().min(1).optional(),
    producer: z.string().optional(),
  })
  .strict();

const entrySchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("mapped"),
    node: z.string().min(1),
    sourceOp: z.string().min(1),
    op: z.string().min(1),
    shapeChecked: z.boolean().default(false),
  }).strict(),
  z.object({
    kind: z.literal("evaluated"),
    node: z.string().min(1),
    sourceName: z.string().min(1),
    attribute: z.string().min(1),
  }).strict(),
  z.object({
    kind: z.literal("rewritten"),
    nodes: z.array(z.string().min(1)).min(1),
    sourceOp: z.string().min(1),
    ops: z.array(z.string().min(1)).min(1),
  }).strict(),
  z.object({
    kind: z.literal("bound"),
    dimension: z.string().min(1),
    value: z.number().int().min(1),
    assumed: z.boolean().default(true),
  }).strict(),
  z.object({
    kind: z.literal("barrier"),
    node: z.string().min(1),
    sourceOp: z.string().min(1),
    reason: z.string().default(""),
  }).strict(),
  z.object({
    kind: z.literal("renamed"),
    tensor: z.string().min(1),
    original: z.string().min(1),
  }).strict(),
]);

const reportSchema = z
  .object({
    origin: originSchema.default({}),
    sourceNodes: z.number().int().min(0).optional(),
    operations: z.number().int().min(0).optional(),
    entries: z.array(entrySchema).default([]),
  })
  .strict();

const envelopeSchema = z
  .object({
    graph: z.unknown(),
    report: reportSchema.optional(),
  })
  .strict();

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
 * The graph is structurally validated here; whether it resolves - shapes agree,
 * operations exist, the DAG is acyclic - remains `resolveGraph`'s question at
 * installation. Report claims are also cross-checked against the graph before
 * this function returns. The same validator runs in preflight for in-memory
 * decoder results, so neither entry path can display a stale or invented claim.
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
  const result = { graph, report };
  const diagnostics = validateImportReport(result);
  if (diagnostics.length) throw new ImportError(diagnostics);
  return result;
}
