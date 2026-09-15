import { beforeEach, describe, expect, it } from "vitest";
import { computeMetrics } from "../core/metrics";
import { count, fromBox } from "../core/region";
import { parseImportJSON } from "../import/json";
import { ImportError, barrierNodes, countByKind } from "../import/types";
import { formatReport, reportHeadline, reportLines } from "../import/report";
import { dslTextOf, exampleIndexOf, useStore } from "../ui/store";

const S = () => useStore.getState();

/**
 * A model the way an exporter writes one.
 *
 * The names matter as much as the graph does. An exporter produces
 * `/fc/MatMul_output_0`, not `Y`, and the whole reason import does not route
 * through the DSL is that names like these cannot survive the printer: `toDSL`
 * identifies a tensor by its name and takes a produced tensor's name from the
 * statement's left-hand side, so there is nowhere for this string to go. A test
 * that used DSL-shaped names would pass while proving nothing about the case
 * that forced the design.
 *
 * The `Resize` is a barrier on purpose. It is the one node in the graph whose
 * semantics nobody modelled, and it is what makes everything downstream of it a
 * bound rather than a count.
 */
const ONNX_SHAPED = {
  graph: {
    nodes: [
      {
        id: "/fc/MatMul",
        op: "matmul",
        inputs: ["input", "fc.weight"],
        outputs: ["/fc/MatMul_output_0"],
        attrs: {},
      },
      {
        id: "/fc/Add",
        op: "elementwise",
        inputs: ["/fc/MatMul_output_0", "fc.bias"],
        outputs: ["/fc/Add_output_0"],
        attrs: { fn: "add", nary: 2 },
      },
      {
        id: "/act/Relu",
        op: "elementwise",
        inputs: ["/fc/Add_output_0"],
        outputs: ["/act/Relu_output_0"],
        attrs: { fn: "relu", nary: 1 },
      },
      {
        id: "/head/Resize",
        op: "opaque",
        inputs: ["/act/Relu_output_0"],
        outputs: ["/head/Resize_output_0"],
        attrs: { op: "Resize", shapes: [[4, 12]], dtypes: ["f32"] },
      },
    ],
    tensors: {
      input: { id: "input", name: "input", shape: [4, 6], dtype: "f32" },
      "fc.weight": { id: "fc.weight", name: "fc.weight", shape: [6, 8], dtype: "f32", role: "weight" },
      "fc.bias": { id: "fc.bias", name: "fc.bias", shape: [8], dtype: "f32", role: "weight" },
      "/fc/MatMul_output_0": {
        id: "/fc/MatMul_output_0",
        name: "/fc/MatMul_output_0",
        shape: [],
        dtype: "f32",
      },
      "/fc/Add_output_0": { id: "/fc/Add_output_0", name: "/fc/Add_output_0", shape: [], dtype: "f32" },
      "/act/Relu_output_0": {
        id: "/act/Relu_output_0",
        name: "/act/Relu_output_0",
        shape: [],
        dtype: "f32",
      },
      "/head/Resize_output_0": {
        id: "/head/Resize_output_0",
        name: "/head/Resize_output_0",
        shape: [],
        dtype: "f32",
      },
    },
    params: {},
  },
  report: {
    origin: { fileName: "tiny.onnx", format: "onnx", opset: 17, producer: "pytorch" },
    sourceNodes: 3,
    operations: 4,
    entries: [
      { kind: "mapped", node: "/fc/MatMul", sourceOp: "Gemm", op: "matmul", shapeChecked: true },
      { kind: "rewritten", nodes: ["/fc/MatMul", "/fc/Add"], sourceOp: "Gemm", ops: ["matmul", "add"] },
      { kind: "mapped", node: "/act/Relu", sourceOp: "Relu", op: "elementwise", shapeChecked: true },
      { kind: "barrier", node: "/head/Resize", sourceOp: "Resize", reason: "no mapping at opset 17" },
      { kind: "renamed", tensor: "fc.weight", original: "fc.weight:0" },
      { kind: "bound", dimension: "batch", value: 4, assumed: true },
    ],
  },
};

const asText = (doc: unknown) => JSON.stringify(doc);

describe("the import contract", () => {
  it("reads an envelope into a graph and a structured report", () => {
    const result = parseImportJSON(asText(ONNX_SHAPED));

    expect(result.graph.nodes).toHaveLength(4);
    expect(result.report.origin).toEqual({
      fileName: "tiny.onnx",
      format: "onnx",
      opset: 17,
      producer: "pytorch",
    });
    // Stated separately because they genuinely differ: the rewrite inserted a
    // node the model does not contain, and one count would hide it.
    expect(result.report.sourceNodes).toBe(3);
    expect(result.report.operations).toBe(4);
    expect(countByKind(result.report)).toMatchObject({ mapped: 2, barrier: 1, rewritten: 1 });
    expect(barrierNodes(result.report)).toEqual(["/head/Resize"]);
  });

  it("accepts a bare graph and derives only its graph-visible barriers", () => {
    const result = parseImportJSON(asText(ONNX_SHAPED.graph), { fileName: "bare.json" });

    expect(result.graph.nodes).toHaveLength(4);
    // Mapping and rewrite claims require a converter report. A barrier is
    // already explicit in the graph and must not disappear from the summary.
    expect(result.report.entries).toEqual([
      {
        kind: "barrier",
        node: "/head/Resize",
        sourceOp: "Resize",
        reason: "present in graph; no conversion report supplied",
      },
    ]);
    expect(result.report.sourceNodes).toBe(4);
    expect(result.report.origin.fileName).toBe("bare.json");
  });

  it("defaults the counts to the graph rather than to zero", () => {
    const { report } = parseImportJSON(
      asText({
        graph: ONNX_SHAPED.graph,
        report: {
          entries: [
            { kind: "barrier", node: "/head/Resize", sourceOp: "Resize", reason: "unmapped" },
          ],
        },
      })
    );
    expect(report.sourceNodes).toBe(4);
    expect(report.operations).toBe(4);
  });

  it("rejects a rewrite whose nodes and operations are not parallel", () => {
    const doc = {
      graph: ONNX_SHAPED.graph,
      report: {
        entries: [
          { kind: "rewritten", nodes: ["/fc/MatMul"], sourceOp: "Gemm", ops: ["matmul", "add"] },
        ],
      },
    };
    expect(() => parseImportJSON(asText(doc))).toThrow(ImportError);
    expect(() => parseImportJSON(asText(doc))).toThrow(/parallel/);
  });

  it("rejects malformed documents by name rather than by throwing something raw", () => {
    expect(() => parseImportJSON("{ not json")).toThrow(ImportError);
    expect(() => parseImportJSON(asText({ graph: { nodes: "no" } }))).toThrow(/at graph\.nodes/);
  });

  it("rejects report claims that disagree with the converted graph", () => {
    const wrongCount = {
      ...ONNX_SHAPED,
      report: { ...ONNX_SHAPED.report, operations: 3 },
    };
    expect(() => parseImportJSON(asText(wrongCount))).toThrow(/operations says 3/);

    const wrongMapping = {
      ...ONNX_SHAPED,
      report: {
        ...ONNX_SHAPED.report,
        entries: ONNX_SHAPED.report.entries.map((entry) =>
          entry.kind === "mapped" && entry.node === "/act/Relu"
            ? { ...entry, op: "relu" }
            : entry
        ),
      },
    };
    expect(() => parseImportJSON(asText(wrongMapping))).toThrow(/claims operation "relu"/);

    const missingBarrier = {
      ...ONNX_SHAPED,
      report: {
        ...ONNX_SHAPED.report,
        entries: ONNX_SHAPED.report.entries.filter((entry) => entry.kind !== "barrier"),
      },
    };
    expect(() => parseImportJSON(asText(missingBarrier))).toThrow(/missing its barrier entry/);
  });

  it("rejects unknown report fields instead of silently stripping converter typos", () => {
    const typo = {
      ...ONNX_SHAPED,
      report: {
        ...ONNX_SHAPED.report,
        entries: ONNX_SHAPED.report.entries.map((entry) =>
          entry.kind === "mapped" ? { ...entry, shapeCheckd: true } : entry
        ),
      },
    };
    expect(() => parseImportJSON(asText(typo))).toThrow(/shapeCheckd/);
  });
});

describe("the import report, rendered", () => {
  const { report } = parseImportJSON(asText(ONNX_SHAPED));

  it("heads with what was opened and what it became", () => {
    expect(reportHeadline(report)).toBe(
      "tiny.onnx · onnx · opset 17 · 3 source nodes → 4 operations"
    );
  });

  it("states every rename, binding, rewrite and barrier", () => {
    const lines = Object.fromEntries(reportLines(report).map((line) => [line.label, line.text]));

    expect(lines.mapped).toBe("2 source nodes, shapes cross-checked against the model");
    expect(lines.rewritten).toBe("1 source node · Gemm → matmul, add");
    expect(lines.bound).toBe("1 dimension · batch = 4 (assumption)");
    expect(lines.barriers).toBe("1 source node · Resize · regions through them are bounds");
    expect(lines.renamed).toBe("1 tensor · originals kept for display");
    // Not a category of its own: the consequence of the barrier above it, said
    // where a reader would otherwise take a total through it as a count.
    expect(lines.unknown).toBe("work in 1 node · totals report status, not a number");
  });

  it("points each line at the nodes it is about, which is what it has instead of a span", () => {
    const byLabel = Object.fromEntries(reportLines(report).map((line) => [line.label, line.nodes]));
    expect(byLabel.barriers).toEqual(["/head/Resize"]);
    expect(byLabel.rewritten).toEqual(["/fc/MatMul", "/fc/Add"]);
    expect(byLabel.renamed).toEqual([]);
  });

  it("does not claim a cross-check that could not run", () => {
    const { report: unchecked } = parseImportJSON(
      asText({
        graph: ONNX_SHAPED.graph,
        report: {
          entries: [
            { kind: "mapped", node: "/act/Relu", sourceOp: "Relu", op: "elementwise", shapeChecked: false },
            { kind: "barrier", node: "/head/Resize", sourceOp: "Resize", reason: "unmapped" },
          ],
        },
      })
    );
    expect(reportLines(unchecked)[0].text).toBe("1 source node, no shapes in the model to cross-check");
  });

  it("formats as text with the same sentences the panel shows", () => {
    const text = formatReport(report);
    expect(text.split("\n")[0]).toBe(`# ${reportHeadline(report)}`);
    for (const line of reportLines(report)) expect(text).toContain(line.text);
  });
});

describe("installing an imported model", () => {
  beforeEach(() => {
    useStore.setState({ theme: "light" });
    S().loadExample(0);
  });

  it("installs, analyses, and reports, with no DSL text in the path", () => {
    expect(S().importJSON(asText(ONNX_SHAPED), { fileName: "tiny.onnx", format: "onnx" })).toBe(true);

    const source = S().source;
    expect(source.kind).toBe("import");
    // The point of the union: there is no text behind this graph, and nothing
    // invented one on the way in.
    expect(dslTextOf(source)).toBeNull();
    expect(exampleIndexOf(source)).toBe(-1);
    expect(S().draftText).toBe("");
    expect(S().loadError).toBeNull();

    // The model's own names survived intact, which is the thing a round trip
    // through the printer would have destroyed.
    const resolved = S().resolved!;
    expect(resolved.tensors["/fc/MatMul_output_0"].resolved).toEqual([4, 8]);
    expect(resolved.tensors["/head/Resize_output_0"].resolved).toEqual([4, 12]);
    expect(resolved.nodes.map((node) => node.id)).toContain("/head/Resize");

    if (source.kind !== "import") throw new Error("unreachable");
    expect(source.report.origin.fileName).toBe("tiny.onnx");
    expect(barrierNodes(source.report)).toEqual(["/head/Resize"]);
  });

  it("answers a cone on the imported graph, and marks what crossed the barrier", () => {
    S().importJSON(asText(ONNX_SHAPED));
    // Drawn through the store rather than propagated by hand, so this is the
    // path the canvas takes.
    S().setSelection(
      "/head/Resize_output_0",
      fromBox([{ lo: 0, hi: 1 }, { lo: 0, hi: 2 }])
    );

    const back = S().backwardRes!;
    // The cone reaches the graph's inputs: a barrier narrows nothing away, it
    // widens. Stopping early would be a subset of the truth.
    expect(back.tensors.has("input")).toBe(true);
    expect(back.tensors.has("fc.weight")).toBe(true);

    // And it reaches them in full, because everything upstream came through an
    // operation nobody modelled.
    const input = back.tensors.get("input")!;
    expect(count(input.region)).toBe(4 * 6);
    expect(input.region.exact).toBe(false);
    expect(input.region.reasons.join(" ")).toContain("Resize");

    const metrics = computeMetrics(S().resolved!, back);
    expect(metrics.exact).toBe(false);
    expect(metrics.reasons.some((reason) => reason.includes("Resize"))).toBe(true);
  });

  it("refuses a graph it cannot resolve, and installs nothing", () => {
    const before = S().resolved;
    const broken = {
      graph: {
        ...ONNX_SHAPED.graph,
        nodes: [
          {
            id: "/fc/MatMul",
            op: "matmul",
            inputs: ["input", "fc.bias"],
            outputs: ["/fc/MatMul_output_0"],
            attrs: {},
          },
        ],
      },
    };

    expect(S().importJSON(asText(broken))).toBe(false);
    // Never the collecting resolver: pruning the bad node and everything below
    // it would install a shorter model than the one that was opened, and every
    // cone drawn on it would be a subset of the truth.
    expect(S().resolved).toBe(before);
    expect(S().source.kind).toBe("dsl");
    expect(S().importDiagnostics).toHaveLength(1);
    expect(S().importDiagnostics[0].subject).toEqual({ kind: "node", id: "/fc/MatMul" });
  });

  it("reports a malformed document without disturbing the workspace", () => {
    const before = S().resolved;
    expect(S().importJSON("{ not json")).toBe(false);
    expect(S().resolved).toBe(before);
    expect(S().source.kind).toBe("dsl");
    expect(S().importDiagnostics[0].message).toMatch(/invalid JSON/);
    expect(S().importDiagnostics[0].subject).toBeUndefined();
  });

  it("surfaces a file read failure before parsing without disturbing the workspace", () => {
    const before = S().resolved;

    S().reportImportError('could not read "model.json": permission denied');

    expect(S().resolved).toBe(before);
    expect(S().importDiagnostics).toEqual([
      { severity: "error", message: 'could not read "model.json": permission denied' },
    ]);
  });

  it("clears an earlier import failure once something installs", () => {
    S().importJSON("{ not json");
    expect(S().importDiagnostics).toHaveLength(1);
    S().importJSON(asText(ONNX_SHAPED));
    expect(S().importDiagnostics).toEqual([]);
  });

  it("refuses to expand a composite in an imported workspace", () => {
    S().importJSON(asText(ONNX_SHAPED));
    const resolved = S().resolved;

    S().expandNodeInPlace("/act/Relu");

    // Expansion rewrites the workspace as generated DSL, which would rename the
    // very nodes the report addresses. The limitation is declared, not silent.
    expect(S().resolved).toBe(resolved);
    expect(S().source.kind).toBe("import");
    expect(S().loadError).toMatch(/generated DSL/);
  });

  it("returns to DSL cleanly when a source is run over an import", () => {
    S().importJSON(asText(ONNX_SHAPED));
    S().applyDSL("X = Tensor(2, 3, dtype=fp32)\nY = relu(X)\n");

    expect(S().source.kind).toBe("dsl");
    expect(dslTextOf(S().source)).toContain("Y = relu(X)");
    expect(S().resolved!.tensors.Y.resolved).toEqual([2, 3]);
  });

  it("replaces an import when an example is chosen instead of staging invisible text", () => {
    S().importJSON(asText(ONNX_SHAPED));
    const target = 0;

    S().chooseExample(target);

    expect(S().source.kind).toBe("dsl");
    expect(exampleIndexOf(S().source)).toBe(target);
    expect(S().draftText).toBe(dslTextOf(S().source));
  });
});
