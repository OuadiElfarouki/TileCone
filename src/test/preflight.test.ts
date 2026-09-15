/**
 * What has to be true before a converted model is safe to install.
 *
 * A barrier is safe only because its output metadata is concrete: total
 * dependence in both directions is a superset by construction, but a superset
 * of *what* needs an extent. There is no conservative guess available for a
 * missing shape - widening in the honest direction is not something a shape
 * supports - so the answer is a named error, or a request for a value.
 */

import { describe, expect, it } from "vitest";
import { parseImportJSON } from "../import/json";
import { preflightImport } from "../import/preflight";
import { computeMetrics } from "../core/metrics";
import { propagateBackward } from "../core/propagate";
import { box, count, fromBox } from "../core/region";
import { dslTextOf, useStore } from "../ui/store";

const S = () => useStore.getState();

type Doc = {
  nodes: unknown[];
  tensors: Record<string, unknown>;
  params?: Record<string, number>;
};

const tensor = (id: string, shape: (string | number)[], dtype = "f32", role?: string) => ({
  id,
  name: id,
  shape,
  dtype,
  ...(role ? { role } : {}),
});

const resultOf = (graph: Doc) => parseImportJSON(JSON.stringify(graph), { fileName: "m.onnx" });

describe("preflight", () => {
  it("passes a model whose metadata is complete", () => {
    const { unbound, errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/Resize",
            op: "opaque",
            inputs: ["x"],
            outputs: ["y"],
            attrs: { op: "Resize", shapes: [[4, 12]], dtypes: ["f32"] },
          },
        ],
        tensors: { x: tensor("x", [4, 6]), y: tensor("y", []) },
      })
    );
    expect(errors).toEqual([]);
    expect(unbound).toEqual([]);
  });

  it("requires a concrete dtype for every imported barrier output", () => {
    const { errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/Missing",
            op: "opaque",
            inputs: ["x"],
            outputs: ["a"],
            attrs: { op: "ArgMax", shapes: [[4]] },
          },
          {
            id: "/Hole",
            op: "opaque",
            inputs: ["a"],
            outputs: ["b"],
            attrs: { op: "Equal", shapes: [[4]], dtypes: [null] },
          },
        ],
        tensors: {
          x: tensor("x", [4]),
          a: tensor("a", []),
          b: tensor("b", []),
        },
      })
    );

    expect(errors).toHaveLength(2);
    expect(errors.map((error) => error.subject)).toEqual([
      { kind: "node", id: "/Missing", attribute: "dtypes" },
      { kind: "node", id: "/Hole", attribute: "dtypes" },
    ]);
    expect(errors[0].message).toMatch(/no declared output dtypes/);
    expect(errors[1].message).toMatch(/no concrete dtype/);
  });

  /* The case this exists for. `resolveGraph` would report the first of these
     and stop, which over four hundred nodes is a fix-one-reload loop. */
  it("reports every problem at once, each naming its node", () => {
    const { errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/A",
            op: "opaque",
            inputs: ["x"],
            outputs: ["a"],
            attrs: { op: "A", dtypes: ["f32"] },
          },
          {
            id: "/B",
            op: "opaque",
            inputs: ["a"],
            outputs: ["b"],
            attrs: { op: "B", shapes: [[4, 0]], dtypes: ["f32"] },
          },
        ],
        tensors: {
          x: tensor("x", [4, 6]),
          a: tensor("a", []),
          b: tensor("b", []),
        },
      })
    );
    expect(errors).toHaveLength(2);
    expect(errors.map((e) => e.subject?.id)).toEqual(["/A", "/B"]);
  });

  it("refuses a barrier with no output shapes, because there is nothing to widen", () => {
    const { errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/Custom",
            op: "opaque",
            inputs: ["x"],
            outputs: ["y"],
            attrs: { op: "Custom", dtypes: ["f32"] },
          },
        ],
        tensors: { x: tensor("x", [4]), y: tensor("y", []) },
      })
    );
    expect(errors[0].message).toContain("no declared output shapes");
    expect(errors[0].subject).toEqual({ kind: "node", id: "/Custom", attribute: "shapes" });
  });

  it("refuses an empty extent, which has no regions to relate", () => {
    const { errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/Split",
            op: "opaque",
            inputs: ["x"],
            outputs: ["y"],
            attrs: { op: "Split", shapes: [[0, 4]], dtypes: ["f32"] },
          },
        ],
        tensors: { x: tensor("x", [4]), y: tensor("y", []) },
      })
    );
    expect(errors[0].message).toContain("/Split");
    expect(errors[0].message).toMatch(/bad dimension 0/);
  });

  /* A free dimension is the ordinary case in an exported model, not a defect.
     Reporting it as an unbound symbol would tell a user their file is broken;
     reporting it as a request lets them answer. */
  it("asks for a value for a dimension the model left free", () => {
    const { unbound, errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/Resize",
            op: "opaque",
            inputs: ["x"],
            outputs: ["y"],
            attrs: { op: "Resize", shapes: [["batch", 12]], dtypes: ["f32"] },
          },
        ],
        tensors: { x: tensor("x", ["batch", 6]), y: tensor("y", []) },
      })
    );
    expect(unbound).toEqual(["batch"]);
    // Not also an error: a shape that is only waiting on a value is waiting,
    // not broken, and saying both would make the request look like a symptom.
    expect(errors).toEqual([]);
  });

  it("is satisfied once the free dimension is bound", () => {
    const { unbound, errors } = preflightImport(
      resultOf({
        nodes: [
          {
            id: "/Resize",
            op: "opaque",
            inputs: ["x"],
            outputs: ["y"],
            attrs: { op: "Resize", shapes: [["batch", 12]], dtypes: ["f32"] },
          },
        ],
        tensors: { x: tensor("x", ["batch", 6]), y: tensor("y", []) },
        params: { batch: 1 },
      })
    );
    expect([unbound, errors]).toEqual([[], []]);
  });

  it("refuses an element type this engine does not hold", () => {
    // Built past the JSON door's schema, the way an in-memory decoder would.
    const { errors } = preflightImport({
      graph: {
        nodes: [],
        tensors: {
          x: { id: "x", name: "x", shape: [4], dtype: "f64" as never },
        },
        params: {},
      },
      report: {
        origin: { fileName: "m.onnx", format: "onnx" },
        sourceNodes: 0,
        operations: 0,
        entries: [],
      },
    });
    expect(errors[0].message).toContain("f64");
    expect(errors[0].subject).toEqual({ kind: "tensor", id: "x" });
  });

  it("refuses a per-output dtype it does not hold", () => {
    const { errors } = preflightImport({
      graph: {
        nodes: [
          {
            id: "/TopK",
            op: "opaque",
            inputs: ["x"],
            outputs: ["v", "i"],
            attrs: { op: "TopK", shapes: [[4], [4]], dtypes: ["f32", "uint64"] },
          },
        ],
        tensors: {
          x: { id: "x", name: "x", shape: [4], dtype: "f32" },
          v: { id: "v", name: "v", shape: [], dtype: "f32" },
          i: { id: "i", name: "i", shape: [], dtype: "f32" },
        },
        params: {},
      },
      report: {
        origin: { fileName: "m.onnx", format: "onnx" },
        sourceNodes: 1,
        operations: 1,
        entries: [],
      },
    });
    expect(errors[0].message).toContain("output 1");
    expect(errors[0].message).toContain("uint64");
  });
});

/**
 * The synthetic graph Phase 03 asks for: a captured-value subgraph, barrier
 * outputs of different types, and an unknown-work path, checked together.
 *
 * `W` is the captured value. In ONNX it would be referenced from inside an `If`
 * or `Loop` body and never appear in the node's own input list, so a converter
 * building the barrier from `node.input` alone would omit it - and a cone that
 * missed a real dependency would be a *subset* of the truth, the one failure
 * this engine treats as critical. Attaching it as a barrier input is what makes
 * the claim honest, and the test is that it is then read in full like any other.
 */
const CAPTURED = {
  graph: {
    nodes: [
      {
        id: "/Loop",
        op: "opaque",
        // `cond` and `x` are the node's own inputs; `W` is the free variable
        // its body closes over.
        inputs: ["cond", "x", "W"],
        outputs: ["/Loop_out_0", "/Loop_out_1"],
        attrs: {
          op: "Loop",
          opset: 17,
          shapes: [[4, 8], [4]],
          dtypes: ["f32", "i64"],
        },
      },
      {
        id: "/Cast",
        op: "cast",
        inputs: ["/Loop_out_1"],
        outputs: ["counts"],
        attrs: { dtype: "f32" },
      },
    ],
    tensors: {
      cond: { id: "cond", name: "cond", shape: [1], dtype: "bool" },
      x: { id: "x", name: "x", shape: [4, 8], dtype: "f32" },
      W: { id: "W", name: "W", shape: [8, 8], dtype: "f32", role: "weight" },
      "/Loop_out_0": { id: "/Loop_out_0", name: "/Loop_out_0", shape: [], dtype: "f32" },
      "/Loop_out_1": { id: "/Loop_out_1", name: "/Loop_out_1", shape: [], dtype: "f32" },
      counts: { id: "counts", name: "counts", shape: [], dtype: "f32" },
    },
    params: {},
  },
  report: {
    origin: { fileName: "loop.onnx", format: "onnx", opset: 17 },
    sourceNodes: 2,
    operations: 2,
    entries: [
      { kind: "barrier", node: "/Loop", sourceOp: "Loop", reason: "subgraph not modelled" },
      { kind: "mapped", node: "/Cast", sourceOp: "Cast", op: "cast", shapeChecked: true },
    ],
  },
};

describe("a captured-value barrier, end to end", () => {
  it("installs, with each output taking its own type", () => {
    S().loadExample(0);
    expect(S().importJSON(JSON.stringify(CAPTURED), { fileName: "loop.onnx", format: "onnx" })).toBe(
      true
    );

    const resolved = S().resolved!;
    // Heterogeneous outputs from one node: the values promote from the inputs,
    // the indices are declared i64. One dtype for both would misreport the
    // bytes of whichever it did not fit.
    expect(resolved.tensors["/Loop_out_0"].dtype).toBe("f32");
    expect(resolved.tensors["/Loop_out_1"].dtype).toBe("i64");
    expect(dslTextOf(S().source)).toBeNull();
  });

  it("reads the captured value in full, like any other dependency", () => {
    S().loadExample(0);
    S().importJSON(JSON.stringify(CAPTURED));
    const resolved = S().resolved!;
    const back = propagateBackward(resolved, {
      tensorId: "counts",
      region: fromBox(box([0, 1])),
    });

    // The free variable is in the cone at all - a barrier built from the node's
    // own input list would have left it out, and the cone would have been a
    // subset of the truth rather than a superset.
    const onW = back.tensors.get("W")!;
    expect(count(onW.region)).toBe(8 * 8);
    expect(onW.region.exact).toBe(false);
    expect(onW.region.reasons).toEqual(['opaque op "Loop"']);
  });

  it("bounds every byte figure and refuses the FLOP total", () => {
    S().loadExample(0);
    S().importJSON(JSON.stringify(CAPTURED));
    const resolved = S().resolved!;
    const m = computeMetrics(
      resolved,
      propagateBackward(resolved, { tensorId: "counts", region: fromBox(box([0, 4])) })
    );

    // Supersets, so the bytes are upper bounds - and they are bounds, which is
    // a claim the oracle-checked region contract supports.
    expect(m.inputBytes.status).toBe("upper");
    expect(m.inputBytes.value).toBe(1 * 1 + 4 * 8 * 4 + 8 * 8 * 4);

    // The one figure that is not a bound in any direction, and so is not a
    // figure at all.
    expect(m.flops.value).toBeNull();
    expect(m.flops.reasons).toEqual(["unknown work in Loop"]);
    expect(m.fusedIntensity.value).toBeNull();
  });
});
