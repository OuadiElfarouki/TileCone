import { describe, expect, it } from "vitest";
import { nodeById, producerNode } from "../../src/core/graph";
import { box, count, sameBox, unionOf } from "../../src/core/region";
import { elementCount } from "../../src/core/shapes";
import { compileDSL } from "../../src/parse/compiler";

describe("the union of several regions, kept as they came", () => {
  it("keeps every box, so overlap is measured on the set and never coarsened", () => {
    const rows = { boxes: [box([0, 1], [0, 4])], exact: true, reasons: [] };
    const cols = { boxes: [box([0, 4], [0, 1])], exact: false, reasons: ["b", "a"] };
    const u = unionOf([rows, cols]);
    expect(u.boxes).toEqual([box([0, 1], [0, 4]), box([0, 4], [0, 1])]);
    expect(count(u)).toBe(7);
    expect(u).toMatchObject({ exact: false, reasons: ["a", "b"] });
    expect(unionOf([])).toEqual({ boxes: [], exact: true, reasons: [] });
  });
});

describe("box equality", () => {
  it("compares rank and every interval", () => {
    expect(sameBox(box([0, 2], [1, 3]), box([0, 2], [1, 3]))).toBe(true);
    expect(sameBox(box([0, 2], [1, 3]), box([0, 2], [1, 4]))).toBe(false);
    expect(sameBox(box([0, 2]), box([0, 2], [0, 1]))).toBe(false);
  });
});

describe("shape and graph lookups", () => {
  it("counts a shape's elements, one for a scalar", () => {
    expect(elementCount([2, 3, 4])).toBe(24);
    expect(elementCount([])).toBe(1);
  });

  it("finds a node by id and the node that computes a tensor", () => {
    const { resolved } = compileDSL("X = Tensor(4)\nY = relu(X)\n");
    const node = producerNode(resolved, "Y")!;
    expect(node.outputs).toEqual(["Y"]);
    expect(nodeById(resolved, node.id)).toBe(node);
    expect(producerNode(resolved, "X")).toBeUndefined(); // a graph input
    expect(nodeById(resolved, "missing")).toBeUndefined();
  });
});
