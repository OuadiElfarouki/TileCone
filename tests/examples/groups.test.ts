import { describe, expect, it } from "vitest";
import { EXAMPLES, exampleGroups, listedName } from "../../src/examples/index";

describe("sample groups", () => {
  it("lists every example once, in groups of first appearance", () => {
    const groups = exampleGroups();
    expect(groups.flatMap((group) => group.indices).sort((a, b) => a - b)).toEqual(EXAMPLES.map((_, i) => i));
    expect(groups.map((group) => group.title)).toEqual(
      ["Matmul", "Attention", "Layer blocks", "Other operations", "Plan experiments"]);
    for (const group of groups)
      for (const index of group.indices) expect(EXAMPLES[index].group).toBe(group.title);
  });

  it("keeps the experiments together, under the heading their prefix repeats", () => {
    const experiments = exampleGroups().find((group) => group.title === "Plan experiments")!;
    expect(experiments.indices.every((i) => EXAMPLES[i].experiment)).toBe(true);
    expect(experiments.indices.map((i) => listedName(EXAMPLES[i]))).toEqual(
      ["Tile size and reuse", "Fusion and recomputation", "Splitting a normalised axis"]);
    expect(listedName(EXAMPLES[0])).toBe("Plain GEMM");
  });
});
