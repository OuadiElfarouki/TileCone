import { describe, expect, it } from "vitest";
import { EXAMPLES } from "../../src/examples/index";
import { planReport, type Work } from "../../src/core/plan/interfaces";
import { useStore } from "../../src/state/store";
import { fmt, formatBytes } from "../../src/view/format";

const S = () => useStore.getState();
const experiment = (name: string) => {
  const index = EXAMPLES.findIndex((example) => example.name === name);
  expect(EXAMPLES[index]?.experiment).toBeDefined();
  return { index, example: EXAMPLES[index] };
};
/** Build the experiment's source as the run button does, and read its three plans. */
const run = (name: string): { dsl: string; p1: Work; p2: Work; current: Work } => {
  const { example } = experiment(name);
  S().applyDSL(example.dsl);
  const [p1, p2] = S().keptPlans.map((kept) => kept.total);
  return { dsl: example.dsl, p1, p2, current: planReport(S().plan!).total! };
};

describe("building an experiment's source", () => {
  it("opens the Plan view on its plan, task, selection and two kept plans", () => {
    const { index, example } = experiment("Experiment: fusion and recomputation");
    S().applyDSL("X = Tensor(2, 3, dtype=float32)\nY = relu(X)\n");
    S().setInspectorTab("dependencies");
    S().applyDSL(example.dsl);
    expect(S().inspectorTab).toBe("plan");
    expect({ ...S().planTiles }).toEqual(example.experiment!.tiles);
    expect(S().planTask).toEqual({ tensorId: "Y", coord: [1, 0] });
    expect(S().keptPlans.map((kept) => kept.tiles)).toEqual(example.experiment!.kept);
    expect(S().selection!.parts.map((part) => part.tensorId)).toEqual(["Y"]);
    expect(S().exampleIndex).toBe(index);
    expect(S().loadError).toBeNull();
  });

  it("builds a plain example as before, with nothing tiled", () => {
    S().applyDSL(EXAMPLES[0].dsl);
    expect(Object.keys(S().planTiles)).toEqual([]);
    expect(S().keptPlans).toEqual([]);
  });

  it("leaves an edited experiment source to build as any other source", () => {
    const { example } = experiment("Experiment: tile size and reuse");
    S().applyDSL(example.dsl.replace("K = 512", "K = 256"));
    expect(S().exampleIndex).toBe(-1);
    expect(Object.keys(S().planTiles)).toEqual([]);
  });
});

/* Each experiment's comments quote figures; these are those figures, so the
   text cannot drift from what the Plan view shows. */
describe("the figures each experiment states", () => {
  it("tile size and reuse: same FLOPs, reads halving as the tile side doubles", () => {
    const { dsl, p1, p2, current } = run("Experiment: tile size and reuse");
    for (const work of [p1, p2, current]) expect(fmt(work.flops.value!)).toBe("67.11M");
    expect([p1, p2, current].map((work) => formatBytes(work.read.value!))).toEqual(["4.0 MB", "2.0 MB", "1.0 MB"]);
    expect(dsl).toContain("67.11M");
    expect(dsl).toContain("4 MB, 2 MB, 1 MB");
  });

  it("fusion and recomputation: written tiles, recomputed fusion, then neither", () => {
    const { p1, p2, current } = run("Experiment: fusion and recomputation");
    expect([p1.tasks, p1.dependencies.value, p1.written.value]).toEqual([24, 32, 320 * 1024]);
    expect([fmt(p2.recomputed.value!), fmt(p2.flops.value!)]).toEqual(["33.55M", "100.66M"]);
    expect([current.tasks, current.recomputed.value, current.written.value]).toEqual([4, 0, 64 * 1024]);
  });

  it("splitting a normalised axis: repeated statistics, or H written instead", () => {
    const { p1, p2, current } = run("Experiment: splitting a normalised axis");
    expect([fmt(p1.flops.value!), p1.recomputed.value]).toEqual(["33.0K", 0]);
    expect([fmt(p2.recomputed.value!), fmt(p2.flops.value!)]).toEqual(["49.9K", "82.9K"]);
    expect([current.recomputed.value, current.written.value, p1.written.value]).toEqual([0, 16 * 1024, 8 * 1024]);
    expect(current.dependencies.value).toBe(16);
  });
});
