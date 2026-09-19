import { describe, expect, it } from "vitest";
import { ExecutionError, executeBoundedQuery } from "../core/executor";
import { computeMetrics } from "../core/metrics";
import { propagateBackward, propagateForward, propagateWithin, PropResult } from "../core/propagate";
import { box, count, fromBox, Region } from "../core/region";
import { compileDSL } from "../parse/compiler";
import { checkGraph, CheckOpts, rng, randInt, randomGraph } from "./harness";

/* The first milestone of NEXT_FEATS, at a size the oracle can enumerate: two
   matmuls, with a consumer after the second so a forward walk has somewhere
   further to go. */
const chain = () =>
  compileDSL(`A = Tensor(8, 8)
B = Tensor(8, 8)
W = Tensor(8, 4)
C = matmul(A, B)
Y = matmul(C, W)
Z = relu(Y)
`);

/** The second matmul as a stage: everything it reads, and what it writes. */
const STAGE = ["C", "W", "Y"];

const ids = (tensors: Map<string, unknown>) => [...tensors.keys()].sort();
const regionOn = (tensors: Map<string, { region: Region }>, id: string) => tensors.get(id)?.region;

describe("a frontier bounds the cone", () => {
  it("stops a consumer tile's demand at the stage boundary", () => {
    const { executor } = chain();
    const tile = fromBox(box([0, 2], [0, 2]));
    const cone = executor.upstreamWithin("Y", tile, STAGE);

    expect(ids(cone.tensors)).toEqual(["C", "W", "Y"]);
    expect(cone.stoppedAt).toEqual(["C", "W"]);
    expect(regionOn(cone.tensors, "C")?.boxes).toEqual([box([0, 2], [0, 8])]);
    expect(regionOn(cone.tensors, "W")?.boxes).toEqual([box([0, 8], [0, 2])]);

    // The bound removes what lies past the boundary, never demand on it.
    const open = executor.upstream("Y", tile);
    expect(ids(open.tensors)).toEqual(["A", "B", "C", "W", "Y"]);
    for (const id of cone.stoppedAt)
      expect(regionOn(cone.tensors, id)).toEqual(regionOn(open.tensors, id));
  });

  it("serves both directions with one boundary, because the seed is never blocked", () => {
    const { executor } = chain();

    const up = executor.upstreamWithin("Y", fromBox(box([0, 2], [0, 2])), STAGE);
    expect(up.frontier).toContain("Y");
    expect(up.stoppedAt).not.toContain("Y");

    const down = executor.downstreamWithin("C", fromBox(box([0, 2], [0, 8])), STAGE);
    expect(down.frontier).toContain("C");
    expect(down.stoppedAt).toEqual(["Y"]);
    expect(ids(down.tensors)).toEqual(["C", "Y"]);
    expect(regionOn(down.tensors, "Y")?.boxes).toEqual([box([0, 2], [0, 4])]);
    expect(executor.downstream("C", fromBox(box([0, 2], [0, 8]))).tensors.has("Z")).toBe(true);
  });

  /* The case that separates "stop at these tensors" from "drop these tensors".
     D[0,1] reads A[0,1] directly and A[1,0] through the transpose. Blocking B
     removes the second route only, so A keeps what the direct route supplies. */
  it("cuts paths through the frontier, not tensors reachable around it", () => {
    const { executor } = compileDSL(`A = Tensor(3, 3)
B = transpose(A, perm=[1, 0])
D = add(B, A)
`);
    const up = executor.upstreamWithin("D", fromBox(box([0, 1], [1, 2])), ["B"]);
    expect(regionOn(up.tensors, "A")?.boxes).toEqual([box([0, 1], [1, 2])]);
    expect(regionOn(up.tensors, "B")?.boxes).toEqual([box([0, 1], [1, 2])]);
    expect(up.stoppedAt).toEqual(["B"]);
    expect(count(executor.upstream("D", fromBox(box([0, 1], [1, 2]))).tensors.get("A")!.region)).toBe(2);

    const down = executor.downstreamWithin("A", fromBox(box([0, 1], [1, 2])), ["B"]);
    expect(regionOn(down.tensors, "D")?.boxes).toEqual([box([0, 1], [1, 2])]);
    expect(regionOn(down.tensors, "B")?.boxes).toEqual([box([1, 2], [0, 1])]);
    expect(count(executor.downstream("A", fromBox(box([0, 1], [1, 2]))).tensors.get("D")!.region)).toBe(2);
  });

  it("reports a tensor read through two operand slots as one stop with two crossings", () => {
    const { executor } = compileDSL(`A = Tensor(4, 4)
X = relu(A)
C = matmul(X, X)
`);
    const tile = fromBox(box([0, 1], [0, 1]));
    const cone = executor.upstreamWithin("C", tile, ["X"]);
    expect(cone.stoppedAt).toEqual(["X"]);
    expect(cone.tensors.has("A")).toBe(false);
    // Both bands, as stored: the row it is read by and the column it is read as.
    expect(count(regionOn(cone.tensors, "X")!)).toBe(7);
    expect(regionOn(cone.tensors, "X")).toEqual(regionOn(executor.upstream("C", tile).tensors, "X"));
    // One crossing per slot keeps the two bands apart.
    expect(cone.crossings.map(({ node, slot, tensorId, region }) => [node, slot, tensorId, region.boxes]))
      .toEqual([
        [cone.crossings[0].node, 0, "X", [box([0, 1], [0, 4])]],
        [cone.crossings[0].node, 1, "X", [box([0, 4], [0, 1])]],
      ]);
  });

  it("is the transitive cone when the frontier is empty", () => {
    const { resolved } = chain();
    const serialize = (tensors: PropResult["tensors"]) =>
      JSON.stringify([...tensors].sort(([a], [b]) => a.localeCompare(b)));
    const up = { tensorId: "Z", region: fromBox(box([1, 3], [0, 4])) };
    const down = { tensorId: "A", region: fromBox(box([2, 3], [0, 8])) };
    expect(serialize(propagateWithin(resolved, up, "backward", []).tensors))
      .toBe(serialize(propagateBackward(resolved, up).tensors));
    expect(serialize(propagateWithin(resolved, down, "forward", []).tensors))
      .toBe(serialize(propagateForward(resolved, down).tensors));
  });

  it("normalizes the frontier it reports", () => {
    const { resolved } = chain();
    const cone = propagateWithin(resolved, { tensorId: "Y", region: fromBox(box([0, 1], [0, 1])) },
      "backward", ["W", "C", "W", "Y"]);
    expect(cone.frontier).toEqual(["C", "W", "Y"]);
    expect(cone.stoppedAt).toEqual(["C", "W"]);
  });
});

describe("bounded cones against the oracle", () => {
  /* Each graph gets its own random frontier and every tensor in it is a seed,
     so seeds land on the frontier, inside it and beyond it. The oracle cuts at
     the same frontier pointwise, from `oracleDeps` alone. */
  const corpus = (seed: number, limits?: CheckOpts["limits"]) => {
    const r = rng(seed);
    let walks = 0;
    let cut = 0;
    for (let trial = 0; trial < 24; trial++) {
      const graph = randomGraph(r, randInt(r, 5, 16));
      const frontier = Object.keys(graph.tensors).filter(() => r() < 0.3);
      const stats = checkGraph(graph, {
        frontier,
        perTensorElementCap: 8,
        boxSelections: 1,
        seed: trial,
        limits,
      });
      walks += stats.walks;
      cut += stats.cut;
    }
    // Agreement means little if the frontiers never cut a path.
    expect(cut).toBeGreaterThan(walks / 20);
  };

  it("agrees on random graphs with random frontiers", () => corpus(7));

  it("stays a superset under tight fallback thresholds", () =>
    corpus(8, { stridedEnum: 2, diagEnum: 2, reshapeRuns: 1, maxBoxes: 2 }));
});

describe("the checked boundary", () => {
  const tile = fromBox(box([0, 1], [0, 1]));

  it("refuses a frontier naming an unknown tensor rather than stopping nowhere", () => {
    const { executor } = chain();
    expect(() => executor.upstreamWithin("Y", tile, ["C", "Q"])).toThrowError(
      expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_FRONTIER" })
    );
  });

  it("refuses a frontier that is not a list", () => {
    const { executor } = chain();
    // A bare string is iterable, so a looser check would read "CW" as ["C", "W"].
    expect(() => executor.upstreamWithin("Y", tile, "CW" as unknown as string[])).toThrowError(
      expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_FRONTIER" })
    );
  });

  it("runs one direction at a time", () => {
    const { resolved } = chain();
    expect(() =>
      executeBoundedQuery(resolved, {
        tensorId: "Y",
        region: tile,
        direction: "both" as never,
        frontier: STAGE,
      })
    ).toThrowError(expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_DIRECTION" }));
  });

  it("validates the seed like any other query", () => {
    const { executor } = chain();
    expect(() => executor.upstreamWithin("Y", fromBox(box([0, 9], [0, 1])), STAGE)).toThrowError(
      expect.objectContaining<Partial<ExecutionError>>({ code: "EXEC_REGION_BOUNDS" })
    );
  });
});

describe("isolation from transitive cones", () => {
  it("leaves the memoized plan to the queries that share it", () => {
    const { executor } = chain();
    const tile = fromBox(box([0, 2], [0, 2]));
    const before = JSON.stringify([...executor.upstream("Z", tile).tensors]);
    executor.upstreamWithin("Z", tile, STAGE);
    expect(JSON.stringify([...executor.upstream("Z", tile).tensors])).toBe(before);
  });

  it("cannot be handed to a consumer that assumes the whole cone", () => {
    const { resolved, executor } = chain();
    const cone = executor.upstreamWithin("Y", fromBox(box([0, 1], [0, 1])), STAGE);
    // Metrics would count C as an intermediate and miss A and B entirely,
    // understating input bytes. The compiler refuses it; this line fails the
    // build if a later change lets it through.
    // @ts-expect-error a bounded cone is not a PropResult
    void (() => computeMetrics(resolved, cone));
  });
});
