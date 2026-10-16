import { describe, expect, it } from "vitest";
import { figure, type Figure } from "../../src/core/metrics";
import { planReport, type Work } from "../../src/core/plan/interfaces";
import { tilePlan } from "../../src/core/plan/plan";
import { compileDSL } from "../../src/parse/compiler";
import { FIGURE_MARK } from "../../src/view/format";
import {
  describePlanChange,
  describeTiling,
  describeTilingChange,
  WORK_ROWS,
  workCell,
  workCellTitle,
  workChange,
  workDirection,
} from "../../src/view/plan-work";

const chain = () =>
  compileDSL(`A = Tensor(256, 256, dtype=fp16)
B = Tensor(256, 256, dtype=fp16)
W = Tensor(256, 128, dtype=fp16)
C = matmul(A, B)
Y = matmul(C, W)
`).resolved;

describe("whole-plan rows", () => {
  const written = planReport(tilePlan(chain(), { C: [64, 64], Y: [64, 64] })).total!;
  const fused = planReport(tilePlan(chain(), { Y: [64, 64] })).total!;
  const row = (label: string) => WORK_ROWS.find((r) => r.label === label)!;

  it("lists every Work figure once, in a fixed order", () => {
    expect(WORK_ROWS.map((r) => r.label)).toEqual([
      "tasks",
      "dependencies",
      "FLOPs",
      "recomputed",
      "read",
      "distinct read",
      "written",
      "FLOP / byte",
    ]);
  });

  it("formats each plan's cell with the mark its status earns", () => {
    expect(workCell(row("tasks"), written)).toBe("24");
    expect(workCell(row("FLOPs"), written)).toBe("50.33M");
    expect(workCell(row("recomputed"), fused)).toBe("33.55M");
    expect(workCell(row("written"), fused)).toBe("64.0 KB");
  });

  it("states a kept plan's figure against the current one", () => {
    expect(workChange(row("FLOPs"), fused, written)).toBe("+67% against the current plan");
    expect(workChange(row("tasks"), fused, written)).toBe("-67% against the current plan");
    expect(workChange(row("written"), written, written)).toBe("same as the current plan");
    // Nothing recomputed in the current plan: no ratio to state.
    expect(workChange(row("recomputed"), fused, written)).toBeNull();
  });
});

describe("changes between bounds", () => {
  const exact = (value: number) => figure(value, "exact");
  const upper = (value: number, reason = "strided conv") => figure(value, "upper", [reason]);
  const work = (read: Figure): Work => ({
    tasks: 4,
    dependencies: exact(0),
    flops: exact(1000),
    recomputed: exact(0),
    read,
    readDistinct: exact(100),
    written: exact(100),
    intensity: exact(1),
  });
  const read = WORK_ROWS.find((r) => r.label === "read")!;
  const approx = FIGURE_MARK.approximate;

  it("does not call two equal bounds the same", () => {
    expect(workChange(read, work(upper(100)), work(upper(100)))).toBe(`${approx}same as the current plan`);
  });

  it("marks a change as approximate when either side is a bound", () => {
    expect(workChange(read, work(upper(150)), work(exact(100)))).toBe(`${approx}+50% against the current plan`);
    expect(workChange(read, work(exact(150)), work(upper(100)))).toBe(`${approx}+50% against the current plan`);
    expect(workChange(read, work(exact(150)), work(exact(100)))).toBe("+50% against the current plan");
  });

  it("has no change against an unknown figure", () => {
    const flops = WORK_ROWS.find((r) => r.label === "FLOPs")!;
    const unknown = { ...work(exact(100)), flops: figure(0, "unknown", ["unknown work in Mystery"]) };
    expect(workChange(flops, unknown, work(exact(100)))).toBeNull();
    expect(workCellTitle(flops, unknown, work(exact(100)))).toBe("unknown work in Mystery");
  });

  it("keeps a figure's reasons beside its change", () => {
    const current = work(exact(100));
    expect(workCellTitle(read, work(upper(150)), current)).toBe(
      `${approx}+50% against the current plan · strided conv`
    );
    // The current plan's own cell states only its reasons.
    expect(workCellTitle(read, current, current)).toBeUndefined();
    expect(workCellTitle(read, work(upper(100)), null)).toBe("strided conv");
  });
});

describe("a tiling in words", () => {
  it("names each tiled tensor with its extents, sorted by name", () => {
    const names: Record<string, string> = { t1: "Y", t2: "C", t3: "S" };
    expect(describeTiling({ t1: [64, 64], t2: [64, 128], t3: [] }, (id) => names[id])).toBe(
      "C 64×128, S scalar, Y 64×64"
    );
    expect(describeTiling({}, (id) => id)).toBe("nothing tiled");
  });
});

describe("what a plan changes from a kept one", () => {
  const isOutput = (id: string) => id === "Y";
  const name = (id: string) => id;

  it("names fusing, materializing, retiling, and starting or stopping a write", () => {
    expect(describeTilingChange({ C: [64, 64], Y: [64, 64] }, { Y: [64, 128] }, isOutput, name))
      .toEqual(["fuses C into its consumers", "retiles Y 64×64 → 64×128"]);
    expect(describeTilingChange({ Y: [64, 64] }, { C: [64, 64], Y: [64, 64] }, isOutput, name))
      .toEqual(["materializes C"]);
    expect(describeTilingChange({ C: [64, 64], Y: [64, 64] }, { C: [64, 64] }, isOutput, name))
      .toEqual(["stops writing Y"]);
    expect(describeTilingChange({ C: [64, 64] }, { C: [64, 64], Y: [64, 64] }, isOutput, name))
      .toEqual(["writes Y"]);
    expect(describeTilingChange({ Y: [64, 64] }, { Y: [64, 64] }, isOutput, name)).toEqual([]);
  });

  it("says which exact figures are lower and higher, and compares no bounds", () => {
    const written = planReport(tilePlan(chain(), { C: [64, 64], Y: [64, 64] })).total!;
    const fused = planReport(tilePlan(chain(), { Y: [64, 64] })).total!;
    const { lower, higher } = workDirection(written, fused);
    expect(lower).toEqual(expect.arrayContaining(["tasks", "dependencies", "written"]));
    expect(higher).toEqual(expect.arrayContaining(["FLOPs", "recomputed"]));
    const bounded: Work = { ...fused, flops: { value: fused.flops.value!, status: "upper", reasons: ["test"] } };
    const both = workDirection(written, bounded);
    expect([...both.lower, ...both.higher]).not.toContain("FLOPs");
  });

  it("puts the decisions first and the figures after, in one sentence", () => {
    const tiles = { C: [64, 64], Y: [64, 64] };
    const written = planReport(tilePlan(chain(), tiles)).total!;
    const fused = planReport(tilePlan(chain(), { Y: [64, 64] })).total!;
    const text = describePlanChange("P1", { tiles, work: written }, { tiles: { Y: [64, 64] }, work: fused }, isOutput, name);
    expect(text).toMatch(/^From P1, the current plan fuses C into its consumers\. Lower: [^.]*\. Higher: [^.]*recomputed[^.]*\.$/);
    expect(describePlanChange("P1", { tiles, work: written }, { tiles: { Y: [64, 64] }, work: null }, isOutput, name))
      .toBe("From P1, the current plan fuses C into its consumers.");
    expect(describePlanChange("P2", { tiles, work: written }, { tiles, work: written }, isOutput, name))
      .toBe("P2 is the current plan.");
  });
});
