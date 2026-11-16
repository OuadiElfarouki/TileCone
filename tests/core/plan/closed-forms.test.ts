import { describe, expect, it } from "vitest";
import { resolveGraph } from "../../../src/core/graph";
import { compileDSL } from "../../../src/parse/compiler";
import { tilePlan } from "../../../src/core/plan/plan";
import { interfaceOf, taskSupplyOf } from "../../../src/core/plan/interfaces";
import { payloadOf, stepOf, stepPayload, taskSteps } from "../../../src/core/plan/steps";
import { closedFormsOf, evaluateClosedForm, formatClosedForm, type AvailableClosedForms } from "../../../src/core/plan/closed-forms";

const graph = (m: number, n: number, k: number) => compileDSL(`M = ${m}\nN = ${n}\nK = ${k}\nA = Tensor(M, K, dtype=float16)\nB = Tensor(K, N, dtype=float16)\nC = matmul(A, B)\n`).resolved;
const available = (v: ReturnType<typeof closedFormsOf>): AvailableClosedForms => { if (v.status !== "available") throw new Error(v.reason); return v; };

describe("closed forms", () => {
  it.each([true, false])("matches freshly rebound concrete tasks and steps (declared chunk %s)", (chunked) => {
    const forms = available(closedFormsOf(tilePlan(graph(5, 7, 5), { C: [3, 4] }, chunked ? { C: { chunk: { k: 1 } } } : {}), "C"));
    for (const [m, n, k] of [[5, 7, 5], [8, 3, 3], [4, 6, 1]]) {
      const plan = tilePlan(graph(m, n, k), { C: [3, 4] }, chunked ? { C: { chunk: { k: 1 } } } : {});
      const fam = plan.families.get("C")!;
      for (let i = 0; i < fam.grid[0]; i++) for (let j = 0; j < fam.grid[1]; j++) {
        const task = { tensorId: "C", coord: [i, j] };
        const steps = taskSteps(plan, task);
        if (!("staging" in steps)) throw new Error(steps.reason);
        const supply = taskSupplyOf(plan, task);
        const bindings = { ...forms.defaults, params: { M: m, N: n, K: k }, taskCoord: { m: i, n: j } };
        const evalAt = (expr: Parameters<typeof evaluateClosedForm>[1]) => evaluateClosedForm(forms, expr, bindings);
        expect(evalAt(forms.taskCount)).toBe(fam.count);
        expect(evalAt(forms.stepCount)).toBe(steps.staging.steps);
        expect(evalAt(forms.task.flops)).toBe(supply.flops.value);
        expect(evalAt(forms.task.readBytes)).toBe(payloadOf(plan, supply.demand).bytes.value);
        expect(evalAt(forms.task.outputBytes)).toBe(steps.accumulator.elements * 2);
        expect(evalAt(forms.task.accumulatorBytes)).toBe(steps.accumulator.bytes.value);
        for (let x = 0; x < steps.staging.steps; x++) {
          const step = stepOf(plan, steps, x), payload = stepPayload(plan, step);
          const evalStep = (expr: Parameters<typeof evaluateClosedForm>[1]) => evaluateClosedForm(forms, expr, { ...bindings, stepCoord: { k: x } });
          expect(evalStep(forms.step.flops)).toBe(step.flops.value);
          expect(evalStep(forms.step.payload)).toBe(payload.bytes.value);
          for (const operand of forms.step.operandBytes) expect(evalStep(operand.bytes)).toBe(payload.tensors.find((t) => t.tensorId === operand.tensorId)!.bytes.value);
        }
      }
    }
    expect(formatClosedForm(forms.step.flops)).toContain("taskCoord[");
  });
  it("rejects invalid coordinates, sizes and arithmetic", () => {
    const f = available(closedFormsOf(tilePlan(graph(5, 7, 5), { C: [3, 4] }, { C: { chunk: { k: 2 } } }), "C"));
    expect(() => evaluateClosedForm(f, f.task.flops, { ...f.defaults, taskCoord: { m: 2, n: 0 } })).toThrow();
    expect(() => evaluateClosedForm(f, f.step.flops, { ...f.defaults, chunk: { k: 0 } })).toThrow();
    expect(() => evaluateClosedForm(f, f.taskCount, { ...f.defaults, params: { M: Number.MAX_SAFE_INTEGER, N: 7, K: 5 }, tile: { m: 1, n: 1 } })).toThrow();
  });

  it("rebinds tile and chunk sizes with simultaneous output and reduction tails", () => {
    const f = available(closedFormsOf(tilePlan(graph(5, 7, 5), { C: [3, 4] }, { C: { chunk: { k: 2 } } }), "C"));
    for (const [m, n, k, bm, bn, bk] of [[5, 7, 5, 3, 4, 2], [7, 8, 7, 2, 3, 3], [3, 2, 4, 8, 8, 3]]) {
      const plan = tilePlan(graph(m, n, k), { C: [bm, bn] }, { C: { chunk: { k: bk } } });
      const family = plan.families.get("C")!;
      const coord = family.grid.map((extent) => extent - 1);
      const task = { tensorId: "C", coord };
      const steps = taskSteps(plan, task);
      if (!("staging" in steps)) throw new Error(steps.reason);
      const step = stepOf(plan, steps, steps.staging.steps - 1);
      const bindings = {
        params: { M: m, N: n, K: k }, tile: { m: bm, n: bn }, chunk: { k: bk },
        taskCoord: { m: coord[0], n: coord[1] }, stepCoord: { k: step.coord[0] },
      };
      expect(evaluateClosedForm(f, f.taskCount, bindings)).toBe(family.count);
      expect(evaluateClosedForm(f, f.task.flops, bindings)).toBe(taskSupplyOf(plan, task).flops.value);
      expect(evaluateClosedForm(f, f.task.accumulatorBytes, bindings)).toBe(steps.accumulator.bytes.value);
      expect(evaluateClosedForm(f, f.step.flops, bindings)).toBe(step.flops.value);
      expect(evaluateClosedForm(f, f.step.payload, bindings)).toBe(stepPayload(plan, step).bytes.value);
    }
  });
  it("refuses diagonal, shared and fused contractions", () => {
    const cases = [
      [`A = Tensor(4, 4, 3)\nB = Tensor(3, 5)\nC = einsum("iik,kj->ij", A, B)`, { C: [2, 2] }],
      [`A = Tensor(4, 4)\nC = matmul(A, A)`, { C: [2, 2] }],
      [`A = Tensor(4, 4)\nB = Tensor(4, 4)\nP = relu(A)\nC = matmul(P, B)`, { C: [2, 2] }],
    ] as const;
    for (const [src, tiles] of cases) expect(closedFormsOf(tilePlan(compileDSL(src).resolved, tiles), "C").status).toBe("unavailable");
  });
  it.each(["Tensor(N, dtype=float16)", "Tensor(M, N, dtype=float16)", "Tensor(dtype=float16)", "Tensor(1, N, dtype=float16)"])("matches a fused add, relu and cast with side %s", (side) => {
    const source = (m: number, n: number, k: number) => `M = ${m}\nN = ${n}\nK = ${k}\nA = Tensor(M, K, dtype=float16)\nB = Tensor(K, N, dtype=float16)\nBias = ${side}\nP = matmul(A, B)\nQ = add(P, Bias)\nR = relu(Q)\nC = astype(R, float32)`;
    const build = (m: number, n: number, k: number) => tilePlan(compileDSL(source(m, n, k)).resolved, { C: [3, 4] }, { C: { chunk: { k: 1 } } });
    const f = available(closedFormsOf(build(5, 1, 5), "C"));
    for (const [m, n, k] of [[5, 1, 5], [8, 7, 3], [4, 6, 1]]) {
      const plan = build(m, n, k);
      const family = plan.families.get("C")!;
      const report = interfaceOf(plan, "C");
      if (report.status !== "evaluated") throw new Error("family not evaluated");
      const bindings = { ...f.defaults, params: { M: m, N: n, K: k } };
      const at = (expr: Parameters<typeof evaluateClosedForm>[1]) => evaluateClosedForm(f, expr, bindings);
      expect(at(f.family.flops)).toBe(report.work.flops.value);
      expect(at(f.family.readBytes)).toBe(report.work.read.value);
      expect(at(f.family.writtenBytes)).toBe(report.work.written.value);
      let sideBytes = 0;
      for (let i = 0; i < family.grid[0]; i++) for (let j = 0; j < family.grid[1]; j++) {
        const supply = taskSupplyOf(plan, { tensorId: "C", coord: [i, j] });
        sideBytes += payloadOf(plan, supply.demand.filter((d) => d.tensorId === "Bias")).bytes.value!;
      }
      expect(at(f.family.operandBytes.find((operand) => operand.tensorId === "Bias")!.bytes)).toBe(sideBytes);
      for (let i = 0; i < family.grid[0]; i++) for (let j = 0; j < family.grid[1]; j++) {
        const task = { tensorId: "C", coord: [i, j] };
        const steps = taskSteps(plan, task);
        if (!("staging" in steps)) throw new Error(steps.reason);
        const supply = taskSupplyOf(plan, task);
        const atTask = (expr: Parameters<typeof evaluateClosedForm>[1]) => evaluateClosedForm(f, expr, { ...bindings, taskCoord: { m: i, n: j } });
        expect(atTask(f.task.flops)).toBe(supply.flops.value);
        expect(atTask(f.task.readBytes)).toBe(payloadOf(plan, supply.demand).bytes.value);
        expect(atTask(f.task.outputBytes)).toBe(steps.accumulator.elements * 4);
        expect(atTask(f.task.accumulatorBytes)).toBe(steps.accumulator.bytes.value);
        if ((i + 1) * 3 <= m && (j + 1) * 4 <= n) {
          expect(atTask(f.interior.task.flops)).toBe(supply.flops.value);
          if (k > 1) expect(atTask(f.interior.accumulating.taskFlops)).toBe(supply.flops.value);
          expect(atTask(f.interior.task.readBytes)).toBe(payloadOf(plan, supply.demand).bytes.value);
        }
        expect(atTask(f.step.flops)).toBe(stepOf(plan, steps, 0).flops.value);
        expect(atTask(f.step.payload)).toBe(stepPayload(plan, stepOf(plan, steps, 0)).bytes.value);
      }
    }
  });

  it("declines unsupported epilogue operands and views", () => {
    const prefix = `M = 4\nN = 1\nK = 4\nQ = 1\nA = Tensor(M, K)\nB = Tensor(K, N)\nBias = Tensor(Q)\nP = matmul(A, B)\n`;
    const examples = [
      `${prefix}C = add(P, Bias)`, // Q only happens to be one
      `${prefix}R = add(P, A)\nC = relu(R)`, // contraction input or shape expansion
      `A = Tensor(4, 4)\nB = Tensor(4, 4)\nBias = Tensor(4)\nP = matmul(A, B)\nR = add(P, Bias)\nC = add(R, Bias)`,
      `A = Tensor(4, 4)\nB = Tensor(4, 4)\nP = matmul(A, B)\nC = add(P, A)`,
      `A = Tensor(4, 4)\nB = Tensor(4, 1)\nBias = Tensor(4, 4)\nP = matmul(A, B)\nC = add(P, Bias)`,
      `A = Tensor(4, 4)\nB = Tensor(4, 4)\nP = matmul(A, B)\nR = relu(P)\nUnused = relu(P)\nC = relu(R)`,
      `A = Tensor(4, 4)\nB = Tensor(4, 4)\nP = matmul(A, B)\nC = permute_dims(P, [1, 0])`,
    ];
    for (const source of examples) {
      const plan = tilePlan(compileDSL(source).resolved, { C: [2, 1] });
      expect(closedFormsOf(plan, "C").status).toBe("unavailable");
    }
  });

  it("refuses aliases equal only at the current binding", () => {
    const g = compileDSL(`M = 4\nN = 4\nA = Tensor(3, M)\nB = Tensor(N, 5)\nC = matmul(A, B)`).resolved;
    expect(closedFormsOf(tilePlan(g, { C: [2, 2] }), "C").status).toBe("unavailable");
  });
  it("proves derived dimensions and supports a tiled operand", () => {
    const source = (h: number, d: number) => `H = ${h}\nD = ${d}\nE = H*D\nA = Tensor(3, E)\nB = Tensor(H*D, 5)\nP = relu(A)\nC = matmul(P, B)`;
    const initial = tilePlan(compileDSL(source(2, 3)).resolved, { P: [2, 3], C: [2, 3] });
    const f = available(closedFormsOf(initial, "C"));
    for (const [h, d] of [[2, 3], [1, 1], [3, 2]]) {
      const plan = tilePlan(compileDSL(source(h, d)).resolved, { P: [2, 3], C: [2, 3] });
      const task = { tensorId: "C", coord: [1, 1] };
      const supply = taskSupplyOf(plan, task);
      const bindings = { ...f.defaults, params: { H: h, D: d }, taskCoord: { m: 1, n: 1 } };
      expect(evaluateClosedForm(f, f.task.flops, bindings)).toBe(supply.flops.value);
      expect(evaluateClosedForm(f, f.task.readBytes, bindings)).toBe(payloadOf(plan, supply.demand).bytes.value);
    }
  });
  it("refuses storage widening", () => {
    const original = graph(4, 4, 4);
    const widened = resolveGraph({ ...original, tensors: { ...original.tensors, A: { ...original.tensors.A, dtypeWidening: { from: "uint16", note: "widened" } } } });
    expect(closedFormsOf(tilePlan(widened, { C: [2, 2] }), "C").status).toBe("unavailable");
  });

  it.each([
    [`A = Tensor(3, 4)\nB = Tensor(3, 4)\nC = einsum("ij,ij->", A, B)`, [], { i: 2, j: 3 }],
    [`A = Tensor(2, 3, 4)\nB = Tensor(2, 4, 5)\nC = matmul(A, B)`, [1, 2, 3], { k: 3 }],
    [`A = Tensor(3, 4)\nB = Tensor(5, 4)\nC = linear(A, B)`, [2, 3], { k: 3 }],
    [`A = Tensor(2, 3, 4)\nB = Tensor(3, 4, 5)\nC = einsum("ijk,jkl->il", A, B)`, [1, 3], { j: 2, k: 3 }],
  ] as const)("matches scalar, batched, linear and multi-axis steps", (source, tile, chunk) => {
    const plan = tilePlan(compileDSL(source).resolved, { C: tile }, { C: { chunk } });
    const f = available(closedFormsOf(plan, "C"));
    const task = { tensorId: "C", coord: tile.map(() => 0) };
    const steps = taskSteps(plan, task);
    if (!("staging" in steps)) throw new Error(steps.reason);
    expect(evaluateClosedForm(f, f.taskCount, f.defaults)).toBe(plan.families.get("C")!.count);
    expect(evaluateClosedForm(f, f.stepCount, f.defaults)).toBe(steps.staging.steps);
    expect(evaluateClosedForm(f, f.task.flops, f.defaults)).toBe(taskSupplyOf(plan, task).flops.value);
    for (let index = 0; index < steps.staging.steps; index++) {
      const step = stepOf(plan, steps, index);
      const coord = Object.fromEntries(steps.staging.reduced.map((axis, i) => [axis.label, step.coord[i]]));
      const bindings = { ...f.defaults, stepCoord: coord };
      expect(evaluateClosedForm(f, f.step.flops, bindings)).toBe(step.flops.value);
      expect(evaluateClosedForm(f, f.step.payload, bindings)).toBe(stepPayload(plan, step).bytes.value);
    }
  });
  it("rejects unknown variables and accepts overrides of full-axis chunks", () => {
    const f = available(closedFormsOf(tilePlan(graph(5, 7, 5), { C: [3, 4] }), "C"));
    expect(evaluateClosedForm(f, f.stepCount, { ...f.defaults, chunk: { k: 2 } })).toBe(3);
    expect(() => evaluateClosedForm(f, f.stepCount, { ...f.defaults, chunk: { typo: 2 } })).toThrow();
    expect(() => evaluateClosedForm(f, f.taskCount, { ...f.defaults, taskCoord: { typo: 0 } })).toThrow();
  });

  it("reuses an undeclared chunk law across rebinding and explicit overrides", () => {
    const f = available(closedFormsOf(tilePlan(graph(5, 7, 5), { C: [3, 4] }), "C"));
    for (const [m, n, k, chunk] of [[5, 7, 5, 2], [8, 3, 3, 3], [4, 6, 1, 1]]) {
      for (const override of [false, true]) {
        const plan = tilePlan(graph(m, n, k), { C: [3, 4] }, override ? { C: { chunk: { k: chunk } } } : {});
        const task = { tensorId: "C", coord: [0, 0] };
        const steps = taskSteps(plan, task);
        if (!("staging" in steps)) throw new Error(steps.reason);
        const bindings = { ...f.defaults, params: { M: m, N: n, K: k }, ...(override ? { chunk: { k: chunk } } : {}) };
        expect(evaluateClosedForm(f, f.stepCount, bindings)).toBe(steps.staging.steps);
        expect(evaluateClosedForm(f, f.step.flops, bindings)).toBe(stepOf(plan, steps, 0).flops.value);
        if (m >= 3 && n >= 4) expect(evaluateClosedForm(f, f.interior.step.flops, bindings)).toBe(stepOf(plan, steps, 0).flops.value);
      }
    }
  });

  it("prints concise interior laws with author names and explicit conditions", () => {
    const g = compileDSL(`M = 6\nN = 8\nK = 4\nA = Tensor(row=M, reduction=K, dtype=float16)\nB = Tensor(reduction=K, column=N, dtype=float16)\nC = matmul(A, B)`).resolved;
    const f = available(closedFormsOf(tilePlan(g, { C: [3, 4] }, { C: { chunk: { k: 2 } } }), "C"));
    expect(f.axes.map((axis) => axis.displayName)).toEqual(["row", "reduction", "column"]);
    expect(f.interior.conditions).toEqual(["every output tile is full", "every reduction chunk is full"]);
    const interior = formatClosedForm(f.interior.step.flops, f);
    expect(interior).toContain("tile[row]");
    expect(interior).toContain("chunk[reduction]");
    expect(interior).not.toContain("taskCoord");
    expect(interior).not.toContain("stepCoord");
    expect(formatClosedForm(f.step.flops, f)).toContain("taskCoord[row]");
    expect(f.interior.accumulating.conditions).toContain("full reduction has more than one term");
    expect(formatClosedForm(f.interior.accumulating.stepFlops, f)).toBe("2 · tile[row] · chunk[reduction] · tile[column]");
    expect(evaluateClosedForm(f, f.interior.accumulating.stepFlops, f.defaults)).toBe(evaluateClosedForm(f, f.step.flops, f.defaults));
    const degenerate = { ...f.defaults, params: { M: 6, N: 8, K: 1 }, chunk: { k: 1 } };
    expect(evaluateClosedForm(f, f.interior.step.flops, degenerate)).toBe(12);
    expect(evaluateClosedForm(f, f.interior.step.flops, f.defaults)).toBe(evaluateClosedForm(f, f.step.flops, f.defaults));
  });

  it("groups compound symbolic extents in printed laws", () => {
    expect(formatClosedForm({ kind: "ceilDiv", left: { kind: "dim", value: "M+N" }, right: { kind: "dim", value: 4 } }))
      .toBe("ceil((M+N) / 4)");
    expect(formatClosedForm({ kind: "sub", left: { kind: "dim", value: 10 }, right: { kind: "dim", value: "M+N" } }))
      .toBe("(10 − (M+N))");
  });


  it("gives family totals that match the engine's sums over every task, tails included", () => {
    const totals = (plan: ReturnType<typeof tilePlan>) => {
      const report = interfaceOf(plan, "C");
      if (report.status !== "evaluated") throw new Error("family not evaluated");
      return report.work;
    };
    const f = available(closedFormsOf(tilePlan(graph(5, 7, 5), { C: [3, 4] }, { C: { chunk: { k: 2 } } }), "C"));
    // Tiles that divide the axes, tiles that leave tails on one or both, a tile
    // wider than its axis, and chunks with and without a tail.
    for (const [m, n, k, bm, bn, bk] of [[5, 7, 5, 3, 4, 2], [8, 6, 4, 2, 3, 4], [7, 8, 7, 2, 3, 3], [3, 2, 4, 8, 8, 3], [6, 5, 1, 1, 5, 1]]) {
      const plan = tilePlan(graph(m, n, k), { C: [bm, bn] }, { C: { chunk: { k: bk } } });
      const work = totals(plan);
      const bindings = { params: { M: m, N: n, K: k }, tile: { m: bm, n: bn }, chunk: { k: bk } };
      const at = (expr: Parameters<typeof evaluateClosedForm>[1]) => evaluateClosedForm(f, expr, bindings);
      expect(at(f.family.flops)).toBe(work.flops.value);
      expect(at(f.family.readBytes)).toBe(work.read.value);
      expect(at(f.family.writtenBytes)).toBe(work.written.value);
      expect(at(f.family.steps)).toBe(work.tasks * at(f.stepCount));
      // Each operand's share: summed over the tasks' own demand rows.
      const family = plan.families.get("C")!;
      for (const operand of f.family.operandBytes) {
        let bytes = 0;
        for (let i = 0; i < family.grid[0]; i++) for (let j = 0; j < family.grid[1]; j++) {
          const supply = taskSupplyOf(plan, { tensorId: "C", coord: [i, j] });
          bytes += payloadOf(plan, supply.demand.filter((d) => d.tensorId === operand.tensorId)).bytes.value!;
        }
        expect(at(operand.bytes)).toBe(bytes);
      }
    }
    // The law itself: A is read once per column of tiles, B once per row of tiles.
    expect(formatClosedForm(f.family.readBytes)).toBe("2 · M · K · ceil(N / tile[n]) + 2 · K · N · ceil(M / tile[m])");
    expect(formatClosedForm(f.family.flops)).toBe("(1 + min(1, (K − 1))) · M · K · N");
  });

  it("gives family totals for a batched contraction whose operands carry different output axes", () => {
    const source = (b: number, q: number, kk: number, d: number) =>
      `B = ${b}\nQ = ${q}\nK = ${kk}\nD = ${d}\nX = Tensor(B, Q, D, dtype=float16)\nY = Tensor(B, K, D, dtype=float32)\nS = einsum("bqd,bkd->bqk", X, Y)`;
    const f = available(closedFormsOf(tilePlan(compileDSL(source(2, 5, 3, 4)).resolved, { S: [1, 2, 2] }, { S: { chunk: { d: 3 } } }), "S"));
    for (const [b, q, kk, d, tile, chunk] of [[2, 5, 3, 4, [1, 2, 2], 3], [3, 4, 6, 5, [2, 3, 4], 2], [1, 7, 2, 3, [1, 7, 1], 3]] as const) {
      const plan = tilePlan(compileDSL(source(b, q, kk, d)).resolved, { S: [...tile] }, { S: { chunk: { d: chunk } } });
      const report = interfaceOf(plan, "S");
      if (report.status !== "evaluated") throw new Error("family not evaluated");
      const bindings = { params: { B: b, Q: q, K: kk, D: d }, tile: { b: tile[0], q: tile[1], k: tile[2] }, chunk: { d: chunk } };
      expect(evaluateClosedForm(f, f.family.flops, bindings)).toBe(report.work.flops.value);
      expect(evaluateClosedForm(f, f.family.readBytes, bindings)).toBe(report.work.read.value);
      expect(evaluateClosedForm(f, f.family.writtenBytes, bindings)).toBe(report.work.written.value);
    }
  });

  it("refuses an extent that has no expression, which a rebinding would change", () => {
    // Both operands are pooled from length L, so their shared axis reads as the
    // literal 8 and agrees with itself. At L = 32 it is 16.
    const source = (l: number) => `N = 4\nL = ${l}\nX1 = Tensor(1, N, L, dtype=float16)\nX2 = Tensor(1, N, L, dtype=float16)\n` +
      `P = max_pool1d(X1, kernel_size=2, stride=2)\nQ = max_pool1d(X2, kernel_size=2, stride=2)\nY = einsum("bcl,bdl->bcd", P, Q)`;
    const g = compileDSL(source(16)).resolved;
    expect(g.tensors.P.symShape).toEqual([1, "N", 8]);
    expect(g.tensors.P.symProven).toEqual([true, true, false]);
    const forms = closedFormsOf(tilePlan(g, { P: [1, 4, 8], Q: [1, 4, 8], Y: [1, 2, 2] }), "Y");
    expect(forms).toEqual({ status: "unavailable", reason: "axis 2 of P has no expression: its extent 8 is known at this binding only" });
    // What a law over the literal would have got wrong.
    const flopsAt = (l: number) => {
      const rebound = compileDSL(source(l)).resolved;
      const plan = tilePlan(rebound, { P: [...rebound.tensors.P.resolved], Q: [...rebound.tensors.Q.resolved], Y: [1, 2, 2] });
      return taskSupplyOf(plan, { tensorId: "Y", coord: [0, 0, 0] }).flops.value;
    };
    expect(flopsAt(32)).toBe(2 * flopsAt(16)!);
  });

  it("keeps a literal the source wrote, and an expression carried through an elementwise operation", () => {
    const g = compileDSL(`M = 4\nA = Tensor(M, 8, dtype=float16)\nB = Tensor(8, 6, dtype=float16)\nP = relu(A)\nC = matmul(P, B)`).resolved;
    expect(g.tensors.P.symProven).toEqual([true, true]);
    expect(closedFormsOf(tilePlan(g, { P: [4, 8], C: [2, 3] }), "C").status).toBe("available");
  });
});
