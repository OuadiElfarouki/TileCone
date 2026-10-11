import { afterEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { contributionAnalysisEnabled, useInspectorAnalysis } from "../../../src/components/inspector/inspector-analysis";
import { compileDSL } from "../../../src/parse/compiler";
import { executeQuery } from "../../../src/core/executor";
import { box, fromBox } from "../../../src/core/region";
import * as metrics from "../../../src/core/metrics";
import * as multiplicity from "../../../src/core/multiplicity";
import * as contribution from "../../../src/core/contribution";
import * as notes from "../../../src/core/notes";
import * as reuse from "../../../src/core/reuse";
import type { InspectorTab } from "../../../src/view/workspace";

afterEach(() => { vi.restoreAllMocks(); });

describe("inspector analysis gating", () => {
  it("runs contribution probes only when downstream rows are visible", () => {
    expect(contributionAnalysisEnabled("both")).toBe(true);
    expect(contributionAnalysisEnabled("forward")).toBe(true);
    expect(contributionAnalysisEnabled("backward")).toBe(false);
    expect(contributionAnalysisEnabled("none")).toBe(false);
  });

  it.each(["dependencies", "execution", "plan"] as const)("computes only analyses used by the %s tab", (tab: InspectorTab) => {
    const { resolved } = compileDSL("A = Tensor(32, 32)\nB = relu(A)\nC = relu(B)\n");
    const tensorId = Object.values(resolved.tensors).find((tensor) => tensor.name === "B")!.id;
    const parts = [box([0, 8], [0, 8]), box([8, 16], [0, 8])].map((tile) => ({ tensorId, box: tile }));
    const perBox = parts.map((part) => {
      const result = executeQuery(resolved, { tensorId, region: fromBox(part.box), direction: "both" });
      return { backward: result.backward, forward: result.forward };
    });
    const compute = vi.spyOn(metrics, "computeMetrics");
    const reads = vi.spyOn(multiplicity, "operandReads");
    const probes = vi.spyOn(contribution, "contributions");
    const findings = vi.spyOn(notes, "coneFindings");
    const sharing = vi.spyOn(reuse, "inputSharing");
    function Probe() {
      const analysis = useInspectorAnalysis({
        resolved, selection: { parts }, perBox, byTensorRes: null,
        hiddenBoxes: new Set(), focusedBox: null, activeTensorId: tensorId, direction: "both", tab,
      });
      if (tab !== "plan") {
        expect(analysis.metrics).not.toBeNull();
        expect(analysis.upstream.length).toBeGreaterThan(0);
        expect(analysis.downstream.length).toBeGreaterThan(0);
      }
      expect(analysis.bounds !== null).toBe(tab === "execution");
      return null;
    }
    renderToStaticMarkup(createElement(Probe));
    // Execution needs the merged metrics and the two per-tile scenarios;
    // Dependencies needs only the merged cost; Plan uses its own analysis.
    expect(compute).toHaveBeenCalledTimes(tab === "execution" ? 3 : tab === "dependencies" ? 1 : 0);
    for (const dependencyOnly of [reads, probes, findings, sharing])
      expect(dependencyOnly).toHaveBeenCalledTimes(tab === "dependencies" ? 1 : 0);
  });
});
