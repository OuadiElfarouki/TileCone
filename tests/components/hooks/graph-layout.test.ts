import { afterEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { useGraphLayout } from "../../../src/components/hooks/useGraphLayout";
import * as jobs from "../../../src/worker/analysis-jobs";

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("graph layout render", () => {
  it("keeps placement usable without running Dagre in a browser render", () => {
    const compiled = jobs.compileArtifact("X = Tensor(4, 32, 256)\nY = relu(X)\n");
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const { resolved, artifact } = compiled;
    vi.stubGlobal("Worker", class {});
    const layout = vi.spyOn(jobs, "layoutArtifact").mockImplementation(() => { throw new Error("layout on UI thread"); });
    function Probe() {
      const state = useGraphLayout(resolved, artifact.layout, 42, artifact.graphPx, { X: [0, 1] });
      expect(state.layout).toBe(artifact.layout);
      expect(state.pending).toBe(true);
      return null;
    }
    renderToStaticMarkup(createElement(Probe));
    expect(layout).not.toHaveBeenCalled();
  });

  it("retains deterministic layout for headless callers without Worker support", () => {
    const compiled = jobs.compileArtifact("X = Tensor(4, 32, 256)\nY = relu(X)\n");
    if (!compiled.ok) throw new Error(compiled.diagnostics[0].message);
    const { resolved, artifact } = compiled;
    vi.stubGlobal("Worker", undefined);
    const axes: Record<string, [number, number]> = { X: [0, 1] };
    const expected = jobs.layoutArtifact(resolved, artifact.graphPx, axes);
    function Probe() {
      const state = useGraphLayout(resolved, null, null, artifact.graphPx, axes);
      expect(state.layout).toEqual(expected);
      expect(state.pending).toBe(false);
      return null;
    }
    renderToStaticMarkup(createElement(Probe));
  });
});
