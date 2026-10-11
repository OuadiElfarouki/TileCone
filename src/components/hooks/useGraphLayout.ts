import { useEffect, useMemo, useState } from "react";
import type { ResolvedGraph } from "../../core/graph";
import type { BaseGraphLayout } from "../../view/graph/graph-scene";
import { layoutArtifact } from "../../worker/analysis-jobs";
import { analysisWorkerAvailable, cancelLayout, isAnalysisCancelled, layoutInWorker } from "../../state/analysis-worker-client";

/** Commit only the current layout request. Keep the previous placement usable
 * while a remap runs, and isolate it from the plan/reuse worker queue. */
export function useGraphLayout(
  graph: ResolvedGraph | null,
  defaultLayout: BaseGraphLayout | null,
  graphId: number | null,
  graphPx: number,
  axes: Record<string, [number, number]>,
) {
  const needsLayout = !!graph && (!defaultLayout || Object.keys(axes).length > 0);
  const workerEnabled = analysisWorkerAvailable();
  const synchronous = useMemo(
    () => graph && needsLayout && !workerEnabled ? layoutArtifact(graph, graphPx, axes) : null,
    [graph, needsLayout, workerEnabled, graphPx, axes],
  );
  type Request = { graph: ResolvedGraph; axes: typeof axes; graphPx: number };
  const [completed, setCompleted] = useState<(Request & { layout: BaseGraphLayout }) | null>(null);
  const [failed, setFailed] = useState<(Request & { message: string }) | null>(null);
  const [attempt, setAttempt] = useState(0);
  const matches = (request: Request | null) => request?.graph === graph && request.axes === axes && request.graphPx === graphPx;

  useEffect(() => {
    if (!graph || !needsLayout || !workerEnabled) return;
    let live = true;
    const request = { graph, axes, graphPx };
    setFailed(null);
    void layoutInWorker({ ...request, graphId }).then((layout) => {
      if (live) setCompleted({ ...request, layout });
    }).catch((error: unknown) => {
      if (!live) return;
      if (isAnalysisCancelled(error)) setAttempt((n) => n + 1);
      else setFailed({ ...request, message: error instanceof Error ? error.message : String(error) });
    });
    return () => { live = false; cancelLayout(); };
  }, [graph, needsLayout, workerEnabled, graphId, graphPx, axes, attempt]);

  return {
    layout: synchronous ?? (needsLayout && completed?.graph === graph ? completed.layout : defaultLayout),
    pending: workerEnabled && needsLayout && !matches(completed) && !matches(failed),
    error: matches(failed) ? failed!.message : null,
  };
}
