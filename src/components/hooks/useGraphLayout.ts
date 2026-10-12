import { useEffect, useMemo, useState } from "react";
import type { ResolvedGraph } from "../../core/graph";
import type { BaseGraphLayout } from "../../view/graph/graph-scene";
import type { CardShapeView } from "../../view/tensor/card-size";
import { layoutArtifact } from "../../worker/analysis-jobs";
import { analysisWorkerAvailable, cancelLayout, isAnalysisCancelled, layoutInWorker } from "../../state/analysis-worker-client";

/** Commit only the current layout request. Keep the previous placement usable
 * while a remap runs, and isolate it from the plan/reuse worker queue. `views`
 * holds the cards whose footprint is not the default one. */
export function useGraphLayout(
  graph: ResolvedGraph | null,
  defaultLayout: BaseGraphLayout | null,
  graphId: number | null,
  graphPx: number,
  views: Record<string, CardShapeView>,
) {
  const needsLayout = !!graph && (!defaultLayout || Object.keys(views).length > 0);
  const workerEnabled = analysisWorkerAvailable();
  const synchronous = useMemo(
    () => graph && needsLayout && !workerEnabled ? layoutArtifact(graph, graphPx, views) : null,
    [graph, needsLayout, workerEnabled, graphPx, views],
  );
  type Request = { graph: ResolvedGraph; views: typeof views; graphPx: number };
  const [completed, setCompleted] = useState<(Request & { layout: BaseGraphLayout }) | null>(null);
  const [failed, setFailed] = useState<(Request & { message: string }) | null>(null);
  const [attempt, setAttempt] = useState(0);
  const matches = (request: Request | null) => request?.graph === graph && request.views === views && request.graphPx === graphPx;

  useEffect(() => {
    if (!graph || !needsLayout || !workerEnabled) return;
    let live = true;
    const request = { graph, views, graphPx };
    setFailed(null);
    void layoutInWorker({ ...request, graphId }).then((layout) => {
      if (live) setCompleted({ ...request, layout });
    }).catch((error: unknown) => {
      if (!live) return;
      if (isAnalysisCancelled(error)) setAttempt((n) => n + 1);
      else setFailed({ ...request, message: error instanceof Error ? error.message : String(error) });
    });
    return () => { live = false; cancelLayout(); };
  }, [graph, needsLayout, workerEnabled, graphId, graphPx, views, attempt]);

  return {
    layout: synchronous ?? (needsLayout && completed?.graph === graph ? completed.layout : defaultLayout),
    pending: workerEnabled && needsLayout && !matches(completed) && !matches(failed),
    error: matches(failed) ? failed!.message : null,
  };
}
