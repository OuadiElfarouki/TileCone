import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ResolvedGraph } from "../../core/graph";
import { fromBox } from "../../core/region";
import { estimateInputReuseSweep, ReuseSurface, ReuseSweep } from "../../core/reuse";
import { sameBox } from "../../core/region";
import { currentReuseRows, ReuseProbe, ReuseRun, sameSurfaces } from "../../view/reuse-rows";
import { Direction, InspectorTab, type Selection, sweepAnchorIndex } from "../../view/workspace";
import { analysisWorkerAvailable, isAnalysisCancelled, reuseInWorker } from "../../state/analysis-worker-client";
import { useStore } from "../../state/store";

/**
 * The reuse sweep behind the Cost model view: the tile it is anchored on, the
 * Worker request, the cached run, and the playback that replays its probes on
 * the cards.
 *
 * The playback's timers live here, with the panel that shows the figures, so
 * the hook also owns retiring them: on leaving Cost model, when the anchor tile
 * changes, and when the panel unmounts.
 */
export function useReuseSweep({
  resolved,
  selection,
  selectedGroup,
  rawFocus,
  hiddenBoxes,
  direction,
  showEntangled,
  tab,
}: {
  resolved: ResolvedGraph | null;
  selection: Selection;
  selectedGroup: string | null;
  rawFocus: number | null;
  hiddenBoxes: Set<number>;
  direction: Direction;
  showEntangled: boolean;
  tab: InspectorTab;
}) {
  /** The one enabled tile that defines a sweep. Keeping this derivation beside
   *  the cache key prevents its button and its displayed result from choosing
   *  subtly different fallbacks. */
  const reuseProbe = useMemo<ReuseProbe | null>(() => {
    if (!selection) return null;
    const index = sweepAnchorIndex(selection.parts, selectedGroup, rawFocus, hiddenBoxes);
    if (index === null) return null;
    const probe = selection.parts[index];
    return probe ? { tensorId: probe.tensorId, box: probe.box, colorIndex: index } : null;
  }, [selection, selectedGroup, rawFocus, hiddenBoxes]);
  /** The relations the sweep should trace: whichever the Dependencies view has
   *  switched on. A probe is painted in each one's own mark, so a relation the
   *  reader has hidden there must not reappear here. */
  const reuseSurfaces = useMemo<ReuseSurface[]>(() => {
    const list: ReuseSurface[] = [];
    if (direction === "backward" || direction === "both") list.push("backward");
    if (direction === "forward" || direction === "both") list.push("forward");
    if (showEntangled) list.push("entangled");
    return list;
  }, [direction, showEntangled]);
  const [reuseRun, setReuseRun] = useState<ReuseRun | null>(null);
  const [reusePending, setReusePending] = useState(false);
  const [reuseError, setReuseError] = useState<string | null>(null);
  const executionPlayback = useStore((s) => s.executionPlayback);
  const setExecutionPlayback = useStore((s) => s.setExecutionPlayback);
  const updateExecutionPlayback = useStore((s) => s.updateExecutionPlayback);
  const playbackTimer = useRef<number | null>(null);
  const fadeTimer = useRef<number | null>(null);
  const reuseRequest = useRef(0);
  const reuseProbeRef = useRef(reuseProbe);
  reuseProbeRef.current = reuseProbe;

  const stopPlaybackTimers = useCallback(() => {
    if (playbackTimer.current !== null) window.clearInterval(playbackTimer.current);
    if (fadeTimer.current !== null) window.clearInterval(fadeTimer.current);
    playbackTimer.current = null;
    fadeTimer.current = null;
  }, []);

  const startPlayback = useCallback((sweep: ReuseSweep, probe: ReuseProbe) => {
    stopPlaybackTimers();
    const tile = probe.box.map((interval) => interval.hi - interval.lo);
    const reduced = typeof matchMedia === "function" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches;
    setExecutionPlayback({
      tensorId: probe.tensorId,
      anchorBox: probe.box,
      tile,
      colorIndex: probe.colorIndex,
      frames: sweep.frames,
      visited: reduced ? sweep.frames.length : Math.min(1, sweep.frames.length),
      phase: reduced ? "settled" : "playing",
      exiting: false,
      opacity: reduced ? 0.72 : 1,
    });
    if (reduced || sweep.frames.length <= 1) {
      updateExecutionPlayback({
        visited: sweep.frames.length,
        phase: "settled",
        opacity: 0.72,
      });
      return;
    }
    let visited = 1;
    const delay = Math.max(42, Math.min(110, Math.round(2100 / sweep.frames.length)));
    playbackTimer.current = window.setInterval(() => {
      visited++;
      if (visited > sweep.frames.length) {
        window.clearInterval(playbackTimer.current!);
        playbackTimer.current = null;
        updateExecutionPlayback({
          visited: sweep.frames.length,
          phase: "settled",
          opacity: 0.72,
        });
      } else {
        updateExecutionPlayback({ visited });
      }
    }, delay);
  }, [setExecutionPlayback, stopPlaybackTimers, updateExecutionPlayback]);

  const fadePlayback = useCallback(() => {
    const current = useStore.getState().executionPlayback;
    if (!current || current.exiting) return;
    stopPlaybackTimers();
    const initial = current.opacity;
    const started = performance.now();
    updateExecutionPlayback({ exiting: true });
    fadeTimer.current = window.setInterval(() => {
      const progress = Math.min(1, (performance.now() - started) / 240);
      updateExecutionPlayback({ opacity: initial * (1 - progress) });
      if (progress >= 1) {
        window.clearInterval(fadeTimer.current!);
        fadeTimer.current = null;
        setExecutionPlayback(null);
      }
    }, 30);
  }, [setExecutionPlayback, stopPlaybackTimers, updateExecutionPlayback]);

  /** Coming back interrupts a departure. Settle rather than resume from where
   *  the fade froze it: the animation explains the estimate once, and picking
   *  a half-watched sweep back up mid-stride explains nothing. `replay` runs
   *  the whole thing again for a reader who wants it. */
  const cancelFade = useCallback(() => {
    const current = useStore.getState().executionPlayback;
    if (!current?.exiting) return;
    stopPlaybackTimers();
    updateExecutionPlayback({
      exiting: false,
      phase: "settled",
      visited: current.frames.length,
      opacity: 0.72,
    });
  }, [stopPlaybackTimers, updateExecutionPlayback]);

  useEffect(() => {
    if (tab === "execution") cancelFade();
    else fadePlayback();
  }, [cancelFade, fadePlayback, tab]);

  useEffect(() => {
    if (!executionPlayback) return;
    if (
      !reuseProbe ||
      executionPlayback.tensorId !== reuseProbe.tensorId ||
      !sameBox(executionPlayback.anchorBox, reuseProbe.box)
    ) fadePlayback();
  }, [executionPlayback, fadePlayback, reuseProbe]);

  useEffect(() => {
    reuseRequest.current++;
    setReusePending(false);
    setReuseError(null);
  }, [reuseProbe]);

  /* The timers that advance and retire a playback live here, so an unmounted
     panel - collapse-to-rail unmounts it - would leave the overlay frozen on
     every card with nothing left able to clear it. The state goes with them,
     and goes at once rather than fading: a fade explains a departure, and
     there is no longer a panel on screen to have departed from. */
  useEffect(() => () => {
    // Invalidate work still in flight before retiring its timers. Otherwise a
    // late result can start playback again after this owner has unmounted.
    reuseRequest.current++;
    stopPlaybackTimers();
    setExecutionPlayback(null);
  }, [setExecutionPlayback, stopPlaybackTimers]);

  /** Reuse factor (§5.5): sample selection-sized output tiles across the selected
   * tensor; count how many touch the current footprint on each input. The sweep
   * is defined by one tile on one tensor, so it follows the anchor part (the
   * focused one) else the last drawn rather than mixing tensors. */
  const runReuse = (
    probe: ReuseProbe,
    graph: ResolvedGraph,
    surfaces: ReuseSurface[],
    request: number,
    retried: boolean
  ) => {
    /** Whether the answer would still be about the tile and the graph it was
     *  asked about. Read fresh: the sweep outlives the click that started it. */
    const stale = () => {
      const current = reuseProbeRef.current;
      return request !== reuseRequest.current ||
        useStore.getState().resolved !== graph ||
        !current ||
        current.tensorId !== probe.tensorId ||
        !sameBox(current.box, probe.box);
    };
    const work = analysisWorkerAvailable()
      ? reuseInWorker({
          graphId: useStore.getState().workerGraphId,
          graph,
          tensorId: probe.tensorId,
          box: probe.box,
          surfaces,
        })
      : Promise.resolve(estimateInputReuseSweep(
          graph,
          { tensorId: probe.tensorId, region: fromBox(probe.box) },
          { surfaces }
        ));
    void work.then((sweep) => {
      if (stale()) return;
      setReusePending(false);
      setReuseRun({ graph, probe, surfaces, rows: sweep.estimates, sweep });
      if (useStore.getState().inspectorTab === "execution") startPlayback(sweep, probe);
    }).catch((error) => {
      if (request !== reuseRequest.current) return;
      /* A cancellation discarded the work, not the question, so re-ask it
         while the tile and the graph are still the ones it was about. Once:
         a second cancellation is something contending for the lane rather
         than the one build that takes it, and a silent button beats a loop. */
      if (isAnalysisCancelled(error)) {
        if (!retried && !stale()) return runReuse(probe, graph, surfaces, request, true);
        setReusePending(false);
        return;
      }
      setReusePending(false);
      setReuseError(error instanceof Error ? error.message : String(error));
    });
  };

  const computeReuse = () => {
    if (!reuseProbe || !resolved) return;
    const cached = reuseRun && currentReuseRows(reuseRun, resolved, reuseProbe) &&
      sameSurfaces(reuseRun.surfaces, reuseSurfaces)
      ? reuseRun.sweep
      : null;
    if (cached) {
      startPlayback(cached, reuseProbe);
      return;
    }
    setReusePending(true);
    setReuseError(null);
    runReuse(reuseProbe, resolved, reuseSurfaces, ++reuseRequest.current, false);
  };

  const reuse = currentReuseRows(reuseRun, resolved, reuseProbe);
  const visiblePlayback = tab === "execution" && executionPlayback && reuseProbe &&
    executionPlayback.tensorId === reuseProbe.tensorId &&
    sameBox(executionPlayback.anchorBox, reuseProbe.box)
      ? executionPlayback
      : null;

  return { reuseProbe, reuse, reusePending, reuseError, visiblePlayback, computeReuse };
}
