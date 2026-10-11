import { useCallback, useEffect, useRef, useState } from "react";
import type { TaskRef, TilePlan } from "../../core/plan/plan";
import { familyReplay, type FamilyReplay } from "../../core/plan/replay";
import type { InspectorTab, ReplayPlayback } from "../../view/workspace";
import { analysisWorkerAvailable, isAnalysisCancelled, replayInWorker } from "../../state/analysis-worker-client";
import { sameTask } from "../../state/history";
import { useStore } from "../../state/store";

/** The tiling a plan was checked with, as the Worker request carries it. */
const tilesOf = (plan: TilePlan): Record<string, number[]> =>
  Object.fromEntries([...plan.families].map(([tensorId, family]) => [tensorId, [...family.tile]]));

/** Whether a playback is about this plan and this task. */
const about = (playback: ReplayPlayback | null, plan: TilePlan | null, task: TaskRef | null) =>
  !!playback && playback.plan === plan && sameTask(playback.replay.task, task);

export type PlanReplayControl = {
  /** Replay the inspected task's family, computing the frames when needed. */
  run: () => void;
  pending: boolean;
  error: string | null;
  /** The playback, while it is about the plan and task on screen. */
  playback: ReplayPlayback | null;
};

/**
 * The family replay behind the Plan view: the Worker request, and the
 * playback that walks its frames on the cards.
 *
 * The playback's timers live here, so the hook also owns retiring them: on
 * leaving the Plan view, when the inspected task or the plan changes, and when
 * the panel unmounts. It lives in the inspector rather than the Plan view so
 * that leaving the view fades the overlay rather than cutting it.
 */
export function usePlanReplay({
  plan,
  task,
  tab,
}: {
  plan: TilePlan | null;
  task: TaskRef | null;
  tab: InspectorTab;
}): PlanReplayControl {
  const playback = useStore((s) => s.replayPlayback);
  const setPlayback = useStore((s) => s.setReplayPlayback);
  const updatePlayback = useStore((s) => s.updateReplayPlayback);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const playbackTimer = useRef<number | null>(null);
  const fadeTimer = useRef<number | null>(null);
  const request = useRef(0);
  const current = useRef({ plan, task, tab });
  current.current = { plan, task, tab };

  const stopTimers = useCallback(() => {
    if (playbackTimer.current !== null) window.clearInterval(playbackTimer.current);
    if (fadeTimer.current !== null) window.clearInterval(fadeTimer.current);
    playbackTimer.current = null;
    fadeTimer.current = null;
  }, []);

  const start = useCallback((replay: FamilyReplay, forPlan: TilePlan) => {
    stopTimers();
    const frames = replay.frames.length;
    const reduced = typeof matchMedia === "function" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced || frames <= 1) {
      setPlayback({ replay, plan: forPlan, visited: frames, phase: "settled", exiting: false, opacity: 0.72 });
      return;
    }
    setPlayback({ replay, plan: forPlan, visited: 1, phase: "playing", exiting: false, opacity: 1 });
    let visited = 1;
    const delay = Math.max(42, Math.min(110, Math.round(2100 / frames)));
    playbackTimer.current = window.setInterval(() => {
      visited++;
      if (visited > frames) {
        window.clearInterval(playbackTimer.current!);
        playbackTimer.current = null;
        updatePlayback({ visited: frames, phase: "settled", opacity: 0.72 });
      } else updatePlayback({ visited });
    }, delay);
  }, [setPlayback, stopTimers, updatePlayback]);

  const fade = useCallback(() => {
    const now = useStore.getState().replayPlayback;
    if (!now || now.exiting) return;
    stopTimers();
    const initial = now.opacity;
    const started = performance.now();
    updatePlayback({ exiting: true });
    fadeTimer.current = window.setInterval(() => {
      const progress = Math.min(1, (performance.now() - started) / 240);
      updatePlayback({ opacity: initial * (1 - progress) });
      if (progress >= 1) {
        window.clearInterval(fadeTimer.current!);
        fadeTimer.current = null;
        setPlayback(null);
      }
    }, 30);
  }, [setPlayback, stopTimers, updatePlayback]);

  /** Coming back interrupts a departure. Settle rather than resume from where
   *  the fade froze it; `run` replays the whole family for a reader who wants it. */
  const cancelFade = useCallback(() => {
    const now = useStore.getState().replayPlayback;
    if (!now?.exiting) return;
    stopTimers();
    updatePlayback({ exiting: false, phase: "settled", visited: now.replay.frames.length, opacity: 0.72 });
  }, [stopTimers, updatePlayback]);

  useEffect(() => {
    if (tab === "plan") cancelFade();
    else fade();
  }, [cancelFade, fade, tab]);

  // A replay is about one task under one plan; another of either retires it.
  useEffect(() => {
    if (playback && !about(playback, plan, task)) fade();
  }, [fade, playback, plan, task]);

  useEffect(() => {
    // Departing also invalidates work in flight: returning to the same task
    // must not revive a request made before the departure.
    request.current++;
    setPending(false);
    setError(null);
  }, [plan, task, tab]);

  /* Collapsing the panel unmounts this hook, which would leave the overlay
     frozen on every card with nothing left able to clear it. The state goes
     with the timers, and at once: there is no longer a panel on screen to
     have departed from. Work still in flight is invalidated first, so a late
     answer cannot start a playback nothing owns. */
  useEffect(() => () => {
    request.current++;
    stopTimers();
    setPlayback(null);
  }, [setPlayback, stopTimers]);

  const ask = (forPlan: TilePlan, forTask: TaskRef, id: number, retried: boolean) => {
    const stale = () =>
      id !== request.current ||
      current.current.tab !== "plan" ||
      current.current.plan !== forPlan ||
      !sameTask(current.current.task, forTask);
    const work = analysisWorkerAvailable()
      ? replayInWorker({
          graphId: useStore.getState().workerGraphId,
          graph: forPlan.graph,
          tiles: tilesOf(forPlan),
          task: forTask,
        })
      : Promise.resolve().then(() => familyReplay(forPlan, forTask));
    void work.then((replay) => {
      if (stale()) return;
      setPending(false);
      start(replay, forPlan);
    }).catch((reason: unknown) => {
      if (stale()) return;
      /* The family evaluation shares the query lane, and either one cancels
         the other. Re-ask once while the question still stands; a second
         cancellation is that evaluation holding the lane, and giving way to
         it beats a loop of each cancelling the other. */
      if (isAnalysisCancelled(reason)) {
        if (!retried && !stale()) return ask(forPlan, forTask, id, true);
        setPending(false);
        return;
      }
      setPending(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  };

  const run = () => {
    if (tab !== "plan" || !plan || !task) return;
    if (playback && about(playback, plan, task)) {
      start(playback.replay, plan);
      return;
    }
    setPending(true);
    setError(null);
    ask(plan, task, ++request.current, false);
  };

  return {
    run,
    pending,
    error,
    playback: tab === "plan" && about(playback, plan, task) ? playback : null,
  };
}
