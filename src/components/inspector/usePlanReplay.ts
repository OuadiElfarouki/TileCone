import { useCallback, useEffect, useRef, useState } from "react";
import type { TaskRef, TilePlan } from "../../core/plan/plan";
import { familyReplay, replayDepthLimit, type FamilyReplay } from "../../core/plan/replay";
import type { InspectorTab, ReplayPlayback } from "../../view/workspace";
import { analysisWorkerAvailable, isAnalysisCancelled, replayInWorker } from "../../state/analysis-worker-client";
import { sameTask } from "../../state/history";
import { useStore } from "../../state/store";

/** The tiling a plan was checked with, as the Worker request carries it. */
const tilesOf = (plan: TilePlan): Record<string, number[]> =>
  Object.fromEntries([...plan.families].map(([tensorId, family]) => [tensorId, [...family.tile]]));

/** How long a whole replay takes, and the bounds on one run's turn within it. */
const REPLAY_MS = 4200;
const MIN_STEP_MS = 84;
const MAX_STEP_MS = 220;
/** How long a departure takes to fade the overlay, and the end of a replay. */
const DEPART_FADE_MS = 240;
const END_FADE_MS = 400;
/** How long the whole trail is held, with reduced motion, before it clears. */
const REDUCED_HOLD_MS = 1500;

/** Whether a playback is about this plan and this task, at this depth. */
const about = (playback: ReplayPlayback | null, plan: TilePlan | null, task: TaskRef | null, depth: number) =>
  !!playback && playback.plan === plan && sameTask(playback.replay.task, task) && playback.replay.depth === depth;

export type PlanReplayControl = {
  /** Replay the inspected task's family, computing the frames when needed. */
  run: () => void;
  /** The depth a replay of this family follows, and the most it can. */
  depth: number;
  depthLimit: number;
  pending: boolean;
  error: string | null;
  /** The playback, while it is about the plan and task on screen. */
  playback: ReplayPlayback | null;
};

/**
 * The family replay behind the Plan view: the Worker request, and the
 * playback that walks its frames on the cards.
 *
 * The playback's timers live here, so the hook also owns retiring them: at the
 * end of the replay, on leaving the Plan view, when the inspected task or the
 * plan changes, and when the panel unmounts. It lives in the inspector rather than the Plan view so
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
  const depthLimit = plan && task ? replayDepthLimit(plan, task.tensorId) : 1;
  const chosen = useStore((s) => (task ? s.replayDepth[task.tensorId] : undefined) ?? 1);
  const depth = Math.min(chosen, depthLimit);
  const setPlayback = useStore((s) => s.setReplayPlayback);
  const updatePlayback = useStore((s) => s.updateReplayPlayback);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const playbackTimer = useRef<number | null>(null);
  const fadeTimer = useRef<number | null>(null);
  const request = useRef(0);
  const current = useRef({ plan, task, tab, depth });
  current.current = { plan, task, tab, depth };

  const stopTimers = useCallback(() => {
    if (playbackTimer.current !== null) window.clearInterval(playbackTimer.current);
    if (fadeTimer.current !== null) window.clearInterval(fadeTimer.current);
    playbackTimer.current = null;
    fadeTimer.current = null;
  }, []);

  const fade = useCallback((duration = DEPART_FADE_MS) => {
    const now = useStore.getState().replayPlayback;
    if (!now || now.exiting) return;
    stopTimers();
    const initial = now.opacity;
    const started = performance.now();
    updatePlayback({ exiting: true });
    fadeTimer.current = window.setInterval(() => {
      const progress = Math.min(1, (performance.now() - started) / duration);
      updatePlayback({ opacity: initial * (1 - progress) });
      if (progress >= 1) {
        window.clearInterval(fadeTimer.current!);
        fadeTimer.current = null;
        setPlayback(null);
      }
    }, 30);
  }, [setPlayback, stopTimers, updatePlayback]);

  /* One run per step, the trail growing behind it; once the last run has had
     its turn, the whole trail clears together. Nothing stays behind on the
     cards. With reduced motion the whole trail is shown at once and held
     rather than walked. */
  const start = useCallback((replay: FamilyReplay, forPlan: TilePlan) => {
    stopTimers();
    const frames = replay.frames.length;
    const reduced = typeof matchMedia === "function" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches;
    let visited = reduced ? frames : 1;
    setPlayback({ replay, plan: forPlan, visited, exiting: false, opacity: 1 });
    const step = reduced
      ? REDUCED_HOLD_MS
      : Math.max(MIN_STEP_MS, Math.min(MAX_STEP_MS, Math.round(REPLAY_MS / frames)));
    playbackTimer.current = window.setInterval(() => {
      if (visited < frames) updatePlayback({ visited: ++visited });
      else fade(END_FADE_MS);
    }, step);
  }, [fade, setPlayback, stopTimers, updatePlayback]);

  useEffect(() => {
    if (tab !== "plan") fade();
  }, [fade, tab]);

  // A replay is about one task under one plan at one depth; another of any of
  // them retires it.
  useEffect(() => {
    if (playback && !about(playback, plan, task, depth)) fade();
  }, [depth, fade, playback, plan, task]);

  useEffect(() => {
    // Departing also invalidates work in flight: returning to the same task
    // must not revive a request made before the departure.
    request.current++;
    setPending(false);
    setError(null);
  }, [plan, task, tab, depth]);

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

  const ask = (forPlan: TilePlan, forTask: TaskRef, forDepth: number, id: number, retried: boolean) => {
    const stale = () =>
      id !== request.current ||
      current.current.tab !== "plan" ||
      current.current.plan !== forPlan ||
      current.current.depth !== forDepth ||
      !sameTask(current.current.task, forTask);
    const work = analysisWorkerAvailable()
      ? replayInWorker({
          graphId: useStore.getState().workerGraphId,
          graph: forPlan.graph,
          tiles: tilesOf(forPlan),
          task: forTask,
          depth: forDepth,
        })
      : Promise.resolve().then(() => familyReplay(forPlan, forTask, { depth: forDepth }));
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
        if (!retried && !stale()) return ask(forPlan, forTask, forDepth, id, true);
        setPending(false);
        return;
      }
      setPending(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  };

  const run = () => {
    if (tab !== "plan" || !plan || !task) return;
    if (playback && about(playback, plan, task, depth)) {
      start(playback.replay, plan);
      return;
    }
    setPending(true);
    setError(null);
    ask(plan, task, depth, ++request.current, false);
  };

  return {
    run,
    depth,
    depthLimit,
    pending,
    error,
    playback: tab === "plan" && about(playback, plan, task, depth) ? playback : null,
  };
}
