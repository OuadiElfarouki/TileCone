import { useEffect, useRef } from "react";

/**
 * Run at most one call per animation frame, keeping the most recent arguments.
 *
 * Pointer events arrive far faster than frames: a 1000Hz mouse produces about
 * sixteen `pointermove` events per painted frame, and the hover preview runs a
 * full dependency query on each one. Fifteen of those sixteen results are
 * overwritten before anything is drawn, so the work is not merely wasteful, it
 * is invisible - the intermediate states never reach a pixel.
 *
 * Coalescing is therefore free of any behavioural change: the last call in a
 * frame is the one whose result would have survived anyway. What it buys is the
 * main thread back, which is what keeps the interaction smooth.
 *
 * The trailing call always runs. Dropping it would leave the preview showing a
 * position the pointer has left, which is precisely the state a reader would
 * misread as the answer to where they are now.
 */

/** Scheduling half, with no React and no DOM, so it can be tested directly. */
export type FrameScheduler = {
  request: (cb: () => void) => number;
  cancel: (handle: number) => void;
};

export type FrameThrottle<A extends unknown[]> = {
  call: (...args: A) => void;
  /** Drop any pending call. Called on unmount. */
  dispose: () => void;
};

/**
 * `scheduler` is null where there are no animation frames (tests, SSR), and the
 * call then runs through synchronously: the throttle changes *when* a caller's
 * function runs, never *whether* it does, so a headless caller observes exactly
 * what it always did.
 */
export function createFrameThrottle<A extends unknown[]>(
  run: (...args: A) => void,
  scheduler: FrameScheduler | null
): FrameThrottle<A> {
  let pending: A | null = null;
  let handle: number | null = null;

  const flush = () => {
    handle = null;
    const args = pending;
    pending = null;
    if (args) run(...args);
  };

  return {
    call: (...args: A) => {
      // Overwriting rather than queueing is the whole mechanism: a later call
      // supersedes an earlier one, so a clear cannot be overtaken by a move
      // that was already pending for this frame.
      pending = args;
      if (!scheduler) {
        flush();
        return;
      }
      if (handle === null) handle = scheduler.request(flush);
    },
    dispose: () => {
      if (handle !== null && scheduler) scheduler.cancel(handle);
      handle = null;
      pending = null;
    },
  };
}

const domScheduler: FrameScheduler | null =
  typeof requestAnimationFrame === "function"
    ? { request: (cb) => requestAnimationFrame(cb), cancel: (h) => cancelAnimationFrame(h) }
    : null;

export function useFrameThrottle<A extends unknown[]>(
  fn: (...args: A) => void
): (...args: A) => void {
  // Keep the callback current without rebuilding the throttle: the caller
  // re-creates its closure on every render, and rescheduling there would defeat
  // the point.
  const latest = useRef(fn);
  latest.current = fn;

  const throttle = useRef<FrameThrottle<A> | null>(null);
  if (!throttle.current)
    throttle.current = createFrameThrottle<A>((...args) => latest.current(...args), domScheduler);

  useEffect(() => () => throttle.current?.dispose(), []);
  return throttle.current.call;
}
