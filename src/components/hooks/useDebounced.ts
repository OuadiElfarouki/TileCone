import { useEffect, useRef } from "react";

/**
 * Run a call once the arguments stop changing, keeping the most recent ones.
 *
 * The counterpart to `useFrameThrottle`, for work that should not run at all
 * until a gesture is over rather than once per frame. Repainting a card at a
 * new zoom is the case it exists for: the canvas is transformed by CSS while
 * the view moves, so an intermediate raster is replaced before anyone reads
 * it, and only the scale the reader stops at is worth rasterising.
 *
 * The trailing call always runs, as it does for the frame throttle. Dropping it
 * would leave the canvas rendered for a scale the view has left.
 */

/** Scheduling half, with no React and no DOM, so it can be tested directly. */
export type Timers = {
  set: (cb: () => void, ms: number) => number;
  clear: (handle: number) => void;
};

export type Debounced<A extends unknown[]> = {
  call: (...args: A) => void;
  /** Run a pending call now. Nothing happens when none is pending. */
  flush: () => void;
  /** Drop any pending call. Called on unmount. */
  dispose: () => void;
};

/**
 * `timers` is null where there is no scheduler (tests, SSR), and the call then
 * runs through synchronously: debouncing changes *when* a caller's function
 * runs, never *whether* it does, so a headless caller observes what it always
 * did.
 */
export function createDebounce<A extends unknown[]>(
  run: (...args: A) => void,
  delayMs: number,
  timers: Timers | null
): Debounced<A> {
  let pending: A | null = null;
  let handle: number | null = null;

  const fire = () => {
    handle = null;
    const args = pending;
    pending = null;
    if (args) run(...args);
  };

  return {
    call: (...args: A) => {
      pending = args;
      if (!timers) {
        fire();
        return;
      }
      // Restart the wait rather than queueing: the point is the value the
      // gesture ends on, and every earlier one is superseded.
      if (handle !== null) timers.clear(handle);
      handle = timers.set(fire, delayMs);
    },
    flush: () => {
      if (handle !== null && timers) timers.clear(handle);
      if (pending) fire();
      handle = null;
    },
    dispose: () => {
      if (handle !== null && timers) timers.clear(handle);
      handle = null;
      pending = null;
    },
  };
}

const domTimers: Timers | null =
  typeof setTimeout === "function"
    ? { set: (cb, ms) => setTimeout(cb, ms) as unknown as number, clear: (h) => clearTimeout(h) }
    : null;

export function useDebounced<A extends unknown[]>(
  fn: (...args: A) => void,
  delayMs: number
): (...args: A) => void {
  // Keep the callback current without rebuilding the debounce, for the same
  // reason the frame throttle does: the caller re-creates its closure on every
  // render, and rescheduling there would defeat the point.
  const latest = useRef(fn);
  latest.current = fn;

  const debounced = useRef<Debounced<A> | null>(null);
  if (!debounced.current)
    debounced.current = createDebounce<A>((...args) => latest.current(...args), delayMs, domTimers);

  useEffect(() => () => debounced.current?.dispose(), []);
  return debounced.current.call;
}
