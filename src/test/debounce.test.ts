/**
 * Settling for card repaints.
 *
 * A card rasterises for one scale and is shown at another while the view
 * moves, so the property that matters is that the raster ends up at the scale
 * the reader stopped on. A dropped trailing call would leave every card drawn
 * for a zoom level nobody is looking at, which is worse than the repaints this
 * exists to avoid.
 */

import { describe, expect, it } from "vitest";
import { createDebounce, Timers } from "../ui/useDebounced";

/** A timer queue held by hand, so a test decides when the wait elapses. */
function fakeTimers() {
  const queued = new Map<number, () => void>();
  let next = 1;
  const timers: Timers = {
    set: (cb) => {
      const handle = next++;
      queued.set(handle, cb);
      return handle;
    },
    clear: (handle) => {
      queued.delete(handle);
    },
  };
  return {
    timers,
    get pending() {
      return queued.size;
    },
    /** Run every timer still waiting. */
    elapse() {
      const due = [...queued.values()];
      queued.clear();
      for (const cb of due) cb();
    },
  };
}

describe("createDebounce", () => {
  it("runs once for a burst, with the last arguments", () => {
    const clock = fakeTimers();
    const calls: unknown[][] = [];
    const { call } = createDebounce((...a: unknown[]) => calls.push(a), 140, clock.timers);

    call(1.0);
    call(1.4);
    call(2.0);
    expect(calls).toEqual([]); // nothing paints mid-gesture
    expect(clock.pending).toBe(1); // and the burst holds one timer, not three

    clock.elapse();
    expect(calls).toEqual([[2.0]]); // the scale the gesture ended on
  });

  it("restarts the wait while the value keeps changing", () => {
    const clock = fakeTimers();
    const calls: unknown[][] = [];
    const { call } = createDebounce((...a: unknown[]) => calls.push(a), 140, clock.timers);

    call("a");
    call("b");
    // The first timer was cleared rather than left to fire: a continuous wheel
    // gesture must not repaint part-way through simply because it started.
    expect(clock.pending).toBe(1);
    clock.elapse();
    expect(calls).toEqual([["b"]]);
  });

  it("always runs the trailing call", () => {
    const clock = fakeTimers();
    const calls: unknown[][] = [];
    const { call } = createDebounce((...a: unknown[]) => calls.push(a), 140, clock.timers);

    call("first");
    clock.elapse();
    call("last");
    clock.elapse();
    expect(calls).toEqual([["first"], ["last"]]);
  });

  it("flushes a pending call on demand", () => {
    const clock = fakeTimers();
    const calls: unknown[][] = [];
    const { call, flush } = createDebounce((...a: unknown[]) => calls.push(a), 140, clock.timers);

    call("pending");
    flush();
    expect(calls).toEqual([["pending"]]);
    // And the timer it was waiting on is gone, so it cannot run twice.
    clock.elapse();
    expect(calls).toEqual([["pending"]]);
  });

  it("flushes nothing when nothing is pending", () => {
    const clock = fakeTimers();
    const calls: unknown[][] = [];
    const { flush } = createDebounce((...a: unknown[]) => calls.push(a), 140, clock.timers);
    flush();
    expect(calls).toEqual([]);
  });

  it("drops a pending call on dispose", () => {
    const clock = fakeTimers();
    const calls: unknown[][] = [];
    const { call, dispose } = createDebounce((...a: unknown[]) => calls.push(a), 140, clock.timers);

    call("gone");
    dispose();
    clock.elapse();
    expect(calls).toEqual([]);
  });

  it("runs synchronously where there are no timers", () => {
    const calls: unknown[][] = [];
    const { call } = createDebounce((...a: unknown[]) => calls.push(a), 140, null);
    call("x");
    call("y");
    // Headless callers observe exactly what they always did; only the timing
    // differs in a browser.
    expect(calls).toEqual([["x"], ["y"]]);
  });
});
