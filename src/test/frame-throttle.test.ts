/**
 * Frame coalescing for the hover preview.
 *
 * The property that matters is not "fewer calls" but "the same last call":
 * coalescing is only safe because the intermediate results were overwritten
 * before anything painted. A dropped trailing call, or a clear overtaken by a
 * pending move, would both be visible bugs.
 */

import { describe, expect, it } from "vitest";
import { createFrameThrottle, FrameScheduler } from "../ui/useFrameThrottle";

/** A frame queue held by hand, so a test decides when frames happen. */
function fakeFrames() {
  const queued: (() => void)[] = [];
  const cancelled: number[] = [];
  const scheduler: FrameScheduler = {
    request: (cb) => {
      queued.push(cb);
      return queued.length;
    },
    cancel: (h) => cancelled.push(h),
  };
  return {
    scheduler,
    cancelled,
    get requested() {
      return queued.length;
    },
    /** Run every frame requested so far. */
    tick() {
      const due = queued.splice(0, queued.length);
      for (const cb of due) cb();
    },
  };
}

describe("createFrameThrottle", () => {
  it("requests one frame for a burst and runs the most recent arguments", () => {
    const frames = fakeFrames();
    const calls: unknown[][] = [];
    const { call } = createFrameThrottle(
      (...args: unknown[]) => calls.push(args),
      frames.scheduler
    );

    call("a", 1);
    call("b", 2);
    call("c", 3);
    expect(calls).toEqual([]); // nothing runs before the frame
    expect(frames.requested).toBe(1); // and the burst asked for one frame

    frames.tick();
    expect(calls).toEqual([["c", 3]]); // the last position, not the first
  });

  it("always runs the trailing call rather than dropping it", () => {
    const frames = fakeFrames();
    const calls: unknown[][] = [];
    const { call } = createFrameThrottle((...a: unknown[]) => calls.push(a), frames.scheduler);

    call("first");
    frames.tick();
    call("last");
    frames.tick();
    // A leading-edge throttle would have swallowed "last" and left the preview
    // pointing at a position the pointer had already left.
    expect(calls).toEqual([["first"], ["last"]]);
  });

  it("lets a clear supersede a move already pending for the frame", () => {
    const frames = fakeFrames();
    const calls: unknown[][] = [];
    const { call } = createFrameThrottle((...a: unknown[]) => calls.push(a), frames.scheduler);

    call("tensorA", { lo: 0, hi: 4 }); // pointer move
    call(null); // pointer down: clear the preview
    frames.tick();
    // If the clear had bypassed the throttle, the pending move would have run
    // after it and put the preview back.
    expect(calls).toEqual([[null]]);
  });

  it("starts a new frame after one has run", () => {
    const frames = fakeFrames();
    const calls: unknown[][] = [];
    const { call } = createFrameThrottle((...a: unknown[]) => calls.push(a), frames.scheduler);

    call(1);
    frames.tick();
    call(2);
    expect(frames.requested).toBe(1); // a fresh request, not a reused handle
    frames.tick();
    expect(calls).toEqual([[1], [2]]);
  });

  it("drops a pending call on dispose and cancels its frame", () => {
    const frames = fakeFrames();
    const calls: unknown[][] = [];
    const { call, dispose } = createFrameThrottle(
      (...a: unknown[]) => calls.push(a),
      frames.scheduler
    );

    call("gone");
    dispose();
    frames.tick();
    expect(calls).toEqual([]);
    expect(frames.cancelled).toHaveLength(1);
  });

  it("runs synchronously where there are no animation frames", () => {
    const calls: unknown[][] = [];
    const { call } = createFrameThrottle((...a: unknown[]) => calls.push(a), null);
    call("x");
    call("y");
    // Headless callers observe exactly what they always did; only the timing
    // differs in a browser.
    expect(calls).toEqual([["x"], ["y"]]);
  });
});
