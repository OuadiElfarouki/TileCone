import { describe, expect, it, vi } from "vitest";
import { cancelDeal, deal, holdPicture } from "../../../src/components/card/deal";

function picture() {
  const animations: { finished: Promise<unknown>; cancel: () => void; finish: () => void; reject: () => void }[] = [];
  const ctx = { setTransform: vi.fn(), clearRect: vi.fn(), drawImage: vi.fn() };
  const canvas = {
    width: 80, height: 60, style: { zIndex: "" },
    getContext: () => ctx,
    getAnimations: () => animations,
    animate: vi.fn(() => {
      let finish!: () => void;
      let reject!: () => void;
      const finished = new Promise<void>((resolve, fail) => { finish = resolve; reject = () => fail(new Error("cancelled")); });
      // Real Animation.finished is lazy; the fake eagerly constructs it.
      void finished.catch(() => {});
      const animation = { finished, finish, reject, cancel: vi.fn(reject) };
      animations.push(animation);
      return animation;
    }),
  };
  return { canvas: canvas as unknown as HTMLCanvasElement, ctx, animations };
}

describe("temporary slice rasters", () => {
  it.each([1, -1] as const)("releases the outgoing picture after a deal in direction %s finishes", async (way) => {
    const front = picture(), out = picture();
    expect(holdPicture(front.canvas, out.canvas)).toBe(true);
    const dispose = deal(front.canvas, out.canvas, way);
    out.animations[0].finish();
    await out.animations[0].finished;
    expect([out.canvas.width, out.canvas.height]).toEqual([0, 0]);
    expect([front.canvas.width, front.canvas.height]).toEqual([80, 60]);
    dispose(); // React cleanup remains safe after completion.
  });

  it("releases an externally cancelled deal", async () => {
    const front = picture(), out = picture();
    deal(front.canvas, out.canvas, 1);
    out.animations[0].reject();
    await out.animations[0].finished.catch(() => {});
    expect([out.canvas.width, out.canvas.height]).toEqual([0, 0]);
    expect(front.animations[0].cancel).toHaveBeenCalled();
  });

  it("does not let a cancelled deal release a newer scrub snapshot", async () => {
    const front = picture(), out = picture();
    const oldDispose = deal(front.canvas, out.canvas, 1);
    expect(holdPicture(front.canvas, out.canvas)).toBe(true);
    const dispose = deal(front.canvas, out.canvas, -1);
    await out.animations[0].finished.catch(() => {});
    oldDispose();
    expect([out.canvas.width, out.canvas.height]).toEqual([80, 60]);
    expect(out.ctx.drawImage).toHaveBeenCalledWith(front.canvas, 0, 0);
    dispose();
    expect([out.canvas.width, out.canvas.height]).toEqual([0, 0]);
  });

  it("cancels both animations and releases the outgoing raster when culled", () => {
    const front = picture(), out = picture();
    deal(front.canvas, out.canvas, 1);
    cancelDeal(front.canvas, out.canvas);
    expect(front.animations[0].cancel).toHaveBeenCalled();
    expect(out.animations[0].cancel).toHaveBeenCalled();
    expect([out.canvas.width, out.canvas.height]).toEqual([0, 0]);
  });

  it("releases a held picture when animation is unavailable", () => {
    const front = picture(), out = picture();
    Object.assign(front.canvas, { animate: undefined });
    deal(front.canvas, out.canvas, 1);
    expect([out.canvas.width, out.canvas.height]).toEqual([0, 0]);
  });
});
