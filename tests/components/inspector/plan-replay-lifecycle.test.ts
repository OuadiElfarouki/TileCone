import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FamilyReplay } from "../../../src/core/plan/replay";
import type { InspectorTab } from "../../../src/view/workspace";

// Drive render and effect commits separately to exercise worker responses that
// arrive between a tab render and its passive effects, without a DOM renderer.
const hooks = vi.hoisted(() => {
  type Slot = { value?: unknown; deps?: readonly unknown[]; cleanup?: () => void };
  let cursor = 0;
  const slots: Slot[] = [];
  let effects: (() => void)[] = [];
  const equal = (a: readonly unknown[] | undefined, b: readonly unknown[]) =>
    a?.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  return {
    begin: () => { cursor = 0; },
    commit: () => { const queued = effects; effects = []; queued.forEach((effect) => effect()); },
    unmount: () => { slots.forEach((slot) => slot.cleanup?.()); slots.length = 0; effects = []; },
    useRef: <T,>(value: T) => {
      const slot = slots[cursor++] ??= { value: { current: value } };
      return slot.value as { current: T };
    },
    useState: <T,>(value: T): [T, (value: T | ((old: T) => T)) => void] => {
      const slot = slots[cursor++] ??= { value };
      return [slot.value as T, (next) => {
        slot.value = typeof next === "function" ? (next as (old: T) => T)(slot.value as T) : next;
      }];
    },
    useCallback: <T,>(fn: T, deps: readonly unknown[]): T => {
      const i = cursor++;
      if (!equal(slots[i]?.deps, deps)) slots[i] = { deps, value: fn };
      return slots[i].value as T;
    },
    useEffect: (fn: () => void | (() => void), deps: readonly unknown[]) => {
      const i = cursor++;
      if (!equal(slots[i]?.deps, deps)) {
        const old = slots[i];
        slots[i] = { deps };
        effects.push(() => { old?.cleanup?.(); slots[i].cleanup = fn() || undefined; });
      }
    },
  };
});
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useRef: hooks.useRef, useState: hooks.useState,
  useCallback: hooks.useCallback, useEffect: hooks.useEffect,
}));
vi.mock("../../../src/state/store", async (original) => {
  const actual = await original<typeof import("../../../src/state/store")>();
  return { ...actual, useStore: Object.assign(
    (select: (state: ReturnType<typeof actual.useStore.getState>) => unknown) => select(actual.useStore.getState()),
    actual.useStore,
  ) };
});
vi.mock("../../../src/state/analysis-worker-client", () => ({
  analysisWorkerAvailable: () => true, isAnalysisCancelled: () => false,
  replayInWorker: vi.fn(),
}));
import { useStore } from "../../../src/state/store";
import { replayInWorker } from "../../../src/state/analysis-worker-client";
import { familyReplay } from "../../../src/core/plan/replay";
import { usePlanReplay } from "../../../src/components/inspector/usePlanReplay";

/** Interval callbacks by id, so a test can step the playback and its fade. */
const ticks = new Map<number, () => void>();
const interval = vi.fn((tick: () => void) => {
  ticks.set(ticks.size + 1, tick);
  return ticks.size;
});
const clear = vi.fn((id: number) => { ticks.delete(id); });
const render = (tab: InspectorTab, commit = true) => {
  useStore.getState().setInspectorTab(tab);
  hooks.begin();
  const { plan, planTask: task } = useStore.getState();
  const control = usePlanReplay({ plan, task, tab });
  if (commit) hooks.commit();
  return control;
};
const response = () => familyReplay(useStore.getState().plan!, useStore.getState().planTask!);
const deferred = () => {
  let resolve!: (result: FamilyReplay) => void;
  const promise = new Promise<FamilyReplay>((done) => { resolve = done; });
  vi.mocked(replayInWorker).mockReturnValueOnce(promise);
  return resolve;
};
const flushResponse = async () => { await Promise.resolve(); await Promise.resolve(); };

beforeEach(() => {
  vi.clearAllMocks();
  ticks.clear();
  vi.stubGlobal("window", { setInterval: interval, clearInterval: clear });
  useStore.getState().applyDSL("X = Tensor(8)\nY = relu(X)\n");
  useStore.getState().setPlanTileAt("Y", [2], [0]);
});
afterEach(() => { hooks.unmount(); vi.unstubAllGlobals(); });

describe("replay responses across tab changes", () => {
  it("starts a requested replay while Plan remains visible", async () => {
    const resolve = deferred();
    render("plan").run();
    expect(render("plan").pending).toBe(true);
    resolve(response());
    await flushResponse();
    expect(render("plan").pending).toBe(false);
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 1, exiting: false });
    expect(interval).toHaveBeenCalledOnce();
  });

  it("ignores a response after leaving Plan, even before departure effects commit", async () => {
    const resolve = deferred();
    render("plan").run();
    render("dependencies", false);
    resolve(response());
    await flushResponse();
    expect(useStore.getState().replayPlayback).toBeNull();
    expect(interval).not.toHaveBeenCalled();
    hooks.commit();
    expect(render("plan").pending).toBe(false);
    expect(useStore.getState().replayPlayback).toBeNull();
  });

  it("does not revive a pending replay when returning to the same task", async () => {
    const resolve = deferred();
    render("plan").run();
    render("dependencies");
    expect(render("plan").pending).toBe(false);
    resolve(response());
    await flushResponse();
    expect(useStore.getState().replayPlayback).toBeNull();
    expect(interval).not.toHaveBeenCalled();
  });

  it("discards the old request after a round trip and preserves a fresh request", async () => {
    const resolveOld = deferred();
    render("plan").run();
    render("dependencies");
    const resolveNew = deferred();
    render("plan").run();
    resolveOld(response());
    await flushResponse();
    expect(render("plan").pending).toBe(true);
    expect(useStore.getState().replayPlayback).toBeNull();
    resolveNew(response());
    await flushResponse();
    expect(render("plan").pending).toBe(false);
    expect(useStore.getState().replayPlayback).not.toBeNull();
    expect(interval).toHaveBeenCalledOnce();
  });
});

describe("the end of a replay", () => {
  const started = async () => {
    const resolve = deferred();
    render("plan").run();
    resolve(response());
    await flushResponse();
    return [...ticks.values()][0];
  };

  it("walks every run, then holds the last one on the cards", async () => {
    const step = await started();
    const frames = useStore.getState().replayPlayback!.replay.frames.length;
    expect(frames).toBe(4);
    for (let visited = 2; visited < frames; visited++) {
      step();
      expect(useStore.getState().replayPlayback).toMatchObject({ visited, paused: false });
    }
    step();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: frames, paused: true, exiting: false });
    expect(ticks.size).toBe(0);
  });

  it("clears a held replay by fading the whole trail together", async () => {
    const step = await started();
    for (let i = 0; i < 3; i++) step();
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    render("plan").clear();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 4, exiting: true });
    const fadeTick = [...ticks.values()].pop()!;
    now = 120;
    fadeTick();
    expect(useStore.getState().replayPlayback!.opacity).toBeCloseTo(0.5);
    now = 240;
    fadeTick();
    expect(useStore.getState().replayPlayback).toBeNull();
    expect(ticks.size).toBe(0);
  });
});

describe("stepping a replay by hand", () => {
  const started = async () => {
    const resolve = deferred();
    render("plan").run();
    resolve(response());
    await flushResponse();
  };

  it("pauses where it is and plays on from there", async () => {
    await started();
    [...ticks.values()][0]();
    render("plan").pause();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 2, paused: true });
    expect(ticks.size).toBe(0);
    render("plan").play();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 2, paused: false });
    [...ticks.values()][0]();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 3 });
  });

  it("seeks within the runs, pausing the walk", async () => {
    await started();
    render("plan").seek(3);
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 3, paused: true });
    expect(ticks.size).toBe(0);
    render("plan").seek(99);
    expect(useStore.getState().replayPlayback!.visited).toBe(4);
    render("plan").seek(0);
    expect(useStore.getState().replayPlayback!.visited).toBe(1);
  });

  it("plays again from the first run once the last is held", async () => {
    await started();
    render("plan").seek(4);
    render("plan").play();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 1, paused: false });
  });

  it("ignores the controls for a replay of another tab", async () => {
    await started();
    render("plan").seek(2);
    const away = render("dependencies", false);
    away.seek(3);
    away.play();
    expect(useStore.getState().replayPlayback).toMatchObject({ visited: 2, paused: true });
  });
});
