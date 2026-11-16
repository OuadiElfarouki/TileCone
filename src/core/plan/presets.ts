/**
 * Machines a budget can be taken from.
 *
 * A preset supplies a default for the number the engineer states - the bytes a
 * task may hold of its operands at a time - and records where it came from. It
 * is not a machine model: nothing is computed from the other fields, which are
 * kept so the source of the default can be read beside it (docs/STAGING.md).
 */

import type { Budget } from "./sweep";

export type MachinePreset = Readonly<{
  name: string;
  /** Shared memory one block may allocate, in bytes: the default step budget. */
  stepBytes: number;
  /** Recorded, and used by no figure. */
  sms?: number;
  l2Bytes?: number;
  /** Fields taken on the user's word and not yet read off the device. */
  unverified: readonly ("stepBytes" | "sms" | "l2Bytes")[];
}>;

export const MACHINE_PRESETS: readonly MachinePreset[] = [
  {
    name: "RTX PRO 6000 Blackwell Workstation Edition",
    stepBytes: 99 * 1024,
    sms: 188,
    l2Bytes: 128 * 1024 * 1024,
    // Confirm with cudaDevAttrMaxSharedMemoryPerBlockOptin and cudaDevAttrL2CacheSize.
    unverified: ["stepBytes", "l2Bytes"],
  },
];

/** The budget a new workspace starts with: the first preset's, and no bound on
 * the accumulator. */
export const DEFAULT_BUDGET: Budget = Object.freeze({ stepBytes: MACHINE_PRESETS[0].stepBytes });

export const sameBudget = (a: Budget, b: Budget): boolean =>
  a.stepBytes === b.stepBytes && a.accumulatorBytes === b.accumulatorBytes;

/** A budget is positive, finite numbers; an absent accumulator bound is none. */
export function isBudget(value: Budget): boolean {
  const ok = (n: unknown) => typeof n === "number" && Number.isFinite(n) && n >= 1;
  return ok(value.stepBytes) && (value.accumulatorBytes === undefined || ok(value.accumulatorBytes));
}
