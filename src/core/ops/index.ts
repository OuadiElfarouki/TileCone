import { OpSpec } from "./types";
import { einsumOp, matmulOp, bmmOp, linearOp } from "./einsum";
import { elementwiseOp } from "./elementwise";
import { reduceOp } from "./reduce";
import { softmaxOp } from "./softmax";
import { normalizeOp } from "./normalize";
import {
  transposeOp,
  sliceOp,
  padOp,
  concatOp,
  splitOp,
  expandOp,
  identityLike,
  castOp,
} from "./shape-ops";
import { reshapeOp } from "./reshape";
import { convOp, poolOp } from "./conv";
import { cumsumOp } from "./scan";
import { gatherOp } from "./gather";

const registry = new Map<string, OpSpec>();

function registerOp(spec: OpSpec): void {
  if (registry.has(spec.name)) throw new Error(`op "${spec.name}" already registered`);
  registry.set(spec.name, spec);
}

export function getOp(name: string): OpSpec | undefined {
  return registry.get(name);
}

/**
 * Every registered operation, in registration order.
 *
 * Exported so that "the op registry" can be a thing tests and tooling iterate
 * rather than a list maintained by hand in parallel with this file. The oracle
 * corpus and the adjointness law both drive off it, so registering an operation
 * without a fixture fails the suite instead of silently going untested.
 */
export function listOps(): OpSpec[] {
  return [...registry.values()];
}

[
  einsumOp,
  matmulOp,
  bmmOp,
  linearOp,
  elementwiseOp,
  reduceOp,
  softmaxOp,
  normalizeOp,
  transposeOp,
  sliceOp,
  padOp,
  concatOp,
  splitOp,
  expandOp,
  reshapeOp,
  convOp,
  poolOp,
  cumsumOp,
  gatherOp,
  identityLike("identity"),
  castOp,
  identityLike("contiguous"),
].forEach(registerOp);
