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
