import { computeMetrics, AggregateReadout } from "./metrics";
import {
  PropResult,
  propagateBackward,
  propagateForward,
  Selection,
} from "./propagate";
import { canonicalize, Region } from "./region";
import { ResolvedGraph } from "./graph";
import { Entanglement, entangledWith } from "./entangle";

export type QueryDirection = "backward" | "forward" | "both";

export type SymbolicQuery = Selection & {
  direction?: QueryDirection;
};

export type SymbolicQueryResult = {
  selection: Selection;
  direction: QueryDirection;
  backward: PropResult | null;
  forward: PropResult | null;
};

export type ExecutionErrorCode =
  | "EXEC_UNKNOWN_TENSOR"
  | "EXEC_DIRECTION"
  | "EXEC_REGION_RANK"
  | "EXEC_REGION_BOUNDS";

export class ExecutionError extends Error {
  constructor(
    public code: ExecutionErrorCode,
    message: string
  ) {
    super(message);
    this.name = "ExecutionError";
  }
}

/** Validate and defensively copy a selection before it enters propagation. */
export function validateSelection(graph: ResolvedGraph, selection: Selection): Selection {
  const tensor = graph.tensors[selection.tensorId];
  if (!tensor)
    throw new ExecutionError("EXEC_UNKNOWN_TENSOR", `unknown tensor "${selection.tensorId}"`);

  const shape = tensor.resolved!;
  const boxes = selection.region.boxes.map((box, boxIndex) => {
    if (box.length !== shape.length)
      throw new ExecutionError(
        "EXEC_REGION_RANK",
        `tensor "${selection.tensorId}" has rank ${shape.length}, but box ${boxIndex} has rank ${box.length}`
      );
    return box.map((interval, axis) => {
      const { lo, hi } = interval;
      if (
        !Number.isSafeInteger(lo) ||
        !Number.isSafeInteger(hi) ||
        lo < 0 ||
        hi < lo ||
        hi > shape[axis]
      )
        throw new ExecutionError(
          "EXEC_REGION_BOUNDS",
          `tensor "${selection.tensorId}" box ${boxIndex}, axis ${axis}: ` +
            `[${lo}, ${hi}) is outside [0, ${shape[axis]})`
        );
      return { lo, hi };
    });
  });
  const region: Region = canonicalize({
    boxes,
    exact: selection.region.exact,
    reasons: selection.region.reasons.slice(),
  });
  return { tensorId: selection.tensorId, region };
}

/** Execute one dependency query against a resolved graph. */
export function executeQuery(
  graph: ResolvedGraph,
  query: SymbolicQuery
): SymbolicQueryResult {
  const selection = validateSelection(graph, query);
  const direction = query.direction ?? "backward";
  if (direction !== "backward" && direction !== "forward" && direction !== "both")
    throw new ExecutionError("EXEC_DIRECTION", `unknown query direction "${String(direction)}"`);
  return {
    selection,
    direction,
    backward:
      direction === "backward" || direction === "both"
        ? propagateBackward(graph, selection)
        : null,
    forward:
      direction === "forward" || direction === "both"
        ? propagateForward(graph, selection)
        : null,
  };
}

/**
 * Headless execution surface for CLIs, tests, and embedders.
 *
 * The low-level propagation functions remain available for op-level work; this
 * class is the checked public boundary for user-authored queries.
 */
export class SymbolicExecutor {
  constructor(readonly graph: ResolvedGraph) {}

  query(query: SymbolicQuery): SymbolicQueryResult {
    return executeQuery(this.graph, query);
  }

  upstream(tensorId: string, region: Region): PropResult {
    return this.query({ tensorId, region, direction: "backward" }).backward!;
  }

  downstream(tensorId: string, region: Region): PropResult {
    return this.query({ tensorId, region, direction: "forward" }).forward!;
  }

  /**
   * What this region is *combined with*: the other operands it meets in the
   * same term, at each operation that reads it.
   *
   * A third relation alongside `upstream` and `downstream`, not a view over
   * them - see `core/entangle.ts` for why the composition of those two answers
   * a different, looser question.
   */
  entangled(tensorId: string, region: Region): Entanglement[] {
    return entangledWith(this.graph, tensorId, validateSelection(this.graph, { tensorId, region }).region);
  }

  metrics(tensorId: string, region: Region, countIntermediates = false): AggregateReadout {
    return computeMetrics(this.graph, this.upstream(tensorId, region), countIntermediates);
  }
}
