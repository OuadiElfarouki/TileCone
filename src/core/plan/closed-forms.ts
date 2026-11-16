/** Exact laws for direct staged contractions. Variables are separate from graph symbols. */
import { compareDims, symProduct } from "../dim-algebra";
import { DTYPE_BYTES } from "../dtypes";
import { nodeById } from "../graph";
import { ELEMENTWISE_FNS } from "../ops/elementwise";
import { getOp } from "../ops";
import { resolveInteger, type Sym } from "../shapes";
import { stagingOf } from "./steps";
import type { TilePlan } from "./plan";

type Domain = "tile" | "chunk" | "taskCoord" | "stepCoord";
export type ClosedExpr =
  | { kind: "dim"; value: Sym }
  | { kind: "variable"; domain: Domain; axis: string }
  | { kind: "product" | "sum" | "min"; args: readonly ClosedExpr[] }
  | { kind: "sub" | "ceilDiv"; left: ClosedExpr; right: ClosedExpr };
const dim = (value: Sym): ClosedExpr => ({ kind: "dim", value });
const variable = (domain: Domain, axis: string): ClosedExpr => ({ kind: "variable", domain, axis });
const product = (...args: ClosedExpr[]): ClosedExpr => {
  const flat = args.flatMap((a) => a.kind === "product" ? a.args : [a]);
  const constants = flat.filter((a) => a.kind === "dim" && typeof a.value === "number");
  const value = constants.reduce((n, a) => n * (a as { value: number }).value, 1);
  if (value === 0) return dim(0);
  const rest = flat.filter((a) => !(a.kind === "dim" && typeof a.value === "number"));
  if (value !== 1 || !rest.length) rest.unshift(dim(value));
  return rest.length === 1 ? rest[0] : { kind: "product", args: rest };
};
const sum = (...args: ClosedExpr[]): ClosedExpr => {
  const flat = args.flatMap((a) => a.kind === "sum" ? a.args : [a]);
  const constants = flat.filter((a) => a.kind === "dim" && typeof a.value === "number");
  const value = constants.reduce((n, a) => n + (a as { value: number }).value, 0);
  const rest = flat.filter((a) => !(a.kind === "dim" && typeof a.value === "number"));
  if (value !== 0 || !rest.length) rest.unshift(dim(value));
  return rest.length === 1 ? rest[0] : { kind: "sum", args: rest };
};
const min = (...args: ClosedExpr[]): ClosedExpr => ({ kind: "min", args });
const sub = (left: ClosedExpr, right: ClosedExpr): ClosedExpr => ({ kind: "sub", left, right });
const ceilDiv = (left: ClosedExpr, right: ClosedExpr): ClosedExpr => ({ kind: "ceilDiv", left, right });

export type ClosedAxis = Readonly<{ label: string; displayName: string; extent: Sym; role: "output" | "reduced"; defaultSize?: number }>;
export type AvailableClosedForms = Readonly<{
  status: "available";
  tensorId: string;
  node: string;
  assumptions: readonly string[];
  axes: readonly ClosedAxis[];
  definitions: Readonly<Record<string, Sym>>;
  defaults: ClosedBindings;
  taskCount: ClosedExpr;
  stepCount: ClosedExpr;
  task: { flops: ClosedExpr; readBytes: ClosedExpr; outputBytes: ClosedExpr; accumulatorBytes: ClosedExpr };
  /**
   * Sums over every task of the family. Tails need no coordinates here: along
   * an output axis the tile widths add up to the axis, so an operand's bytes
   * are its own extents times the number of tiles along each output axis it
   * does not carry - each of those tiles reads it again.
   */
  family: {
    flops: ClosedExpr;
    readBytes: ClosedExpr;
    operandBytes: readonly { tensorId: string; bytes: ClosedExpr }[];
    writtenBytes: ClosedExpr;
    steps: ClosedExpr;
  };
  step: { flops: ClosedExpr; operandBytes: readonly { tensorId: string; bytes: ClosedExpr }[]; payload: ClosedExpr };
  /** Valid only when every output tile and reduction chunk is full. */
  interior: {
    conditions: readonly string[];
    task: AvailableClosedForms["task"];
    step: AvailableClosedForms["step"];
    /** More compact FLOP laws when the full reduction contains multiple terms. */
    accumulating: { conditions: readonly string[]; taskFlops: ClosedExpr; stepFlops: ClosedExpr };
  };
}>;
export type ClosedForms = AvailableClosedForms | Readonly<{ status: "unavailable"; reason: string }>;
export type ClosedBindings = Readonly<{
  params: Readonly<Record<string, number>>;
  tile?: Readonly<Record<string, number>>;
  chunk?: Readonly<Record<string, number>>;
  taskCoord?: Readonly<Record<string, number>>;
  stepCoord?: Readonly<Record<string, number>>;
}>;

export function closedFormsOf(plan: TilePlan, tensorId: string): ClosedForms {
  const unavailable = (reason: string): ClosedForms => ({ status: "unavailable", reason });
  const staged = stagingOf(plan, tensorId);
  if (!staged.staged) return unavailable(staged.reason);
  const node = nodeById(plan.graph, staged.node)!;
  if (!["einsum", "matmul", "linear"].includes(node.op)) return unavailable("operation is not a supported contraction");
  if (new Set(node.inputs).size !== node.inputs.length) return unavailable("shared operands are not supported");
  if (node.inputs.some((id) => !plan.frontier.includes(id))) return unavailable("fused prologue is not supported");
  const domain = staged.domain;
  if (domain.inputs.some((access) => new Set(access).size !== access.length)) return unavailable("diagonal access is not supported");
  const syms: (Sym | undefined)[] = domain.axes.map(() => undefined);
  const definitions = plan.graph.derived ?? {};
  // `symShape` writes an axis it has no expression for as its current extent,
  // which reads as a literal. A law built on it would hold at this binding
  // alone, so every axis used must be one the resolver proved (`symProven`).
  const used = new Set([...node.inputs, staged.output, tensorId]);
  const sides: { id: string; carried: number[] }[] = [];
  let chain = staged.output;
  let epilogueCost = 0;
  const baseShape = plan.graph.tensors[staged.output].symShape;
  for (const nodeId of [...staged.epilogue].reverse()) {
    const op = nodeById(plan.graph, nodeId)!;
    const consumers = plan.graph.consumers[chain] ?? [];
    if (consumers.length !== 1 || consumers[0].nodeId !== op.id)
      return unavailable("epilogue chain has a shared intermediate");
    const fn = op.attrs.fn as string;
    const unary = op.op === "elementwise" && ELEMENTWISE_FNS[fn]?.minInputs === 1 && ELEMENTWISE_FNS[fn]?.maxInputs === 1;
    const add = op.op === "elementwise" && fn === "add" && op.inputs.length === 2;
    const passthrough = ["identity", "contiguous", "astype"].includes(op.op);
    if ((!unary && !add && !passthrough) || op.outputs.length !== 1 || op.inputs.filter((id) => id === chain).length !== 1 || (!add && op.inputs.length !== 1))
      return unavailable("unsupported epilogue operation or chain");
    const output = op.outputs[0];
    const shape = plan.graph.tensors[output].symShape;
    if (shape.length !== baseShape.length || shape.some((extent, i) => compareDims(extent, baseShape[i], { definitions }) !== "same"))
      return unavailable("epilogue output shape is not symbolically equal to contraction output");
    used.add(output);
    if (add) {
      if (sides.length) return unavailable("multiple epilogue additions are not supported");
      const side = op.inputs.find((id) => id !== chain)!;
      if (!plan.frontier.includes(side) || used.has(side)) return unavailable("epilogue side operand must be a distinct frontier tensor");
      const tensor = plan.graph.tensors[side];
      if (tensor.symShape.length > baseShape.length) return unavailable("epilogue side rank exceeds output rank");
      const offset = baseShape.length - tensor.symShape.length;
      const carried: number[] = [];
      for (let i = 0; i < tensor.symShape.length; i++) {
        const extent = tensor.symShape[i];
        if (compareDims(extent, 1, { definitions }) === "same") continue;
        if (compareDims(extent, baseShape[offset + i], { definitions }) !== "same")
          return unavailable("epilogue side shape is not symbolically broadcastable");
        carried.push(domain.output[offset + i]);
      }
      sides.push({ id: side, carried });
      used.add(side);
    }
    const spec = getOp(op.op)!;
    if (!spec.flopsPerElement) return unavailable("epilogue cost is unavailable");
    epilogueCost += spec.flopsPerElement(0, { inShapes: plan.graph.shapesOf(op.inputs), outShapes: plan.graph.shapesOf(op.outputs), attrs: op.attrs });
    chain = output;
  }
  if (chain !== tensorId) return unavailable("epilogue chain does not reach output");
  if ([...used].some((id) => plan.graph.tensors[id].dtypeWidening)) return unavailable("widened dtype has no exact byte count");
  for (const id of used) {
    const tensor = plan.graph.tensors[id];
    const axis = tensor.symProven.findIndex((proven) => !proven);
    if (axis >= 0)
      return unavailable(
        `axis ${axis} of ${tensor.name ?? id} has no expression: its extent ${tensor.resolved[axis]} is known at this binding only`
      );
  }
  const check = (axis: number, sym: Sym): boolean => {
    const existing = syms[axis];
    if (existing === undefined) { syms[axis] = sym; return true; }
    return compareDims(existing, sym, { definitions }) === "same";
  };
  for (let slot = 0; slot < node.inputs.length; slot++)
    for (let i = 0; i < domain.inputs[slot].length; i++)
      if (!check(domain.inputs[slot][i], plan.graph.tensors[node.inputs[slot]].symShape[i]))
        return unavailable("axis extents are not symbolically equal");
  for (let i = 0; i < domain.output.length; i++)
    if (!check(domain.output[i], plan.graph.tensors[staged.output].symShape[i]))
      return unavailable("output extents are not symbolically equal to operand extents");
  if (syms.some((s) => s === undefined)) return unavailable("axis has no symbolic extent");
  const family = plan.families.get(tensorId)!;
  const declared = plan.steps.get(tensorId)?.chunk ?? {};
  const outputIndex = new Map(domain.output.map((axis, index) => [axis, index]));
  const names = domain.axes.map(() => new Set<string>());
  for (let slot = 0; slot < node.inputs.length; slot++)
    domain.inputs[slot].forEach((axis, i) => {
      const name = plan.graph.tensors[node.inputs[slot]].axisNames?.[i];
      if (name) names[axis].add(name);
    });
  domain.output.forEach((axis, i) => {
    const name = plan.graph.tensors[tensorId].axisNames?.[i];
    if (name) names[axis].add(name);
  });
  const candidates = names.map((set, i) => set.size === 1 ? [...set][0] : String(syms[i]));
  const displayNames = candidates.map((name, i) => candidates.filter((other) => other === name).length === 1 ? name : domain.axes[i].label);
  // A fallback label may itself collide with another axis's preferred name.
  if (new Set(displayNames).size !== displayNames.length) domain.axes.forEach((axis, i) => { displayNames[i] = axis.label; });
  const axes: ClosedAxis[] = domain.axes.map((axis, i) => ({
    label: axis.label,
    displayName: displayNames[i],
    role: axis.role,
    extent: syms[i]!,
    defaultSize: axis.role === "output" ? family.tile[outputIndex.get(i)!] : declared[axis.label],
  }));
  const size = (axis: ClosedAxis): ClosedExpr =>
    variable(axis.role === "output" ? "tile" : "chunk", axis.label);
  const width = (axis: ClosedAxis): ClosedExpr => min(
    size(axis),
    sub(dim(axis.extent), product(variable(axis.role === "output" ? "taskCoord" : "stepCoord", axis.label), size(axis)))
  );
  const taskWidth = (axis: ClosedAxis): ClosedExpr => axis.role === "output" ? width(axis) : dim(axis.extent);
  const outAxes = axes.filter((axis) => axis.role === "output");
  const reduced = axes.filter((axis) => axis.role === "reduced");
  const reducedVolume = dim(symProduct(reduced.map((axis) => axis.extent)));
  const factor = sum(dim(node.inputs.length - 1), min(dim(1), sub(reducedVolume, dim(1))));
  const outputVolume = product(...outAxes.map(width));
  const taskReads = node.inputs.map((id, slot) => product(
    dim(DTYPE_BYTES[plan.graph.tensors[id].dtype]),
    ...domain.inputs[slot].map((i) => taskWidth(axes[i]))
  ));
  const operandBytes = node.inputs.map((id, slot) => ({
    tensorId: id,
    bytes: product(dim(DTYPE_BYTES[plan.graph.tensors[id].dtype]), ...domain.inputs[slot].map((i) => width(axes[i]))),
  }));
  const tilesAlong = (axis: ClosedAxis): ClosedExpr => ceilDiv(dim(axis.extent), size(axis));
  const taskCount = product(...outAxes.map(tilesAlong));
  const stepCount = product(...reduced.map(tilesAlong));
  const familyOperands = node.inputs.map((id, slot) => {
    const carried = new Set(domain.inputs[slot]);
    return {
      tensorId: id,
      bytes: product(
        dim(DTYPE_BYTES[plan.graph.tensors[id].dtype]),
        ...domain.inputs[slot].map((i) => dim(axes[i].extent)),
        ...axes.filter((axis, i) => axis.role === "output" && !carried.has(i)).map(tilesAlong)
      ),
    };
  });
  for (const side of sides) {
    const carried = new Set(side.carried);
    familyOperands.push({
      tensorId: side.id,
      bytes: product(dim(DTYPE_BYTES[plan.graph.tensors[side.id].dtype]),
        ...side.carried.map((i) => dim(axes[i].extent)),
        ...domain.output.filter((i) => !carried.has(i)).map((i) => tilesAlong(axes[i]))),
    });
  }
  const sideReads = (axisWidth: (axis: ClosedAxis) => ClosedExpr): ClosedExpr[] => sides.map((side) =>
    product(dim(DTYPE_BYTES[plan.graph.tensors[side.id].dtype]), ...side.carried.map((i) => axisWidth(axes[i])))
  );
  const interiorOperands = node.inputs.map((id, slot) => ({
    tensorId: id,
    bytes: product(dim(DTYPE_BYTES[plan.graph.tensors[id].dtype]), ...domain.inputs[slot].map((i) => size(axes[i]))),
  }));
  const interiorOutputVolume = product(...outAxes.map(size));
  const interiorTaskWidth = (axis: ClosedAxis): ClosedExpr => axis.role === "output" ? size(axis) : dim(axis.extent);
  const defaults: ClosedBindings = {
    params: { ...plan.graph.params },
    tile: Object.fromEntries(outAxes.map((axis) => [axis.label, axis.defaultSize!])),
    chunk: Object.fromEntries(reduced.filter((axis) => axis.defaultSize !== undefined).map((axis) => [axis.label, axis.defaultSize!])),
    taskCoord: Object.fromEntries(outAxes.map((axis) => [axis.label, 0])),
    stepCoord: Object.fromEntries(reduced.map((axis) => [axis.label, 0])),
  };
  return {
    status: "available",
    tensorId,
    node: node.id,
    assumptions: [
      staged.epilogue.length ? "staged contraction with distinct frontier operands and a supported pointwise epilogue" : "direct staged contraction with distinct frontier operands",
      "symbolically equal axis extents and exact storage widths",
      "an undeclared reduction chunk defaults to the full rebound axis unless overridden",
    ],
    axes,
    definitions,
    defaults,
    taskCount,
    stepCount,
    family: {
      flops: sum(product(factor, ...axes.map((axis) => dim(axis.extent))), product(dim(epilogueCost), ...outAxes.map((axis) => dim(axis.extent)))),
      readBytes: sum(...familyOperands.map((operand) => operand.bytes)),
      operandBytes: familyOperands,
      writtenBytes: product(dim(DTYPE_BYTES[plan.graph.tensors[tensorId].dtype]), ...outAxes.map((axis) => dim(axis.extent))),
      steps: product(taskCount, stepCount),
    },
    task: {
      flops: sum(product(factor, ...axes.map(taskWidth)), product(dim(epilogueCost), outputVolume)),
      readBytes: sum(...taskReads, ...sideReads(width)),
      outputBytes: product(dim(DTYPE_BYTES[plan.graph.tensors[tensorId].dtype]), outputVolume),
      accumulatorBytes: product(dim(DTYPE_BYTES[staged.accumulator]), outputVolume),
    },
    step: {
      flops: product(factor, ...axes.map(width)),
      operandBytes,
      payload: sum(...operandBytes.map((a) => a.bytes)),
    },
    interior: {
      conditions: ["every output tile is full", "every reduction chunk is full"],
      task: {
        flops: sum(product(factor, ...axes.map(interiorTaskWidth)), product(dim(epilogueCost), interiorOutputVolume)),
        readBytes: sum(...node.inputs.map((id, slot) => product(
          dim(DTYPE_BYTES[plan.graph.tensors[id].dtype]), ...domain.inputs[slot].map((i) => interiorTaskWidth(axes[i]))
        )), ...sideReads(size)),
        outputBytes: product(dim(DTYPE_BYTES[plan.graph.tensors[tensorId].dtype]), interiorOutputVolume),
        accumulatorBytes: product(dim(DTYPE_BYTES[staged.accumulator]), interiorOutputVolume),
      },
      step: {
        flops: product(factor, ...axes.map(size)),
        operandBytes: interiorOperands,
        payload: sum(...interiorOperands.map((operand) => operand.bytes)),
      },
      accumulating: {
        conditions: ["every output tile is full", "every reduction chunk is full", "full reduction has more than one term"],
        taskFlops: sum(product(dim(node.inputs.length), ...axes.map(interiorTaskWidth)), product(dim(epilogueCost), interiorOutputVolume)),
        stepFlops: product(dim(node.inputs.length), ...axes.map(size)),
      },
    },
  };
}

/** Evaluate with explicit bindings. Coordinates and chunk sizes are checked against
 * the rebound dimensions before evaluating even an expression that omits them. */
export function evaluateClosedForm(forms: AvailableClosedForms, expr: ClosedExpr, bindings: ClosedBindings): number {
  const params: Record<string, number> = { ...bindings.params };
  for (const [name, value] of Object.entries(params)) if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`invalid parameter ${name}`);
  for (const [name, definition] of Object.entries(forms.definitions)) {
    const value = resolveInteger(definition, params);
    if (value <= 0) throw new Error(`invalid derived dimension ${name}`);
    params[name] = value;
  }
  const checked = (n: number): number => { if (!Number.isSafeInteger(n)) throw new Error("unsafe closed-form arithmetic"); return n; };
  const values: Record<Domain, Record<string, number>> = { tile: {}, chunk: {}, taskCoord: {}, stepCoord: {} };
  for (const domain of ["tile", "chunk", "taskCoord", "stepCoord"] as const) {
    for (const key of Object.keys(bindings[domain] ?? {})) {
      const axis = forms.axes.find((a) => a.label === key);
      const allowed = axis && (domain === "tile" || domain === "taskCoord" ? axis.role === "output" : axis.role === "reduced");
      if (!allowed) throw new Error(`unknown ${domain} variable ${key}`);
    }
  }
  for (const axis of forms.axes) {
    const extent = resolveInteger(axis.extent, params);
    if (!Number.isSafeInteger(extent) || extent <= 0) throw new Error(`invalid extent for ${axis.label}`);
    const sizeDomain = axis.role === "output" ? "tile" : "chunk";
    const coordDomain = axis.role === "output" ? "taskCoord" : "stepCoord";
    const size = bindings[sizeDomain]?.[axis.label] ?? axis.defaultSize ?? (axis.role === "reduced" ? extent : undefined);
    if (size === undefined || !Number.isSafeInteger(size) || size <= 0 || (axis.role === "reduced" && size > extent))
      throw new Error(`invalid ${sizeDomain} for ${axis.label}`);
    const coord = bindings[coordDomain]?.[axis.label] ?? 0;
    if (!Number.isSafeInteger(coord) || coord < 0 || coord >= Math.ceil(extent / size)) throw new Error(`invalid ${coordDomain} for ${axis.label}`);
    values[sizeDomain][axis.label] = size;
    values[coordDomain][axis.label] = coord;
  }
  const visit = (e: ClosedExpr): number => {
    switch (e.kind) {
      case "dim": return checked(resolveInteger(e.value, params));
      case "variable": {
        const n = values[e.domain][e.axis];
        if (n === undefined) throw new Error(`unknown variable ${e.domain}.${e.axis}`);
        return n;
      }
      case "sum": return e.args.reduce((n, a) => checked(n + visit(a)), 0);
      case "product": return e.args.reduce((n, a) => checked(n * visit(a)), 1);
      case "min": {
        if (!e.args.length) throw new Error("min needs an argument");
        return checked(Math.min(...e.args.map(visit)));
      }
      case "sub": return checked(visit(e.left) - visit(e.right));
      case "ceilDiv": {
        const left = visit(e.left), right = visit(e.right);
        if (left < 0 || right <= 0) throw new Error("invalid ceilDiv");
        return checked(Math.floor(left / right) + (left % right === 0 ? 0 : 1));
      }
    }
  };
  return visit(expr);
}

export function formatClosedForm(expr: ClosedExpr, forms?: AvailableClosedForms): string {
  const render = (e: ClosedExpr, parent: "product" | "other" = "other"): string => {
    switch (e.kind) {
      case "dim": return typeof e.value === "string" && /[+*/-]/.test(e.value)
        ? `(${e.value})` : String(e.value);
      case "variable": return `${e.domain}[${forms?.axes.find((axis) => axis.label === e.axis)?.displayName ?? e.axis}]`;
      case "product": return e.args.length ? e.args.map((a) => render(a, "product")).join(" · ") : "1";
      case "sum": {
        const text = e.args.length ? e.args.map((a) => render(a)).join(" + ") : "0";
        return parent === "product" ? `(${text})` : text;
      }
      case "min": return `min(${e.args.map((a) => render(a)).join(", ")})`;
      case "sub": return `(${render(e.left)} − ${render(e.right)})`;
      case "ceilDiv": return `ceil(${render(e.left)} / ${render(e.right)})`;
    }
  };
  return render(expr);
}
