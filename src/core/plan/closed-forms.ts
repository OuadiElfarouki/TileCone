/** Exact laws for direct staged contractions. Variables are separate from graph symbols. */
import { compareDims, symProduct } from "../dim-algebra";
import { DTYPE_BYTES } from "../dtypes";
import { nodeById } from "../graph";
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
const product = (...args: ClosedExpr[]): ClosedExpr => ({ kind: "product", args });
const sum = (...args: ClosedExpr[]): ClosedExpr => ({ kind: "sum", args });
const min = (...args: ClosedExpr[]): ClosedExpr => ({ kind: "min", args });
const sub = (left: ClosedExpr, right: ClosedExpr): ClosedExpr => ({ kind: "sub", left, right });
const ceilDiv = (left: ClosedExpr, right: ClosedExpr): ClosedExpr => ({ kind: "ceilDiv", left, right });

export type ClosedAxis = Readonly<{ label: string; extent: Sym; role: "output" | "reduced"; defaultSize?: number }>;
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
  if (staged.epilogue.length || staged.output !== tensorId) return unavailable("fused epilogue is not supported");
  const node = nodeById(plan.graph, staged.node)!;
  if (!["einsum", "matmul", "linear"].includes(node.op)) return unavailable("operation is not a supported contraction");
  if (new Set(node.inputs).size !== node.inputs.length) return unavailable("shared operands are not supported");
  if (node.inputs.some((id) => !plan.frontier.includes(id))) return unavailable("fused prologue is not supported");
  if ([...node.inputs, tensorId].some((id) => plan.graph.tensors[id].dtypeWidening)) return unavailable("widened dtype has no exact byte count");
  const domain = staged.domain;
  if (domain.inputs.some((access) => new Set(access).size !== access.length)) return unavailable("diagonal access is not supported");
  const syms: (Sym | undefined)[] = domain.axes.map(() => undefined);
  const definitions = plan.graph.derived ?? {};
  // `symShape` writes an axis it has no expression for as its current extent,
  // which reads as a literal. A law built on it would hold at this binding
  // alone, so every axis used must be one the resolver proved (`symProven`).
  for (const id of [...node.inputs, tensorId]) {
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
    if (!check(domain.output[i], plan.graph.tensors[tensorId].symShape[i]))
      return unavailable("output extents are not symbolically equal to operand extents");
  if (syms.some((s) => s === undefined)) return unavailable("axis has no symbolic extent");
  const family = plan.families.get(tensorId)!;
  const declared = plan.steps.get(tensorId)?.chunk ?? {};
  const outputIndex = new Map(domain.output.map((axis, index) => [axis, index]));
  const axes: ClosedAxis[] = domain.axes.map((axis, i) => ({
    label: axis.label,
    role: axis.role,
    extent: syms[i]!,
    defaultSize: axis.role === "output" ? family.tile[outputIndex.get(i)!] : declared[axis.label],
  }));
  const size = (axis: ClosedAxis): ClosedExpr => {
    if (axis.role === "output") return variable("tile", axis.label);
    return axis.defaultSize === undefined ? dim(axis.extent) : variable("chunk", axis.label);
  };
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
      "direct staged contraction with distinct frontier operands",
      "symbolically equal axis extents and exact storage widths",
      "an undeclared reduction chunk spans the full rebound axis",
    ],
    axes,
    definitions,
    defaults,
    taskCount,
    stepCount,
    family: {
      flops: product(factor, ...axes.map((axis) => dim(axis.extent))),
      readBytes: sum(...familyOperands.map((operand) => operand.bytes)),
      operandBytes: familyOperands,
      writtenBytes: product(dim(DTYPE_BYTES[plan.graph.tensors[tensorId].dtype]), ...outAxes.map((axis) => dim(axis.extent))),
      steps: product(taskCount, stepCount),
    },
    task: {
      flops: product(factor, ...axes.map(taskWidth)),
      readBytes: sum(...taskReads),
      outputBytes: product(dim(DTYPE_BYTES[plan.graph.tensors[tensorId].dtype]), outputVolume),
      accumulatorBytes: product(dim(DTYPE_BYTES[staged.accumulator]), outputVolume),
    },
    step: {
      flops: product(factor, ...axes.map(width)),
      operandBytes,
      payload: sum(...operandBytes.map((a) => a.bytes)),
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
      const allowed = axis && (domain === "tile" || domain === "taskCoord" ? axis.role === "output" : axis.role === "reduced") &&
        !(domain === "chunk" && axis.defaultSize === undefined);
      if (!allowed) throw new Error(`unknown ${domain} variable ${key}`);
    }
  }
  for (const axis of forms.axes) {
    const extent = resolveInteger(axis.extent, params);
    if (!Number.isSafeInteger(extent) || extent <= 0) throw new Error(`invalid extent for ${axis.label}`);
    const sizeDomain = axis.role === "output" ? "tile" : "chunk";
    const coordDomain = axis.role === "output" ? "taskCoord" : "stepCoord";
    const size = axis.role === "reduced" && axis.defaultSize === undefined ? extent : bindings[sizeDomain]?.[axis.label];
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

export function formatClosedForm(expr: ClosedExpr): string {
  switch (expr.kind) {
    // A symbolic extent may itself be a sum or quotient. Keep it grouped
    // inside a subtraction or ceiling division so the printed law agrees.
    case "dim": return typeof expr.value === "string" && /[+*/-]/.test(expr.value)
      ? `(${expr.value})` : String(expr.value);
    case "variable": return `${expr.domain}[${expr.axis}]`;
    case "product": return expr.args.length ? expr.args.map((a) => `(${formatClosedForm(a)})`).join(" · ") : "1";
    case "sum": return expr.args.length ? expr.args.map(formatClosedForm).join(" + ") : "0";
    case "min": return `min(${expr.args.map(formatClosedForm).join(", ")})`;
    case "sub": return `(${formatClosedForm(expr.left)} − ${formatClosedForm(expr.right)})`;
    case "ceilDiv": return `ceil(${formatClosedForm(expr.left)} / ${formatClosedForm(expr.right)})`;
  }
}
