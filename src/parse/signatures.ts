/**
 * The DSL's vocabulary: every function a source can call, as a signature over
 * the operation registry.
 *
 * Names and keywords follow the Python Array API Standard - the part NumPy,
 * PyTorch, JAX and CuPy share - and `torch.nn.functional` for the network
 * operations the standard does not cover. Each concept has one spelling:
 * `axis`, never `dim`; `keepdims`, never `keepdim`.
 *
 * A signature says three things and nothing else:
 *
 * - which parameters the function takes, in order, and which are keyword-only
 *   (the `"*"` marker, as in a Python signature);
 * - for a parameter that is an operation attribute, which one. Its kind - what
 *   values it accepts, its default, how a value is shortened - is the
 *   operation's, from the registry, so no value rule is defined here;
 * - which attributes the name itself stands for: `subtract` is the
 *   elementwise operation with `fn=subtract`, `conv2d` is a convolution over
 *   two spatial axes.
 *
 * Binding a call and printing a node both read this table, so whatever the
 * DSL accepts it can write back, and the reverse.
 */

import { ELEMENTWISE_FNS } from "../core/ops/elementwise";
import { getOp } from "../core/ops/index";
import type { Kind } from "../core/ops/kinds";
import type { Attrs } from "../core/ops/types";

/** A parameter that takes tensors rather than an attribute. */
export type TensorRole =
  /** One tensor. */
  | "input"
  /** One tensor, which may be left out. */
  | "optional-input"
  /** A bracketed list of tensors: `concat([A, B])`. */
  | "inputs"
  /** Every remaining positional argument, as after Python's `*args`. */
  | "variadic";

export type Param =
  /** `present` names the attribute recording whether an optional input was
   * given, which is how a printer tells which inputs a node has. */
  | { name: string; role: TensorRole; present?: string }
  | { name: string; attr: string };

export type Signature = {
  name: string;
  /** The registry operation a call builds. */
  op: string;
  params: Param[];
  /** Index of the first keyword-only parameter; `params.length` when none is. */
  keywordOnly: number;
  /** Attributes the name stands for. A node is this function written back when
   * its attributes carry these values. */
  implied: Attrs;
  /** Attributes that follow from which tensors a call passed. */
  fromInputs?: (bound: { count: number; present: ReadonlySet<string> }) => Attrs;
  /** The keys `fromInputs` produces, which a printer leaves out like `implied`. */
  inputAttrs?: string[];
};

type ParamSpec = Param | "*";

const input = (name: string): Param => ({ name, role: "input" });
const optionalInput = (name: string, present: string): Param => ({ name, role: "optional-input", present });
const inputs = (name: string): Param => ({ name, role: "inputs" });
const variadic = (name: string): Param => ({ name, role: "variadic" });
const attr = (name: string, attribute = name): Param => ({ name, attr: attribute });

function define(
  name: string,
  op: string,
  specs: ParamSpec[],
  extra: Partial<Pick<Signature, "implied" | "fromInputs" | "inputAttrs">> = {}
): Signature {
  const star = specs.indexOf("*");
  const params = specs.filter((p): p is Param => p !== "*");
  // After a variadic parameter every later one is keyword-only, as in Python.
  const afterVariadic = params.findIndex((p) => "role" in p && p.role === "variadic");
  const keywordOnly =
    star >= 0 ? star : afterVariadic >= 0 ? afterVariadic + 1 : params.length;
  return { name, op, params, keywordOnly, implied: extra.implied ?? {}, ...extra };
}

// ------------------------------------------------------------- the table

const elementwise = Object.entries(ELEMENTWISE_FNS).map(([fn, spec]) =>
  define(
    fn,
    "elementwise",
    spec.maxInputs === 1
      ? [input("x")]
      : spec.maxInputs === 2
        ? [input("x1"), input("x2")]
        : // Associative: more operands than two mean the same as a chain.
          [input("x1"), input("x2"), variadic("more")],
    {
      implied: { fn },
      fromInputs: ({ count }) => ({ nary: count }),
      inputAttrs: ["nary"],
    }
  )
);

const reductions = ["sum", "prod", "mean", "max", "min", "logsumexp"].map((fn) =>
  define(fn, "reduce", [input("x"), "*", attr("axis", "axes"), attr("keepdims", "keepdim")], {
    implied: { fn },
  })
);

const normalizations = [
  define(
    "layer_norm",
    "normalize",
    [input("x"), optionalInput("weight", "hasWeight"), optionalInput("bias", "hasBias"), "*", attr("axis", "axes")],
    {
      implied: { kind: "layer_norm" },
      fromInputs: ({ present }) => ({ hasWeight: present.has("weight"), hasBias: present.has("bias") }),
      inputAttrs: ["hasWeight", "hasBias"],
    }
  ),
  define("rms_norm", "normalize", [input("x"), optionalInput("weight", "hasWeight"), "*", attr("axis", "axes")], {
    implied: { kind: "rms_norm" },
    fromInputs: ({ present }) => ({ hasWeight: present.has("weight"), hasBias: false }),
    inputAttrs: ["hasWeight", "hasBias"],
  }),
];

const spatialRanks = [1, 2, 3];

const convolutions = spatialRanks.map((n) =>
  define(
    `conv${n}d`,
    "conv",
    [
      input("input"),
      input("weight"),
      "*",
      attr("stride"),
      attr("padding", "pads"),
      attr("dilation"),
      attr("groups"),
    ],
    { implied: { spatialRank: n } }
  )
);

const pools = (["max", "avg"] as const).flatMap((kind) =>
  spatialRanks.map((n) =>
    define(
      `${kind}_pool${n}d`,
      "pool",
      [
        input("input"),
        attr("kernel_size", "kernelShape"),
        "*",
        attr("stride"),
        attr("padding", "pads"),
        attr("dilation"),
      ],
      { implied: { kind, spatialRank: n } }
    )
  )
);

export const SIGNATURES: readonly Signature[] = [
  define("einsum", "einsum", [attr("subscripts", "equation"), variadic("operands")]),
  define("matmul", "matmul", [input("x1"), input("x2")]),
  define("linear", "linear", [input("input"), input("weight")]),
  ...elementwise,
  ...reductions,
  define("softmax", "softmax", [input("x"), "*", attr("axis")]),
  ...normalizations,
  define("reshape", "reshape", [input("x"), attr("shape")]),
  define("permute_dims", "permute_dims", [input("x"), attr("axes", "perm")]),
  define("broadcast_to", "broadcast_to", [input("x"), attr("shape")]),
  define("concat", "concat", [inputs("arrays"), "*", attr("axis")]),
  define("split", "split", [input("x"), attr("sizes"), "*", attr("axis")]),
  define("pad", "pad", [input("x"), attr("pad_width", "pads"), "*", attr("mode")]),
  ...convolutions,
  ...pools,
  define("cumulative_sum", "cumulative_sum", [input("x"), "*", attr("axis"), attr("reverse")]),
  define("take", "take", [input("x"), input("indices"), "*", attr("axis"), attr("index_values", "indexValues")]),
  define("astype", "astype", [input("x"), attr("dtype")]),
  define("identity", "identity", [input("x")]),
  define("contiguous", "contiguous", [input("x")]),
  define("opaque", "opaque", [
    variadic("inputs"),
    attr("op"),
    attr("shapes"),
    attr("dtypes"),
    attr("domain"),
    attr("opset"),
    attr("source_name", "sourceName"),
  ]),
];

/**
 * The registry operation written with subscript syntax, `X[a:b, ::2]`, rather
 * than as a call. The grammar form is the one every array library uses.
 */
export const SUBSCRIPT_OP = "slice";

const byName = new Map(SIGNATURES.map((sig) => [sig.name, sig]));

const own = <T>(table: Readonly<Record<string, T>>, key: string): T | undefined =>
  Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;

/** What another library calls `name`, in this vocabulary. */
export const foreignName = (name: string): string | undefined => own(FOREIGN_NAMES, name);

/** The keywords another library's `keyword` may mean here. */
export const foreignKeyword = (keyword: string): readonly string[] => own(FOREIGN_KEYWORDS, keyword) ?? [];

/** The signature a call name refers to, if the DSL has that function. */
export function signatureFor(name: string): Signature | undefined {
  return byName.get(name);
}

/** The kind of an attribute parameter, from the operation that declares it. */
export function kindOf(sig: Signature, param: Param & { attr: string }): Kind {
  const kind = getOp(sig.op)?.attrs[param.attr];
  if (!kind) throw new Error(`signature ${sig.name}: ${sig.op} has no attribute "${param.attr}"`);
  return kind;
}

/**
 * The function a node is written as: the signature for its operation whose
 * implied attributes the node carries. `attrs` should be canonical where
 * possible, since an implied attribute may be one the source left to a default.
 */
export function signatureForNode(op: string, attrs: Attrs): Signature | undefined {
  return SIGNATURES.find(
    (sig) =>
      sig.op === op &&
      Object.entries(sig.implied).every(([key, value]) => attrs[key] === value)
  );
}

/** `conv2d(input, weight, *, stride, padding, dilation, groups)`, for errors and help. */
export function signatureLine(sig: Signature): string {
  const parts: string[] = [];
  sig.params.forEach((param, i) => {
    if (i === sig.keywordOnly && !sig.params.slice(0, i).some((p) => "role" in p && p.role === "variadic"))
      parts.push("*");
    if ("role" in param) {
      parts.push(
        param.role === "variadic"
          ? `*${param.name}`
          : param.role === "optional-input"
            ? `${param.name}=None`
            : param.name
      );
      return;
    }
    const kind = kindOf(sig, param);
    parts.push(kind.default || kind.optional ? `${param.name}=…` : param.name);
  });
  return `${sig.name}(${parts.join(", ")})`;
}

/**
 * Other libraries' spellings, and this DSL's earlier ones, for the concepts
 * above. Read only to suggest the right name in an error; a source that uses
 * one of these is refused like any other unknown name.
 */
export const FOREIGN_NAMES: Readonly<Record<string, string>> = {
  transpose: "permute_dims",
  permute: "permute_dims",
  expand: "broadcast_to",
  cast: "astype",
  gather: "take (it selects whole slices along an axis, like index_select)",
  index_select: "take",
  cumsum: "cumulative_sum",
  layernorm: "layer_norm",
  rmsnorm: "rms_norm",
  sub: "subtract",
  mul: "multiply",
  div: "divide",
  neg: "negative",
  amax: "max",
  amin: "min",
  cat: "concat",
  concatenate: "concat",
  bmm: "matmul",
  conv: "conv1d, conv2d or conv3d",
  pool: "max_pool2d or avg_pool2d (or the 1d and 3d forms)",
  slice: "a subscript: X[a:b, :, ::2]",
};

/** Keywords the same way; a keyword maps to every name it may mean here, and
 * the one the signature at hand takes is suggested. */
export const FOREIGN_KEYWORDS: Readonly<Record<string, readonly string[]>> = {
  dim: ["axis", "axes"],
  dims: ["axes"],
  axes: ["axis"],
  axis: ["axes"],
  keepdim: ["keepdims"],
  perm: ["axes"],
  kernelShape: ["kernel_size"],
  kernel_shape: ["kernel_size"],
  pads: ["padding", "pad_width"],
  padding: ["pad_width"],
  indexValues: ["index_values"],
  sourceName: ["source_name"],
  equation: ["subscripts"],
};
