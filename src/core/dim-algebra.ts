/**
 * Equality of dimension expressions as expressions, not as numbers.
 *
 * Shape inference evaluates every extent against the bound parameters, so two
 * axes that an operation requires to match are compared as integers. That
 * accepts `M*2` against `N*2` whenever `M` and `N` happen to be bound to the
 * same value, which is a property of one binding rather than of the program.
 * This module decides whether two extents are equal for *every* binding.
 *
 * An expression is put into a canonical form: a sum of monomials, each a
 * product of symbols raised to nonzero integer powers, with an exact rational
 * coefficient. `+ - * /` stay closed in that form as long as every divisor is a
 * single monomial, which covers the dimension arithmetic people write (`E/H`,
 * `H*D`, `N/2*2`). Two canonical forms are equal exactly when the polynomials
 * are, and distinct polynomials differ for some binding, so a difference is a
 * proof that the equality depends on the values chosen.
 *
 * Anything the form cannot represent becomes an opaque atom: a division by a
 * sum, or an extent the caller marks as unknown. Opaque atoms cancel against
 * themselves, but a difference that still contains one proves nothing, and the
 * answer is `undecided` rather than a claim the algebra cannot back.
 *
 * A derived symbol (`E = H*D`) is not an atom: it is replaced by its
 * definition, so `E` and `H*D` are the same expression and `E` against `M*N`
 * is compared as `H*D` against `M*N`.
 */

import { DimNode, readDimExpr, Sym } from "./shapes";

/** Exact rational in lowest terms, denominator positive. */
export type Rat = { n: bigint; d: bigint };

const absBig = (v: bigint): bigint => (v < 0n ? -v : v);

function gcdBig(a: bigint, b: bigint): bigint {
  a = absBig(a);
  b = absBig(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

function rat(n: bigint, d = 1n): Rat {
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const g = gcdBig(n, d) || 1n;
  return { n: n / g, d: d / g };
}

const ratAdd = (a: Rat, b: Rat): Rat => rat(a.n * b.d + b.n * a.d, a.d * b.d);
const ratMul = (a: Rat, b: Rat): Rat => rat(a.n * b.n, a.d * b.d);

/** Atom name to power, sorted by name so equal monomials print equal keys. */
export type Monomial = [string, number][];
export type Term = { mono: Monomial; coef: Rat };
/** Keyed by the monomial, so like terms combine on insertion. */
export type Poly = Map<string, Term>;

const monoKey = (mono: Monomial): string => mono.map(([name, pow]) => `${name}^${pow}`).join("*");

function monoMul(a: Monomial, b: Monomial): Monomial {
  const powers = new Map(a);
  for (const [name, pow] of b) {
    const next = (powers.get(name) ?? 0) + pow;
    if (next) powers.set(name, next);
    else powers.delete(name);
  }
  return [...powers].sort(([x], [y]) => (x < y ? -1 : x > y ? 1 : 0));
}

function addTerm(poly: Poly, mono: Monomial, coef: Rat): void {
  if (!coef.n) return;
  const key = monoKey(mono);
  const prev = poly.get(key);
  if (!prev) {
    poly.set(key, { mono, coef });
    return;
  }
  const sum = ratAdd(prev.coef, coef);
  if (sum.n) poly.set(key, { mono, coef: sum });
  else poly.delete(key);
}

const constant = (value: Rat): Poly => {
  const poly: Poly = new Map();
  addTerm(poly, [], value);
  return poly;
};

function polyAdd(a: Poly, b: Poly, sign: 1n | -1n = 1n): Poly {
  const out: Poly = new Map(a);
  for (const { mono, coef } of b.values()) addTerm(out, mono, rat(sign * coef.n, coef.d));
  return out;
}

/**
 * A product of two sums grows as the product of their sizes, and a reshape over
 * a high-rank tensor of sums multiplies many. Past this many terms the answer
 * is given up as `undecided`; the numeric check has already passed, so giving
 * up costs a missed warning and never a false one.
 */
const MAX_TERMS = 4096;

function polyMul(a: Poly, b: Poly): Poly | null {
  if (a.size * b.size > MAX_TERMS) return null;
  const out: Poly = new Map();
  for (const x of a.values())
    for (const y of b.values()) addTerm(out, monoMul(x.mono, y.mono), ratMul(x.coef, y.coef));
  return out;
}

/** The canonical text of a polynomial, also used to name an opaque quotient. */
function formatPoly(poly: Poly): string {
  if (!poly.size) return "0";
  const terms = [...poly.values()].sort((x, y) =>
    monoKey(x.mono) < monoKey(y.mono) ? -1 : monoKey(x.mono) > monoKey(y.mono) ? 1 : 0
  );
  return terms
    .map(({ mono, coef }, i) => {
      const up = mono.flatMap(([name, pow]) => Array<string>(Math.max(pow, 0)).fill(name));
      const down = mono.flatMap(([name, pow]) => Array<string>(Math.max(-pow, 0)).fill(name));
      const n = absBig(coef.n);
      const numerator = [...(n !== 1n || !up.length ? [String(n)] : []), ...up].join("*");
      const denominator = [...(coef.d !== 1n ? [String(coef.d)] : []), ...down].join("*");
      const body = denominator ? `${numerator}/${denominator}` : numerator;
      const negative = coef.n < 0n;
      if (i === 0) return negative ? `-${body}` : body;
      return negative ? ` - ${body}` : ` + ${body}`;
    })
    .join("");
}

/** Marks an atom the algebra treats as an unknown quantity. */
const OPAQUE_PREFIX = "\u0000";

/**
 * Canonical forms of derived symbols, computed once each. A definition is
 * expanded wherever it is read, so without this a chain `D1 = D0+D0`,
 * `D2 = D1+D1`, ... re-expands every link at every use and the work doubles
 * per definition, though the result is one term. Valid while the definitions
 * it was filled from are unchanged and name no opaque symbol, which holds for
 * the resolver: its opaque symbols are minted under names nothing defines.
 */
export type DimCache = Map<string, Poly | null>;

export const dimCache = (): DimCache => new Map();

export type DimContext = {
  /** Symbols standing for unknown quantities rather than free parameters. */
  opaque?: ReadonlySet<string>;
  /** Derived symbols and the expressions that define them. */
  definitions?: Readonly<Record<string, Sym>>;
  /** Shared across comparisons over the same definitions. */
  cache?: DimCache;
};

/**
 * Expression nodes one comparison may visit, counting each definition once.
 * Shapes in a graph are a handful of nodes each; past this the answer is
 * `undecided`, which costs a missed warning and never a false one, rather
 * than letting an authored or imported program stall compilation.
 */
const MAX_WORK = 100_000;

type Scope = DimContext & { expanding: Set<string>; cache: DimCache; work: { left: number } };

function canonical(node: DimNode, scope: Scope): Poly | null {
  if (--scope.work.left < 0) return null;
  switch (node.kind) {
    case "num": {
      if (!Number.isSafeInteger(node.value)) return null;
      return constant(rat(BigInt(node.value)));
    }
    case "sym": {
      const { definitions, expanding, cache } = scope;
      if (definitions && Object.prototype.hasOwnProperty.call(definitions, node.name)) {
        if (cache.has(node.name)) return cache.get(node.name)!;
        // A cyclic definition has no value; resolution rejects one, and here
        // it is simply not something to reason about.
        if (expanding.has(node.name)) return null;
        expanding.add(node.name);
        const expanded = canonicalSym(definitions[node.name], scope);
        expanding.delete(node.name);
        // A result cut short by the budget is not the definition's form, and
        // caching it would hand the shortfall to every later comparison.
        if (scope.work.left >= 0) cache.set(node.name, expanded);
        return expanded;
      }
      const atom = scope.opaque?.has(node.name) ? OPAQUE_PREFIX + node.name : node.name;
      const poly: Poly = new Map();
      addTerm(poly, [[atom, 1]], rat(1n));
      return poly;
    }
    case "neg": {
      const arg = canonical(node.arg, scope);
      return arg && polyAdd(new Map(), arg, -1n);
    }
    case "bin": {
      const left = canonical(node.left, scope);
      const right = canonical(node.right, scope);
      if (!left || !right) return null;
      if (node.op === "+") return polyAdd(left, right);
      if (node.op === "-") return polyAdd(left, right, -1n);
      if (node.op === "*") return polyMul(left, right);
      if (!right.size) return null;
      // Dividing by one monomial is multiplying by its inverse. Dividing by a
      // sum has no polynomial form, so the quotient is named after both sides
      // and becomes an atom: equal quotients still cancel, and nothing else is
      // claimed about it.
      if (right.size === 1) {
        const [{ mono, coef }] = right.values();
        const inverse: Poly = new Map();
        addTerm(
          inverse,
          mono.map(([name, pow]) => [name, -pow]),
          rat(coef.d, coef.n)
        );
        return polyMul(left, inverse);
      }
      const quotient: Poly = new Map();
      addTerm(
        quotient,
        [[`${OPAQUE_PREFIX}(${formatPoly(left)})/(${formatPoly(right)})`, 1]],
        rat(1n)
      );
      return quotient;
    }
  }
}

function canonicalSym(s: Sym, scope: Scope): Poly | null {
  if (typeof s === "number") return Number.isSafeInteger(s) ? constant(rat(BigInt(s))) : null;
  const parsed = readDimExpr(s);
  if (!parsed || s.slice(parsed.end).trim()) return null;
  return canonical(parsed.node, scope);
}

export type DimComparison = "same" | "different" | "undecided";

/**
 * Whether two extents are equal for every binding of their symbols.
 *
 * `context.opaque` names symbols that stand for unknown quantities rather than
 * free parameters: an extent whose expression was lost along the way still has
 * a value, but not one this module can reason about.
 */
export function compareDims(a: Sym, b: Sym, context: DimContext = {}): DimComparison {
  const scope: Scope = { ...context, expanding: new Set(), cache: context.cache ?? dimCache(), work: { left: MAX_WORK } };
  const left = canonicalSym(a, scope);
  const right = canonicalSym(b, scope);
  if (!left || !right) return "undecided";
  const difference = polyAdd(left, right, -1n);
  if (!difference.size) return "same";
  for (const { mono } of difference.values())
    if (mono.some(([name]) => name.startsWith(OPAQUE_PREFIX))) return "undecided";
  return "different";
}

/** Product of extents as one expression the dimension grammar reads back. */
export function symProduct(syms: readonly Sym[]): Sym {
  if (!syms.length) return 1;
  if (syms.every((s) => typeof s === "number"))
    return (syms as number[]).reduce((a, b) => a * b, 1);
  return syms.map((s) => (typeof s === "string" && /[+-]/.test(s) ? `(${s})` : String(s))).join("*");
}

/** Sum of extents as one expression the dimension grammar reads back. */
export function symSum(syms: readonly Sym[]): Sym {
  if (!syms.length) return 0;
  if (syms.every((s) => typeof s === "number"))
    return (syms as number[]).reduce((a, b) => a + b, 0);
  return syms.map(String).join("+");
}
