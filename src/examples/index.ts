export type Example = {
  name: string;
  dsl: string;
  /** initial selection: tensor name + box as [lo, hi] pairs */
  defaultSelection?: { tensor: string; box: [number, number][] };
  /**
   * A worked experiment: building this exact source opens the Plan view on
   * `tiles`, inspecting `task`, with `kept` already kept for comparison as P1
   * and P2. The question and what the figures show are in the source's
   * comments, which name the plans by those labels.
   */
  experiment?: {
    tiles: Record<string, number[]>;
    task: { tensor: string; coord: number[] };
    kept: Record<string, number[]>[];
  };
};

export const EXAMPLES: Example[] = [
  {
    name: "Plain GEMM",
    dsl: `M = 256
N = 256
K = 512

A = Tensor(M, K, dtype=float16)
B = Tensor(K, N, dtype=float16)

C = matmul(A, B)
`,
    defaultSelection: { tensor: "C", box: [[64, 128], [0, 64]] },
  },
  {
    name: "Shared operand (A @ A)",
    dsl: `N = 256

# A is read through both operand slots, so a tile of C needs a row band and a
# column band of A. The bands share a square, and that square is read twice.
# Both bands are reported whole; the shared elements are counted once.

A = Tensor(N, N, dtype=float16)

C = matmul(A, A)
`,
    defaultSelection: { tensor: "C", box: [[64, 128], [32, 96]] },
  },
  {
    name: "Multi-head attention",
    dsl: `B = 1
H = 4
S = 128
D = 32

X = Tensor(batch=B, seq=S, emb=H*D, dtype=float16)
Wq = Parameter(emb=H*D, proj=H*D, dtype=float16)
Wk = Parameter(emb=H*D, proj=H*D, dtype=float16)
Wv = Parameter(emb=H*D, proj=H*D, dtype=float16)
Wo = Parameter(emb=H*D, out=H*D, dtype=float16)

Qp = einsum("bse,ef->bsf", X, Wq)
Kp = einsum("bse,ef->bsf", X, Wk)
Vp = einsum("bse,ef->bsf", X, Wv)
Q4 = reshape(Qp, [B, S, H, D])
K4 = reshape(Kp, [B, S, H, D])
V4 = reshape(Vp, [B, S, H, D])
Qh = permute_dims(Q4, [0, 2, 1, 3])
Kh = permute_dims(K4, [0, 2, 1, 3])
Vh = permute_dims(V4, [0, 2, 1, 3])
Scores = einsum("bhqd,bhkd->bhqk", Qh, Kh)
P = softmax(Scores, axis=-1)
Z = einsum("bhqk,bhkd->bhqd", P, Vh)
Zt = permute_dims(Z, [0, 2, 1, 3])
Zm = reshape(Zt, [B, S, H*D])
Out = einsum("bse,ef->bsf", Zm, Wo)
`,
    defaultSelection: { tensor: "Out", box: [[0, 1], [17, 18], [0, 128]] },
  },
  {
    name: "KV-cache decode step",
    dsl: `B = 1
H = 4
P = 96
T = 32
D = 32

# One decode step: T new tokens attend over a P-token cache and themselves.
# Concatenating the cache is why a single new token's output depends on the
# whole of it, and the mask and bias arrive by broadcast rather than as
# full-sized tensors.

Kc = Tensor(batch=B, head=H, kv=P, dim=D, dtype=float16)
Vc = Tensor(batch=B, head=H, kv=P, dim=D, dtype=float16)
X = Tensor(batch=B, seq=T, emb=H*D, dtype=float16)
Wq = Parameter(emb=H*D, proj=H*D, dtype=float16)
Wk = Parameter(emb=H*D, proj=H*D, dtype=float16)
Wv = Parameter(emb=H*D, proj=H*D, dtype=float16)
Bq = Parameter(proj=H*D, dtype=float16)
Mk = Tensor(q=T, k=P+T, dtype=float16)

Qp = einsum("bse,ef->bsf", X, Wq)
Qb = add(Qp, Bq)
Kp = einsum("bse,ef->bsf", X, Wk)
Vp = einsum("bse,ef->bsf", X, Wv)
Q4 = reshape(Qb, [B, T, H, D])
K4 = reshape(Kp, [B, T, H, D])
V4 = reshape(Vp, [B, T, H, D])
Qh = permute_dims(Q4, [0, 2, 1, 3])
Kh = permute_dims(K4, [0, 2, 1, 3])
Vh = permute_dims(V4, [0, 2, 1, 3])
Kf = concat([Kc, Kh], axis=2)
Vf = concat([Vc, Vh], axis=2)
Sc = einsum("bhqd,bhkd->bhqk", Qh, Kf)
Sm = add(Sc, Mk)
Pr = softmax(Sm, axis=-1)
Z  = einsum("bhqk,bhkd->bhqd", Pr, Vf)
Zt = permute_dims(Z, [0, 2, 1, 3])
Out = reshape(Zt, [B, T, H*D])
`,
    defaultSelection: { tensor: "Out", box: [[0, 1], [31, 32], [0, 128]] },
  },
  {
    name: "Grouped-query decode step",
    dsl: `B = 1
KVH = 2
G = 4
P = 96
T = 4
D = 32

# Grouped-query attention: KVH*G query heads share KVH key/value heads, G query
# heads to each one. broadcast_to is where that sharing is written down - the cache is
# stored once per KV head and read by G query heads. So a tile of Out covering
# one query head pulls one KV head band of the cache, and the four heads in
# columns 0..4*D pull the same band rather than four of them.
#
# The three projections are one fused weight cut by split, the way a served
# model stores them: a tile of Out reaches three column bands of the same Wqkv
# rather than bands of three separate tensors.

Kc = Tensor(batch=B, kv_head=KVH, kv=P, dim=D, dtype=float16)
Vc = Tensor(batch=B, kv_head=KVH, kv=P, dim=D, dtype=float16)
X = Tensor(batch=B, seq=T, emb=KVH*G*D, dtype=float16)
Wqkv = Parameter(emb=KVH*G*D, qkv=KVH*G*D+2*KVH*D, dtype=float16)

QKV = einsum("bse,ef->bsf", X, Wqkv)
Qp, Kp, Vp = split(QKV, [KVH*G*D, KVH*D, KVH*D], axis=-1)
Q4 = reshape(Qp, [B, T, KVH*G, D])
K4 = reshape(Kp, [B, T, KVH, D])
V4 = reshape(Vp, [B, T, KVH, D])
Qh = permute_dims(Q4, [0, 2, 1, 3])
Kh = permute_dims(K4, [0, 2, 1, 3])
Vh = permute_dims(V4, [0, 2, 1, 3])
Kf = concat([Kc, Kh], axis=2)
Vf = concat([Vc, Vh], axis=2)
K5 = reshape(Kf, [B, KVH, 1, P+T, D])
V5 = reshape(Vf, [B, KVH, 1, P+T, D])
Kg = broadcast_to(K5, [B, KVH, G, P+T, D])
Vg = broadcast_to(V5, [B, KVH, G, P+T, D])
Kx = reshape(Kg, [B, KVH*G, P+T, D])
Vx = reshape(Vg, [B, KVH*G, P+T, D])
Sc = einsum("bhqd,bhkd->bhqk", Qh, Kx)
Pr = softmax(Sc, axis=-1)
Z = einsum("bhqk,bhkd->bhqd", Pr, Vx)
Zt = permute_dims(Z, [0, 2, 1, 3])
Out = reshape(Zt, [B, T, KVH*G*D])
`,
    defaultSelection: { tensor: "Out", box: [[0, 1], [3, 4], [0, 32]] },
  },
  {
    name: "SwiGLU feed-forward",
    dsl: `S = 128
E = 256
F = 704

# The feed-forward half of a transformer block. Xn fans out to two projections
# and the branches rejoin at the elementwise multiply, so a tile of Y reads the
# same band of Xn twice: once for the gate, once for the value it scales. The
# norm is why that band is whole rows of X; the residual adds the tile's own
# columns of X, which those rows already cover.

X = Tensor(seq=S, emb=E, dtype=float16)
Wn = Parameter(emb=E, dtype=float16)
Wg = Parameter(emb=E, ff=F, dtype=float16)
Wu = Parameter(emb=E, ff=F, dtype=float16)
Wd = Parameter(ff=F, emb=E, dtype=float16)

Xn = rms_norm(X, Wn)
G = matmul(Xn, Wg)
U = matmul(Xn, Wu)
Ga = silu(G)
Hd = multiply(Ga, U)
Y = matmul(Hd, Wd)
Out = add(Y, X)
`,
    defaultSelection: { tensor: "Out", box: [[16, 24], [0, 64]] },
  },
  {
    name: "MoE expert dispatch",
    dsl: `T = 8
E = 128
F = 256
N = 8
K = 2

# A mixture-of-experts step. The expert weights are one stacked tensor and the
# routing picks slabs out of it, which is the only place in these examples where
# a dependency is not decidable from the graph: take's indices are data, so
# every expert is reported as needed and the cone is marked inexact.
#
# Declaring the choice makes it exact - add index_values=[3, 5] to both takes
# and the same tile narrows to those two slabs. The router's own top-k cannot be
# written here (no argmax), so Idx arrives as an input and Pr is left as an
# output of its own: what the graph can state is that the choice is data.

X = Tensor(seq=T, emb=E, dtype=float16)
Wr = Parameter(emb=E, expert=N, dtype=float16)
W1 = Parameter(expert=N, emb=E, ff=F, dtype=float16)
W2 = Parameter(expert=N, ff=F, emb=E, dtype=float16)
Idx = Tensor(chosen=K, dtype=int32)
Gate = Tensor(seq=T, chosen=K, dtype=float16)

Sc = einsum("se,en->sn", X, Wr)
Pr = softmax(Sc, axis=-1)
E1 = take(W1, Idx, axis=0)
E2 = take(W2, Idx, axis=0)
Hd = einsum("se,kef->skf", X, E1)
Ac = silu(Hd)
Yk = einsum("skf,kfe->ske", Ac, E2)
Gw = reshape(Gate, [T, K, 1])
Yg = multiply(Yk, Gw)
Y = sum(Yg, axis=1, keepdims=false)
`,
    defaultSelection: { tensor: "Y", box: [[3, 4], [0, 64]] },
  },
  {
    name: "Conv2d 3x3 stride 2 (stacked)",
    dsl: `N = 1
C = 3
F1 = 8
F2 = 16
H = 64
W = 64

X = Tensor(N, C, H, W, dtype=float16)
W1 = Parameter(F1, C, 3, 3, dtype=float16)
W2 = Parameter(F2, F1, 3, 3, dtype=float16)

Y1 = conv2d(X, W1, stride=2, padding=1)
Y2 = conv2d(Y1, W2, stride=2, padding=1)
`,
    defaultSelection: { tensor: "Y2", box: [[0, 1], [0, 1], [7, 9], [7, 9]] },
  },
  {
    name: "Reshape trap",
    dsl: `X = Tensor(4, 4, dtype=float32)

F = reshape(X, [16])
Y = reshape(F, [2, 8])
`,
    defaultSelection: { tensor: "F", box: [[6, 10]] },
  },
  {
    name: "Layernorm + residual",
    dsl: `S = 64
E = 64

X = Tensor(S, E, dtype=float16)
W = Parameter(E, dtype=float16)
Bb = Parameter(E, dtype=float16)

H = layer_norm(X, W, Bb)
Y = add(H, X)
`,
    defaultSelection: { tensor: "Y", box: [[10, 11], [20, 24]] },
  },
  {
    name: "Cumulative sum",
    dsl: `S = 48

X = Tensor(S, dtype=float32)

Y = cumulative_sum(X, axis=0)
Z = cumulative_sum(Y, axis=0)
`,
    defaultSelection: { tensor: "Z", box: [[20, 24]] },
  },
  {
    name: "Batched matmul",
    dsl: `B = 2
H = 4
M = 200
K = 64
N = 128

# Four-dimensional operands: the card draws the last two axes, and batch and
# head are hidden. The Axes table in the inspector names every axis of the
# tile; M = 200 does not divide by 64, so the last row tile is 8 rows.

A = Tensor(batch=B, head=H, m=M, k=K, dtype=float16)
W = Tensor(batch=B, head=H, k=K, n=N, dtype=float16)

C = matmul(A, W)
`,
    defaultSelection: { tensor: "C", box: [[0, 1], [0, 2], [64, 128], [0, 64]] },
  },
  {
    name: "Experiment: tile size and reuse",
    dsl: `M = 256
N = 256
K = 512

# Question: how does a larger tile increase operand reuse?
#
# Three tilings of C are compared in the Plan view: P1 is 32x32, P2 is 64x64,
# and the current plan is 128x128. FLOPs are 67.11M in all three. Bytes read
# halve each time the tile side doubles - 4 MB, 2 MB, 1 MB - because a task
# reads a band of A as tall as its tile and a band of B as wide, and uses each
# element of A once per column of its tile. "Operand reuse in this task" shows
# that count: 128 uses per element here, 32 under P1.

A = Tensor(M, K, dtype=float16)
B = Tensor(K, N, dtype=float16)

C = matmul(A, B)
`,
    defaultSelection: { tensor: "C", box: [[128, 256], [0, 128]] },
    experiment: {
      tiles: { C: [128, 128] },
      task: { tensor: "C", coord: [1, 0] },
      kept: [{ C: [32, 32] }, { C: [64, 64] }],
    },
  },
  {
    name: "Experiment: fusion and recomputation",
    dsl: `# Question: when does fusion introduce recomputation?
#
# P1 writes H in 64x128 tiles and Y reads them: 24 tasks, 32 dependencies,
# 320 KB written. P2 fuses H into 64x64 tiles of Y: each task computes the
# 64 rows of H it needs, and both Y tasks in a row band compute the same rows,
# so 33.55M of its 100.66M FLOPs are recomputed. The current plan fuses H into
# full-width 64x128 tiles of Y: one task per row band, nothing recomputed, and
# 64 KB written. Fusion recomputes when several tasks need the same part of
# the fused tensor; press fuse or materialize beside H to move between them.

X = Tensor(256, 128, dtype=float16)
W1 = Parameter(128, 512, dtype=float16)
W2 = Parameter(512, 128, dtype=float16)

H = matmul(X, W1)
Y = matmul(H, W2)
`,
    defaultSelection: { tensor: "Y", box: [[64, 128], [0, 128]] },
    experiment: {
      tiles: { Y: [64, 128] },
      task: { tensor: "Y", coord: [1, 0] },
      kept: [{ H: [64, 128], Y: [64, 64] }, { Y: [64, 64] }],
    },
  },
  {
    name: "Experiment: splitting a normalised axis",
    dsl: `S = 64
E = 64

# Question: why does splitting the normalised axis repeat statistics?
#
# layer_norm needs a whole row of X for the mean and variance of any element in
# that row. P1 tiles Y in 16x64 row bands with H fused: 33.0K FLOPs, nothing
# recomputed. P2 splits each band into four 16x16 tiles, still fused: each of
# the four computes the same row statistics, so 49.9K of its 82.9K FLOPs are
# recomputed. The current plan keeps the 16x16 tiles of Y but materializes H
# in row bands: nothing is recomputed, and the cost moves to memory - 16 KB
# written instead of 8 KB, and 16 dependencies on H's tasks.

X = Tensor(S, E, dtype=float16)
W = Parameter(E, dtype=float16)
Bb = Parameter(E, dtype=float16)

H = layer_norm(X, W, Bb)
Y = add(H, X)
`,
    defaultSelection: { tensor: "Y", box: [[16, 32], [16, 32]] },
    experiment: {
      tiles: { Y: [16, 16], H: [16, 64] },
      task: { tensor: "Y", coord: [1, 1] },
      kept: [{ Y: [16, 64] }, { Y: [16, 16] }],
    },
  },
];
