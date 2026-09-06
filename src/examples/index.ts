type Example = {
  name: string;
  dsl: string;
  /** initial selection: tensor name + box as [lo, hi] pairs */
  defaultSelection?: { tensor: string; box: [number, number][] };
};

export const EXAMPLES: Example[] = [
  {
    name: "Plain GEMM",
    dsl: `M = 256
N = 256
K = 512

A = Tensor(M, K, dtype=fp16)
B = Tensor(K, N, dtype=fp16)

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

A = Tensor(N, N, dtype=fp16)

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

X = Tensor(batch=B, seq=S, emb=H*D, dtype=fp16)
Wq = Parameter(emb=H*D, proj=H*D, dtype=fp16)
Wk = Parameter(emb=H*D, proj=H*D, dtype=fp16)
Wv = Parameter(emb=H*D, proj=H*D, dtype=fp16)
Wo = Parameter(emb=H*D, out=H*D, dtype=fp16)

Qp = einsum("bse,ef->bsf", X, Wq)
Kp = einsum("bse,ef->bsf", X, Wk)
Vp = einsum("bse,ef->bsf", X, Wv)
Q4 = reshape(Qp, shape=[B, S, H, D])
K4 = reshape(Kp, shape=[B, S, H, D])
V4 = reshape(Vp, shape=[B, S, H, D])
Qh = transpose(Q4, perm=[0, 2, 1, 3])
Kh = transpose(K4, perm=[0, 2, 1, 3])
Vh = transpose(V4, perm=[0, 2, 1, 3])
Scores = einsum("bhqd,bhkd->bhqk", Qh, Kh)
P = softmax(Scores, axis=-1)
Z = einsum("bhqk,bhkd->bhqd", P, Vh)
Zt = transpose(Z, perm=[0, 2, 1, 3])
Zm = reshape(Zt, shape=[B, S, H*D])
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

Kc = Tensor(batch=B, head=H, kv=P, dim=D, dtype=fp16)
Vc = Tensor(batch=B, head=H, kv=P, dim=D, dtype=fp16)
X = Tensor(batch=B, seq=T, emb=H*D, dtype=fp16)
Wq = Parameter(emb=H*D, proj=H*D, dtype=fp16)
Wk = Parameter(emb=H*D, proj=H*D, dtype=fp16)
Wv = Parameter(emb=H*D, proj=H*D, dtype=fp16)
Bq = Parameter(proj=H*D, dtype=fp16)
Mk = Tensor(q=T, k=P+T, dtype=fp16)

Qp = einsum("bse,ef->bsf", X, Wq)
Qb = add(Qp, Bq)
Kp = einsum("bse,ef->bsf", X, Wk)
Vp = einsum("bse,ef->bsf", X, Wv)
Q4 = reshape(Qb, shape=[B, T, H, D])
K4 = reshape(Kp, shape=[B, T, H, D])
V4 = reshape(Vp, shape=[B, T, H, D])
Qh = transpose(Q4, perm=[0, 2, 1, 3])
Kh = transpose(K4, perm=[0, 2, 1, 3])
Vh = transpose(V4, perm=[0, 2, 1, 3])
Kf = concat(Kc, Kh, axis=2)
Vf = concat(Vc, Vh, axis=2)
Sc = einsum("bhqd,bhkd->bhqk", Qh, Kf)
Sm = add(Sc, Mk)
Pr = softmax(Sm, axis=-1)
Z  = einsum("bhqk,bhkd->bhqd", Pr, Vf)
Zt = transpose(Z, perm=[0, 2, 1, 3])
Out = reshape(Zt, shape=[B, T, H*D])
`,
    defaultSelection: { tensor: "Out", box: [[0, 1], [31, 32], [0, 128]] },
  },
  {
    name: "Conv2d 3x3 stride 2 (stacked)",
    dsl: `N = 1
C = 3
F1 = 8
F2 = 16
H = 64
W = 64

X = Tensor(N, C, H, W, dtype=fp16)
W1 = Parameter(F1, C, 3, 3, dtype=fp16)
W2 = Parameter(F2, F1, 3, 3, dtype=fp16)

Y1 = conv(X, W1, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
Y2 = conv(Y1, W2, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
`,
    defaultSelection: { tensor: "Y2", box: [[0, 1], [0, 1], [7, 9], [7, 9]] },
  },
  {
    name: "Reshape trap",
    dsl: `X = Tensor(4, 4, dtype=fp32)

F = reshape(X, shape=[16])
Y = reshape(F, shape=[2, 8])
`,
    defaultSelection: { tensor: "F", box: [[6, 10]] },
  },
  {
    name: "Layernorm + residual",
    dsl: `S = 64
E = 64

X = Tensor(S, E, dtype=fp16)
W = Parameter(E, dtype=fp16)
Bb = Parameter(E, dtype=fp16)

H = layernorm(X, W, Bb, axes=[-1])
Y = add(H, X)
`,
    defaultSelection: { tensor: "Y", box: [[10, 11], [20, 24]] },
  },
  {
    name: "Cumsum",
    dsl: `S = 48

X = Tensor(S, dtype=fp32)

Y = cumsum(X, axis=0, reverse=false)
Z = cumsum(Y, axis=0, reverse=false)
`,
    defaultSelection: { tensor: "Z", box: [[20, 24]] },
  },
];
