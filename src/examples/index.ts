type Example = {
  name: string;
  dsl: string;
  /** initial selection: tensor name + box as [lo, hi] pairs */
  defaultSelection?: { tensor: string; box: [number, number][] };
};

export const EXAMPLES: Example[] = [
  {
    name: "Plain GEMM",
    dsl: `params M=256 N=256 K=512

input A [M, K] f16
input B [K, N] f16

C = matmul(A, B)
`,
    defaultSelection: { tensor: "C", box: [[64, 128], [0, 64]] },
  },
  {
    name: "Multi-head attention",
    dsl: `params B=1 H=4 S=128 D=32 E=128

input X  [B, S, E] f16
input Wq [E, E] f16
input Wk [E, E] f16
input Wv [E, E] f16
input Wo [E, E] f16

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
Zm = reshape(Zt, shape=[B, S, E])
Out = einsum("bse,ef->bsf", Zm, Wo)
`,
    defaultSelection: { tensor: "Out", box: [[0, 1], [17, 18], [0, 128]] },
  },
  {
    name: "Conv2d 3x3 stride 2 (stacked)",
    dsl: `params N=1 C=3 F1=8 F2=16 H=64 W=64

input X  [N, C, H, W] f16
input W1 [F1, C, 3, 3] f16
input W2 [F2, F1, 3, 3] f16

Y1 = conv(X, W1, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
Y2 = conv(Y1, W2, stride=[2, 2], pads=[[1, 1], [1, 1]], dilation=[1, 1], groups=1)
`,
    defaultSelection: { tensor: "Y2", box: [[0, 1], [0, 1], [7, 9], [7, 9]] },
  },
  {
    name: "Reshape trap",
    dsl: `input X [4, 4] f32

F = reshape(X, shape=[16])
Y = reshape(F, shape=[2, 8])
`,
    defaultSelection: { tensor: "F", box: [[6, 10]] },
  },
  {
    name: "Layernorm + residual",
    dsl: `params S=64 E=64

input X [S, E] f16
input W [E] f16
input Bb [E] f16

H = layernorm(X, W, Bb, axes=[-1])
Y = add(H, X)
`,
    defaultSelection: { tensor: "Y", box: [[10, 11], [20, 24]] },
  },
  {
    name: "Cumsum",
    dsl: `params S=48

input X [S] f32

Y = cumsum(X, axis=0, reverse=false)
Z = cumsum(Y, axis=0, reverse=false)
`,
    defaultSelection: { tensor: "Z", box: [[20, 24]] },
  },
];
