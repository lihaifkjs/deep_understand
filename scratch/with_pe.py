# -*- coding: utf-8 -*-
"""加了位置编码后，置换等变为什么就破了。"""
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
d = 4
Wq, Wk, Wv = rng.normal(size=(3, d, d))
E = rng.normal(size=(3, d))


def sinusoidal_pe(n, d):
    """原论文写法：偶数维 sin，奇数维 cos。"""
    pos = np.arange(n)[:, None]
    div = 10000.0 ** (np.arange(0, d, 2) / d)
    ang = pos / div
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def attn(X):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(d)
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    return A @ V, A, S


P = np.array([[0., 0., 1.],
              [0., 1., 0.],
              [1., 0., 0.]])
PE = sinusoidal_pe(3, d)
print("PE =\n", PE)

Xe = E[[0, 1, 2]]          # 纯内容：wo da ni

# ---------- (A) 正确：PE 钉在原地，不跟着置换 ----------
XA1 = Xe + PE
XA2 = P @ Xe + PE          # 注意：PE 没有 P 左乘
OA1, AA1, SA1 = attn(XA1)
OA2, AA2, SA2 = attn(XA2)
print("\n=== (A) X2 = P@X1 + PE   （PE 固定）===")
print("max|O2 - P@O1|        =", np.abs(OA2 - P @ OA1).max(), "  <- 不再等变")
print("max|S2 - P@S1@P^T|    =", np.abs(SA2 - P @ SA1 @ P.T).max(), "  <- 交叉项进来了")
print("O1 =\n", OA1)
print("O2 =\n", OA2)
print("P@O1 =\n", P @ OA1)

# ---------- (B) 错误：把 PE 也一起置换 ----------
XB1 = Xe + PE
XB2 = P @ (Xe + PE)        # 等价于 P@X1 + P@PE，整个输入被改名
OB1, _, _ = attn(XB1)
OB2, _, _ = attn(XB2)
print("\n=== (B) X2 = P@(X1+PE)  （PE 跟着置换，等于没加）===")
print("max|O2 - P@O1|        =", np.abs(OB2 - P @ OB1).max(), "  <- 又等变了")

# ---------- (C) 拆开看 S 多出来的交叉项 ----------
Qe, Qp = Xe @ Wq, PE @ Wq
Ke, Kp = Xe @ Wk, PE @ Wk
S_cross = (P @ Qe) @ Kp.T + Qp @ (P @ Ke).T    # 与位置有关的项
print("\n=== (C) S = P·Se·P^T  +  交叉项  +  Qp·Kp^T ===")
print("Qp@Kp^T（纯位置项，完全不随置换变）=\n", Qp @ Kp.T)
