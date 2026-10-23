# -*- coding: utf-8 -*-
"""演示：纯 self-attention 对 token 顺序是置换等变的。"""
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


d = 4
Wq, Wk, Wv = rng.normal(size=(3, d, d))   # 三个投影矩阵：与位置无关


def attn(X, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(d)               # [n, n] 注意力分数
    if causal:
        n = X.shape[0]
        S = np.where(np.tril(np.ones((n, n), bool)), S, -np.inf)
    A = softmax(S)                         # 逐行归一化
    return A @ V, A


E = rng.normal(size=(3, d))                # 词嵌入（不含任何位置信息）
w, d_, y = E[0], E[1], E[2]               # 我 / 打 / 你

X1 = np.stack([w, d_, y])                  # 我 打 你
X2 = np.stack([y, d_, w])                  # 你 打 我
perm = np.array([2, 1, 0])                 # X2[i] == X1[perm[i]]

# ---- 1. 双向 attention：输出只是被同样地置换 ----
O1, A1 = attn(X1)
O2, A2 = attn(X2)
print("=== 双向 attention ===")
print("max|O2 - O1[perm]| =", np.abs(O2 - O1[perm]).max())
print("A1 =\n", A1)
print("A2 =\n", A2)
print("A2 == A1[perm][:, perm] ?", np.allclose(A2, A1[np.ix_(perm, perm)]))

# ---- 2. 后面接任何逐位置运算 + 池化，结果完全一样 ----
Wf, bf = rng.normal(size=(d, 8)), rng.normal(size=8)
f = lambda O: np.maximum(O @ Wf + bf, 0)   # 逐位置 FFN

print("\n=== 池化后（模型最终读到的）===")
print("mean(O1) =", O1.mean(0))
print("mean(O2) =", O2.mean(0))
print("mean(FFN(O1)) =", f(O1).mean(0))
print("mean(FFN(O2)) =", f(O2).mean(0))

# ---- 3. 因果掩码会打破它 ----
O1c, _ = attn(X1, causal=True)
O2c, _ = attn(X2, causal=True)
print("\n=== 因果掩码 attention ===")
print("max|O2c - O1c[perm]| =", np.abs(O2c - O1c[perm]).max())
