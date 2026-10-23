# -*- coding: utf-8 -*-
"""用显式置换矩阵把整个证明跑一遍。"""
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
d = 4
Wq, Wk, Wv = rng.normal(size=(3, d, d))
E = rng.normal(size=(3, d))


def attn(X):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(d)
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    return A @ V, A


X1 = E[[0, 1, 2]]

# ---- 你要的那个 P：交换首尾 ----
P = np.array([[0., 0., 1.],
              [0., 1., 0.],
              [1., 0., 0.]])

print("=== X2 = P @ X1 是否等于 E[[2,1,0]] ===")
print("P @ X1 =\n", P @ X1)
print("E[[2,1,0]] =\n", E[[2, 1, 0]])
print("allclose:", np.allclose(P @ X1, E[[2, 1, 0]]))

print("\n=== P 是正交的，所以 P^T P = I（两个 P 就靠这个抵消）===")
print("P^T @ P =\n", P.T @ P)
print("P^T == P^-1 :", np.allclose(P.T, np.linalg.inv(P)))

# ---- 代入 attn ----
X1 = E[[0, 1, 2]]
X2 = P @ X1
O1, A1 = attn(X1)
O2, A2 = attn(X2)

print("\n=== 三层各自成立 ===")
print("max|X2 - P@X1|           =", np.abs(X2 - P @ X1).max())
print("max|S2 - P@S1@P^T|       =", np.abs(
    (X2 @ Wq) @ (X2 @ Wk).T - P @ ((X1 @ Wq) @ (X1 @ Wk).T) @ P.T).max() / np.sqrt(d))
print("max|A2 - P@A1@P^T|       =", np.abs(A2 - P @ A1 @ P.T).max())
print("max|O2 - P@O1|           =", np.abs(O2 - P @ O1).max())

# ---- 换一个不是对合的置换（轮换），排除"P 恰好对称"的侥幸 ----
Pc = np.array([[0., 1., 0.],
               [0., 0., 1.],
               [1., 0., 0.]])
print("\n=== 轮换置换 Pc（Pc != Pc^T，不是对合）===")
print("Pc == Pc^T ?", np.array_equal(Pc, Pc.T))
Oc, Ac = attn(Pc @ X1)
print("max|Oc - Pc@O1| =", np.abs(Oc - Pc @ O1).max())
print("max|Ac - Pc@A1@Pc^T| =", np.abs(Ac - Pc @ A1 @ Pc.T).max())
