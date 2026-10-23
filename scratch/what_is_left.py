# -*- coding: utf-8 -*-
"""三个词，检验"没位置编码时，模型到底能读出什么"。"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def softmax(S):
    A = np.exp(S - S.max(1, keepdims=True))
    return A / A.sum(1, keepdims=True)


def attn(X):
    """W=I, d=3"""
    S = X @ X.T / np.sqrt(3)
    A = softmax(S)
    return A, A @ X


WO = np.array([1., 0., 0.])   # 我
DA = np.array([0., 1., 0.])   # 打
NI = np.array([0., 0., 1.])   # 你

X1 = np.stack([WO, DA, NI])   # 我 打 你
X2 = np.stack([WO, NI, DA])   # 我 你 打   <- 只有后两位对调

A1, O1 = attn(X1)
A2, O2 = attn(X2)

print("注意力矩阵 A1 =\n", A1)
print("注意力矩阵 A2 =\n", A2)
print("\n“我打你” O1 =\n", O1)
print("“我你打” O2 =\n", O2)

print("\n=== 第 1 个位置（都是“我”）的输出 ===")
print("O1[0] =", O1[0])
print("O2[0] =", O2[0])
print("最大差 =", np.abs(O1[0] - O2[0]).max(), "  <- 完全相同")

print("\n=== 整体：O2 只是 O1 换了两行吗 ===")
print("O2 == O1[[0,2,1]] ?", np.allclose(O2, O1[[0, 2, 1]]))

print("\n=== 两句的输出行，作为集合是一样的 ===")
print("sorted rows equal:", sorted(map(tuple, O1)) == sorted(map(tuple, O2)))
print("mean(O1) =", O1.mean(0))
print("mean(O2) =", O2.mean(0), " <- 池化后也相同")

print("\n=== 但两个 2 词句子的 O 确实不同（所以能区分）===")
Y1 = np.stack([WO, DA])       # 我打
Y2 = np.stack([DA, WO])       # 打我
S1 = Y1 @ Y1.T / np.sqrt(3)
S2 = Y2 @ Y2.T / np.sqrt(3)
_, Z1 = attn(Y1)
_, Z2 = attn(Y2)
print("“我打” O =\n", Z1)
print("“打我” O =\n", Z2)
print("max|O1 - O2| =", np.abs(Z1 - Z2).max(), " <- 不是 0，确实不同")
