# -*- coding: utf-8 -*-
"""把 multiset 那句话的每一个数字都算出来。"""
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

# 复刻 perm_attn.py 的取数顺序：先 3 个投影矩阵，再取词嵌入表
rng.normal(size=(3, 4, 4))
E = rng.normal(size=(3, 4))

names = ["wo", "da", "ni"]
d = E.shape[1]

print("=== embedding table E (indexed by token id, NOT by position) ===")
for i, nm in enumerate(names):
    print(f"  E[{i}] {nm}: {E[i]}")

ids1 = [0, 1, 2]   # wo da ni
ids2 = [2, 1, 0]   # ni da wo
X1, X2 = E[ids1], E[ids2]

print("\n=== X1 = 'wo da ni'  (ids=[0,1,2]) ===")
for i, t in enumerate(ids1):
    print(f"  row {i}  {names[t]}: {X1[i]}")

print("\n=== X2 = 'ni da wo'  (ids=[2,1,0]) ===")
for i, t in enumerate(ids2):
    print(f"  row {i}  {names[t]}: {X2[i]}")

print("\n=== 逐元素比较：X1 != X2 ===")
print("X1 - X2 =\n", X1 - X2)

print("\n=== 多重集比较：排序后逐个比 ===")
s1 = sorted(map(tuple, X1))
s2 = sorted(map(tuple, X2))
for a, b in zip(s1, s2):
    print(f"  {np.array(a)}  ==  {np.array(b)}   -> {a == b}")
print("multiset(X1) == multiset(X2):", s1 == s2)

print("\n=== 同一个 token 在不同位置拿到的向量 ===")
print("X1[0] (wo 在第 0 位) =", X1[0])
print("X2[2] (wo 在第 2 位) =", X2[2])
print("bitwise identical:", np.array_equal(X1[0], X2[2]))
print("max abs diff      :", np.abs(X1[0] - X2[2]).max())
