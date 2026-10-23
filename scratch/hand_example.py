# -*- coding: utf-8 -*-
"""最小的、能手算的例子：两句“我打”/“打我”，d=2，W=I，词向量 one-hot。"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def softmax(S):
    A = np.exp(S - S.max(1, keepdims=True))
    return A / A.sum(1, keepdims=True)


def attn(X):
    """Wq=Wk=Wv=I，d=2，所以 S = X·Xᵀ/√2，没有任何矩阵乘法。"""
    S = X @ X.T / np.sqrt(2)
    A = softmax(S)
    return A, A @ X


WO = np.array([1.0, 0.0])   # 我
DA = np.array([0.0, 1.0])   # 打

# ================= 不加位置编码 =================
print("########## 不加位置编码 ##########")
X1 = np.stack([WO, DA])     # 我 打
X2 = np.stack([DA, WO])     # 打 我

for name, X in [('"我打"', X1), ('"打我"', X2)]:
    A, O = attn(X)
    print(f"\n{name}  输入 X = {X.tolist()}")
    print("  注意力权重 A =\n", A)
    print("  输出 O =\n", O)

print("\n--- 关键：同一个字“我”最终拿到的向量 ---")
print('  “我打” 里 “我”在第 1 位 -> O =', attn(X1)[1][0])
print('  “打我” 里 “我”在第 2 位 -> O =', attn(X2)[1][1])
print("  两者最大差 =", np.abs(attn(X1)[1][0] - attn(X2)[1][1]).max(), " <- 完全一样")

# ================= 加位置编码 =================
print("\n\n########## 加位置编码 ##########")
# d=2 时 10000^(2i/d)=1，所以 PE(pos) = [sin(pos), cos(pos)]，单位圆上的点
PE = np.stack([np.array([np.sin(p), np.cos(p)]) for p in range(2)])
print("位置牌 PE(第1位) =", PE[0], "   PE(第2位) =", PE[1])
print("  注：这两张牌是固定的，跟句子内容无关")

Y1 = np.stack([WO + PE[0], DA + PE[1]])   # 我 打
Y2 = np.stack([DA + PE[0], WO + PE[1]])   # 打 我

for name, Y in [('"我打"', Y1), ('"打我"', Y2)]:
    A, O = attn(Y)
    print(f"\n{name}  输入 = 词 + 位置牌 = {Y.tolist()}")
    print("  输出 O =\n", O)

print("\n--- 同一个字“我”最终拿到的向量 ---")
print('  “我打” 里 “我”在第 1 位 -> O =', attn(Y1)[1][0])
print('  “打我” 里 “我”在第 2 位 -> O =', attn(Y2)[1][1])
print("  两者最大差 =", np.abs(attn(Y1)[1][0] - attn(Y2)[1][1]).max(), " <- 完全不同")
