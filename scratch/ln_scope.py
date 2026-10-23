import torch
import torch.nn as nn

x = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])   # [1, 3, 2]

print("输入 =", x[0].tolist())
print()

# ---- A. Transformer 的用法：normalized_shape = (d,) = (2,) ----
ln_tok = nn.LayerNorm(2, elementwise_affine=False)
y = ln_tok(x)
print("A) nn.LayerNorm(2)  —— 逐位置")
for i, name in enumerate(["我", "爱", "我"]):
    row = x[0, i]
    mu, var = row.mean(), row.var(unbiased=False)
    print(f"   pos{i} {name}: 输入={row.tolist()}  mu={mu:.4f} var={var:.4f} -> {y[0,i].tolist()}")
print()

# ---- B. 对整个序列归一化：normalized_shape = (T, d) = (3, 2) ----
ln_seq = nn.LayerNorm([3, 2], elementwise_affine=False)
y2 = ln_seq(x)
allv = x[0].reshape(-1)
print("B) nn.LayerNorm([3, 2]) —— 整句 6 个数当一组")
print(f"   所有元素 = {allv.tolist()}")
print(f"   mu={allv.mean():.4f} var={allv.var(unbiased=False):.4f} sd={allv.std(unbiased=False):.4f}")
for i, name in enumerate(["我", "爱", "我"]):
    print(f"   pos{i} {name} -> {y2[0,i].tolist()}")

print()
print("A 里两个我的 diff =", (y[0, 0] - y[0, 2]).abs().max().item())
print("B 里两个我的 diff =", (y2[0, 0] - y2[0, 2]).abs().max().item())
