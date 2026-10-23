import math

import torch
import torch.nn as nn
import torch.nn.functional as F

d, T = 8, 3
torch.manual_seed(0)


def fmt(v):
    return "[" + ", ".join(f"{x:+.3f}" for x in v.tolist()) + "]"


wo = torch.randn(d)          # "我" 的词向量
ai = torch.randn(d)          # "爱" 的词向量
x = torch.stack([wo, ai, wo])  # "我 爱 我" → [T, d]

ln = nn.LayerNorm(d)

# ---- 1. 直接 norm，无位置信息 ----
y = ln(x)
print("1) 裸 LayerNorm（无位置编码）")
print("   我@0:", fmt(y[0]))
print("   我@2:", fmt(y[2]))
print("   max|diff| =", (y[0] - y[2]).abs().max().item())

# ---- 2. 加正弦位置编码后 ----
pos = torch.arange(T).unsqueeze(1)
freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
pe = torch.zeros(T, d)
pe[:, 0::2] = torch.sin(pos * freq)
pe[:, 1::2] = torch.cos(pos * freq)

y2 = ln(x + pe)
print("\n2) 先加 sin/cos 位置编码，再 LayerNorm")
print("   我@0:", fmt(y2[0]))
print("   我@2:", fmt(y2[2]))
print("   max|diff| =", (y2[0] - y2[2]).abs().max().item())

# ---- 3. 无位置编码，但过一个因果 attention 层 ----
attn = F.scaled_dot_product_attention(x[None], x[None], x[None], is_causal=True)[0]
y3 = ln(x + attn)            # pre-norm 结构的简化版：残差 + LN
print("\n3) 无 PE，但做一次因果 attention 后")
print("   我@0:", fmt(y3[0]))
print("   我@2:", fmt(y3[2]))
print("   max|diff| =", (y3[0] - y3[2]).abs().max().item())
