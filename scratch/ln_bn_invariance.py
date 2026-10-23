import torch
import torch.nn as nn

ln = nn.LayerNorm(3, elementwise_affine=False)
bn = nn.BatchNorm1d(3, affine=False)

wo = torch.tensor([1.0, 2.0, 3.0])   # 我
ai = torch.tensor([5.0, 1.0, 0.0])   # 爱
hi = torch.tensor([0.0, 4.0, 2.0])   # 恨

# batch 里两句：我爱我 / 我恨我
x0 = torch.stack([torch.stack([wo, ai, wo]), torch.stack([wo, hi, wo])])


def L(x):
    return ln(x)


def B(x):
    return bn(x.permute(0, 2, 1)).permute(0, 2, 1)


def fmt(v):
    return "[" + ", ".join(f"{t:+.4f}" for t in v.tolist()) + "]"


yl, yb = L(x0), B(x0)
print("基准 LN:")
print("  句0 我(pos0) =", fmt(yl[0, 0]), " 句1 我(pos0) =", fmt(yl[1, 0]))
print("基准 BN:")
print("  句0 我(pos0) =", fmt(yb[0, 0]), " 句1 我(pos0) =", fmt(yb[1, 0]))
print("  句0 爱(pos1) =", fmt(yb[0, 1]), " 句1 恨(pos1) =", fmt(yb[1, 1]))

print("\n" + "=" * 72)
print("实验 A：只改句1 的'恨' [0,4,2] -> [0,4,6]，句0 一个字都没动")
x1 = x0.clone(); x1[1, 1] = torch.tensor([0.0, 4.0, 6.0])
yl1, yb1 = L(x1), B(x1)
print("  句0 我 LN:", fmt(yl[0, 0]), "->", fmt(yl1[0, 0]))
print("  句0 我 BN:", fmt(yb[0, 0]), "->", fmt(yb1[0, 0]))
for k in range(3):
    print(f"    维度{k}: LN变化={(yl1[0,0,k]-yl[0,0,k]).abs():.4f}   "
          f"BN变化={(yb1[0,0,k]-yb[0,0,k]).abs():.4f}")

print("\n" + "=" * 72)
print("实验 B：只改句0 自己的'我' [1,2,3] -> [1,3,5]")
x2 = x0.clone()
x2[0, 0] = torch.tensor([1.0, 3.0, 5.0]); x2[0, 2] = torch.tensor([1.0, 3.0, 5.0])
yl2, yb2 = L(x2), B(x2)
print("  句0 我 LN:", fmt(yl[0, 0]), "->", fmt(yl2[0, 0]))
print("  句1 我 LN:", fmt(yl[1, 0]), "->", fmt(yl2[1, 0]), " <- 不受影响")
print("  句0 我 BN:", fmt(yb[0, 0]), "->", fmt(yb2[0, 0]))
print("  句1 我 BN:", fmt(yb[1, 0]), "->", fmt(yb2[1, 0]), " <- 被连锁改动")
print("  句1 恨 BN:", fmt(yb[1, 1]), "->", fmt(yb2[1, 1]))

print("\n" + "=" * 72)
print("实验 B2：改句0 自己的'我' [1,2,3] -> [2,1,4]（不是缩放+平移）")
x2b = x0.clone()
x2b[0, 0] = torch.tensor([2.0, 1.0, 4.0]); x2b[0, 2] = torch.tensor([2.0, 1.0, 4.0])
yl2b, yb2b = L(x2b), B(x2b)
print("  句0 我 LN:", fmt(yl[0, 0]), "->", fmt(yl2b[0, 0]), " <- 自己变了")
print("  句1 我 LN:", fmt(yl[1, 0]), "->", fmt(yl2b[1, 0]), " <- 不受影响")
print("  句0 我 BN:", fmt(yb[0, 0]), "->", fmt(yb2b[0, 0]))
print("  句1 我 BN:", fmt(yb[1, 0]), "->", fmt(yb2b[1, 0]), " <- 连锁")
print("  句1 恨 BN:", fmt(yb[1, 1]), "->", fmt(yb2b[1, 1]), " <- 连锁")

print("\n" + "=" * 72)
print("不变性测试：句0 的'我'整行 +10（其他不动）")
x3 = x0.clone(); x3[0, 0] = wo + 10; x3[0, 2] = wo + 10
yl3, yb3 = L(x3), B(x3)
print(f"  LN 变化 = {(yl3[0,0]-yl[0,0]).abs().max():.4f}   <- 平移不变")
print(f"  BN 变化 = {(yb3[0,0]-yb[0,0]).abs().max():.4f}")

print("\n不变性测试：整个 batch 所有 token 全部 +10")
yl4, yb4 = L(x0 + 10), B(x0 + 10)
print(f"  LN 变化 = {(yl4[0,0]-yl[0,0]).abs().max():.4f}   <- 每行各加常数，不变")
print(f"  BN 变化 = {(yb4[0,0]-yb[0,0]).abs().max():.4f}   <- 全部平移，不变")

print("\n不变性测试：整个 batch 所有 token 全部 ×3")
yl5, yb5 = L(x0 * 3), B(x0 * 3)
print(f"  LN 变化 = {(yl5[0,0]-yl[0,0]).abs().max():.4f}")
print(f"  BN 变化 = {(yb5[0,0]-yb[0,0]).abs().max():.4f}")
