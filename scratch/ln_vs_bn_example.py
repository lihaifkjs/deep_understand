import torch
import torch.nn as nn

torch.manual_seed(0)
ln = nn.LayerNorm(3, elementwise_affine=False)
bn = nn.BatchNorm1d(3, affine=False)


def L(x):                       # x: [1, T, 3]
    return ln(x)[0]


def B(x):
    return bn(x.permute(0, 2, 1)).permute(0, 2, 1)[0]


def fmt(v):
    return "[" + ", ".join(f"{t:+.3f}" for t in v.tolist()) + "]"


def case(tag, x):
    y_l, y_b = L(x), B(x)
    print(f"\n{tag}")
    for i in range(x.shape[1]):
        print(f"  pos{i} 输入{fmt(x[0,i])}  LN{fmt(y_l[i])}  BN{fmt(y_b[i])}")
    return y_l, y_b


x0 = torch.tensor([[[1.0, 2.0, 3.0],
                    [5.0, 1.0, 0.0],
                    [1.0, 2.0, 3.0]]])

print("=" * 70)
print("基准：我爱我，其中 我=[1,2,3]  爱=[5,1,0]")
yl0, yb0 = case("基准", x0)

print("\n" + "=" * 70)
print("实验 1：只改中间的词 [5,1,0] -> [0,1,5]，看 pos0 的'我'")
x1 = x0.clone(); x1[0, 1] = torch.tensor([0.0, 1.0, 5.0])
yl1, yb1 = case("改中间词后", x1)
print(f"  pos0 的 LN: {fmt(yl0[0])} -> {fmt(yl1[0])}   diff={(yl1[0]-yl0[0]).abs().max():.3f}")
print(f"  pos0 的 BN: {fmt(yb0[0])} -> {fmt(yb1[0])}   diff={(yb1[0]-yb0[0]).abs().max():.3f}")
for k in range(3):
    print(f"    通道{k}: LN变化={(yl1[0,k]-yl0[0,k]).abs().item():.3f}  "
          f"BN变化={(yb1[0,k]-yb0[0,k]).abs().item():.3f}")

print("\n" + "=" * 70)
print("实验 2：改'我'自己 [1,2,3] -> [1.5,2,3]，看 pos1(爱) 受不受影响")
x2 = x0.clone(); x2[0, 0] = torch.tensor([1.5, 2.0, 3.0]); x2[0, 2] = torch.tensor([1.5, 2.0, 3.0])
yl2, yb2 = case("改我之后", x2)
print(f"  pos1 的 LN diff = {(yl2[1]-yl0[1]).abs().max():.3f}")
print(f"  pos1 的 BN diff = {(yb2[1]-yb0[1]).abs().max():.3f}   <- 连锁")

print("\n" + "=" * 70)
print("实验 3：'我'整行 +10（[11,12,13]），其他不动")
x3 = x0.clone(); x3[0, 0] = x0[0, 0] + 10; x3[0, 2] = x0[0, 2] + 10
yl3, yb3 = case("平移后", x3)
print(f"  pos0 LN diff = {(yl3[0]-yl0[0]).abs().max():.3f}   <- 平移不变")
print(f"  pos0 BN diff = {(yb3[0]-yb0[0]).abs().max():.3f}")

print("\n" + "=" * 70)
print("实验 4：'我'整行 ×3，其他不动")
x4 = x0.clone(); x4[0, 0] = x0[0, 0] * 3; x4[0, 2] = x0[0, 2] * 3
yl4, yb4 = case("放大后", x4)
print(f"  pos0 LN diff = {(yl4[0]-yl0[0]).abs().max():.3f}   <- 缩放不变")
print(f"  pos0 BN diff = {(yb4[0]-yb0[0]).abs().max():.3f}")

print("\n" + "=" * 70)
print("实验 5：整个 batch 所有位置同通道 ×3")
x5 = x0 * 3
yl5, yb5 = L(x5), B(x5)
print(f"  LN pos0 diff = {(yl5[0]-yl0[0]).abs().max():.3f}")
print(f"  BN pos0 diff = {(yb5[0]-yb0[0]).abs().max():.3f}   <- 全局缩放不变")
