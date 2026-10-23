import torch
import torch.nn as nn

E = 1e-5
wo = torch.tensor([1.0, 0.0])
ai = torch.tensor([2.0, 0.0])
hi = torch.tensor([3.0, 0.0])
low = torch.tensor([0.5, 0.0])          # 比"我"小的词

ln = nn.LayerNorm(2, eps=E, elementwise_affine=False)
bn = nn.BatchNorm1d(2, eps=E, affine=False)


def run(mod, x):
    if isinstance(mod, nn.BatchNorm1d):
        return mod(x.permute(0, 2, 1)).permute(0, 2, 1)
    return mod(x)


def show(tag, vec):
    print(f"   {tag}: [{vec[0]:+.4f}, {vec[1]:+.4f}]")


print("=== 你给的例子：我[1,0] 爱[2,0] 恨[3,0] ===")
s1 = torch.stack([wo, ai, wo])       # 我 爱 我
s2 = torch.stack([wo, hi, wo])       # 我 恨 我

print("[LN] 逐 token 归一化")
for tag, seq in [("我爱我", s1), ("我恨我", s2)]:
    y = run(ln, seq[None])[0]
    print(f"  {tag}:")
    for i, name in enumerate(["我", "爱" if tag == "我爱我" else "恨", "我"]):
        show(f"pos{i} {name}", y[i])

print("\n[BN] 分开两次 forward，各自 B=1")
for tag, seq in [("我爱我", s1), ("我恨我", s2)]:
    y = run(bn, seq[None])[0]
    show(f"{tag} pos0 我", y[0])
    show(f"{tag} pos2 我", y[2])

print("\n[BN] 合成一个 batch=2")
y = run(bn, torch.stack([s1, s2]))
show("样本1 pos0 我", y[0, 0])
show("样本2 pos0 我", y[1, 0])

print("\n=== 让'恨'落在'我'的另一侧：我[1,0] 爱[2,0] 恨[0.5,0] ===")
s3 = torch.stack([wo, low, wo])
print("[LN]")
for tag, seq in [("我爱我", s1), ("我恨我(0.5)", s3)]:
    y = run(ln, seq[None])[0]
    show(f"{tag} pos0 我", y[0])
print("[BN] 分开两次 forward")
for tag, seq in [("我爱我", s1), ("我恨我(0.5)", s3)]:
    y = run(bn, seq[None])[0]
    show(f"{tag} pos0 我", y[0])
