import torch
import torch.nn as nn

torch.manual_seed(0)
d = 8
wo = torch.randn(d)          # 我
ai = torch.randn(d)          # 爱
hi = torch.randn(d)          # 恨

bn = nn.BatchNorm1d(d, affine=False)
ln = nn.LayerNorm(d, elementwise_affine=False)

s1 = torch.stack([wo, ai, wo])   # 我 爱 我
s2 = torch.stack([wo, hi, wo])   # 我 恨 我


def run(mod, x):                 # x: [B, T, d]
    if isinstance(mod, nn.BatchNorm1d):
        return mod(x.permute(0, 2, 1)).permute(0, 2, 1)
    return mod(x)


print("=== 分开两次 forward（每次 B=1）===")
for name, mod in [("LN", ln), ("BN", bn)]:
    y1 = run(mod, s1[None])
    y2 = run(mod, s2[None])
    print(f"{name}: 我@0  batch1={y1[0,0,0]:+.4f}  batch2={y2[0,0,0]:+.4f}  "
          f"diff={(y1[0,0]-y2[0,0]).abs().max():.1e}")

print("\n=== 合成一个 batch=2 forward ===")
for name, mod in [("LN", ln), ("BN", bn)]:
    y = run(mod, torch.stack([s1, s2]))
    print(f"{name}: 我@0  batch1={y[0,0,0]:+.4f}  batch2={y[1,0,0]:+.4f}  "
          f"diff={(y[0,0]-y[1,0]).abs().max():.1e}")

print("\n=== BN：batch=2 时，batch1 的'我'受 batch2 影响吗 ===")
y_a = run(bn, torch.stack([s1, s2]))
s3 = torch.stack([wo, torch.randn(d), wo])   # 把第二句换个词
y_b = run(bn, torch.stack([s1, s3]))
print(f"BN  batch1 的我@0: 配'恨'={y_a[0,0,0]:+.4f}  配别的={y_b[0,0,0]:+.4f}  "
      f"diff={(y_a[0,0]-y_b[0,0]).abs().max():.4f}")
y_c = run(ln, torch.stack([s1, s3]))
print(f"LN  batch1 的我@0: 配'恨'={run(ln, torch.stack([s1, s2]))[0,0,0]:+.4f}  "
      f"配别的={y_c[0,0,0]:+.4f}  diff={(run(ln, torch.stack([s1, s2]))[0,0]-y_c[0,0]).abs().max():.1e}")
