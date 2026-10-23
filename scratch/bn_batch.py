import torch
import torch.nn as nn

torch.manual_seed(0)
d = 8
wo, ai = torch.randn(d), torch.randn(d)

bn = nn.BatchNorm1d(d, affine=False)
ln = nn.LayerNorm(d, elementwise_affine=False)


def out(mod, seqs):
    x = torch.stack(seqs)[None] if seqs[0].dim() == 1 else torch.stack(seqs)  # [B,T,d]
    if isinstance(mod, nn.BatchNorm1d):
        return mod(x.permute(0, 2, 1)).permute(0, 2, 1)
    return mod(x)


s1 = torch.stack([wo, ai, wo])          # 样本1：我 爱 我
s2 = torch.stack([torch.randn(d) for _ in range(3)])   # 样本2：随机一句话
s3 = torch.stack([torch.randn(d) for _ in range(3)])   # 样本3

for name, batch in [("B=1", [s1]), ("B=2", [s1, s2]), ("B=3", [s1, s2, s3])]:
    y = out(bn, batch)
    print(f"BN {name}: 我@0={y[0,0,0]:+.4f}  句内两个我 diff={(y[0,0]-y[0,2]).abs().max():.1e}")

print()
for name, batch in [("B=1", [s1]), ("B=2", [s1, s2]), ("B=3", [s1, s2, s3])]:
    y = out(ln, batch)
    print(f"LN {name}: 我@0={y[0,0,0]:+.4f}  句内两个我 diff={(y[0,0]-y[0,2]).abs().max():.1e}")

# eval 模式：统计量锁死，跨 batch 稳定
bn.eval()
print("\nBN eval 模式（用 running stats）:")
for name, batch in [("B=1", [s1]), ("B=2", [s1, s2]), ("B=3", [s1, s2, s3])]:
    y = out(bn, batch)
    print(f"   {name}: 我@0={y[0,0,0]:+.4f}")
