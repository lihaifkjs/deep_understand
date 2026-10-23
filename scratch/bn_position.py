import torch
import torch.nn as nn

torch.manual_seed(0)
d = 8
wo = torch.randn(d)          # "我"
ai = torch.randn(d)          # "爱"

bn = nn.BatchNorm1d(d, affine=False)   # 对 [B, d, T] 的 T 维做统计
ln = nn.LayerNorm(d, elementwise_affine=False)


def bn_out(seq):             # seq: [B, T, d] -> 当成 [B, d, T] 喂给 BN1d
    return bn(seq.permute(0, 2, 1)).permute(0, 2, 1)


x  = torch.stack([wo, ai,    wo])[None]        # "我 爱 我"
x2 = torch.stack([wo, torch.randn(d), wo])[None]  # 只改中间的词

y, y2 = bn_out(x), bn_out(x2)
print("BN  : 句内两个我 max|diff| = %.3e" % (y[0, 0] - y[0, 2]).abs().max())
print("BN  : 换掉中间词后 pos0 我 变化 = %.4f   <- 泄漏" % (y2[0, 0] - y[0, 0]).abs().max())

yl, yl2 = ln(x), ln(x2)
print("LN  : 句内两个我 max|diff| = %.3e" % (yl[0, 0] - yl[0, 2]).abs().max())
print("LN  : 换掉中间词后 pos0 我 变化 = %.3e" % (yl2[0, 0] - yl[0, 0]).abs().max())

# batch 里混进别的句子，同一个"我"也会变
other = torch.stack([torch.randn(d), torch.randn(d), torch.randn(d)])[None]
y_batch = bn_out(torch.cat([x, other], 0))
print("BN  : batch 里多一句别的，pos0 我 变化 = %.4f" % (y_batch[0, 0] - y[0, 0]).abs().max())
yl_batch = ln(torch.cat([x, other], 0))
print("LN  : batch 里多一句别的，pos0 我 变化 = %.3e" % (yl_batch[0, 0] - yl[0, 0]).abs().max())

# 训练时整句一起算统计量 -> 位置 0 的表示里混入了未来 token 的信息
prefix = torch.stack([wo, ai])[None]              # 只看前两个词
print("\n因果性检查：只喂 [我, 爱] 时 pos0 的表示")
y_pre = bn_out(prefix)
print("   与原整句相比 max|diff| = %.4f   <- 未来的'我'影响了 pos0" % (y_pre[0, 0] - y[0, 0]).abs().max())
yl_pre = ln(prefix)
print("   LN 对照 max|diff| = %.3e" % (yl_pre[0, 0] - yl[0, 0]).abs().max())
