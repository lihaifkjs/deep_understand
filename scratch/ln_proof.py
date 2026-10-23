import sys

import torch

sys.path.insert(0, r"C:\Users\fkjslee\Desktop\deep_understand\demo")
from model import TinyGPT  # noqa: E402

torch.manual_seed(0)
m = TinyGPT(vocab_size=26, d_model=16, n_layer=1, max_len=16)
m.eval()
idx = torch.randint(0, 26, (2, 5))

cap = {}
h = m.blocks[0].ln1.register_forward_hook(
    lambda mod, inp, out: cap.update(i=inp[0].detach(), o=out.detach())
)
with torch.no_grad():
    m(idx)
h.remove()

i, o = cap["i"], cap["o"]
print("ln1 输入 shape:", tuple(i.shape), " (B=2, T=5, C=16)")

print("\n第一个 ln1 输出的每个位置统计量（逐位置归一化则 mean≈0, std≈1）:")
for t in range(5):
    row = o[0, t]
    print(f"   pos{t}: mean={row.mean():+.6f}  std={row.std(unbiased=False):.6f}")

print(f"\n整个输出张量（B*T*C 全部元素）的 mean = {o.mean():+.6f}, std = {o.std(unbiased=False):.6f}")
print("→ 若按整句 T*C 归一化，上面的 pos 行均值不会全是 0")

# 手工逐位置复算，验证和 PyTorch 输出一致
mu = i[0, 0].mean()
var = i[0, 0].var(unbiased=False)
manual = (i[0, 0] - mu) / torch.sqrt(var + 1e-5)
print("\n手工对 pos0 逐位置复算（只用该位置自己的 16 个数）:")
print(f"   mu={mu:+.6f} var={var:.6f}  与 nn.LayerNorm 输出的最大差 = "
      f"{(manual - o[0, 0]).abs().max().item():.2e}")

# 对照：整句归一化会是什么样
whole_mu = i[0].mean()
whole_var = i[0].var(unbiased=False)
whole = (i[0] - whole_mu) / torch.sqrt(whole_var + 1e-5)
print(f"\n若强行按整句（5*16=80 个数）归一化：mu={whole_mu:+.6f} var={whole_var:.6f}")
for t in range(5):
    r = whole[t]
    print(f"   pos{t}: mean={r.mean():+.6f}  std={r.std(unbiased=False):.6f}  ← 非 0 / 非 1")
