"""三张图：为什么 eps（梯度衰减因子）随 sqrt(d) 指数衰减。

打分 s_i = c * z_i，z ~ N(0,1) 标准化。
  c = 1       -> 已除 sqrt(d)（分数标准差 1）
  c = sqrt(d) -> 没除（标准差 sqrt(d)）
eps = 1 - p_max = softmax 分给"非赢家"的总质量，梯度整体乘它。

为了让直线干净，用一次代表性初始化（前两名间距取 500 次抽样的中位数），
不跨种子平均 —— 平均会把罕见的"近并列"混进来，曲线就不直了。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

N = 1024
DRAWS = 500
rng = np.random.default_rng(0)
Zs = rng.normal(0, 1, (DRAWS, N))
gaps = np.sort(Zs, axis=1)[:, ::-1][:, 0] - np.sort(Zs, axis=1)[:, ::-1][:, 1]
k = int(np.argsort(gaps)[DRAWS // 2])          # 取间距为中位数的那次初始化
z = Zs[k]
gap = gaps[k]
print(f"代表性初始化：前两名间距 Δ = {gap:.3f}")


def eps(c):
    c = np.atleast_1d(np.asarray(c, float))
    s = c[:, None] * z[None, :]
    s = s - s.max(axis=1, keepdims=True)
    return np.exp(s).sum(axis=1) - 1.0


def l10(c):
    return np.log10(np.maximum(eps(c), 1e-30))


fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))

# ---- (A) softmax 值 vs 倍率：只有"饱和 ----
ax = axs[0]
cs = np.linspace(0, 8, 400)
e = eps(cs)
pmax = 1.0 / (1.0 + e)
ax.plot(cs, pmax, lw=2.2, c="C0", label="赢家权重  p$_{max}$")
ax.plot(cs, e / (1 + e), lw=2.0, c="C1", label="ε = 1 − p$_{max}$")
ax.axvline(1, ls=":", c="k", lw=1)
ax.annotate("c = 1：已除 √d\n注意力近乎均匀", (1, 0.5), xytext=(1.4, 0.72),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("不除 √d 落在 c = √d\n(本图右边很远)", (8, 0.62), xytext=(4.2, 0.18),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set(xlabel="倍率 c（打分 = c·z）", ylabel="softmax 权重", ylim=(-0.02, 1.02))
ax.set_title("(A) 你提的那种图：softmax 值 vs 倍率\n只看得到“饱和”，看不出衰减速率", fontsize=11)
ax.legend(fontsize=9, loc="center left")
ax.grid(alpha=0.3)

# ---- (B) 纵轴取 log：直线 ----
ax = axs[1]
cs = np.linspace(0.5, 70, 400)
ax.plot(cs, l10(cs), lw=2.2, c="C3")
ax.axvline(1, ls=":", c="k", lw=1)
slope = -gap / np.log(10)
ax.plot([6, 66], [l10(6)[0], l10(6)[0] + slope * 60], "k--", lw=1,
        label=f"斜率 −Δ/ln10 ≈ {slope:.2f}")
for d, xy in [(512, (14, -4.2)), (4096, (30, -8.4))]:
    c0 = np.sqrt(d)
    y = l10(c0)[0]
    ax.plot([c0], [y], "o", c="C3", ms=7)
    ax.annotate(f"d={d}, c=√d → ε≈{10**y:.0e}", (c0, y), xytext=xy, fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("c = 1：ε≈%.0f\n与 d 无关" % eps(1)[0], (1, l10(1)[0]), xytext=(4, 1.4),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set(xlabel="倍率 c", ylabel=r"$\log_{10}\,\varepsilon$", ylim=(-13, 2.6))
ax.set_title("(B) 纵轴换 log：一条直线 = 指数衰减\n斜率就是 −(前两名间距)", fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3)

# ---- (C) 不除 sqrt(d)：ε 对 sqrt(d) 也是直线 ----
ax = axs[2]
ds = 2.0 ** np.arange(6, 15)
xs = np.sqrt(ds)
ys = np.array([l10(np.sqrt(d))[0] for d in ds])
ax.plot(xs, ys, "o-", lw=2.2, ms=6, c="C0", label="不除 √d（取 c=√d）")
ax.axhline(l10(1)[0], ls="--", c="C2", lw=2, label="除了 √d（c=1，恒定）")
for d, x, y, off in [(64, 8, ys[0], (6, -18)), (512, np.sqrt(512), ys[3], (8, 12)),
                     (4096, 64, ys[6], (-52, -20)), (16384, 128, ys[-1], (-40, 10))]:
    ax.annotate(f"d={d}", (x, y), textcoords="offset points", xytext=off, fontsize=9)
ax.set(xlabel=r"$\sqrt{d}$", ylabel=r"$\log_{10}\,\varepsilon$", ylim=(-13, 2.6))
ax.set_title(r"(C) 横轴换成 √d：还是直线，所以 $\varepsilon \propto e^{-k\sqrt{d}}$,"
             "\n指数底是 √d，不是 d", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("scratch/softmax_scale.png", dpi=150)

print("\n d      eps(不除)     log10     eps(除)")
for d in ds:
    print(f"{int(d):6d}  {eps(np.sqrt(d))[0]:10.2e}  {l10(np.sqrt(d))[0]:7.2f}   {eps(1)[0]:8.1f}")
print(f"\n eps(c=1)={eps(1)[0]:.1f}  -> p_max={1/(1+eps(1)[0]):.3f}")
