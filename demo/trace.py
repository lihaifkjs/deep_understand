"""因果追踪（破坏-恢复）：定位「哪一层、哪个位置、哪个子层」承载了预测答案。

协议出自 Meng et al. 2022《Locating and Editing Factual Associations in GPT》，三步：
  1. 干净跑：正常前向，记录各位置对正确答案的概率 p_clean
  2. 破坏跑：给 token embedding 加高斯噪声（std = noise_scale × 词向量标准差），得到 p_corrupt
  3. 恢复跑：每次只把「某一层、某一位置、某一处激活」换回干净值，得到 p_restore

  recovery = (p_restore - p_corrupt) / (p_clean - p_corrupt)
  1 = 完全救回干净答案，0 = 一点没帮上，负 = 帮倒忙

恢复矩阵 M[i][j]：把干净激活恢复到「位置 i」，看「位置 j」的答案恢复多少。
  - 落在对角 j=i  -> 这条信息在该位置本地就有（不需要搬运）
  - 落在偏移 j=i+k -> 这条信息是从 k 步之前的某处搬过来的

三个干预点，从粗到细：
  resid  进入第 l 个 block 的残差流  —— 问「信息到这里到位了吗」
  attn   第 l 层 attention 的写回量  —— 问「是 attention 从别处搬来的吗」
  ffn    第 l 层 FFN 的写回量        —— 问「是 FFN 在本位置算出来的吗」

在凯撒链上（y[t] = shift(x[t])，答案只依赖当前字符），预期：
  - 三个矩阵的非对角都 ≈ 0：没有任何跨位置搬运
  - resid 与 ffn 的对角明显 > attn 的对角：答案由本位置的 FFN 算出
这是「FFN 算内容、attention 搬运」的最小对照组（echo 任务里 answer 与 source 分离，
非对角才会亮起来）。

用法：
    python trace.py --checkpoint ckpt_caesar.pt
    python trace.py --checkpoint ckpt_echo.pt
"""
import argparse
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import data
from model import TinyGPT

COMPONENTS = ("resid", "attn", "ffn")


class Tracer:
    """挂钩子，支持缓存干净激活与单点覆盖。

    - resid 用 Block 的 forward_pre_hook：hook 到的正是进入该 block 的残差流
    - attn/ffn 用子层的 forward_hook：hook 到的正是要写回残差流的增量
    覆盖发生在残差相加之前，语义是「这一处本应写入这个干净值」。
    """

    def __init__(self, model):
        self.model = model
        self.clean = {}          # (comp, layer) -> [B,T,C] 干净激活
        self.capture = False     # True 时缓存激活
        self.write = None        # (comp, layer, pos) 时覆盖该位置
        self._noise = None       # (sigma, seed) 时给 embedding 加噪声
        for li, blk in enumerate(model.blocks):
            self._resid_hook(li, blk)
            self._sub_hook(li, "attn", blk.attn)
            self._sub_hook(li, "ffn", blk.ffn)
        model.tok_emb.register_forward_hook(self._emb_hook)

    def _resid_hook(self, li, block):
        key = ("resid", li)

        def fn(mod, args):
            x = args[0]
            if self.capture:
                self.clean[key] = x.detach()
            if self.write is not None and self.write[:2] == key:
                x = x.clone()
                x[:, self.write[2]] = self.clean[key][:, self.write[2]]
            return (x,)

        block.register_forward_pre_hook(fn)

    def _sub_hook(self, li, comp, module):
        key = (comp, li)

        def fn(mod, inp, out):
            if self.capture:
                self.clean[key] = out.detach()
            if self.write is not None and self.write[:2] == key:
                out = out.clone()
                out[:, self.write[2]] = self.clean[key][:, self.write[2]]
            return out

        module.register_forward_hook(fn)

    def _emb_hook(self, mod, inp, out):
        if self._noise is None:
            return out
        sigma, seed = self._noise
        # 每次前向都用同一个 seed 重放同一份噪声：
        # 保证「破坏」这个基准在所有恢复跑里完全一致，差异只来自干预本身
        gen = torch.Generator().manual_seed(seed)
        return out + torch.randn(out.shape, generator=gen) * sigma

    def add_noise(self, sigma, seed):
        self._noise = (sigma, seed)


def run(model, idx, ans):
    """前向一次，返回 (每位置正确答案的概率 [B,T], 预测是否正确 [B,T])。"""
    logits, _ = model(idx)
    logp = F.log_softmax(logits, dim=-1)
    p = logp.gather(-1, ans.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp()
    return p, logits.argmax(-1) == ans.clamp(min=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ckpt_caesar.pt")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--noise-scale", type=float, default=2.0,
                   help="破坏噪声的 std，相对词向量 std 的倍数（论文用 3，太大则单点恢复救不回来）")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    task, seq_len = cfg["task"], cfg["seq_len"]
    distance = cfg.get("distance", 4)
    model = TinyGPT(data.VOCAB_SIZE, cfg["d_model"], cfg["n_layer"], max_len=seq_len)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x, y = data.make_batch(task, args.batch, seq_len, random.Random(args.seed), distance=distance)
    T = x.size(1)
    valid = y[0] >= 0                        # echo 前 distance-1 个目标在因果 mask 下不可见
    n_layer = len(model.blocks)

    tracer = Tracer(model)
    with torch.no_grad():
        tracer.capture = True
        p_clean, ok_clean = run(model, x, y)
        tracer.capture = False

        sigma = model.tok_emb.weight.std().item() * args.noise_scale
        tracer.add_noise(sigma, args.seed)
        p_corr, ok_corr = run(model, x, y)
        if ok_corr[:, valid].float().mean() > 0.9:
            print(f"[警告] 破坏后准确率仍 {ok_corr[:, valid].float().mean():.2f}，"
                  f"加大 --noise-scale；否则分母接近 0，恢复率没有意义")

        # 逐点恢复：每次只覆盖一处 (干预点, 层, 位置)
        rec = {}
        for comp in COMPONENTS:
            for li in range(n_layer):
                mat = torch.zeros(T, T)
                for i in range(T):
                    tracer.write = (comp, li, i)
                    p_r, _ = run(model, x, y)
                    tracer.write = None
                    # 分子分母都在 batch 上取均值，得到平均恢复比例
                    mat[i] = (p_r - p_corr).mean(0) / (p_clean - p_corr).mean(0).clamp(min=1e-6)
                mat[:, ~valid] = float("nan")
                rec[(comp, li)] = mat

    # ---------------------------------------------------------------- 报告
    print(f"task={task} T={T} layer={n_layer} noise_scale={args.noise_scale}")
    print(f"干净准确率 {ok_clean[:, valid].float().mean():.3f}  "
          f"破坏后 {ok_corr[:, valid].float().mean():.3f}")
    print()

    # 掩码必须在完整 T×T 网格上算。echo 的有效答案列从 d-1 才开始，
    # 若先把列裁掉再取对角线，对角线会整体左移 d-1 格（踩过的坑）
    colok = valid.unsqueeze(0).expand(T, T)          # 列 j 是否为有效答案位置
    diag_m = torch.eye(T, dtype=torch.bool) & colok
    off_m = ~torch.eye(T, dtype=torch.bool) & colok
    neg_inf = float("-inf")

    print("             对角平均   离对角平均   非对角最大")
    for (comp, li), mat in rec.items():
        m = mat.masked_fill(~colok, neg_inf)
        print(f" layer{li} {comp:5s} {mat[diag_m].mean():>8.3f} "
              f"{mat[off_m].mean():>11.3f} {m.max():>11.3f}")

    # 最强恢复点落在哪里：偏移 0 = 本地算出；偏移 k = 从 k 步之前搬过来
    hits = []
    for (comp, li), mat in rec.items():
        m = mat.masked_fill(~colok, neg_inf)
        i, j = divmod(int(m.argmax()), T)
        hits.append((float(m.max()), comp, li, i, j))
    hits.sort(reverse=True)
    print("\n最强恢复点：")
    for v, comp, li, i, j in hits[:4]:
        print(f"  {v:.3f}  layer{li} {comp:5s}  restore i={i:2d} -> query j={j:2d}  偏移 {j - i:+d}")

    # ---------------------------------------------------------------- 画图
    fig, axes = plt.subplots(len(COMPONENTS), n_layer,
                             figsize=(5.0 * n_layer, 4.3 * len(COMPONENTS)), squeeze=False)
    for row, comp in enumerate(COMPONENTS):
        for li in range(n_layer):
            ax = axes[row][li]
            im = ax.imshow(rec[(comp, li)].numpy(), cmap="magma", vmin=0, vmax=1)
            ax.plot(range(T), range(T), color="cyan", lw=0.8, alpha=0.8)  # 对角线参考
            # 图内用英文：matplotlib 默认字体没有中文字形，会渲染成方框
            ax.set_xlabel("query position j (whose answer)")
            ax.set_ylabel("restore position i (where)")
            ax.set_title(f"layer {li} / {comp}")
            fig.colorbar(im, ax=ax)
    fig.suptitle(f"corruption-restoration  recovery=(p_r-p_c)/(p_clean-p_c)  task={task}")
    plt.tight_layout()
    out = f"trace_{task}.png"
    plt.savefig(out, dpi=120)
    print(f"\n恢复矩阵 -> {out}")


if __name__ == "__main__":
    main()
