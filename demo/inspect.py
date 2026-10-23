"""注意力可视化：加载 checkpoint，画注意力热力图。

用法：
    python inspect.py --checkpoint checkpoint.pt
"""
import argparse
import random

import matplotlib.pyplot as plt
import torch

import data
from model import TinyGPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--seq-len", type=int, default=None, help="默认取 checkpoint 里保存的值")
    p.add_argument("--distance", type=int, default=None, help="默认取 checkpoint 里保存的值")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    seq_len = args.seq_len or cfg.get("seq_len", 16)
    distance = args.distance or cfg.get("distance", 4)
    model = TinyGPT(data.VOCAB_SIZE, cfg["d_model"], cfg["n_layer"], max_len=seq_len)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 回声任务训好的模型，query 位置 t 应主要 attend key 位置 t-(distance-1)
    task = cfg.get("task", "echo")
    rng = random.Random(7)
    x, _ = data.make_batch(task, 1, seq_len, rng, distance=distance)
    chars = [data.itos[int(t)] for t in x[0]]

    with torch.no_grad():
        model(x)

    # 每个 Block 各画一张，并量化「正确 key」（t-(distance-1)）上的注意力占比
    n_layer = len(model.blocks)
    fig, axes = plt.subplots(1, n_layer, figsize=(5 * n_layer, 4.5), squeeze=False)
    for li, blk in enumerate(model.blocks):
        attn = blk.attn.last_attn[0, 0]        # [T, T]  行=query，列=key
        ax = axes[0][li]
        im = ax.imshow(attn, cmap="viridis", vmin=0)
        ax.set_xticks(range(len(chars)), chars)
        ax.set_yticks(range(len(chars)), chars)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        ax.set_title(f"layer {li}")
        fig.colorbar(im, ax=ax)
        hit = [attn[t, t - (distance - 1)].item() for t in range(distance - 1, attn.size(0))]
        print(f"layer {li}: 正确 key 上的平均注意力权重 = {sum(hit) / len(hit):.2f}")
        for t in range(distance - 1, attn.size(0)):
            j = attn[t].argmax().item()
            mark = " <-- 正确" if j == t - (distance - 1) else ""
            print(f"  query {t:2d} -> key {j:2d}  w={attn[t, j]:.2f}{mark}")
    fig.suptitle(f"attention weights (task={task})")
    plt.tight_layout()
    plt.savefig("attention.png", dpi=120)
    print("attention map -> attention.png")


if __name__ == "__main__":
    main()
