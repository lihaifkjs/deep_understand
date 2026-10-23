"""训练循环：所有任务共用。只换 --task。

用法：
    python train.py --task caesar --steps 2000
    python train.py --task echo    --steps 4000
"""
import argparse
import random

import matplotlib.pyplot as plt
import torch

import data
from model import TinyGPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(data.TASKS), required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--distance", type=int, default=4, help="echo 任务的回溯距离")
    p.add_argument("--out", default="checkpoint.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyGPT(data.VOCAB_SIZE, args.d_model, args.n_layer, max_len=args.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"task={args.task} device={device} params={sum(q.numel() for q in model.parameters())}")
    losses = []
    for step in range(1, args.steps + 1):
        x, y = data.make_batch(args.task, args.batch, args.seq_len, rng, distance=args.distance)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # loss 曲线
    plt.figure(figsize=(6, 4))
    window = max(1, len(losses) // 100)
    smooth = [sum(losses[i : i + window]) / len(losses[i : i + window]) for i in range(0, len(losses), window)]
    plt.plot(range(0, len(losses), window), smooth)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"task={args.task}")
    plt.tight_layout()
    loss_png = f"loss_{args.task}.png"
    plt.savefig(loss_png, dpi=120)
    print(f"loss curve -> {loss_png}")

    # 抽查：对比预测与真实目标
    model.eval()
    x, y = data.make_batch(args.task, 3, args.seq_len, rng, distance=args.distance)
    with torch.no_grad():
        logits, _ = model(x.to(device))
    pred = logits.argmax(-1).cpu()
    for i in range(3):
        print("in :", "".join(data.itos[int(t)] for t in x[i]))
        print("pred:", "".join(data.itos[int(t)] for t in pred[i]))
        print("tgt:", "".join(data.itos[int(t)] if int(t) >= 0 else "?" for t in y[i]))

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
