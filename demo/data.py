"""数据生成：demo 中唯一会随阶段更换的部分。

所有任务都统一成「给定前缀，预测下一个字符」的标准语言建模形式：
    x = seq[:-1]   输入
    y = seq[1:]    目标（每个位置的下一个字符）
"""
import random

import torch

VOCAB = "abcdefghijklmnopqrstuvwxyz"
VOCAB_SIZE = len(VOCAB)
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for c, i in stoi.items()}


def _random_seq(n, rng):
    return [rng.choice(VOCAB) for _ in range(n)]


def make_batch(task, batch_size, seq_len, rng=None, **kwargs):
    """返回 (x, y)，形状均为 [batch_size, seq_len-1] 的 long 张量。

    caesar: y[t] = x[t] 的移位        —— 下一字符，bigram 可解
    echo:   目标是回溯序列中更早的字符。两个坐标系注意区分：
            「预测下一字符」视角：y[t] = seq[t+1] 的前 d 个字符处，即 seq[t+1-d]
            「y 对比 x」视角：    y[t] = x[t-(d-1)]，即偏移 d-1 步
            （attention 图按后者画，inspect.py 中的偏移 = distance - 1）
    中间隔着随机字符，bigram 命中率仅 1/26，必须靠 attention 回溯
    """
    rng = rng or random.Random()
    # 注意 distance 只影响 echo 的目标 y 的构造（见下方），不传给任务函数本身
    seqs = [TASKS[task](seq_len, rng) for _ in range(batch_size)]
    ids = torch.tensor([[stoi[c] for c in s] for s in seqs], dtype=torch.long)
    x = ids[:, :-1]
    y = ids[:, 1:]
    if task == "echo":
        d = kwargs.get("distance", 4)
        # y[t] = seq[t-(d-1)]：x 保持完整，前 d-1 个位置的目标在因果 mask 下
        # 不可见，用 -100 填充（model 的交叉熵设了 ignore_index=-100）
        y = torch.full_like(x, -100)
        y[:, d - 1 :] = ids[:, : x.size(1) - (d - 1)]
    return x, y


# ---------------------------------------------------------------- 阶段一
def caesar(seq_len, rng, shift=1):
    """凯撒移位链：每个字符的下一个字符是字母表中后移 shift 位的结果。

    即 seq[t+1] = shift(seq[t])，预测下一字符只需看当前字符
    （bigram 查表即可解），用于验证训练管线。
    """
    seq = [rng.choice(VOCAB)]
    for _ in range(seq_len - 1):
        seq.append(VOCAB[(stoi[seq[-1]] + shift) % VOCAB_SIZE])
    return seq


# ---------------------------------------------------------------- 阶段二
def echo(seq_len, rng, distance=4):
    """回声任务：序列完全随机，目标是回溯——预测「distance 步之前」的字符。

    关键设计：序列不做周期重复（见 plan.md 坑点记录），中间隔着随机字符，
    bigram 命中率仅 1/26，模型必须借助 attention 找到固定偏移处的 token ——
    训好后注意力图会呈现一条清晰的偏移对角线（偏移 = distance-1）。
    """
    return _random_seq(seq_len, rng)


TASKS = {"caesar": caesar, "echo": echo}


if __name__ == "__main__":
    # 冒烟测试：打印两个任务的样本
    rng = random.Random(0)
    for task in ("caesar", "echo"):
        x, y = make_batch(task, batch_size=2, seq_len=10, rng=rng)
        print(f"--- {task} ---")
        for row in x:
            print("  x:", "".join(itos[i.item()] for i in row))
        for row in y:
            print("  y:", "".join(itos[i.item()] for i in row))
