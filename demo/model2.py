"""极简 decoder-only Transformer（GPT 风格），n_head=4 的多头版本。

由 model.py 复制而来，只把单头换成 4 头，其余结构完全一致。
注意：n_head 必须整除 d_model（4 头要求 d_model 是 4 的倍数，64 可以）。
张量形状约定：B=batch, T=序列长, C=通道维(模型宽度), V=词表大小。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 位置编码
class SinusoidalPositionEncoding(nn.Module):
    """原版 Transformer 的固定位置编码，无可训练参数。

    位置 pos 的编码向量：偶数维 sin(pos / 10000^(2i/C))，奇数维取 cos。
    不同维度频率不同，形成一组正交基；相对位置可通过线性变换得到。
    """

    def __init__(self, max_len, d_model):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)          # [T, 1]
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)               # 偶数维
        pe[:, 1::2] = torch.cos(pos * freq)               # 奇数维
        self.register_buffer("pe", pe)                    # [T, C]

    def forward(self, x):
        # x: [B, T, C]，直接按位置相加
        return x + self.pe[: x.size(1)]


# ---------------------------------------------------------------- 注意力
class CausalSelfAttention(nn.Module):
    """4 头因果自注意力。

    Q·K^T / sqrt(head_dim) 得到 [T, T] 的打分矩阵（每个头一份），
    行 = 查询位置，列 = 被看的键位置；
    下三角 mask 保证每个位置只能看到自己和过去（causal）。
    softmax 归一化后对 V 加权求和 —— 这就是「按相关性取上下文」。
    """

    def __init__(self, d_model, max_len, n_head=4):
        super().__init__()
        assert d_model % n_head == 0, (
            f"d_model={d_model} 必须能被 n_head={n_head} 整除，否则切不整齐"
        )
        self.n_head = n_head
        self.head_dim = d_model // n_head                  # 每个头负责的宽度
        self.qkv = nn.Linear(d_model, 3 * d_model)        # 一次性算 Q,K,V
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(max_len, max_len))   # 下三角 = 1
        self.register_buffer("causal_mask", mask.view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)             # 各 [B, T, C]
        # 把宽度 C 拆成 (n_head, head_dim)，再把头维提前：
        # [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        # 这样最后两维才是 (T, head_dim)，才能做批量矩阵乘；每个头独立打分。
        q, k, v = (
            t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        )                                                 # [B, n_head, T, head_dim]
        # 打分：点积后除以 sqrt(head_dim) 防数值过大导致 softmax 饱和
        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # [B, n_head, T, T]
        score = score.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(score, dim=-1)                   # 每行和为 1
        self.last_attn = attn                             # 留给可视化，形状 [B, n_head, T, T]
        out = attn @ v                                    # [B, n_head, T, head_dim]
        # 拼回头：头维退回 C 里面，变回 [B, T, C] 再过输出投影
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


# ---------------------------------------------------------------- 前馈网络
class FeedForward(nn.Module):
    """逐位置的 MLP：attention 负责「搬运信息」，FFN 负责「加工信息」。"""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- Transformer 块
class Block(nn.Module):
    """pre-norm：先 LayerNorm 再进子层，残差相加。"""

    def __init__(self, d_model, max_len, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, max_len, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差：梯度高速路
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------- 完整模型
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layer=2, max_len=32, n_head=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)      # [V] -> [C] 查表
        self.pos_enc = SinusoidalPositionEncoding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, max_len, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)            # [C] -> [V] logits

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)              # [B, T]    -> [B, T, C]
        x = self.pos_enc(x)                # 加位置信息 -> [B, T, C]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)              # [B, T, C] -> [B, T, V]
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        return logits, loss


if __name__ == "__main__":
    # 形状自检：随机输入跑一遍 forward，逐层打印 shape
    m = TinyGPT(vocab_size=26, d_model=64, n_layer=2, max_len=32)
    x = torch.randint(0, 26, (4, 16))
    logits, loss = m(x, torch.randint(0, 26, (4, 16)))
    print("logits:", tuple(logits.shape), " loss:", float(loss))
    print("attn:", tuple(m.blocks[0].attn.last_attn.shape))
    for n, p in m.named_parameters():
        print(f"  {n:24s} {tuple(p.shape)}")
    print("total params:", sum(p.numel() for p in m.parameters()))
