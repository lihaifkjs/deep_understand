"""极简 decoder-only Transformer（GPT 风格）。

刻意保持单头、少量参数，目标是每个部件都能单步看懂。
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
        assert d_model % 2 == 0, "sin/cos 位置编码要求 d_model 为偶数（奇数维时 sin/cos 切片不等长）"
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
    """单头因果自注意力。

    Q·K^T / sqrt(d) 得到 [T, T] 的打分矩阵，行 = 查询位置，列 = 被看的键位置；
    下三角 mask 保证每个位置只能看到自己和过去（causal）。
    softmax 归一化后对 V 加权求和 —— 这就是「按相关性取上下文」。
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)        # 一次性算 Q,K,V
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(max_len, max_len))   # 下三角 = 1
        self.register_buffer("causal_mask", mask.view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)             # 各 [B, T, C]
        # 拆成多头只需 reshape 成 [B, T, n_head, head_dim] 再转置，
        # 本 demo 保持单头：把 C 视作 1 个头
        q, k, v = (t.view(B, T, 1, C).transpose(1, 2) for t in (q, k, v))  # [B, 1, T, C]
        # 打分：点积后除以 sqrt(C) 防数值过大导致 softmax 饱和
        score = q @ k.transpose(-2, -1) / math.sqrt(C)    # [B, 1, T, T]
        score = score.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(score, dim=-1)                   # 每行和为 1
        self.last_attn = attn.detach()                    # 调试旁路，切断梯度图
        out = attn @ v                                    # [B, 1, T, C]
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
    """pre-norm：LayerNorm 在残差分支内部，相加发生在分支外。

    与 post-norm（先算子层、相加后再 norm）相比，深层时梯度更稳，
    是 GPT-2 及之后的主流写法。
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差：梯度高速路
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------- 完整模型
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layer=2, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)      # [V] -> [C] 查表
        self.pos_enc = SinusoidalPositionEncoding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, max_len) for _ in range(n_layer))
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
            # reshape 而非 view：targets 可能是 ids[:, 1:] 这类非连续列切片
            # （.to(device) 到 CPU 时不拷贝，view 会炸；reshape 对连续张量零开销）
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss


if __name__ == "__main__":
    # 形状自检：随机输入跑一遍 forward，逐层打印 shape
    m = TinyGPT(vocab_size=26, d_model=64, n_layer=2, max_len=32)
    x = torch.randint(0, 26, (4, 16))
    logits, loss = m(x, torch.randint(0, 26, (4, 16)))
    print("logits:", tuple(logits.shape), " loss:", float(loss))
    for n, p in m.named_parameters():
        print(f"  {n:24s} {tuple(p.shape)}")
    print("total params:", sum(p.numel() for p in m.parameters()))
