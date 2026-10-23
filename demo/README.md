# 极简 Transformer Demo

手写 decoder-only Transformer，用于理解原理。设计见 [plan.md](plan.md)。

## 环境

PyTorch（开发用的是 conda 环境 `myllm`：torch 2.11 + CUDA）+ matplotlib。

## 文件

| 文件 | 职责 |
|---|---|
| `data.py` | 数据生成，唯一会随阶段更换的部分（`caesar` / `echo`） |
| `model.py` | 模型：sin/cos 位置编码 + 单头因果注意力 + post-norm 块 |
| `train.py` | 训练循环 + loss 曲线 + 抽查预测 |
| `inspect.py` | 注意力热力图 |
| `trace.py` | 因果追踪（破坏-恢复），定位答案来自哪一层/哪个位置 |

## 运行

```bash
python data.py                                      # 冒烟：看两个任务的样本
python model.py                                     # 形状自检 + 参数量
python train.py --task caesar --steps 2000          # 阶段一：loss 应降到 ≈0
python train.py --task echo    --steps 4000         # 阶段二
python inspect.py --checkpoint ckpt_echo.pt         # 画注意力图（双层对比）
python trace.py --checkpoint ckpt_caesar.pt        # 因果追踪：答案在哪一层/哪个位置
python trace.py --checkpoint ckpt_echo.pt          # 对照：任务本身需要搬运时矩阵怎么变
```

产物：`ckpt_<task>.pt`（权重）、`loss_<task>.png`（loss 曲线）、`attention.png`（注意力图）、
`trace_<task>.png`（恢复矩阵）。

## Shape 速查（B=batch, T=序列长, C=模型宽度, V=词表）

| 阶段 | 形状 |
|---|---|
| token embedding | `[B,T]` → `[B,T,C]` |
| + 位置编码 | `[B,T,C]` |
| Q / K / V | 各 `[B,1,T,C]`（单头） |
| 注意力打分 QKᵀ/√C | `[B,1,T,T]` |
| causal mask 后 softmax | `[B,1,T,T]`（每行和为 1） |
| 加权求和 ·V | `[B,1,T,C]` → `[B,T,C]` |
| LM head logits | `[B,T,C]` → `[B,T,V]` |

## 两阶段的预期现象

- **caesar（查表）**：loss 快速降到接近 0；注意力图不重要，bigram 就能解。
- **echo（回溯任务，distance=4）**：序列完全随机，目标是预测「3 步前」的字符，
  bigram 命中率仅 1/26，必须靠 attention；
  训好后 layer 1 注意力图呈现偏移 3 的锐利对角线（实测正确 key 权重 0.97），
  layer 0 则呈弥散分布（正确 key 仅 0.18）——两层分工一目了然。
  前 `distance-1` 个位置的目标在因果 mask 下不可见，目标用 -100 填充、损失忽略。

## 因果追踪的预期现象（实测 2026-10-22）

`recovery=(p_restore-p_corrupt)/(p_clean-p_corrupt)`，矩阵 `[restore 位置 i][query 位置 j]`。

- **caesar**：非对角恒为 0，全部落在「偏移 0」的对角线上 —— 答案只依赖当前字符，不需要搬运。
  FFN 对角 0.44–0.47，attention 对角仅 0.07–0.13，说明这一步是本位置 FFN 算出来的。
- **echo**：亮带整体偏移 **+3**（= distance−1）——「restore 位置 5 → query 位置 8」即直接证据；
  由 layer 1 承担（attn 对角 0.76），layer 0 的 attention 是空转（0.006）。
- `resid` 干预的对角恢复恒为 1.0（等价于把该位置下游全变干净），只当基线看。
