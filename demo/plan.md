# 极简 Transformer Demo — 组件清单

## 目标

手写一个 decoder-only 极简 Transformer，用于理解原理。分两阶段：

- **阶段一（管线验证）**：凯撒移位字符预测（`a→b, b→c…`），bigram 查表即可解，目标是跑通训练管线。
- **阶段二（attention 验证）**：回声任务（输入 `abXcd`，要求模型 attend 到远处 token），目标是看到注意力图从均匀变锐利。

两阶段共用全部模型与训练代码，**只换数据生成函数**。

## 文件结构

```
demo/
├── plan.md            # 本文档：组件清单与设计决策
├── data.py            # 数据生成（唯一会换的部分）
│   ├── caesar()       # 阶段一：凯撒移位
│   └── echo()         # 阶段二：回声任务
├── model.py           # 模型定义
├── train.py           # 训练循环 + loss 曲线
├── plot_attention.py  # 注意力图可视化（阶段二用；不能叫 inspect.py，会遮蔽标准库）
├── trace.py           # 因果追踪：破坏-恢复，定位答案来自哪一层/哪个位置
└── README.md          # 运行说明 + 每个部件的 shape 速查
```

## 模型组件（model.py）

GPT 风格 decoder-only，按依赖顺序：

| 组件 | 输入 → 输出 shape | 职责 |
|---|---|---|
| Token Embedding | `[B,T]` → `[B,T,C]` | 查表，把字符 id 变成向量 |
| 位置编码（sin/cos 固定） | 公式生成 `[T,C]` buffer | 让模型感知顺序；无参数，偶数维 sin、奇数维 cos |
| Causal Self-Attention（单头） | `[B,T,C]` → `[B,T,C]` | QKV 投影 + mask + 加权聚合；**先单头，看懂后再升多头** |
| 残差连接 + LayerNorm | `[B,T,C]` → `[B,T,C]` | 梯度通路 + 稳定训练；实现用 pre-norm（norm 在残差分支内） |
| FFN | `[B,T,C]` → `[B,T,C]` | 逐位置非线性变换 |
| LM Head | `[B,T,C]` → `[B,T,V]` | 线性层输出每个字符的 logits |
| 交叉熵 loss | logits + 目标 → 标量 | 标准 GPT 训练目标 |

## 训练（train.py）

- 固定循环：取 batch → forward → loss → backward → step
- 每 N 步打印 loss，画 loss 曲线
- 评估：抽查样本的逐字符预测 + （阶段二）注意力权重热力图
- 不写死训练步数，看 loss 曲线判断收敛

## 配置建议

| 参数 | 值 | 理由 |
|---|---|---|
| 词表 V | ~30（`a-z` + 特殊符） | 极小，embedding 矩阵肉眼可察 |
| 维度 C | 32–64 | CPU 秒级训练 |
| 层数 | 1–2 | 先单层，阶段二视情况加 |
| 注意力头数 | 先 1 | 多头只是 reshape，看懂单头再升 |
| 序列长度 T | 8–16 | 回声任务要足够长才能看清远距离 attend |
| batch | 32–64 | 训练稳定 |
| 优化器 | AdamW，lr ~1e-3 | 标准配置 |

## 阶段检查点

1. 随机输入 forward 一遍，打印每个 stage 的 shape，全部对上才继续
2. 阶段一：loss 稳定降到接近 0 → 管线正确
3. 阶段二：loss 下降 + 注意力图出现非均匀结构 → attention 真的在工作
4. 消融（可选）：attention 换成「取平均」，阶段二任务应显著变差，反向验证 attention 的必要性

## 因果追踪（trace.py）

回答「答案是哪一层、哪个位置算出来的」。协议：Meng et al. 2022（ROME）的破坏-恢复。

1. **干净跑**：记录各位置对正确答案的概率 `p_clean`
2. **破坏跑**：给 token embedding 加高斯噪声（`std = noise_scale × 词向量 std`），得 `p_corrupt`
3. **恢复跑**：每次只把一处激活换回干净值，得 `p_restore`，
   `recovery = (p_r - p_c) / (p_clean - p_c)`

三个干预点，粗→细：`resid`（进入第 l 个 block 的残差流）/ `attn`（第 l 层 attention 写回量）/
`ffn`（第 l 层 FFN 写回量）。恢复矩阵 `M[i][j]` = 把干净激活放到位置 i，位置 j 的答案恢复多少：
**偏移 0 = 本地算出，偏移 k = 从 k 步之前搬来**。

关键设计决定：

- 噪声 seed 固定并在每次前向重放同一份噪声，保证破坏基准一致，差异只来自干预
- 只干预「写回残差流的增量」，正好对应「attention 搬来的」与「FFN 本地算出的」
- 掩码在完整 `T×T` 网格上算（见坑点记录）

## 验证结论（2026-10-20）

- 两阶段 loss 均收敛至 ≈0.0001，可见位置预测 100% 正确。
- 回声任务 layer 1 注意力正确 key 权重 0.97（偏移 3 对角线），layer 0 仅 0.18（弥散）——两层分工清晰。

## 验证结论（2026-10-22，因果追踪）

`noise_scale=2.0`，干净/破坏准确率：caesar 1.000/0.504，echo 1.000/0.109。

| | resid 对角 | attn 对角 | ffn 对角 | 非对角最大 | 最强离位恢复 |
|---|---|---|---|---|---|
| caesar | 1.000 | 0.069 / 0.130 | 0.435 / 0.470 | 0.002 | **无**（全在偏移 0） |
| echo | 0.071 / 0.083 | **0.760**(L1) | 0.589(L1) | — | **偏移 +3**（restore 5 → query 8） |

- **caesar 全落在对角**：答案只依赖当前字符，不需要任何跨位置搬运；信息由本位置 FFN 算出
  （ffn 对角 0.44–0.47 远高于 attn 0.07–0.13），attention 在这种任务上是空转的。
- **echo 出现偏移 +3 带**（= distance−1）：resid 恢复在源位置 i 就救回了 query j=i+3 的答案，
  这就是「信息从 3 步之前搬来」的直接证据；且由 layer 1 承担（layer 0 attn 对角 0.006，
  layer 1 attn 0.760），与注意力图结论一致。
- 两组对照说明：`resid` 对角恢复恒为 1.0，因为恢复它等于把该位置下游全部变干净——
  它是基线，真正区分分工的是 attn / ffn 的对角高低。

## 坑点记录

- **捷径学习**：echo 初版用「周期重复序列」，模型退化为 bigram 循环链解法，
  注意力图弥散却 loss ≈0。教训：设计任务时必须检查「目标是否绕远路也能被解出」。
- **单批过拟合测试的陷阱**：固定一批数据能把随机噪声也背下来（实测 loss 可到 0.08），
  不能作为「任务可学」的证据；要用「新鲜采样批次上 loss 下降」判断。
- **caesar 初版数据 bug**：先移位再滑窗导致目标与输入独立（loss 恒 = ln 26）；
  正确做法是「序列本身构成移位链」，seq[t+1] = shift(seq[t])。
- **因果可见性**：回声任务前 distance-1 个位置的目标在因果 mask 下不可见，
  目标须用 -100 填充（ignore_index），裁剪窗口会把目标裁没。
- **非连续 targets 的 view 陷阱**：`ids[:, 1:]` 列切片 stride 不连续，`.view(-1)` 非法；
  CUDA 训练被 `.to(device)` 的拷贝掩盖，CPU 上才炸。交叉熵前统一用 `reshape`
  （对连续张量零开销）。
- **因果追踪里「先裁列再取对角线」的错位**：echo 的有效答案列从 `d-1` 才开始，
  若先 `mat[:, valid]` 裁掉无效列再 `.diagonal()`，压缩后的对角线整体左移 `d-1` 格，
  会把 echo 的偏移 +3 误读成「全在对角」。掩码必须在完整 `T×T` 网格上算（`trace.py` 已按此实现）。
- **破坏强度不足则恢复率无意义**：`noise_scale=1.0` 时 caesar 破坏后准确率仍 1.000，
  分母 `p_clean - p_corrupt ≈ 0`，恢复率变成噪声比。`trace.py` 会在破坏后准确率 >0.9 时告警；
  默认 2.0（论文用 3，但单点恢复救不回来，对角恢复率被压到 0.2 量级）。
- **matplotlib 中文字形**：默认 DejaVu Sans 无 CJK，图内中文渲染成方框。图内用英文，中文只留在注释/终端。
- **本地文件遮蔽标准库**：可视化脚本原名 `inspect.py`，与标准库 `inspect` 同名；直接运行本目录脚本时
  `sys.path[0]` 是本目录，torch 启动时 `import inspect` 会拿到本地文件，报
  `ImportError: cannot import name 'Parameter' from partially initialized module 'inspect'`。
  已改名为 `plot_attention.py`。以后新增脚本避开标准库模块名（现有 `trace.py` 也撞了标准库 `trace`，
  只是启动路径不 import 它，暂未触发；真撞上就同样改名）。
