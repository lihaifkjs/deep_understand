import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))   # γ 初始化为 1
        self.bias   = nn.Parameter(torch.zeros(num_features))  # β 初始化为 0
        # 训练/推理两套统计量的切换靠这两个 buffer，不是参数，不参与梯度
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

    def forward(self, x):                        # x: [B, C, H, W]
        if self.training:
            # 每个通道跨 (N, H, W) 求统计，结果 shape [C]
            mean = x.mean(dim=(0, 2, 3))
            var  = x.var(dim=(0, 2, 3), unbiased=False)   # 归一化用有偏
            with torch.no_grad():
                n = x.numel() // x.size(1)                # N*H*W
                if n > 1:
                    unbiased_var = var * n / (n - 1)      # running 用无偏
                    self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
        else:
            # 推理：直接用累积的全局统计量，等效于固定的仿射变换
            mean, var = self.running_mean, self.running_var

        shape = (1, -1, 1, 1)                    # 广播到 [B, C, H, W]
        xhat = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return xhat * self.weight.view(shape) + self.bias.view(shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    mine = BatchNorm2d(3)
    ref = nn.BatchNorm2d(3)
    ref.load_state_dict({"weight": torch.randn(3), "bias": torch.randn(3),
                         "running_mean": torch.zeros(3), "running_var": torch.ones(3),
                         "num_batches_tracked": torch.tensor(0)})
    sd = {k: v for k, v in ref.state_dict().items() if k != "num_batches_tracked"}
    mine.load_state_dict(sd)

    # 训练模式跑 3 个 batch，逐次比较输出与 running stats
    for step in range(3):
        x = torch.randn(4, 3, 8, 8)
        out_m, out_r = mine(x), ref(x)
        print(f"train step{step}: out diff={ (out_m - out_r).abs().max():.2e} "
              f"run_mean diff={(mine.running_mean - ref.running_mean).abs().max():.2e} "
              f"run_var diff={(mine.running_var - ref.running_var).abs().max():.2e}")

    # 推理模式
    mine.eval(); ref.eval()
    x = torch.randn(2, 3, 8, 8)
    print(f"eval: out diff={(mine(x) - ref(x)).abs().max():.2e}")

    # 训练/推理输出确实不同（证明统计量切换生效）
    mine.train(); print(f"train vs eval out diff={(mine(x) - ref(x)).abs().max():.2e}")
