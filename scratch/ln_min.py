import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):                       # x: [B, T, d_model]
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 16, 8)

    mine = LayerNorm(8)
    ref = nn.LayerNorm(8)
    ref.load_state_dict({"weight": torch.randn(8), "bias": torch.randn(8)})
    mine.load_state_dict(ref.state_dict())

    print("max |diff| =", (mine(x) - ref(x)).abs().max().item())
    print("out[0,0]   =", mine(x)[0, 0].detach())
