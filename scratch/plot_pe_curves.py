"""按「横轴 i（位置）、纵轴 pe[i, j]」画曲线：每个固定的 j 一条线。

输出：
  scratch/pe_curves.png   两个面板：低维（频率高）/ 高维（频率低）
  scratch/pe_values.csv   完整 PE 矩阵（精确数值，可用 Excel 打开）
"""

import csv
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 1330, 800
D_MODEL = 128
T_SHORT, T_LONG = 64, 1024      # 上排看 64 个位置，下排看 1024 个位置

TXT = (38, 43, 55)
GRAY = (110, 118, 132)
AXIS = (150, 156, 168)
GRID = (235, 237, 242)
COLORS = [(40, 78, 168), (178, 40, 45), (22, 122, 92), (196, 120, 20), (120, 70, 160), (10, 130, 150)]


def _font_path():
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(p):
            return p
    return None


FONT_PATH = _font_path()
_fc = {}


def font(size):
    k = int(round(size * S))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONT_PATH, k) if FONT_PATH else ImageFont.load_default()
    return _fc[k]


def sinusoidal_pe(max_len, d_model, base=10000.0):
    pos = np.arange(max_len)[:, None]
    idx = np.arange(0, d_model, 2)
    omega = np.exp(idx * (-math.log(base) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe, omega


img = Image.new("RGB", (W * S, H * S), (255, 255, 255))
draw = ImageDraw.Draw(img)


def text(x, y, s, size=17, fill=TXT, anchor="la"):
    draw.text((x * S, y * S), s, font=font(size), fill=fill, anchor=anchor)


def line(pts, fill=TXT, width=2):
    draw.line([(float(a) * S, float(b) * S) for a, b in pts], fill=fill, width=max(1, int(width * S)))


def dot(x, y, r, fill):
    draw.ellipse([(x - r - 1.5) * S, (y - r - 1.5) * S, (x + r + 1.5) * S, (y + r + 1.5) * S], fill=(255, 255, 255))
    draw.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=fill)


def rect(x0, y0, x1, y1):
    draw.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], outline=AXIS, width=max(1, int(2 * S)))


class Axes:
    def __init__(self, x, y, w, h, xr, yr, xticks, yticks):
        self.x, self.y, self.w, self.h, self.xr, self.yr = x, y, w, h, xr, yr
        rect(x, y, x + w, y + h)
        for tx in xticks:
            px = self.px(tx)
            line([(px, y), (px, y + h)], fill=GRID, width=2)
            line([(px, y + h), (px, y + h + 5)], fill=AXIS, width=2)
            text(px, y + h + 9, f"{tx:g}", size=15, fill=GRAY, anchor="ma")
        for ty in yticks:
            py = self.py(ty)
            line([(x, py), (x + w, py)], fill=GRID, width=2)
            line([(x - 5, py), (x, py)], fill=AXIS, width=2)
            text(x - 9, py, f"{ty:g}", size=15, fill=GRAY, anchor="rm")

    def px(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def curve(self, xs, ys, fill, width=2.4):
        line([(self.px(a), self.py(b)) for a, b in zip(xs, ys)], fill=fill, width=width)


pe_short, omega = sinusoidal_pe(T_SHORT, D_MODEL)
pe_long, _ = sinusoidal_pe(T_LONG, D_MODEL)

text(W / 2, 28, "pe[i, j] 随 i 变化（j 固定；横轴 i = 位置，纵轴 = 该维度的取值）", size=24, anchor="ma")
text(W / 2, 64, "d_model = 128：sin 在偶数列 j=2m，cos 在奇数列 j=2m+1，两者共用同一个频率 ω_m",
     size=16, fill=GRAY, anchor="ma")

# ---------------------------------------------------------------- 上排：低维（快）
AX, AY, AW, AH = 130, 130, 900, 250
text(AX, 100, "低维 j = 0…3（ω ≈ 1 到 0.75：约 6 步绕一圈）", size=18)
ax1 = Axes(AX, AY, AW, AH, (0, T_SHORT - 1), (-1.12, 1.12), [0, 10, 20, 30, 40, 50, 63], [-1, 0, 1])
for k, j in enumerate([0, 1, 2, 3]):
    m = j // 2
    w = omega[m]
    ax1.curve(np.arange(T_SHORT), pe_short[:, j], COLORS[k], width=2.4)
    for i in range(0, T_SHORT, 5):
        dot(ax1.px(i), ax1.py(pe_short[i, j]), 3, COLORS[k])
    text(AX + AW + 14, AY + 14 + 30 * k,
         f"j={j}  {'sin' if j % 2 == 0 else 'cos'}({w:.3g}·i)", size=15, fill=COLORS[k])
text(AX + AW / 2, AY + AH + 32, "位置 i", size=16, anchor="ma")
text(AX - 52, AY + AH / 2, "pe[i, j]", size=16, anchor="ma")

# ---------------------------------------------------------------- 下排：高维（慢）
BY, BH = 480, 250
text(AX, 450, "高维 j = 32 / 64 / 126（ω = 0.1 / 0.01 / 0.000115：1024 个位置也只走一点点）", size=18)
ax2 = Axes(AX, BY, AW, BH, (0, T_LONG - 1), (-1.12, 1.12), [0, 256, 512, 768, 1023], [-1, 0, 1])
for k, j in enumerate([32, 64, 126]):
    w = omega[j // 2]
    ax2.curve(np.arange(T_LONG), pe_long[:, j], COLORS[k], width=2.4)
    text(AX + AW + 14, BY + 14 + 30 * k, f"j={j}  sin({w:.3g}·i)", size=15, fill=COLORS[k])
text(AX + AW / 2, BY + BH + 32, "位置 i", size=16, anchor="ma")
text(AX - 52, BY + BH / 2, "pe[i, j]", size=16, anchor="ma")

img.resize((W, H), Image.LANCZOS).save("scratch/pe_curves.png")

# ---------------------------------------------------------------- 精确数值
pe_all, _ = sinusoidal_pe(D_MODEL, D_MODEL)   # 128 × 128
with open("scratch/pe_values.csv", "w", newline="", encoding="utf-8") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["i\\j"] + [f"j={j}" for j in range(D_MODEL)])
    for i in range(D_MODEL):
        wcsv.writerow([i] + [f"{v:.6f}" for v in pe_all[i]])

np.set_printoptions(precision=4, suppress=True, linewidth=150)
print("pe[i, 0..5], i = 0..5:")
print(pe_all[:6, :6])
print("cols:", [(j, "sin" if j % 2 == 0 else "cos", round(float(omega[j // 2]), 6)) for j in range(6)])
