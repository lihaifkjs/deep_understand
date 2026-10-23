"""渲染 Sinusoidal 位置编码（PE）示意图，仅用 numpy + Pillow。

输出 scratch/pe_visualization.png：
  A  PE 矩阵热力图（位置 × 维度）
  B  每个维度对的角频率 ω_i（对数坐标）
  C  4 组 (sin, cos) 曲线，展示不同频率的快慢
  D  PE(0) 与 PE(pos) 的余弦相似度随距离衰减
  E  一对维度落在单位圆上：推进 pos = 固定角度的旋转
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 参数与常量
S = 2                        # 超采样倍数（抗锯齿）
W, H = 1600, 1530
D_MODEL, T = 128, 128

TXT = (38, 43, 55)
GRAY = (110, 118, 132)
AXIS = (150, 156, 168)
GRID = (235, 237, 242)
BLUE = (40, 78, 168)
RED = (178, 40, 45)
GREEN = (22, 122, 92)


def _font_path():
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(p):
            return p
    return None


FONT_PATH = _font_path()
_font_cache = {}


def font(size):
    key = int(round(size * S))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_PATH, key) if FONT_PATH else ImageFont.load_default()
    return _font_cache[key]


# ---------------------------------------------------------------- 位置编码本体
def sinusoidal_pe(max_len, d_model, base=10000.0):
    """与 PyTorch 版逐元素等价：偶数维 sin(pos·ω_i)，奇数维 cos(pos·ω_i)。"""
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d_model, 2)
    omega = np.exp(i * (-math.log(base) / d_model))     # ω_i = 10000^(-2i/d)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe, omega


DIVERGING = [
    (0.00, (33, 62, 140)),
    (0.25, (108, 148, 210)),
    (0.50, (247, 247, 243)),
    (0.75, (226, 148, 118)),
    (1.00, (150, 26, 34)),
]


def colorize(values):
    """[-1,1] 线性映射到发散色标 RGB。"""
    v = np.clip((values + 1) / 2, 0, 1)
    keys = np.array([a[0] for a in DIVERGING])
    cols = np.array([a[1] for a in DIVERGING], dtype=float) / 255.0
    idx = np.clip(np.searchsorted(keys, v, side="right") - 1, 0, len(keys) - 2)
    t = (v - keys[idx]) / (keys[idx + 1] - keys[idx])
    return ((cols[idx] + (cols[idx + 1] - cols[idx]) * t[..., None]) * 255).astype(np.uint8)


# ---------------------------------------------------------------- 绘图原语
img = Image.new("RGB", (W * S, H * S), (255, 255, 255))
draw = ImageDraw.Draw(img)


def text(x, y, s, size=17, fill=TXT, anchor="la"):
    draw.text((x * S, y * S), s, font=font(size), fill=fill, anchor=anchor)


def text_w(s, size=17):
    """文本宽度（基础像素）。"""
    return draw.textlength(s, font=font(size)) / S


def line(pts, fill=TXT, width=2):
    draw.line([(float(px) * S, float(py) * S) for px, py in pts], fill=fill, width=max(1, int(width * S)))


def rect(x0, y0, x1, y1, outline=AXIS, width=2):
    draw.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], outline=outline, width=max(1, int(width * S)))


def dot(x, y, r, fill, halo=True):
    if halo:
        draw.ellipse([(x - r - 2) * S, (y - r - 2) * S, (x + r + 2) * S, (y + r + 2) * S], fill=(255, 255, 255))
    draw.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=fill)


def paste_rgb(arr, x, y):
    img.paste(Image.fromarray(arr, "RGB"), (int(x * S), int(y * S)))


class Axes:
    """绘图区：数据坐标 → 像素坐标，并绘制边框/刻度/网格。"""

    def __init__(self, x, y, w, h, xr, yr, xticks, yticks, grid=True, xfmt="{:g}", yfmt="{:g}"):
        self.x, self.y, self.w, self.h, self.xr, self.yr = x, y, w, h, xr, yr
        rect(x, y, x + w, y + h)
        for tx in xticks:
            px = self.px(tx)
            if grid:
                line([(px, y), (px, y + h)], fill=GRID, width=2)
            line([(px, y + h), (px, y + h + 5)], fill=AXIS, width=2)
            text(px, y + h + 9, xfmt.format(tx), size=15, fill=GRAY, anchor="ma")
        for ty in yticks:
            py = self.py(ty)
            if grid:
                line([(x, py), (x + w, py)], fill=GRID, width=2)
            line([(x - 5, py), (x, py)], fill=AXIS, width=2)
            text(x - 9, py, yfmt.format(ty), size=15, fill=GRAY, anchor="rm")

    def px(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def curve(self, xs, ys, fill, width=2.4):
        line([(self.px(a), self.py(b)) for a, b in zip(xs, ys)], fill=fill, width=width)


# ---------------------------------------------------------------- 图题
text(W / 2, 32, "正弦位置编码 Sinusoidal Position Encoding", size=30, anchor="ma")
text(W / 2, 76,
     "PE(pos, 2i) = sin( pos / 10000^(2i/d) )        PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )"
     "        d_model = 128, T = 128",
     size=17, fill=GRAY, anchor="ma")

pe, omega = sinusoidal_pe(T, D_MODEL)

# ---------------------------------------------------------------- A. 热力图
AX_, AY, AW, AH = 150, 150, 640, 410
text(AX_, 114, "A   PE 矩阵热力图：行 = 位置 pos，列 = 维度\n", size=19)
rows = np.minimum((np.arange(AH * S) / (AH * S) * T).astype(int), T - 1)
cols = np.minimum((np.arange(AW * S) / (AW * S) * D_MODEL).astype(int), D_MODEL - 1)
paste_rgb(colorize(pe)[rows][:, cols], AX_, AY)
rect(AX_, AY, AX_ + AW, AY + AH)
for tpos in (0, 32, 64, 96, 127):
    text(AX_ - 8, AY + AH - tpos / (T - 1) * AH, str(tpos), size=15, fill=GRAY, anchor="rm")
for tdim in (0, 32, 64, 96, 127):
    text(AX_ + tdim / (D_MODEL - 1) * AW, AY + AH + 9, str(tdim), size=15, fill=GRAY, anchor="ma")
text(AX_ + AW / 2, AY + AH + 33, "维度索引 →", size=16, anchor="ma")
text(AX_ - 62, AY + AH / 2, "位置 pos", size=16, anchor="ma")

# 色标
CBX = AX_ + AW + 22
paste_rgb(colorize(np.linspace(-1, 1, 400)[:, None].repeat(40, axis=1)), CBX, AY)
rect(CBX, AY, CBX + 40, AY + AH)
for v in (-1.0, 0.0, 1.0):
    text(CBX + 50, AY + AH - (v + 1) / 2 * AH, f"{v:+.0f}", size=15, fill=GRAY, anchor="lm")
text(CBX + 20, AY - 20, "取值", size=15, fill=GRAY, anchor="ma")
text(AX_, AY + AH + 62,
     "低维列逐位置剧烈翻转（快），高维列几乎不动（慢）——一张表里并存多个时间尺度。", size=16, fill=GREEN)

# ---------------------------------------------------------------- B. 频率谱
BX, BY, BW, BH = 990, 150, 540, 410
text(BX, 114, "B   每个维度对的角频率 ω_i = 10000^(−2i/d)", size=19)
axb = Axes(BX, BY, BW, BH, (0, 63), (-4.15, 0.15), [0, 16, 32, 48, 63], [], grid=False)
for exp in (-4, -3, -2, -1, 0):
    y = axb.py(exp)
    line([(BX, y), (BX + BW, y)], fill=GRID, width=2)
    line([(BX - 5, y), (BX, y)], fill=AXIS, width=2)
    text(BX - 9, y, f"1e{exp}", size=15, fill=GRAY, anchor="rm")
axb.curve(np.arange(64), np.log10(omega), BLUE, width=3)
for i in range(0, 64, 9):
    dot(axb.px(i), axb.py(math.log10(omega[i])), 4, GREEN)
text(BX + BW / 2, BY + BH + 33, "维度对索引 i（0 … 63）", size=16, anchor="ma")
text(BX + BW / 2, BY + BH + 62,
     "对数坐标下是一条直线：ω 从 1 到 1e-4，波长跨 4 个数量级。", size=16, fill=GREEN, anchor="ma")

# ---------------------------------------------------------------- C. 曲线
pairs = [0, 8, 24, 48]
xs = np.arange(0, 60)
CW, CH, CG = 320, 252, 40
CY = 716
_cx = 150
for _s, _c in (("C   四组维度对的位置曲线：i 越大频率越低，", TXT), ("sin(2i)", BLUE),
               (" 与 ", TXT), ("cos(2i+1)", RED), (" 恒差 90°", TXT)):
    text(_cx, 668, _s, size=19, fill=_c)
    _cx += text_w(_s, 19)
for k, i in enumerate(pairs):
    cx0 = 150 + k * (CW + CG)
    w = omega[i]
    text(cx0, 692, f"i = {i}    ω = {w:.4g} rad/pos", size=16, fill=GRAY)
    ax = Axes(cx0, CY, CW, CH, (0, 59), (-1.25, 1.25), [0, 15, 30, 45, 59], [-1, 0, 1])
    ax.curve(xs, np.sin(xs * w), BLUE, width=2.6)
    ax.curve(xs, np.cos(xs * w), RED, width=2.6)
    text(cx0 + CW / 2, CY + CH + 33, "位置 pos", size=16, anchor="ma")
    if k == 0:
        text(cx0 - 52, CY + CH / 2, "编码值", size=16, anchor="ma")

# ---------------------------------------------------------------- D. 相似度衰减
DX, DY, DW, DH = 150, 1078, 640, 330
text(DX, 1042, "D   PE(0) 与 PE(pos) 的余弦相似度：随距离衰减", size=19)
pe_long, _ = sinusoidal_pe(1024, D_MODEL)
sim = pe_long @ pe_long[0] / (np.linalg.norm(pe_long, axis=1) * np.linalg.norm(pe_long[0]))
axd = Axes(DX, DY, DW, DH, (0, 1023), (-0.35, 1.0), [0, 256, 512, 768, 1023], [-0.2, 0, 0.4, 0.7, 1.0])
axd.curve(np.arange(1024), sim, BLUE, width=2.4)
text(DX + DW / 2, DY + DH + 33, "相对距离 pos", size=16, anchor="ma")
text(DX - 52, DY + DH / 2, "cos 相似度", size=16, anchor="ma")
text(axd.px(50), axd.py(0.90), "近距离：几乎重合", size=15, fill=GREEN)
text(axd.px(330), axd.py(0.80), "远距离：进入 0~0.4 的振荡带，并不归零", size=15, fill=GREEN)
text(axd.px(330), axd.py(0.72), "（最低频的那几十个维度几乎没变）", size=15, fill=GREEN)

# ---------------------------------------------------------------- E. 单位圆
EX, EY, EW, EH = 990, 1078, 540, 330
text(EX, 1042, "E   一对维度 (sin, cos) 落在单位圆上：推进 pos = 旋转", size=19)
cx, cy = EX + EW / 2 - 10, EY + EH / 2 - 6
R = 108
line([(cx + R * math.cos(a / 200 * 2 * math.pi), cy + R * math.sin(a / 200 * 2 * math.pi))
      for a in range(201)], fill=(186, 192, 206), width=2)
line([(cx - R - 24, cy), (cx + R + 24, cy)], fill=(215, 219, 228), width=1.5)
line([(cx, cy - R - 24), (cx, cy + R + 24)], fill=(215, 219, 228), width=1.5)
w8 = omega[8]
pts = [(cx + R * math.sin(m * w8), cy - R * math.cos(m * w8)) for m in range(9)]
line(pts, fill=(158, 167, 186), width=1.4)
for m, (px, py) in enumerate(pts):
    color = BLUE if m == 0 else (RED if m == 8 else (70, 80, 100))
    dot(px, py, 6 if m else 8, color)
    text(cx + (px - cx) * 1.26, cy + (py - cy) * 1.26, f"{m}", size=15,
         fill=color if m in (0, 8) else GRAY, anchor="mm")
text(cx, cy + 44, "圆心 = 原点", size=14, fill=(160, 166, 178), anchor="ma")
text(EX, EY + EH + 12, f"i=8 时 ω≈{w8:.3f} rad/pos：每推进一个位置就转同一个角，", size=16, fill=GREEN)
text(EX, EY + EH + 36, "所以两位置的“相对角度”只由 Δpos 决定。", size=16, fill=GREEN)

# ---------------------------------------------------------------- 输出与自检
img.resize((W, H), Image.LANCZOS).save("scratch/pe_visualization.png")

ver = np.zeros_like(pe)
for p in range(T):
    for j in range(0, D_MODEL, 2):
        ver[p, j] = math.sin(p / 10000 ** (j / D_MODEL))
        ver[p, j + 1] = math.cos(p / 10000 ** (j / D_MODEL))
print("formula check (allclose):", np.allclose(pe, ver, atol=1e-12))
print("pos=1, dims 0..5:", np.round(pe[1, :6], 5))
print("omega[0], omega[32], omega[63]:", np.round([omega[0], omega[32], omega[63]], 6))
print("sim floor at pos>=900:", np.round(sim[900:].mean(), 3))
