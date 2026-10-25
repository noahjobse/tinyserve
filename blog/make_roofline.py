"""Roofline diagram for the tinyserve blog, RTX 4050 Laptop GPU.
Run from repo root:  .venv/bin/python blog/make_roofline.py  -> blog/part1-roofline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "part1-roofline.png"

# RTX 4050 Laptop GPU
BW = 192.0     # GB/s   (192e9 bytes/s)
PEAK = 8.986   # FP16 TFLOP/s  (8.986e12 FLOP/s)
RIDGE = PEAK * 1e12 / (BW * 1e9)   # FLOPs per byte ~= 46.8

INK, MUTED = "#1f2937", "#6b7280"
BLUE, AMBER, GRID = "#2563eb", "#d97706", "#eceff1"

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

x = np.logspace(-0.3, 3, 400)        # 0.5 .. 1000 FLOPs/byte
y_mem = BW * x / 1000.0              # memory-bandwidth roof, in TFLOP/s
y_roof = np.minimum(y_mem, PEAK)     # the actual roofline

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.5, 1000); ax.set_ylim(0.05, 20)

# regions
ax.axvspan(0.5, RIDGE, color=BLUE, alpha=0.05)
ax.axvspan(RIDGE, 1000, color=AMBER, alpha=0.05)

# the two roofs (dashed) and the roofline itself (bold)
ax.plot(x, y_mem, color=BLUE, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.hlines(PEAK, 0.5, 1000, color=AMBER, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.plot(x, y_roof, color=INK, lw=2.6, zorder=5)

# key points
x_sr, y_sr = 1.0, BW * 1.0 / 1000.0          # single request ~1 FLOP/byte
ax.plot([x_sr], [y_sr], "o", color=BLUE, ms=8, zorder=6)
ax.plot([RIDGE], [PEAK], "o", color=INK, ms=8, zorder=6)

# batching arrow along the slope, single request -> ridge
ax.annotate("", xy=(RIDGE, PEAK), xytext=(x_sr, y_sr),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8,
                            shrinkA=10, shrinkB=10))
ax.text(4.5, 2.7, "batching raises\narithmetic intensity", color=MUTED,
        fontsize=9.5, style="italic", ha="left", va="center")

# labels
ax.text(0.62, 0.075, "memory-bound", color=BLUE, fontsize=11, weight="bold")
ax.text(150, 12.5, "compute-bound", color=AMBER, fontsize=11, weight="bold", ha="center")
ax.text(260, PEAK * 1.12, "compute roof: 8.986 FP16 TFLOP/s", color=AMBER, fontsize=9, ha="center")
ax.text(0.78, 1.5, "bandwidth roof\n192 GB/s", color=BLUE, fontsize=9, ha="left", va="center")

ax.annotate("single request\n~1 FLOP/byte", (x_sr, y_sr),
            xytext=(1.25, 0.11), fontsize=9.5, color=INK, va="center")
ax.annotate(f"ridge ≈ {RIDGE:.0f} FLOP/byte\n(≈ batch 47)", (RIDGE, PEAK),
            xytext=(60, 4.2), fontsize=9.5, color=INK, weight="bold",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))

ax.set_title("Roofline, RTX 4050 Laptop GPU", fontsize=14, weight="bold", color=INK, pad=12)
ax.set_xlabel("arithmetic intensity (FLOPs per byte)", fontsize=10.5, color=INK)
ax.set_ylabel("attainable performance (TFLOP/s)", fontsize=10.5, color=INK)
ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
