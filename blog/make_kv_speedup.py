"""KV-cache speedup chart for the tinyserve blog (post 3).
Decode throughput vs sequence length: the cache holds flat, naive degrades.
Run:  python blog/make_kv_speedup.py  -> blog/kv-speedup.png

Data: measured on a 6GB laptop GPU (RTX 4050), 51M-param model, greedy decode
from a short prompt. Two measured lengths (64, 256 new tokens).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the locked chart identity lives in the goals content folder
sys.path.insert(0, "/home/noah/projects/personal/goals/content")
from chart_style import apply_identity, style_axes, BLUE, AMBER, MUTED, INK  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "kv-speedup.png"

# measured tokens/sec at two generation lengths
x = [64, 256]
cache = [250, 250]   # KV cache: flat
naive = [242, 179]   # recompute-everything: degrades with length

apply_identity()
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)

ax.plot(x, cache, color=BLUE, lw=2.6, marker="o", ms=6, label="KV cache", zorder=5)
ax.plot(x, naive, color=AMBER, lw=2.0, ls=(0, (4, 3)), marker="o", ms=6,
        label="naive (recompute each step)", zorder=4)

# endpoint value labels
ax.annotate("250", (256, 250), textcoords="offset points", xytext=(10, 2),
            color=BLUE, fontsize=11, fontweight="bold", fontfamily="monospace")
ax.annotate("179", (256, 179), textcoords="offset points", xytext=(10, -4),
            color=AMBER, fontsize=11, fontweight="bold", fontfamily="monospace")

# the gap = 1.4x
ax.annotate("1.4x", (256, 214), textcoords="offset points", xytext=(34, 0),
            color=INK, fontsize=12, fontweight="bold", fontfamily="monospace", va="center")
ax.annotate("", xy=(256, 250), xytext=(256, 179),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))

ax.set_xlim(40, 300); ax.set_ylim(150, 275)
ax.set_xticks([64, 256])
ax.set_xlabel("new tokens generated", fontsize=12)
ax.set_ylabel("tokens / sec", fontsize=12)
ax.set_title("Decode throughput as the sequence grows", loc="left", pad=12)
ax.legend(loc="lower left", fontsize=11)
style_axes(ax)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
assert OUT.stat().st_size > 0, "chart did not render"
print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB)")
