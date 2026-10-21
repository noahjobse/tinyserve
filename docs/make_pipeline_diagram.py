"""Living diagram of tinyserve's generation pipeline.
We update STAGES (and later add detail / new boxes) as the project progresses.
Status per stage: "done" (understood/built), "current" (what we're on now), "todo" (coming up).
Run: python docs/make_pipeline_diagram.py  -> writes docs/pipeline.png
"""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "pipeline.png"
DOWNLOADS = Path("/mnt/c/Users/noahj/Downloads")

# colors
INK = "#1f2937"
MUTED = "#6b7280"
BLUE = "#2563eb"
BLUE_FILL = "#dbeafe"
AMBER = "#d97706"
AMBER_FILL = "#fef3c7"
GRAY = "#9ca3af"
GRAY_FILL = "#f3f4f6"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# The pipeline. Edit statuses / labels / add entries as we go.
STAGES = [
    ("Text", '"The cat sat"', "done"),
    ("Tokenize", "text <-> ID numbers", "done"),
    ("Embeddings", "IDs -> vectors (512-d)", "done"),
    ("Transformer", "attention + MLP", "current"),
    ("Next-token scores", "~50k token scores", "todo"),
    ("Pick a token", "sample one", "todo"),
]

STYLE = {
    "done":    dict(fill=BLUE_FILL,  edge=BLUE,  tc=INK,   lw=1.8, ls="solid"),
    "current": dict(fill=AMBER_FILL, edge=AMBER, tc=INK,   lw=2.4, ls="solid"),
    "todo":    dict(fill=GRAY_FILL,  edge=GRAY,  tc=MUTED, lw=1.4, ls=(0, (4, 3))),
}


def draw():
    n = len(STAGES)
    fig, ax = plt.subplots(figsize=(2.6 * n, 5), dpi=200)
    W = 2.6 * n
    ax.set_xlim(0, W)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.15, 4.65, "tinyserve: the generation pipeline", fontsize=17, weight="bold", color=INK)
    ax.text(0.15, 4.25, "What we understand so far, and where we're headed. Updated as we build.",
            fontsize=10.5, color=MUTED)

    bw, bh, cy = 2.0, 1.15, 2.7
    centers = [1.3 + i * 2.6 for i in range(n)]

    for (title, sub, status), cx in zip(STAGES, centers):
        s = STYLE[status]
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=s["lw"], edgecolor=s["edge"], facecolor=s["fill"],
            linestyle=s["ls"], zorder=2,
        ))
        ax.text(cx, cy + 0.17, title, ha="center", va="center", fontsize=11.5, weight="bold", color=s["tc"], zorder=3)
        ax.text(cx, cy - 0.27, sub, ha="center", va="center", fontsize=9, color=s["tc"], zorder=3)
        if status == "current":
            ax.text(cx, cy - bh / 2 - 0.28, "you are here", ha="center", va="center",
                    fontsize=9.5, weight="bold", color=AMBER, style="italic")

    # forward arrows
    for a, b in zip(centers, centers[1:]):
        ax.add_patch(FancyArrowPatch((a + bw / 2, cy), (b - bw / 2, cy),
                     arrowstyle="-|>", mutation_scale=15, linewidth=1.8, color=MUTED, zorder=1))

    # generation loop: last box -> down -> back to first -> up
    y_lo = 1.15
    first, last = centers[0], centers[-1]
    ax.add_patch(FancyArrowPatch((last, cy - bh / 2), (last, y_lo),
                 arrowstyle="-", linewidth=1.6, color=GRAY, zorder=1))
    ax.add_patch(FancyArrowPatch((last, y_lo), (first, y_lo),
                 arrowstyle="-", linewidth=1.6, color=GRAY, zorder=1))
    ax.add_patch(FancyArrowPatch((first, y_lo), (first, cy - bh / 2),
                 arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color=GRAY, zorder=1))
    ax.text((first + last) / 2, y_lo - 0.28, "append the new token, feed it all back in, repeat  (this loop is text generation)",
            ha="center", va="center", fontsize=9.5, color=GRAY, style="italic")

    # legend
    lx, ly = 0.3, 0.45
    for label, st in [("understood", "done"), ("working on now", "current"), ("coming up", "todo")]:
        s = STYLE[st]
        ax.add_patch(FancyBboxPatch((lx, ly - 0.12), 0.35, 0.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                     linewidth=s["lw"], edgecolor=s["edge"], facecolor=s["fill"], linestyle=s["ls"]))
        ax.text(lx + 0.5, ly, label, fontsize=9.5, color=INK, va="center")
        lx += 3.0

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    if DOWNLOADS.exists():
        shutil.copy(OUT, DOWNLOADS / "tinyserve-pipeline.png")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    draw()
