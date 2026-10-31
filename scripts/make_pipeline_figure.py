"""Column-width pipeline figure for the report: paper/fig/pipeline_compact.pdf (vector)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, PINK, GREY, LINE, TXT = "#C00000", "#FBE9E9", "#F2F2F2", "#595959", "#262626"
TITLE, SUB = 5.6, 5.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": TITLE, "mathtext.fontset": "dejavusans"})

fig = plt.figure(figsize=(3.3, 1.42))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 43)
ax.axis("off")


def box(x, y, w, h, title, sub="", fc=GREY, ec=LINE, lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", weight="bold", color=TXT)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", color=TXT, fontsize=SUB)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", weight="bold", color=TXT)


def arrow(p, q, color=LINE, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=0.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# row 1: observation -> prefix (run once per re-plan) -> KV cache
box(0.5, 25, 21, 15, "Observation $o_k$", "2 cameras, text,\n8-D robot state")
ax.add_patch(FancyBboxPatch((24, 23.5), 52, 18, boxstyle="round,pad=0,rounding_size=1.2",
                            fc="white", ec=RED, lw=0.7, ls=(0, (3, 2))))
ax.text(50, 39.0, "prefix: once per re-plan", ha="center", va="center", color=RED, fontsize=SUB, weight="bold")
box(25.5, 25.5, 12, 11, "SigLIP", "86M")
box(39.5, 25.5, 14, 11, "Connector", "12M")
box(55.5, 25.5, 19, 11, "SmolLM2", "362M, 32 layers")
box(80, 27, 19.5, 11, "KV cache", "per layer", fc=PINK, ec=RED)
arrow((21.5, 32.5), (25.5, 31))
arrow((37.5, 31), (39.5, 31))
arrow((53.5, 31), (55.5, 31))
arrow((74.5, 31), (80, 32.5))

# row 2: action expert (S passes) -> chunk -> simulator -> next observation
box(61, 2, 38.5, 15, r"Action expert $\times S$ (97M)",
    r"$\mathbf{A}\leftarrow\mathbf{A}-\frac{1}{S}v_\theta(\mathbf{A},\tau\mid o)$, $\tau{:}\,1\to0$",
    fc=PINK, ec=RED, lw=0.8)
ax.text(93.5, 22, r"noise $\epsilon$", ha="center", va="center", color=TXT, fontsize=SUB)
arrow((89, 27), (87, 17), color=RED)
box(35.5, 3.5, 22, 12, "Chunk 50×7", "1st action executed")
box(0.5, 3.5, 28, 12, "LIBERO simulator", "20 Hz, then re-plan")
arrow((61, 9.5), (57.5, 9.5))
arrow((35.5, 9.5), (28.5, 9.5))
arrow((11, 15.5), (11, 25))

out = Path(__file__).resolve().parents[1] / "paper" / "fig" / "pipeline_compact.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=300)
print("saved", out)
