"""Figure 2 — the three representation configurations compared in the paper.

The classifier chain and the stacking stage are shared; only the fusion vector that
feeds the chain differs, which is what makes the comparison in Table 2 an ablation.

Colour: palette slots 3 (aqua, 2D) and 7 (violet, 3D). Deliberately NOT the blue/orange
of Figures 4-5, where those hues already denote the threshold protocol. Every box carries
a text label, which supplies the relief the aqua fill's contrast WARN requires.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C2D, C3D = "#1baf7a", "#4a3aa7"
F2D, F3D = "#e8f7f1", "#eceaf6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GREY, FGREY = "#6f6e69", "#f2f1ee"
SURFACE = "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
fig, ax = plt.subplots(figsize=(5.833, 3.339), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x0, x1, y0, y1, title, sub=None, edge=GREY, face=FGREY, tsize=7.0, ssize=5.9):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2))
    cx, cy = (x0+x1)/2, (y0+y1)/2
    if sub:
        ax.text(cx, cy+2.4, title, ha="center", va="center", fontsize=tsize,
                color=INK, fontweight="bold", zorder=3)
        ax.text(cx, cy-3.0, sub, ha="center", va="center", fontsize=ssize,
                color=INK2, zorder=3, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize,
                color=INK, fontweight="bold", zorder=3)

def arrow(x0, y0, x1, y1, color=MUTED, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
                 color=color, zorder=1,
                 connectionstyle=f"arc3,rad={rad}",
                 shrinkA=0, shrinkB=1))

# band marking the ablation axis
ax.add_patch(FancyBboxPatch((1.5, 41.5), 97, 18,
             boxstyle="round,pad=0,rounding_size=1.6",
             linewidth=0, facecolor="#f7f6f3", zorder=0))

# row 1 — inputs
box(5, 47, 84, 98, "916-d 2D feature vector",
    "Morgan 729 bit  +  MACCS 167  +  20 descriptors", C2D, F2D)
box(53, 95, 84, 98, "3D conformer graph",
    "ETKDGv3 / MMFF94; covalent + ≤ 5 Å proximity edges,\nweighted by inverse interatomic distance", C3D, F3D)

# row 2 — branches
box(5, 47, 67, 79, "2-layer MLP branch", "→  h₂D  (128-d)", C2D, F2D)
box(53, 95, 67, 79, "3 × GCNConv + global add pool", "→  h₃D  (128-d)", C3D, F3D)

# row 3 — the three configurations
box(3, 31, 44, 57, "2D-only", "h₂D   (128-d)", C2D, F2D)
box(36, 64, 44, 57, "2D–3D hybrid", "[ h₂D ‖ h₃D ]   (256-d)", GREY, "#f0eef7")
box(69, 97, 44, 57, "3D-only", "h₃D   (128-d)", C3D, F3D)

# row 4 — shared chain
box(3, 97, 25, 38, "Differentiable 22-class classifier chain",
    "head k receives the fusion vector concatenated with the raw logits of heads 1 … k−1,\n"
    "so label dependencies are learned end-to-end", GREY, FGREY, 7.0, 5.9)

# row 5 — output
box(3, 97, 5, 19, "Per-class blend with the 2D tree ensemble, then calibrated thresholds",
    "blend weight αₖ and threshold τₖ tuned jointly per class on validation  →  22 GHS hazard labels",
    GREY, FGREY, 7.0, 5.9)

# arrows
arrow(26, 84, 26, 79.4, C2D)
arrow(74, 84, 74, 79.4, C3D)
arrow(20, 67, 17, 57.4, C2D)
arrow(32, 67, 44, 57.4, C2D, rad=-0.12)
arrow(80, 67, 83, 57.4, C3D)
arrow(68, 67, 56, 57.4, C3D, rad=0.12)
for x in (17, 50, 83):
    arrow(x, 44, x, 38.4, MUTED)
arrow(50, 25, 50, 19.4, MUTED)

fig.tight_layout(pad=0.15)
fig.savefig("/tmp/exp/figure2.png", dpi=300)
print("figure2.png written")
