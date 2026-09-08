"""Regenerate Figures 4 and 5 from the calibrated results.

Figure 4  macro-F1 for every configuration under both threshold protocols.
Figure 5  what threshold calibration is worth under each evaluation protocol.

Colour: categorical slots 1 and 2 of the validated reference palette
(#2a78d6 / #eb6834; CVD dE 24.7, normal-vision dE 33.6, both PASS).
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_HALF, C_CAL = "#2a78d6", "#eb6834"          # series 1 = fixed 0.5, series 2 = calibrated
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

T2 = json.load(open("/tmp/exp/new_tables.json"))["table2"]

# ---------------------------------------------------------------- Figure 4
ROWS = [("Stacked: tree + 2D–3D hybrid", "stack_both"),
        ("Stacked: tree + 2D-only",           "stack_d2"),
        ("Stacked: tree + 3D-only",           "stack_d3"),
        ("2D tree ensemble",                  "tree"),
        ("2D-only MLP-ECC",                   "d2_alone"),
        ("2D–3D hybrid GNN-ECC",         "both_alone"),
        ("3D-only GNN-ECC",                   "d3_alone")]

fig, ax = plt.subplots(figsize=(6.0, 3.634), dpi=300)
h = 0.34
ys = list(range(len(ROWS)))[::-1]
for y, (label, key) in zip(ys, ROWS):
    v05, vcal = T2[key]["macro_f1_at_0.5"], T2[key]["macro_f1"]
    ax.barh(y + h/2 + 0.012, v05,  height=h, color=C_HALF, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.barh(y - h/2 - 0.012, vcal, height=h, color=C_CAL,  edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.text(v05 + 0.006,  y + h/2 + 0.012, f"{v05:.3f}",  va="center", ha="left", fontsize=7, color=INK2)
    ax.text(vcal + 0.006, y - h/2 - 0.012, f"{vcal:.3f}", va="center", ha="left",
            fontsize=7, color=INK, fontweight="bold")

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], color=INK)
ax.set_xlim(0, 0.78); ax.set_xlabel("Macro-F1 (test set, n = 2,803)", color=INK2)
ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.set_ylim(-0.72, 6.78)
ax.axhline(3.5, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
ax.text(0.757, 6.52, "stacked with the tree ensemble", ha="right", fontsize=7,
        color=MUTED, style="italic", va="center")
ax.text(0.757, 3.30, "single model", ha="right", fontsize=7,
        color=MUTED, style="italic", va="center")
ax.legend(handles=[Line2D([], [], color=C_HALF, lw=6, label="Fixed 0.5 threshold"),
                   Line2D([], [], color=C_CAL,  lw=6, label="Per-class calibrated threshold")],
          loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
          fontsize=7.5, labelcolor=INK2, handlelength=1.1, borderpad=0.2,
          columnspacing=1.6)
fig.tight_layout(pad=0.5)
fig.savefig("/tmp/exp/figure4.png", dpi=300)
print("figure4.png written")

# ---------------------------------------------------------------- Figure 5
PROTO = [("Murcko\nscaffold-disjoint", 0.4527, 0.5390, None,   None),
         ("Five-fold\ncross-validation", 0.5672, 0.6439, 0.0170, 0.0149),
         ("Random split",                0.5608, 0.6264, None,   None)]

fig, ax = plt.subplots(figsize=(4.2, 4.013), dpi=300)
for y, (label, a, b, sa, sb) in enumerate(PROTO):
    ax.plot([a, b], [y, y], color=MUTED, linewidth=2, zorder=2, solid_capstyle="round")
    if sa: ax.errorbar([a], [y], xerr=[sa], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2.5, zorder=3)
    if sb: ax.errorbar([b], [y], xerr=[sb], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2.5, zorder=3)
    ax.scatter([a], [y], s=95, color=C_HALF, zorder=4, edgecolor=SURFACE, linewidth=1.4)
    ax.scatter([b], [y], s=95, color=C_CAL,  zorder=4, edgecolor=SURFACE, linewidth=1.4)
    ax.text(a - 0.013 - (sa or 0), y, f"{a:.3f}", va="center", ha="right",
            fontsize=7.5, color=INK2)
    ax.text(b + 0.013 + (sb or 0), y, f"{b:.3f}", va="center", ha="left",
            fontsize=7.5, color=INK, fontweight="bold")
    ax.text((a + b) / 2, y + 0.15, f"+{b - a:.3f}", va="bottom", ha="center",
            fontsize=8, color=INK, fontweight="bold")

ax.set_yticks(range(len(PROTO))); ax.set_yticklabels([p[0] for p in PROTO], color=INK)
ax.set_ylim(-0.42, len(PROTO) - 0.52)
ax.set_xlim(0.405, 0.695); ax.set_xlabel("Macro-F1, 2D tree ensemble", color=INK2)
ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", color=SURFACE, markerfacecolor=C_HALF,
                          markersize=8, label="Fixed 0.5 threshold"),
                   Line2D([], [], marker="o", color=SURFACE, markerfacecolor=C_CAL,
                          markersize=8, label="Per-class calibrated")],
          loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
          fontsize=7.5, labelcolor=INK2, handlelength=1.0, ncol=2, borderpad=0.2,
          columnspacing=1.4)
fig.tight_layout(pad=0.5)
fig.savefig("/tmp/exp/figure5.png", dpi=300)
print("figure5.png written")
