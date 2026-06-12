"""
Génère les figures du rapport à partir des résultats sauvegardés.

Figures produites (dans figures/) :
  1. fig_evolution.png       : courbes de fitness/score au fil des générations.
  2. fig_comparaison.png     : barres comparatives score moyen + survie.
  3. fig_distribution.png    : distribution des scores de l'agent AG (test).
  4. fig_robustesse.png      : score selon le nombre d'obstacles.
"""

from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})

# Palette cohérente.
C_AG = "#2563eb"
C_HEUR = "#16a34a"
C_RAND = "#dc2626"
C_BEST = "#2563eb"
C_MEAN = "#f59e0b"

with open(os.path.join(RESULTS, "history.json")) as f:
    history = json.load(f)
with open(os.path.join(RESULTS, "results.json")) as f:
    results = json.load(f)

gens = [h["generation"] for h in history]

# --------------------------------------------------------------------------- #
# Figure 1 : évolution                                                        #
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(gens, [h["best_fitness"] for h in history], color=C_BEST,
        label="Meilleure fitness", linewidth=1.8)
ax.plot(gens, [h["mean_fitness"] for h in history], color=C_MEAN,
        label="Fitness moyenne", linewidth=1.5)
ax.set_xlabel("Génération")
ax.set_ylabel("Fitness")
ax.set_title("Évolution de la fitness")
ax.legend()

ax = axes[1]
ax.plot(gens, [h["best_score"] for h in history], color=C_BEST,
        label="Meilleur score", linewidth=1.8)
ax.plot(gens, [h["mean_score"] for h in history], color=C_MEAN,
        label="Score moyen population", linewidth=1.5)
ax.set_xlabel("Génération")
ax.set_ylabel("Score (nourriture mangée)")
ax.set_title("Évolution du score")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_evolution.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 2 : comparaison des agents                                           #
# --------------------------------------------------------------------------- #
names = ["Aléatoire", "Heuristique", "AG (neuroévolution)"]
colors = [C_RAND, C_HEUR, C_AG]
ev = results["evaluation"]
mean_scores = [ev[n]["mean_score"] for n in names]
std_scores = [ev[n]["std_score"] for n in names]
mean_steps = [ev[n]["mean_steps"] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
bars = ax.bar(names, mean_scores, yerr=std_scores, capsize=5, color=colors,
              alpha=0.85)
ax.set_ylabel("Score moyen (± écart-type)")
ax.set_title("Score moyen sur 200 épisodes de test")
for b, v in zip(bars, mean_scores):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", labelrotation=12)

ax = axes[1]
bars = ax.bar(names, mean_steps, color=colors, alpha=0.85)
ax.set_ylabel("Survie moyenne (nb de pas)")
ax.set_title("Durée de survie moyenne")
for b, v in zip(bars, mean_steps):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
            f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", labelrotation=12)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_comparaison.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 3 : distribution des scores AG                                       #
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 4.2))
ag_scores = ev["AG (neuroévolution)"]["scores"]
heur_scores = ev["Heuristique"]["scores"]
bins = np.arange(0, max(max(ag_scores), max(heur_scores)) + 2) - 0.5
ax.hist(heur_scores, bins=bins, alpha=0.6, color=C_HEUR, label="Heuristique")
ax.hist(ag_scores, bins=bins, alpha=0.7, color=C_AG, label="AG (neuroévolution)")
ax.axvline(np.mean(ag_scores), color=C_AG, linestyle="--", linewidth=1.5)
ax.axvline(np.mean(heur_scores), color=C_HEUR, linestyle="--", linewidth=1.5)
ax.set_xlabel("Score (nourriture mangée par épisode)")
ax.set_ylabel("Nombre d'épisodes")
ax.set_title("Distribution des scores sur 200 épisodes de test")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_distribution.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 4 : robustesse vs obstacles                                          #
# --------------------------------------------------------------------------- #
levels = results["obstacle_levels"]
rob = results["robustness"]
fig, ax = plt.subplots(figsize=(7, 4.2))
for n, c in zip(names, colors):
    ys = [rob[n][str(l)] for l in levels]
    ax.plot(levels, ys, marker="o", color=c, label=n, linewidth=1.8)
ax.set_xlabel("Nombre d'obstacles")
ax.set_ylabel("Score moyen")
ax.set_title("Robustesse : score selon la densité d'obstacles")
ax.set_xticks(levels)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_robustesse.png"), bbox_inches="tight")
plt.close(fig)

print("Figures générées dans", FIGS)
for f in sorted(os.listdir(FIGS)):
    print(" -", f)
