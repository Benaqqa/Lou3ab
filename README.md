# Snake évolutif sur grille dynamique — Projet final (Sujet 2)

**Module :** M2442 — Jeux Vidéo IA — ENSIAS
**Équipe :** Benaqqa Moubarak, Bensmina Anass

Agent neuroévolutionnaire (réseau de neurones optimisé par algorithme génétique)
jouant à une variante dynamique du jeu Snake, dans un environnement Gymnasium
personnalisé.

## Structure du projet

```
snake_evolutif/
├── snake_evo/              # Code source du projet
│   ├── env.py              # Environnement Gymnasium personnalisé (Snake dynamique)
│   ├── network.py          # Politique : perceptron multicouche (MLP)
│   ├── agents.py           # Agents : génétique, aléatoire, heuristique
│   ├── evolution.py        # Algorithme génétique (sélection, croisement, mutation, élitisme)
│   ├── evaluate.py         # Protocole d'évaluation (métriques multi-épisodes)
│   └── render_pygame.py    # Rendu graphique (fenêtre animée) pour la démo
├── experiments/
│   ├── train.py            # Entraînement + évaluation comparative + robustesse
│   ├── make_figures.py     # Génération des figures du rapport
│   └── play.py             # Démonstration du meilleur agent (fenêtre graphique ou texte)
├── results/                # Génome, historique, résultats JSON, log d'entraînement
├── figures/                # Graphiques générés
├── report/                 # Rapport LaTeX (rapport.tex + rapport.pdf)
├── slides/                 # Slides de soutenance (soutenance.tex + soutenance.pdf)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproduire l'expérience

```bash
# 1. Entraîner l'agent et évaluer contre les baselines (génère results/)
python experiments/train.py

# 2. Générer les figures à partir des résultats
python experiments/make_figures.py

# 3. (Optionnel) Voir le meilleur agent jouer
python experiments/play.py                 # fenêtre graphique (par défaut)
python experiments/play.py 900011          # graine donnant un bon score
python experiments/play.py 900011 --text   # repli en mode texte (terminal)
python experiments/play.py --fps 8         # ralentir l'animation
```

### Démonstration graphique (soutenance)

Par défaut, `play.py` ouvre une **fenêtre Pygame** animée : grille colorée,
serpent en dégradé, nourriture, obstacles et panneau d'informations en direct
(score, pas, longueur, obstacles). Touches **Échap** ou **Q** pour quitter.
Si Pygame n'est pas disponible, l'affichage bascule automatiquement en mode
texte. Pour une démo sans risque, choisissez à l'avance une graine au bon
rendu (ex. `900011`).

## Résumé de l'environnement

- **Grille :** 12×12, obstacles fixes en nombre variable (3 à 10) selon l'épisode.
- **Actions (discrètes) :** tout droit / tourner à droite / tourner à gauche.
- **Observation (11 valeurs) :** danger immédiat (3), direction one-hot (4),
  position relative de la nourriture (4).
- **Récompense :** +1 manger, −1 collision, −0.5 famine, ±0.1 rapprochement,
  −0.01 coût de temps, +5 victoire.
- **Fin d'épisode :** collision, famine, victoire (grille pleine), troncature.

## Agent et algorithme génétique

- **Politique :** MLP 11 → 16 (tanh) → 3, action = argmax (déterministe).
- **Génome :** 243 poids du réseau.
- **Fitness :** `100·score_moyen + 0.5·pas_moyen + récompense_moyenne`.
- **Opérateurs :** sélection par tournoi (5), croisement uniforme (0.7),
  mutation gaussienne (taux 0.1, σ décroissant 0.25→0.10), élitisme (12 %).
- **Population :** 150 — **Générations :** 150.

## Principaux résultats (200 épisodes de test indépendants)

| Agent               | Score moyen | Score max | Survie moyenne | Taux de succès |
|---------------------|:-----------:|:---------:|:--------------:|:--------------:|
| Aléatoire           | 0.09        | 2         | 15.4           | 8.5 %          |
| **AG (neuroévol.)** | **6.30**    | 25        | **172.8**      | 91.5 %         |
| Heuristique (réf.)  | 15.97       | 36        | 153.1          | 100 %          |

L'agent évolué multiplie le score par ~70 par rapport au hasard et survit plus
longtemps que l'heuristique. Sans obstacle, il égale presque l'heuristique
(18.9 contre 19.9) : c'est la gestion des obstacles denses qui constitue sa
principale marge de progrès.
```
