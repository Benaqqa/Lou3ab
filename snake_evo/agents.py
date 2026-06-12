"""
Agents pour l'environnement Snake.

Contient :
  - GeneticAgent : agent neuroévolutionnaire (politique = MLP, poids optimisés
    par algorithme génétique). C'est l'agent principal du projet.
  - RandomAgent : baseline aléatoire.
  - HeuristicAgent : baseline heuristique simple (greedy vers la nourriture,
    évite le danger immédiat).
"""

from __future__ import annotations

import numpy as np

from .network import MLPPolicy
from .env import ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT


class GeneticAgent:
    """Agent dont la politique est un MLP paramétré par un génome évolué."""

    def __init__(self, genome: np.ndarray, policy: MLPPolicy):
        self.genome = np.asarray(genome, dtype=np.float64)
        self.policy = policy

    def act(self, obs: np.ndarray) -> int:
        return self.policy.act(self.genome, obs)


class RandomAgent:
    """Baseline : choisit une action uniformément au hasard."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, 3))


class HeuristicAgent:
    """Baseline heuristique.

    Règle : parmi les actions ne menant pas à un danger immédiat (lu dans
    l'observation), choisir celle qui rapproche de la nourriture. En cas
    d'égalité ou de blocage, aller tout droit si possible.

    L'observation encode :
      [0] danger tout droit, [1] danger droite, [2] danger gauche
      [3:7] direction one-hot, [7..10] nourriture gauche/droite/haut/bas
    """

    # Pour chaque direction courante (index 0..3), donne la direction absolue
    # résultant d'une action (tout droit / droite / gauche).
    # Directions absolues : 0=HAUT,1=DROITE,2=BAS,3=GAUCHE
    def act(self, obs: np.ndarray) -> int:
        danger = {ACTION_STRAIGHT: obs[0], ACTION_RIGHT: obs[1], ACTION_LEFT: obs[2]}
        cur_dir = int(np.argmax(obs[3:7]))
        food_left, food_right, food_up, food_down = obs[7], obs[8], obs[9], obs[10]

        # Direction absolue souhaitée (vers la nourriture) — on privilégie l'axe
        # avec le plus grand écart ; ici on teste simplement les deux axes.
        desired = set()
        if food_up:
            desired.add(0)
        if food_down:
            desired.add(2)
        if food_right:
            desired.add(1)
        if food_left:
            desired.add(3)

        # Direction absolue pour chaque action.
        action_to_absdir = {
            ACTION_STRAIGHT: cur_dir,
            ACTION_RIGHT: (cur_dir + 1) % 4,
            ACTION_LEFT: (cur_dir - 1) % 4,
        }

        # Candidats sûrs.
        safe = [a for a in (ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT)
                if danger[a] < 0.5]
        if not safe:
            return ACTION_STRAIGHT  # condamné de toute façon

        # Parmi les sûrs, préférer ceux qui vont vers la nourriture.
        toward = [a for a in safe if action_to_absdir[a] in desired]
        if toward:
            # privilégier tout droit si possible
            if ACTION_STRAIGHT in toward:
                return ACTION_STRAIGHT
            return toward[0]

        # Sinon, action sûre, en privilégiant tout droit.
        if ACTION_STRAIGHT in safe:
            return ACTION_STRAIGHT
        return safe[0]
