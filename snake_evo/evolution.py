"""
Algorithme génétique pour l'évolution des poids du MLP.

Opérateurs implémentés :
  - Sélection : par tournoi.
  - Croisement : croisement uniforme (BLX-like par mélange de gènes parents).
  - Mutation : bruit gaussien additif avec taux de mutation par gène.
  - Élitisme : conservation des meilleurs individus d'une génération à l'autre.

Fonction de fitness : moyenne, sur plusieurs épisodes (graines fixées), d'un
score combinant la nourriture mangée (objectif principal), la survie (nombre de
pas) et la récompense cumulée. Voir `evaluate_genome`.
"""

from __future__ import annotations

import numpy as np

from .env import SnakeEnv
from .network import MLPPolicy
from .agents import GeneticAgent


def evaluate_genome(
    genome: np.ndarray,
    policy: MLPPolicy,
    env_kwargs: dict,
    eval_seeds: list[int],
    max_steps: int = 500,
) -> dict:
    """Évalue un génome sur un ensemble d'épisodes (graines fixées).

    Renvoie un dictionnaire avec fitness moyenne et métriques détaillées.
    La fitness pondère fortement la nourriture (score) tout en valorisant la
    survie, ce qui correspond à l'objectif du sujet : « maximiser survie et
    score ».
    """
    agent = GeneticAgent(genome, policy)
    scores, steps_list, rewards = [], [], []

    for seed in eval_seeds:
        env = SnakeEnv(**env_kwargs, seed=seed)
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        scores.append(info["score"])
        steps_list.append(env.steps)
        rewards.append(total_reward)

    mean_score = float(np.mean(scores))
    mean_steps = float(np.mean(steps_list))
    mean_reward = float(np.mean(rewards))

    # Fitness : la nourriture domine (x100), la survie est un bonus secondaire
    # (évite la récompense pour tourner en rond sans manger).
    fitness = mean_score * 100.0 + mean_steps * 0.5 + mean_reward

    return {
        "fitness": fitness,
        "mean_score": mean_score,
        "mean_steps": mean_steps,
        "mean_reward": mean_reward,
        "scores": scores,
    }


class GeneticAlgorithm:
    """Algorithme génétique à population fixe avec élitisme."""

    def __init__(
        self,
        policy: MLPPolicy,
        env_kwargs: dict,
        pop_size: int = 80,
        elite_frac: float = 0.15,
        tournament_size: int = 4,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.3,
        crossover_rate: float = 0.7,
        n_eval_episodes: int = 5,
        max_steps: int = 500,
        seed: int = 0,
    ):
        self.policy = policy
        self.env_kwargs = env_kwargs
        self.pop_size = pop_size
        self.n_elite = max(1, int(round(elite_frac * pop_size)))
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)
        self.n_params = policy.n_params

        # Population initiale : poids gaussiens.
        self.population = self.rng.normal(
            0.0, 0.5, size=(pop_size, self.n_params)
        )

        # Graines d'évaluation fixées (mêmes conditions pour tous les individus
        # d'une génération => comparaison équitable). Renouvelées chaque génération
        # pour éviter le surapprentissage à un jeu de cartes.
        self.history = []  # liste de dicts par génération

    def _eval_seeds(self, generation: int) -> list[int]:
        base = 10_000 + generation * 100
        return [base + i for i in range(self.n_eval_episodes)]

    def _tournament(self, fitnesses: np.ndarray) -> int:
        idx = self.rng.integers(0, self.pop_size, size=self.tournament_size)
        best = idx[np.argmax(fitnesses[idx])]
        return int(best)

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.crossover_rate:
            return p1.copy()
        mask = self.rng.random(self.n_params) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.n_params) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_scale, size=self.n_params)
        return genome + mask * noise

    def evolve(self, n_generations: int, verbose: bool = True):
        best_genome = None
        best_fitness = -np.inf
        base_scale = self.mutation_scale

        for gen in range(n_generations):
            # Décroissance linéaire de l'amplitude de mutation : exploration
            # forte au début, affinage en fin d'évolution.
            frac = gen / max(1, n_generations - 1)
            self.mutation_scale = base_scale * (1.0 - 0.6 * frac)
            seeds = self._eval_seeds(gen)
            results = [
                evaluate_genome(ind, self.policy, self.env_kwargs, seeds,
                                self.max_steps)
                for ind in self.population
            ]
            fitnesses = np.array([r["fitness"] for r in results])
            scores = np.array([r["mean_score"] for r in results])
            steps = np.array([r["mean_steps"] for r in results])

            order = np.argsort(fitnesses)[::-1]
            gen_best_idx = order[0]
            gen_best_fit = fitnesses[gen_best_idx]

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_genome = self.population[gen_best_idx].copy()

            rec = {
                "generation": gen,
                "best_fitness": float(fitnesses[gen_best_idx]),
                "mean_fitness": float(np.mean(fitnesses)),
                "best_score": float(scores[gen_best_idx]),
                "mean_score": float(np.mean(scores)),
                "best_steps": float(steps[gen_best_idx]),
                "mean_steps": float(np.mean(steps)),
            }
            self.history.append(rec)

            if verbose:
                print(
                    f"Gen {gen:3d} | best_fit={rec['best_fitness']:8.2f} "
                    f"mean_fit={rec['mean_fitness']:8.2f} "
                    f"best_score={rec['best_score']:5.2f} "
                    f"mean_score={rec['mean_score']:5.2f} "
                    f"best_steps={rec['best_steps']:6.1f}",
                    flush=True,
                )

            # Construction de la nouvelle population.
            new_pop = []
            # Élitisme.
            for i in range(self.n_elite):
                new_pop.append(self.population[order[i]].copy())
            # Reproduction.
            while len(new_pop) < self.pop_size:
                i1 = self._tournament(fitnesses)
                i2 = self._tournament(fitnesses)
                child = self._crossover(self.population[i1], self.population[i2])
                child = self._mutate(child)
                new_pop.append(child)

            self.population = np.array(new_pop)

        return best_genome, best_fitness, self.history
