"""
Script d'entraînement principal.

Entraîne l'agent neuroévolutionnaire sur l'environnement Snake évolutif,
sauvegarde le meilleur génome, l'historique d'évolution, et lance l'évaluation
comparative contre les baselines (aléatoire, heuristique) sur un jeu de test
indépendant. Tous les résultats sont écrits dans results/ au format JSON.

Usage :
    python experiments/train.py
"""

from __future__ import annotations

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snake_evo.network import MLPPolicy
from snake_evo.evolution import GeneticAlgorithm
from snake_evo.agents import RandomAgent, HeuristicAgent, GeneticAgent
from snake_evo.evaluate import evaluate_agent


# --------------------------------------------------------------------------- #
# Hyperparamètres de l'expérience                                             #
# --------------------------------------------------------------------------- #
CONFIG = {
    "grid_size": 12,
    "n_obstacles": 6,
    "dynamic": True,
    "obstacle_range": (3, 10),
    "hidden_size": 16,
    "pop_size": 150,
    "n_generations": 150,
    "elite_frac": 0.12,
    "tournament_size": 5,
    "mutation_rate": 0.1,
    "mutation_scale": 0.25,
    "crossover_rate": 0.7,
    "n_eval_episodes": 6,
    "max_steps": 400,
    "n_test_episodes": 200,
    "ga_seed": 42,
}

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env_kwargs = dict(
        grid_size=CONFIG["grid_size"],
        n_obstacles=CONFIG["n_obstacles"],
        dynamic=CONFIG["dynamic"],
        obstacle_range=CONFIG["obstacle_range"],
    )

    policy = MLPPolicy(11, CONFIG["hidden_size"], 3)
    print(f"Taille du génome : {policy.n_params} paramètres")

    # -------------------- Entraînement de l'agent AG -------------------- #
    print("\n=== Entraînement de l'agent neuroévolutionnaire ===")
    t0 = time.time()
    ga = GeneticAlgorithm(
        policy=policy,
        env_kwargs=env_kwargs,
        pop_size=CONFIG["pop_size"],
        elite_frac=CONFIG["elite_frac"],
        tournament_size=CONFIG["tournament_size"],
        mutation_rate=CONFIG["mutation_rate"],
        mutation_scale=CONFIG["mutation_scale"],
        crossover_rate=CONFIG["crossover_rate"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        max_steps=CONFIG["max_steps"],
        seed=CONFIG["ga_seed"],
    )
    best_genome, best_fit, history = ga.evolve(
        CONFIG["n_generations"], verbose=True
    )
    train_time = time.time() - t0
    print(f"\nEntraînement terminé en {train_time:.1f}s. "
          f"Meilleure fitness = {best_fit:.2f}")

    # Sauvegarde du génome et de l'historique.
    np.save(os.path.join(RESULTS_DIR, "best_genome.npy"), best_genome)
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # -------------------- Évaluation comparative ----------------------- #
    # Jeu de test INDÉPENDANT des graines d'entraînement.
    test_seeds = list(range(900_000, 900_000 + CONFIG["n_test_episodes"]))

    print("\n=== Évaluation sur le jeu de test "
          f"({CONFIG['n_test_episodes']} épisodes) ===")

    agents = {
        "Aléatoire": RandomAgent(seed=123),
        "Heuristique": HeuristicAgent(),
        "AG (neuroévolution)": GeneticAgent(best_genome, policy),
    }

    eval_results = {}
    for name, agent in agents.items():
        res = evaluate_agent(agent, env_kwargs, test_seeds, CONFIG["max_steps"])
        eval_results[name] = res
        print(
            f"{name:22s} | score moy = {res['mean_score']:.2f} "
            f"± {res['std_score']:.2f} | max = {res['max_score']} | "
            f"survie moy = {res['mean_steps']:.1f} pas | "
            f"succès = {100*res['success_rate']:.0f}%"
        )

    # -------------------- Robustesse vs nombre d'obstacles ------------- #
    print("\n=== Étude de robustesse (score selon le nombre d'obstacles) ===")
    robustness = {}
    obstacle_levels = [0, 4, 8, 12]
    rob_seeds = list(range(700_000, 700_000 + 60))
    for name, agent in agents.items():
        robustness[name] = {}
        for n_obs in obstacle_levels:
            ek = dict(env_kwargs)
            ek["dynamic"] = False
            ek["n_obstacles"] = n_obs
            r = evaluate_agent(agent, ek, rob_seeds, CONFIG["max_steps"])
            robustness[name][n_obs] = r["mean_score"]
        line = " | ".join(
            f"{n}obs:{robustness[name][n]:.2f}" for n in obstacle_levels
        )
        print(f"{name:22s} | {line}")

    # -------------------- Sauvegarde globale --------------------------- #
    output = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in CONFIG.items()},
        "n_params": policy.n_params,
        "train_time_s": train_time,
        "best_fitness": best_fit,
        "test_seeds_range": [test_seeds[0], test_seeds[-1]],
        "evaluation": eval_results,
        "robustness": {
            name: {str(k): v for k, v in d.items()}
            for name, d in robustness.items()
        },
        "obstacle_levels": obstacle_levels,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats sauvegardés dans {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
