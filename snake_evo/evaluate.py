"""
Protocole d'évaluation expérimentale.

Évalue n'importe quel agent (génétique, aléatoire, heuristique) sur un jeu
d'épisodes de TEST distinct des épisodes d'entraînement, et calcule des
indicateurs quantitatifs : score moyen, écart-type, survie moyenne, score max,
taux de réussite (au moins 1 nourriture).
"""

from __future__ import annotations

import numpy as np

from .env import SnakeEnv


def run_episode(agent, env_kwargs: dict, seed: int, max_steps: int = 500) -> dict:
    env = SnakeEnv(**env_kwargs, seed=seed)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {
        "score": info["score"],
        "steps": env.steps,
        "reward": total_reward,
        "n_obstacles": info["n_obstacles"],
    }


def evaluate_agent(agent, env_kwargs: dict, test_seeds: list[int],
                   max_steps: int = 500) -> dict:
    """Évalue un agent sur un ensemble d'épisodes de test."""
    scores, steps_list, rewards = [], [], []
    for seed in test_seeds:
        r = run_episode(agent, env_kwargs, seed, max_steps)
        scores.append(r["score"])
        steps_list.append(r["steps"])
        rewards.append(r["reward"])
    scores = np.array(scores)
    steps_list = np.array(steps_list)
    rewards = np.array(rewards)
    return {
        "n_episodes": len(test_seeds),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "max_score": int(scores.max()),
        "min_score": int(scores.min()),
        "median_score": float(np.median(scores)),
        "mean_steps": float(steps_list.mean()),
        "std_steps": float(steps_list.std()),
        "mean_reward": float(rewards.mean()),
        "success_rate": float((scores >= 1).mean()),
        "scores": scores.tolist(),
        "steps": steps_list.tolist(),
    }
