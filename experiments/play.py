"""
Démonstration : joue une partie avec le meilleur agent évolué.

Deux modes d'affichage :
  - graphique (Pygame) : fenêtre animée avec cases colorées  [par défaut]
  - texte (ANSI)       : déroulé dans le terminal             [--text]

Usage :
    python experiments/play.py                 # fenêtre graphique, graine par défaut
    python experiments/play.py 900011          # fenêtre graphique, graine choisie
    python experiments/play.py 900011 --text   # mode texte
    python experiments/play.py --fps 8         # ralentir l'animation

Touches dans la fenêtre : Échap ou Q pour quitter.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snake_evo.env import SnakeEnv
from snake_evo.network import MLPPolicy
from snake_evo.agents import GeneticAgent

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def parse_args(argv):
    seed = 900000
    text_mode = "--text" in argv
    fps = 12
    if "--fps" in argv:
        i = argv.index("--fps")
        fps = int(argv[i + 1])
    # Première valeur numérique = graine.
    for a in argv[1:]:
        if a.isdigit():
            seed = int(a)
            break
    return seed, text_mode, fps


def load_agent():
    genome = np.load(os.path.join(RESULTS, "best_genome.npy"))
    policy = MLPPolicy(11, 16, 3)
    return GeneticAgent(genome, policy)


def play_text(agent, seed):
    env = SnakeEnv(grid_size=12, n_obstacles=6, dynamic=True,
                   render_mode="ansi", seed=seed)
    obs, info = env.reset(seed=seed)
    print(env.render())
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        os.system("clear")
        print(env.render())
        time.sleep(0.08)
    print(f"Partie terminée — score : {info['score']}, "
          f"pas : {env.steps}, cause : {info.get('cause', 'fin')}")


def play_graphic(agent, seed, fps):
    try:
        from snake_evo.render_pygame import PygameRenderer
    except Exception as exc:  # pragma: no cover
        print(f"[Pygame indisponible : {exc}]\nBascule en mode texte.\n")
        return play_text(agent, seed)

    env = SnakeEnv(grid_size=12, n_obstacles=6, dynamic=True, seed=seed)
    obs, info = env.reset(seed=seed)
    renderer = PygameRenderer(env, fps=fps)

    done = False
    open_ = renderer.render(info)
    while not done and open_:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        open_ = renderer.render(info)

    if open_:
        cause = info.get("cause", "fin")
        status = f"Fin — score {env.score} ({cause})"
        renderer.render(info, status=status)
        print(f"Partie terminée — score : {env.score}, pas : {env.steps}, "
              f"cause : {cause}")
        renderer.wait_close()
    else:
        renderer.close()


def main():
    seed, text_mode, fps = parse_args(sys.argv)
    agent = load_agent()
    if text_mode:
        play_text(agent, seed)
    else:
        play_graphic(agent, seed, fps)


if __name__ == "__main__":
    main()
