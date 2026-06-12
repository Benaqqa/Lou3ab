"""
Environnement Gymnasium personnalisé : Snake évolutif sur grille dynamique.

Sujet 2 du projet final (M2442 Jeux Vidéo IA).

Caractéristiques :
  - Grille de taille paramétrable.
  - Nourriture à collecter (augmente la longueur et le score).
  - Obstacles statiques disposés aléatoirement à chaque épisode.
  - Conditions variables selon les épisodes : nombre d'obstacles, position
    initiale et budget de pas sans nourriture (« faim ») peuvent varier,
    ce qui rend la grille *dynamique* d'un épisode à l'autre.
  - Observation vectorielle compacte (perception locale + direction + vecteur
    vers la nourriture) adaptée à un contrôleur de type réseau de neurones.

Conforme à l'API Gymnasium : observations, actions, fonction de récompense
et conditions de fin d'épisode sont définies explicitement.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Directions encodées en (delta_ligne, delta_colonne).
# Ordre : HAUT, DROITE, BAS, GAUCHE (sens horaire).
DIRECTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)

# Actions relatives au serpent : 0 = tout droit, 1 = tourner à droite,
# 2 = tourner à gauche. Ce choix évite les demi-tours suicidaires et rend
# l'espace d'actions discret et simple, conformément au cadrage du sujet.
ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2


class SnakeEnv(gym.Env):
    """Environnement Snake évolutif sur grille dynamique."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 12,
        n_obstacles: int = 6,
        dynamic: bool = True,
        max_steps_without_food: int | None = None,
        obstacle_range: tuple[int, int] = (3, 10),
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.base_n_obstacles = int(n_obstacles)
        self.dynamic = bool(dynamic)
        self.obstacle_range = obstacle_range
        self.render_mode = render_mode

        # Budget de pas sans manger avant la mort par « faim ».
        if max_steps_without_food is None:
            self.base_hunger = self.grid_size * self.grid_size
        else:
            self.base_hunger = int(max_steps_without_food)

        # Espace d'actions : 3 actions relatives discrètes.
        self.action_space = spaces.Discrete(3)

        # Espace d'observations : vecteur de 11 composantes (voir _get_obs).
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)

        # État interne (initialisé dans reset).
        self.snake: list[tuple[int, int]] = []
        self.direction_idx: int = 1
        self.obstacles: set[tuple[int, int]] = set()
        self.food: tuple[int, int] = (0, 0)
        self.steps: int = 0
        self.steps_since_food: int = 0
        self.hunger_limit: int = self.base_hunger
        self.n_obstacles: int = self.base_n_obstacles
        self.score: int = 0

    # ------------------------------------------------------------------ #
    # Gestion de l'aléatoire                                             #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)

        gs = self.grid_size

        # Conditions variables selon l'épisode (grille dynamique).
        if self.dynamic:
            lo, hi = self.obstacle_range
            self.n_obstacles = int(self._rng.integers(lo, hi + 1))
            # Petite variation du budget de faim.
            self.hunger_limit = int(self.base_hunger * self._rng.uniform(0.8, 1.2))
        else:
            self.n_obstacles = self.base_n_obstacles
            self.hunger_limit = self.base_hunger

        # Serpent initial : longueur 3, placé horizontalement au centre.
        cy = gs // 2
        cx = gs // 2
        self.direction_idx = 1  # vers la droite
        self.snake = [(cy, cx), (cy, cx - 1), (cy, cx - 2)]

        # Placement des obstacles (en évitant le corps du serpent et la zone
        # immédiate devant la tête).
        occupied = set(self.snake)
        forbidden = set(self.snake)
        for k in range(1, 4):
            forbidden.add((cy, cx + k))
        self.obstacles = set()
        attempts = 0
        while len(self.obstacles) < self.n_obstacles and attempts < 1000:
            attempts += 1
            pos = (int(self._rng.integers(0, gs)), int(self._rng.integers(0, gs)))
            if pos in occupied or pos in forbidden or pos in self.obstacles:
                continue
            self.obstacles.add(pos)
            occupied.add(pos)

        # Placement de la nourriture.
        self.food = self._place_food()

        self.steps = 0
        self.steps_since_food = 0
        self.score = 0

        obs = self._get_obs()
        info = {"score": self.score, "n_obstacles": self.n_obstacles}
        return obs, info

    def _place_food(self) -> tuple[int, int]:
        gs = self.grid_size
        occupied = set(self.snake) | self.obstacles
        free = [
            (r, c)
            for r in range(gs)
            for c in range(gs)
            if (r, c) not in occupied
        ]
        if not free:
            # Grille pleine : situation de victoire, on garde l'ancienne case.
            return self.food
        idx = int(self._rng.integers(0, len(free)))
        return free[idx]

    # ------------------------------------------------------------------ #
    # Dynamique                                                          #
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        action = int(action)
        # Mise à jour de la direction (relative).
        if action == ACTION_RIGHT:
            self.direction_idx = (self.direction_idx + 1) % 4
        elif action == ACTION_LEFT:
            self.direction_idx = (self.direction_idx - 1) % 4
        # ACTION_STRAIGHT : pas de changement.

        dr, dc = DIRECTIONS[self.direction_idx]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        self.steps += 1
        self.steps_since_food += 1

        terminated = False
        reward = 0.0

        # Récompense de rapprochement (shaping) : encourage à aller vers la
        # nourriture, accélère et stabilise l'évolution.
        old_dist = self._manhattan((head_r, head_c), self.food)
        new_dist = self._manhattan(new_head, self.food)

        # Collisions : murs, obstacles ou corps.
        if self._is_collision(new_head):
            terminated = True
            reward = -1.0
            obs = self._get_obs()
            info = {"score": self.score, "n_obstacles": self.n_obstacles,
                    "cause": "collision"}
            return obs, reward, terminated, False, info

        # Déplacement.
        ate = (new_head == self.food)
        self.snake.insert(0, new_head)
        if ate:
            self.score += 1
            self.steps_since_food = 0
            reward += 1.0
            self.food = self._place_food()
        else:
            self.snake.pop()  # avance sans grandir
            # shaping : +/- petit selon rapprochement
            reward += 0.1 if new_dist < old_dist else -0.15
            # petit coût de temps pour décourager les boucles
            reward -= 0.01

        truncated = False
        # Mort par faim.
        if self.steps_since_food >= self.hunger_limit:
            terminated = True
            reward -= 0.5
            info = {"score": self.score, "n_obstacles": self.n_obstacles,
                    "cause": "starvation"}
            return self._get_obs(), reward, terminated, truncated, info

        # Victoire : grille pleine.
        if len(self.snake) >= self.grid_size * self.grid_size:
            terminated = True
            reward += 5.0

        info = {"score": self.score, "n_obstacles": self.n_obstacles}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Outils                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_collision(self, pos: tuple[int, int], ignore_tail: bool = True) -> bool:
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        if pos in self.obstacles:
            return True
        # Le corps : on ignore la queue qui va bouger (sauf si on vient de manger,
        # cas géré par simplification : on considère tout le corps sauf la dernière case).
        body = set(self.snake[:-1]) if ignore_tail else set(self.snake)
        if pos in body:
            return True
        return False

    def _danger(self, dir_idx: int) -> float:
        """Renvoie 1.0 s'il y a danger immédiat dans la direction donnée."""
        dr, dc = DIRECTIONS[dir_idx]
        head_r, head_c = self.snake[0]
        nxt = (head_r + dr, head_c + dc)
        return 1.0 if self._is_collision(nxt) else 0.0

    def _get_obs(self) -> np.ndarray:
        """Vecteur d'observation de 11 composantes.

        [0] danger tout droit
        [1] danger à droite
        [2] danger à gauche
        [3:7] direction courante (one-hot HAUT/DROITE/BAS/GAUCHE)
        [7] nourriture à gauche (relatif au repère absolu)
        [8] nourriture à droite
        [9] nourriture en haut
        [10] nourriture en bas
        """
        di = self.direction_idx
        straight = di
        right = (di + 1) % 4
        left = (di - 1) % 4

        head_r, head_c = self.snake[0]
        food_r, food_c = self.food

        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_onehot[di] = 1.0

        obs = np.array(
            [
                self._danger(straight),
                self._danger(right),
                self._danger(left),
                dir_onehot[0],
                dir_onehot[1],
                dir_onehot[2],
                dir_onehot[3],
                1.0 if food_c < head_c else 0.0,  # nourriture à gauche
                1.0 if food_c > head_c else 0.0,  # nourriture à droite
                1.0 if food_r < head_r else 0.0,  # nourriture en haut
                1.0 if food_r > head_r else 0.0,  # nourriture en bas
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------ #
    # Rendu texte                                                        #
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode != "ansi":
            return None
        gs = self.grid_size
        grid = [["." for _ in range(gs)] for _ in range(gs)]
        for (r, c) in self.obstacles:
            grid[r][c] = "#"
        fr, fc = self.food
        grid[fr][fc] = "*"
        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = "H" if i == 0 else "o"
        lines = [" ".join(row) for row in grid]
        return "\n".join(lines) + f"\nScore: {self.score}  Steps: {self.steps}\n"
