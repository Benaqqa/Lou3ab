"""
Rendu graphique Pygame pour l'environnement Snake évolutif.

Fournit une fenêtre animée avec :
  - une grille de jeu (cases colorées) ;
  - le serpent en dégradé (tête mise en évidence) ;
  - la nourriture et les obstacles ;
  - un panneau latéral (HUD) affichant score, pas, nombre d'obstacles et FPS.

Le module est volontairement indépendant de l'environnement : il lit
simplement l'état public de `SnakeEnv` (snake, food, obstacles, score...).
Si Pygame n'est pas installé, l'import échoue proprement et la démo retombe
sur le rendu texte.
"""

from __future__ import annotations

import pygame


# --- Palette (couleurs cohérentes avec le rapport / les slides) ---
# Snake colors — earthy dark yellow with warm amber gradient
COL_HEAD   = (184, 134, 11)   # tête : ocre doré profond (dark goldenrod)
COL_BODY1  = (161, 117, 10)   # corps proche tête : un ton plus sombre
COL_BODY2  = (107,  79,  7)   # corps loin : brun-ocre foncé

# Environment — warm slate, less "midnight UI template"
COL_BG     = (18,  22,  30)   # fond : gris ardoise très sombre (plus chaud que bleu nuit)
COL_PANEL  = (26,  31,  42)   # panneau latéral : légèrement surélevé
COL_GRID   = (38,  45,  60)   # grille : subtile, pas trop contrastée
COL_CELL   = (26,  31,  42)   # case vide : identique au panneau

# Accents — désaturés et distinctifs
COL_FOOD   = (200,  72,  54)  # nourriture : rouge brique (plus chaud que rouge pur)
COL_OBST   = (82,   88,  98)  # obstacle : gris bleuté discret
COL_TEXT   = (210, 215, 225)  # texte : blanc cassé légèrement froid
COL_ACCENT = (168, 130,  50)  # accent/titres : doré désaturé, cohérent avec le serpent


class PygameRenderer:
    """Affiche une partie de SnakeEnv dans une fenêtre Pygame."""

    def __init__(self, env, cell_size: int = 42, panel_width: int = 240,
                 fps: int = 12, title: str = "Snake évolutif — démo"):
        self.env = env
        self.cell = cell_size
        self.panel_w = panel_width
        self.fps = fps
        self.margin = 16

        gs = env.grid_size
        self.board_px = gs * self.cell
        self.width = self.board_px + self.panel_w + 3 * self.margin
        self.height = self.board_px + 2 * self.margin

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
        self.font = pygame.font.SysFont("DejaVu Sans", 22)
        self.font_small = pygame.font.SysFont("DejaVu Sans", 17)

    # ------------------------------------------------------------------ #
    def _cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = self.margin + c * self.cell
        y = self.margin + r * self.cell
        return pygame.Rect(x + 1, y + 1, self.cell - 2, self.cell - 2)

    def _draw_board(self):
        gs = self.env.grid_size
        # Cases vides + grille.
        for r in range(gs):
            for c in range(gs):
                pygame.draw.rect(self.screen, COL_CELL, self._cell_rect(r, c),
                                 border_radius=4)

        # Obstacles.
        for (r, c) in self.env.obstacles:
            pygame.draw.rect(self.screen, COL_OBST, self._cell_rect(r, c),
                             border_radius=4)

        # Nourriture (cercle).
        fr, fc = self.env.food
        rect = self._cell_rect(fr, fc)
        pygame.draw.circle(self.screen, COL_FOOD, rect.center,
                           self.cell // 3)

        # Serpent (dégradé tête -> queue).
        n = len(self.env.snake)
        for i, (r, c) in enumerate(self.env.snake):
            if i == 0:
                color = COL_HEAD
            else:
                t = i / max(1, n - 1)
                color = tuple(
                    int(COL_BODY1[k] + t * (COL_BODY2[k] - COL_BODY1[k]))
                    for k in range(3)
                )
            pygame.draw.rect(self.screen, color, self._cell_rect(r, c),
                             border_radius=6)
            # Yeux sur la tête.
            if i == 0:
                cx, cy = self._cell_rect(r, c).center
                off = self.cell // 7
                for dx in (-off, off):
                    pygame.draw.circle(self.screen, COL_TEXT,
                                       (cx + dx, cy - off), 3)

    def _draw_panel(self, info: dict, status: str = ""):
        gs = self.env.grid_size
        px = self.board_px + 2 * self.margin
        panel = pygame.Rect(px, self.margin, self.panel_w, self.board_px)
        pygame.draw.rect(self.screen, COL_PANEL, panel, border_radius=10)

        x = px + 18
        y = self.margin + 20

        title = self.font_big.render("SNAKE IA", True, COL_ACCENT)
        self.screen.blit(title, (x, y))
        y += 50
        sub = self.font_small.render("Agent neuroévolutionnaire", True, COL_TEXT)
        self.screen.blit(sub, (x, y))
        y += 45

        lines = [
            ("Score", str(self.env.score)),
            ("Pas", str(self.env.steps)),
            ("Longueur", str(len(self.env.snake))),
            ("Obstacles", str(info.get("n_obstacles", "-"))),
        ]
        for label, value in lines:
            lbl = self.font_small.render(label, True, (148, 163, 184))
            self.screen.blit(lbl, (x, y))
            val = self.font.render(value, True, COL_TEXT)
            self.screen.blit(val, (x, y + 18))
            y += 56

        if status:
            y += 10
            st = self.font_small.render(status, True, COL_FOOD)
            self.screen.blit(st, (x, y))

        # Pied de panneau.
        foot = self.font_small.render("Benaqqa · Bensmina", True,
                                      (100, 116, 139))
        self.screen.blit(foot, (x, self.margin + self.board_px - 30))

    def render(self, info: dict, status: str = ""):
        # Gestion des événements (fermeture, pause/quit clavier).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE, pygame.K_q
            ):
                return False

        self.screen.fill(COL_BG)
        self._draw_board()
        self._draw_panel(info, status)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def wait_close(self):
        """Maintient la fenêtre ouverte jusqu'à fermeture par l'utilisateur."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE, pygame.K_q, pygame.K_RETURN, pygame.K_SPACE
                ):
                    running = False
            self.clock.tick(30)
        pygame.quit()

    def close(self):
        pygame.quit()
