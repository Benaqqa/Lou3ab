"""
Réseau de neurones feed-forward minimal utilisé comme politique du serpent.

Le réseau prend l'observation (11 composantes) et produit des scores pour les
3 actions. L'action choisie est l'argmax (politique déterministe), ce qui rend
l'évaluation reproductible.

Le génome de l'agent évolutionnaire est exactement le vecteur aplati de tous
les poids et biais de ce réseau.
"""

from __future__ import annotations

import numpy as np


class MLPPolicy:
    """Perceptron multicouche : entrée -> couche cachée (tanh) -> sortie."""

    def __init__(self, input_size: int = 11, hidden_size: int = 16,
                 output_size: int = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    @property
    def n_params(self) -> int:
        """Nombre total de paramètres (taille du génome)."""
        return (
            self.input_size * self.hidden_size  # W1
            + self.hidden_size                  # b1
            + self.hidden_size * self.output_size  # W2
            + self.output_size                  # b2
        )

    def _unpack(self, genome: np.ndarray):
        i, h, o = self.input_size, self.hidden_size, self.output_size
        idx = 0
        W1 = genome[idx:idx + i * h].reshape(i, h); idx += i * h
        b1 = genome[idx:idx + h]; idx += h
        W2 = genome[idx:idx + h * o].reshape(h, o); idx += h * o
        b2 = genome[idx:idx + o]; idx += o
        return W1, b1, W2, b2

    def act(self, genome: np.ndarray, obs: np.ndarray) -> int:
        """Renvoie l'action (argmax) pour un génome et une observation donnés."""
        W1, b1, W2, b2 = self._unpack(genome)
        h = np.tanh(obs @ W1 + b1)
        logits = h @ W2 + b2
        return int(np.argmax(logits))
