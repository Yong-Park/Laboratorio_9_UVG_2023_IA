"""
Universidad del Valle de Guatemala
(CC3085) Inteligencia Artificial
Laboratorio 09: Aprendizaje por Refuerzo

Miembros del equipo:
- Pedro Pablo Arriola Jiménez (20188)
- Oscar Fernando López Barrios (20679)
- Yong Bum Park (20117)
- Santiago Taracena Puga (20017)
"""

# Clase NamedTuple que contiene los parámetros que se le pasan a la clase FrozenLake.
from typing import NamedTuple

# Clase Params que se tiene qué pasar a la clase FrozenLake.
class FrozenLakeParams(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen

# Clase Params que se tiene qué pasar a la clase Boxing.
class BoxingParams(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    alpha: float  # Learning rate
    max_epsilon: float  # Exploration probability at start
    min_epsilon: float  # Minimum exploration probability
    decay_rate: float  # Exponential decay rate for exploration prob
    max_steps: int  # Max steps per episode
