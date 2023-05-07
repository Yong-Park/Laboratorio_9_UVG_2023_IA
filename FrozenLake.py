# referencia de https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#sphx-glr-tutorials-training-agents-frozenlake-tuto-py

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import NamedTuple

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Clase Params que se tiene qué pasar a la clase FrozenLake.
class Params(NamedTuple):
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

params = Params(
    total_episodes=10000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
)
params

# Set the seed
rng = np.random.default_rng(params.seed)

env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(size=4),
)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action

learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
)
explorer = EpsilonGreedy(
    epsilon=params.epsilon,
)

def run_env():
    rewards = np.zeros((params.total_episodes))
    steps = np.zeros((params.total_episodes))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.state_size, params.action_size))
    all_states = []
    all_actions = []
    points = 0
    learner.reset_qtable()
    for episode in tqdm(
        episodes, desc="Episodes", leave=False
    ):
        state = env.reset(seed=params.seed)[0]  # Reset the environment
        step = 0
        done = False
        total_rewards = 0

        while not done:
            action = explorer.choose_action(
                action_space=env.action_space, state=state, qtable=learner.qtable
            )

            # Log all states and actions
            all_states.append(state)
            all_actions.append(action)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            learner.qtable[state, action] = learner.update(
                state, action, reward, new_state
            )

            total_rewards += reward
            step += 1

            # Our new state is state
            state = new_state
        
        if reward > 0:
            print(" ganaste")
            points +=1
        else:
            print(" perdiste")
        if points == 20:
            break
        print(" puntos totales: ",points)
        print("==================================")
    return rewards, steps, episodes, qtables, all_states, all_actions


res_all = pd.DataFrame()
st_all = pd.DataFrame()

#numero de veces que lo hara, en este caso son 5 mapas distintos
for _ in range(5):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="human",
        desc=generate_random_map(size=4),
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )

    rewards, steps, episodes, qtables, all_states, all_actions = run_env()

    env.close()

