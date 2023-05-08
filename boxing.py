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

# Librerías necesarias.
import gym
import numpy as np
from tqdm import tqdm
from params import BoxingParams

# Configuración del entorno de entrenamiento.
environment = gym.make('ALE/Boxing-v5')
environment.observation_space = gym.spaces.flatten_space(environment.observation_space)

# Configuración de los parámetros de entrenamiento.
state_space = environment.observation_space.shape[0]
action_space = environment.action_space.n
qtable = np.zeros((state_space, action_space))

# Parámetros de la ejecución.
parameters = BoxingParams(
    total_episodes=250,
    learning_rate=0.8,
    gamma=0.99,
    epsilon=1.0,
    map_size=5,
    seed=123,
    action_size=None,
    state_size=None,
    alpha=0.77,
    max_epsilon=1.0,
    min_epsilon=0.01,
    decay_rate=0.001,
    max_steps=100,
)

# Entrenamiento del agente con el número de episodios dado.
for episode in tqdm(range(parameters.total_episodes)):

    # Reinicio del entorno.
    state = environment.reset()

    # Variables de control.
    finished = False
    current_step = 0

    # Se aplana el espacio de observación.
    state = gym.spaces.flatten(environment.observation_space, state[0])

    # Se ejecuta el episodio.
    for current_step in range(parameters.max_steps):

        # Se elige una acción.
        experience_tradeoff = np.random.uniform(0, 1)

        # Se elige la acción con base en la política.
        if (experience_tradeoff > parameters.epsilon):
            action = np.argmax(qtable[state, :])
        else:
            action = environment.action_space.sample()

        # Se ejecuta la acción.
        new_state, reward, finished, _, _ = environment.step(action)
        qtable[state, action] = qtable[state, action] + parameters.alpha * (reward + parameters.gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state

        # Se termina el episodio.
        if (finished):
            break

    # Se reduce la probabilidad de exploración.
    epsilon = parameters.min_epsilon + (parameters.max_epsilon - parameters.min_epsilon) * np.exp((-1 * parameters.decay_rate) * episode)

# Se cierra el entorno.
environment.close()

# Ejecución del agente.
environment = gym.make('ALE/Boxing-v5', render_mode='human')
environment.observation_space = gym.spaces.flatten_space(environment.observation_space)

# Variables de control.
finished = False
state = environment.reset()

# Ciclo que ejecuta el agente hasta que se termine el episodio.
while (not finished):

    # Se aplana el espacio de observación.
    state = gym.spaces.flatten(environment.observation_space, state[0])
    action = np.argmax(qtable[state, :])
    new_state, reward, finished, _, _ = environment.step(action)
    state = new_state

# Se cierra el entorno.
print(f"Finished with reward {reward}")
environment.close()
