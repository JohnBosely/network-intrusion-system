import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FastNetworkDefenseEnv(gym.Env):
    """
    A fast AI training simulator that uses 
    pre-saved data so the AI doesn't have to 
    pause and calculate things step-by-step.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, precomputed_observations: np.ndarray, data_labels: np.ndarray, benign_idx: int):
        super(FastNetworkDefenseEnv, self).__init__()
        
        self.obs_matrix = precomputed_observations
        self.labels = data_labels
        self.benign_idx = benign_idx
        self.total_packets = len(data_labels)
        self.current_idx = 0

        # Define Action Space (0: ALLOW, 1: THROTTLE, 2: DROP, 3: HONEYPOT)
        self.action_space = spaces.Discrete(4)

        # Define Observation Space dynamically based on the input matrix dimensions
        num_signals = self.obs_matrix.shape[1]
        low_bound = np.array([-2.0] * num_signals, dtype=np.float32)
        high_bound = np.array([2.0] * num_signals, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bound, high=high_bound, dtype=np.float32)

    def _get_observation(self):
        return self.obs_matrix[self.episode_order[self.current_idx]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_order = np.arange(self.total_packets)  # sequential, not shuffled
        self.current_idx = 0
        return self.obs_matrix[self.episode_order[0]], {}

    def step(self, action):
        true_label_idx = self.labels[self.episode_order[self.current_idx]]
        is_attack = (true_label_idx != self.benign_idx)

        if action == 0:  # ALLOW
            reward = 5.0 if not is_attack else -15.0
        elif action == 1:  # THROTTLE
            reward = -0.5 if not is_attack else 5.0
        elif action == 2:  # DROP
            reward = -2.0 if not is_attack else 8.0
        elif action == 3:  # HONEYPOT
            reward = -1.0 if not is_attack else 6.0

        self.current_idx += 1
        terminated = (self.current_idx >= self.total_packets - 1)
        truncated = False

        next_observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "true_identity": "ATTACK" if is_attack else "BENIGN",
            "action_executed": action
        }
        return next_observation, reward, terminated, truncated, info