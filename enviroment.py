from typing import Optional
import numpy as np
import gymnasium as gym
from math import pi, cos

class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 100):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([1, 2], dtype=np.float32)
        self.theta = pi/4
        self.speed = 5
        # This will be set by user later
        self._target_location = np.array([-1, -1], dtype=np.float32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.float32),   # [x, y] coordinates
                "theta": gym.spaces.Box(0, 2*np.pi, shape=(1,), dtype=np.float32),  # [x, y] coordinates
            }
        )

        # Define what actions are available 
        self.action_space = gym.spaces.Discrete(3)

        # Change theta based on action
        self._action_to_direction = {
            0: .1,   # soft left
            1: 0,  # don't rotate
            2: -.1 # soft right
        }


    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "theta": np.array([self.theta], dtype=np.float32)}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
            }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = np.array([1, 2], dtype=np.float32)
        self.theta = np.pi/4

        # Randomly place target, ensuring it's different from agent position
        # User will change this later
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(0, self.size, size=2).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update direction and normalize
        self.theta += direction
        self.theta = self.theta % (2 * np.pi)

        # Update location
        # np.clip prevents the agent from walking off the edge

        self._agent_location = np.clip(
            self._agent_location + [self.speed*np.cos(self.theta),
                                    self.speed*np.sin(self.theta)], 0, self.size - 1
        )


        # Check if agent reached the target
        terminated = np.linalg.norm(self._agent_location - self._target_location)<5

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info