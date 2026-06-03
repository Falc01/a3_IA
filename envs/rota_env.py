from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RotaConfig:
    width: int = 8
    height: int = 8
    start: Sequence[int] = field(default_factory=lambda: [0, 0])
    goal: Sequence[int] = field(default_factory=lambda: [7, 7])
    obstacles: Sequence[Sequence[int]] = field(default_factory=list)
    step_penalty: float = -1.0
    collision_penalty: float = -10.0
    goal_reward: float = 100.0
    max_steps_factor: int = 2

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers")

        self._validate_position(self.start, "start")
        self._validate_position(self.goal, "goal")

        if self.start == self.goal:
            raise ValueError("start and goal positions must be different")

        if any(tuple(obs) == tuple(self.start) or tuple(obs) == tuple(self.goal) 
               for obs in self.obstacles):
            raise ValueError("obstacle cannot overlap start or goal")

    def _validate_position(self, position: Sequence[int], name: str) -> None:
        if len(position) != 2:
            raise ValueError(f"{name} must be a sequence of two integers")
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} {position} is outside the grid bounds")

    @property
    def obstacle_set(self) -> frozenset[tuple[int, int]]:
        return frozenset((int(x), int(y)) for x, y in self.obstacles)

    @property
    def max_steps(self) -> int:
        return self.max_steps_factor * (self.width * self.height)


class RotaEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Optional[RotaConfig] = None):
        self.config = config or RotaConfig()

        # Observation: posição (x, y)
        self.observation_space = spaces.MultiDiscrete([
            self.config.width,
            self.config.height
        ])
        
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right

        self._action_map = {
            0: np.array([0, -1], dtype=int),   # up
            1: np.array([0, 1], dtype=int),    # down
            2: np.array([-1, 0], dtype=int),   # left
            3: np.array([1, 0], dtype=int),    # right
        }

        self._state = np.array(self.config.start, dtype=int)
        self._steps = 0
        self._terminated = False

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._state = np.array(self.config.start, dtype=int)
        self._steps = 0
        self._terminated = False

        observation = self._get_observation()
        info = {
            "goal": tuple(self.config.goal),
            "obstacles": self.config.obstacle_set,
            "start": tuple(self.config.start)
        }
        return observation, info

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("Cannot call step() on a terminated environment. Call reset() first.")

        self._steps += 1
        delta = self._action_map[action]
        next_pos = self._state + delta

        # Movimento inválido (parede ou obstáculo)
        if not self._is_valid_position(next_pos):
            reward = self.config.collision_penalty
            terminated = False
            # estado não muda
        # Chegou no objetivo
        elif np.array_equal(next_pos, self.config.goal):
            self._state = next_pos
            reward = self.config.goal_reward
            terminated = True
        # Movimento normal
        else:
            self._state = next_pos
            reward = self.config.step_penalty
            terminated = False

        truncated = self._steps >= self.config.max_steps
        self._terminated = terminated or truncated

        observation = self._get_observation()
        info = {
            "steps": self._steps,
            "truncated": truncated,
            "goal": tuple(self.config.goal),
        }

        return observation, reward, terminated, truncated, info

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        x, y = pos
        return (0 <= x < self.config.width and 
                0 <= y < self.config.height and 
                (int(x), int(y)) not in self.config.obstacle_set)

    def _get_observation(self) -> np.ndarray:
        return np.array(self._state, dtype=np.int32)

    def render(self, mode: str = "human"):
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {mode}")

        grid = np.full((self.config.height, self.config.width), " ", dtype=str)

        # Obstáculos
        for x, y in self.config.obstacle_set:
            grid[y, x] = "#"

        # Start, Goal e Agente
        sx, sy = self.config.start
        gx, gy = self.config.goal
        cx, cy = self._state

        grid[sy, sx] = "S"
        grid[gy, gx] = "G"
        grid[cy, cx] = "A"

        print("\n".join("".join(row) for row in grid))
        print(f"Steps: {self._steps} | Reward so far: pending")

    def close(self):
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Cria o ambiente a partir de um dicionário"""
        return cls(config=RotaConfig(**config_dict))