from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Any

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
    use_reward_shaping: bool = True

    def __post_init__(self) -> None:
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

    def __init__(self, config: Optional[RotaConfig] = None) -> None:
        self.config = config or RotaConfig()

        # Observation space: Flat cell index
        self.observation_space = spaces.Discrete(self.config.width * self.config.height)
        
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
        self._dist_matrix = self._compute_distance_matrix()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = np.array(self.config.start, dtype=int)
        self._steps = 0
        self._terminated = False

        start_pos = tuple(self.config.start)
        shortest_path_len = self._dist_matrix[start_pos[0], start_pos[1]]
        if shortest_path_len < 9999.0:
            self._max_possible_reward = 101.0 - 0.5 * shortest_path_len
        else:
            self._max_possible_reward = -9999.0

        observation = self._get_observation()
        info = {
            "goal": tuple(self.config.goal),
            "obstacles": self.config.obstacle_set,
            "start": tuple(self.config.start),
            "max_possible_reward": self._max_possible_reward
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Cannot call step() on a terminated environment. Call reset() first.")

        # Distância real BFS antes do movimento
        old_dist = self._dist_matrix[self._state[0], self._state[1]]

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

        # Distância real BFS após o movimento
        new_dist = self._dist_matrix[self._state[0], self._state[1]]

        # Reward shaping: diferença de distância BFS (positivo se aproximar no caminho mais curto, negativo se afastar)
        if self.config.use_reward_shaping and old_dist < 9999.0 and new_dist < 9999.0:
            shaping = float(old_dist - new_dist) * 0.5
            reward += shaping

        truncated = self._steps >= self.config.max_steps
        self._terminated = terminated or truncated

        observation = self._get_observation()
        info = {
            "steps": self._steps,
            "truncated": truncated,
            "goal": tuple(self.config.goal),
            "max_possible_reward": self._max_possible_reward
        }

        return observation, reward, terminated, truncated, info

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        x, y = pos
        return (0 <= x < self.config.width and 
                0 <= y < self.config.height and 
                (int(x), int(y)) not in self.config.obstacle_set)

    def _get_observation(self) -> int:
        cx, cy = self._state
        # Retorna o índice 1D (linear) da célula
        return int(cx * self.config.height + cy)

    def render(self, mode: str = "human") -> None:
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

    def close(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> RotaEnv:
        """Cria o ambiente a partir de um dicionário"""
        return cls(config=RotaConfig(**config_dict))

    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Calcula a distância do caminho mais curto livre de obstáculos (BFS)
        a partir de cada célula da grade até o objetivo.
        Retorna uma matriz 2D de tamanho (width, height).
        """
        from collections import deque
        w, h = self.config.width, self.config.height
        goal = tuple(self.config.goal)
        obstacles = self.config.obstacle_set
        
        # Inicializa matriz com valor alto para células inacessíveis
        dist_matrix = np.full((w, h), 9999.0, dtype=float)
        
        if 0 <= goal[0] < w and 0 <= goal[1] < h:
            dist_matrix[goal] = 0.0
            queue = deque([goal])
            visited = {goal}
            
            while queue:
                cx, cy = queue.popleft()
                cdist = dist_matrix[cx, cy]
                
                # Movimentos possíveis nas 4 direções discretas
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        npos = (nx, ny)
                        if npos not in visited and npos not in obstacles:
                            visited.add(npos)
                            dist_matrix[nx, ny] = cdist + 1.0
                            queue.append(npos)
                            
        return dist_matrix