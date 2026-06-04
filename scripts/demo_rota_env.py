from __future__ import annotations

import argparse
from typing import Sequence

from envs.rota_env import RotaConfig, RotaEnv


def parse_positions(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Positions must be two integers separated by a comma, e.g. 0,0")
    return [int(parts[0]), int(parts[1])]


def parse_obstacles(value: str) -> list[list[int]]:
    obstacles: list[list[int]] = []
    if not value:
        return obstacles
    for item in value.split(";"):
        obstacles.append(parse_positions(item))
    return obstacles


def render_episode(env: RotaEnv, actions: Sequence[int]) -> None:
    print("=== Ambiente inicial ===")
    env.render()
    for step, action in enumerate(actions, start=1):
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"\nPasso {step}: ação={action} | obs={observation.tolist()} | reward={reward} | terminated={terminated} | truncated={truncated}")
        env.render()
        if terminated or truncated:
            print("Episódio finalizado.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrador visual do ambiente RotaEnv")
    parser.add_argument("--width", type=int, default=6, help="Largura do grid")
    parser.add_argument("--height", type=int, default=6, help="Altura do grid")
    parser.add_argument("--start", type=parse_positions, default=[0, 0], help="Posição inicial no formato x,y")
    parser.add_argument("--goal", type=parse_positions, default=[5, 5], help="Posição de destino no formato x,y")
    parser.add_argument(
        "--obstacles",
        type=parse_obstacles,
        default=[[1, 0], [1, 1], [2, 1], [3, 3]],
        help="Obstáculos separados por ponto e vírgula no formato x,y;x,y" 
    )
    parser.add_argument(
        "--actions",
        type=int,
        nargs="*",
        default=[3, 3, 1, 1, 3, 3, 1, 1],
        help="Sequência de ações para simular (0=Cima, 1=Baixo, 2=Esquerda, 3=Direita)"
    )
    args = parser.parse_args()

    config = RotaConfig(
        width=args.width,
        height=args.height,
        start=args.start,
        goal=args.goal,
        obstacles=args.obstacles,
    )
    env = RotaEnv(config)
    env.reset()
    render_episode(env, args.actions)


if __name__ == "__main__":
    main()
