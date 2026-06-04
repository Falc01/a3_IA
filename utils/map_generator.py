import random
from collections import deque
from typing import Sequence


def is_solvable(w: int, h: int, start: Sequence[int], goal: Sequence[int], obstacles: Sequence[Sequence[int]]) -> bool:
    """
    Verifica via BFS (Busca em Largura) se existe pelo menos um caminho válido 
    ligando a posição inicial à posição de destino sem passar por obstáculos.
    """
    start_t = tuple(start)
    goal_t = tuple(goal)
    obs_set = {tuple(o) for o in obstacles}
    
    if start_t in obs_set or goal_t in obs_set:
        return False
        
    queue = deque([start_t])
    visited = {start_t}
    
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal_t:
            return True
            
        # 4 Direções discretas (Cima, Baixo, Esquerda, Direita)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                n_t = (nx, ny)
                if n_t not in visited and n_t not in obs_set:
                    visited.add(n_t)
                    queue.append(n_t)
                    
    return False


def generate_map_obstacles(method: str, w: int, h: int, start: Sequence[int], goal: Sequence[int], density: float) -> list[list[int]]:
    """
    Gera dinamicamente uma distribuição de obstáculos (paredes/barreiras) de acordo
    com o padrão e densidade selecionados, garantindo que o mapa gerado seja solucionável.
    """
    start_t = tuple(start)
    goal_t = tuple(goal)
    
    # Loop de tentativas para garantir a solucionabilidade do mapa
    for _ in range(100):
        obstacles = []
        if method == "Aleatório":
            for x in range(w):
                for y in range(h):
                    pos = (x, y)
                    if pos != start_t and pos != goal_t:
                        if random.random() < density:
                            obstacles.append([x, y])
                            
        elif method == "Paredes Alternadas":
            # Cria paredes verticais alternadas com uma fenda randômica de passagem
            for x in range(2, w - 1, 2):
                gap_y = random.randint(0, h - 1)
                for y in range(h):
                    pos = (x, y)
                    if pos != start_t and pos != goal_t and y != gap_y:
                        obstacles.append([x, y])
                        
        elif method == "Barreira Central":
            # Cria uma barreira vertical sólida no centro, deixando passagens nas extremidades
            mid_x = w // 2
            for y in range(1, h - 1):
                pos = (mid_x, y)
                if pos != start_t and pos != goal_t:
                    obstacles.append([mid_x, y])
                    
        # Se for solucionável, retorna a lista de obstáculos
        if is_solvable(w, h, start, goal, obstacles):
            return obstacles
            
    return []
