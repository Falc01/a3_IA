import unittest
from utils.map_generator import is_solvable, generate_map_obstacles


class TestMapGenerator(unittest.TestCase):
    """
    Suíte de testes automatizados para validar a busca de caminhos (BFS) 
    e a solucionabilidade dos mapas gerados proceduralmente.
    """
    def setUp(self) -> None:
        self.w = 6
        self.h = 6
        self.start = [0, 0]
        self.goal = [5, 5]

    def test_is_solvable_open_path(self) -> None:
        """Testa se o BFS retorna True para um mapa totalmente aberto ou com caminho livre."""
        # Mapa sem obstáculos
        self.assertTrue(is_solvable(self.w, self.h, self.start, self.goal, []))

        # Mapa com obstáculos mas contornável
        obstacles = [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]  # deixa a linha inferior livre
        self.assertTrue(is_solvable(self.w, self.h, self.start, self.goal, obstacles))

    def test_is_solvable_blocked_path(self) -> None:
        """Testa se o BFS retorna False quando todos os caminhos para o objetivo estão bloqueados."""
        # Paredes bloqueando totalmente a coluna 2
        obstacles = [[2, y] for y in range(self.h)]
        self.assertFalse(is_solvable(self.w, self.h, self.start, self.goal, obstacles))

        # Ponto final diretamente bloqueado
        self.assertFalse(is_solvable(self.w, self.h, self.start, self.goal, [[5, 5]]))

    def test_generate_map_obstacles_always_solvable(self) -> None:
        """Testa se as funções de geração de obstáculos retornam layouts válidos e solucionáveis."""
        # Padrão Aleatório
        obs_rand = generate_map_obstacles("Aleatório", self.w, self.h, self.start, self.goal, 0.3)
        self.assertTrue(is_solvable(self.w, self.h, self.start, self.goal, obs_rand))

        # Padrão Paredes Alternadas
        obs_walls = generate_map_obstacles("Paredes Alternadas", self.w, self.h, self.start, self.goal, 0.2)
        self.assertTrue(is_solvable(self.w, self.h, self.start, self.goal, obs_walls))

        # Padrão Barreira Central
        obs_center = generate_map_obstacles("Barreira Central", self.w, self.h, self.start, self.goal, 0.2)
        self.assertTrue(is_solvable(self.w, self.h, self.start, self.goal, obs_center))


if __name__ == "__main__":
    unittest.main()
