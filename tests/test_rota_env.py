import unittest
import numpy as np
from envs.rota_env import RotaEnv, RotaConfig


class TestRotaEnv(unittest.TestCase):
    """
    Suíte de testes automatizados para validar a lógica de transição, 
    recompensas, colisões e observações discretas do ambiente RotaEnv.
    """
    def setUp(self) -> None:
        self.config = RotaConfig(
            width=6,
            height=6,
            start=[0, 0],
            goal=[5, 5],
            obstacles=[[1, 1], [2, 2]]
        )
        self.env = RotaEnv(self.config)

    def test_reset(self) -> None:
        """Testa se o método reset retorna o índice do estado inicial correto."""
        obs, info = self.env.reset()
        np.testing.assert_array_equal(self.env._state, np.array([0, 0], dtype=int))
        self.assertEqual(obs, 0)  # Célula (0,0) na grade 6x6 -> 0 * 6 + 0 = 0
        self.assertEqual(info["goal"], (5, 5))
        self.assertEqual(info["start"], (0, 0))
        self.assertIn((1, 1), info["obstacles"])

    def test_valid_step_penalty(self) -> None:
        """Testa se movimentos válidos retornam a nova observação correta e aplicam penalidade de passo."""
        self.env.reset()
        # Mover para a direita (ação 3)
        obs, reward, terminated, truncated, info = self.env.step(3)
        np.testing.assert_array_equal(self.env._state, np.array([1, 0], dtype=int))
        self.assertEqual(obs, 6)  # Célula (1,0) na grade 6x6 -> 1 * 6 + 0 = 6
        self.assertEqual(reward, self.config.step_penalty + 0.5)  # -1 + 0.5 = -0.5
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_collision_penalty(self) -> None:
        """Testa se movimentos contra obstáculos mantêm o agente parado e aplicam -10."""
        self.env.reset()
        
        # Colisão com parede: mover para cima (ação 0) em [0,0]
        obs, reward, terminated, truncated, info = self.env.step(0)
        np.testing.assert_array_equal(self.env._state, np.array([0, 0], dtype=int))
        self.assertEqual(obs, 0)
        self.assertEqual(reward, self.config.collision_penalty)  # -10
        self.assertFalse(terminated)

        # Mover para a direita para [1,0] (observação deve virar 6)
        obs, _, _, _, _ = self.env.step(3)
        self.assertEqual(obs, 6)
        
        # Colisão com obstáculo em [1,1]: mover para baixo (ação 1)
        obs, reward, terminated, truncated, info = self.env.step(1)
        np.testing.assert_array_equal(self.env._state, np.array([1, 0], dtype=int))
        self.assertEqual(obs, 6)
        self.assertEqual(reward, self.config.collision_penalty)  # -10
        self.assertFalse(terminated)

    def test_goal_reward(self) -> None:
        """Testa se alcançar o objetivo encerra o episódio com recompensa de +100."""
        self.env.reset()
        # Teleporta o agente para perto do objetivo para testar a colisão de vitória
        self.env._state = np.array([4, 5], dtype=int)
        
        # Mover para a direita (ação 3) em direção a [5,5]
        obs, reward, terminated, truncated, info = self.env.step(3)
        np.testing.assert_array_equal(self.env._state, np.array([5, 5], dtype=int))
        self.assertEqual(obs, 35)  # Célula (5,5) na grade 6x6 -> 5 * 6 + 5 = 35
        self.assertEqual(reward, self.config.goal_reward + 0.5)  # +100 + 0.5 = 100.5
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_truncation(self) -> None:
        """Testa se o limite máximo de passos encerra o episódio por truncamento."""
        self.env.reset()
        max_steps = self.config.max_steps
        
        # Alterna movimentos válidos de ir e voltar (direita/esquerda)
        for _ in range(max_steps // 2):
            self.env.step(3)  # direita
            self.env.step(2)  # esquerda
            
        self.assertTrue(self.env._terminated)
        self.assertEqual(self.env._steps, max_steps)


if __name__ == "__main__":
    unittest.main()