import os
import random
import numpy as np
import gymnasium as gym
from typing import Optional, Any


class GerenciadorAgenteSarsa:
    """
    Classe responsável por centralizar a configuração, treinamento e 
    inferência do agente utilizando o algoritmo SARSA Tabular.
    """
    def __init__(self, env: gym.Env, q_table: Optional[np.ndarray] = None) -> None:
        self.env = env
        self.n_estados = env.observation_space.n
        self.n_acoes = env.action_space.n
        
        # Inicializa a Q-table com zeros
        if q_table is not None:
            self.Q = q_table
        else:
            self.Q = np.zeros((self.n_estados, self.n_acoes))

    def escolher_acao(self, estado: int, epsilon: float) -> int:
        """Escolhe uma ação usando a estratégia epsilon-greedy com desempate aleatório."""
        if random.random() < epsilon:
            return random.randint(0, self.n_acoes - 1)
        
        q_valores = self.Q[estado]
        max_q = np.max(q_valores)
        # Identifica todas as ações com o valor máximo para desempatar aleatoriamente
        acoes_candidatas = np.where(q_valores == max_q)[0]
        return int(random.choice(acoes_candidatas))

    def configurar_agente(self, kwargs_personalizados: Optional[dict[str, Any]] = None, seed: int = 42) -> None:
        """Mantido para compatibilidade de interface, reinicializa a Q-table e define sementes."""
        random.seed(seed)
        np.random.seed(seed)
        self.Q = np.zeros((self.n_estados, self.n_acoes))

    def treinar(
        self,
        total_episodios: int = 1000,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_inicial: float = 0.3,
        epsilon_min: float = 0.01,
        decaimento_epsilon: float = 0.99,
        callback: Optional[Any] = None
    ) -> list[float]:
        """
        Treina o agente utilizando a regra de atualização do SARSA:
        Q(s, a) = Q(s, a) + alpha * [R + gamma * Q(s', a') - Q(s, a)]
        """
        historico_recompensas = []
        epsilon = epsilon_inicial

        for ep in range(1, total_episodios + 1):
            obs, info = self.env.reset()
            estado = int(obs)
            acao = self.escolher_acao(estado, epsilon)
            
            recompensa_total = 0.0
            passos_do_episodio = 0
            done = False

            while not done:
                prox_obs, reward, terminated, truncated, info = self.env.step(acao)
                prox_estado = int(prox_obs)
                prox_acao = self.escolher_acao(prox_estado, epsilon)

                # Atualização SARSA
                q_atual = self.Q[estado, acao]
                q_proximo = self.Q[prox_estado, prox_acao]
                self.Q[estado, acao] = q_atual + learning_rate * (reward + gamma * q_proximo - q_atual)

                estado = prox_estado
                acao = prox_acao
                recompensa_total += reward
                passos_do_episodio += 1
                done = terminated or truncated

            # Decaimento do epsilon
            epsilon = max(epsilon_min, epsilon * decaimento_epsilon)
            historico_recompensas.append(recompensa_total)

            # Notifica o callback do progresso no Streamlit
            if callback is not None:
                callback.registrar_episodio(ep, total_episodios, recompensa_total, passos_do_episodio)
                if callback.should_stop():
                    break

        return historico_recompensas

    def predict(self, estado: int, deterministic: bool = True) -> tuple[int, None]:
        """Prediz a melhor ação com base na Q-table atualizada (usado na inferência)."""
        q_valores = self.Q[estado]
        max_q = np.max(q_valores)
        acoes_candidatas = np.where(q_valores == max_q)[0]
        return int(random.choice(acoes_candidatas)), None

    def salvar_modelo(self, caminho_completo: str = "data/models/agente_sarsa.npy") -> None:
        os.makedirs(os.path.dirname(caminho_completo), exist_ok=True)
        np.save(caminho_completo, self.Q)
        print(f"Modelo SARSA salvo com sucesso em: {caminho_completo}")

    def carregar_modelo(self, caminho_completo: str = "data/models/agente_sarsa.npy") -> None:
        if not caminho_completo.endswith(".npy"):
            if caminho_completo.endswith(".zip"):
                caminho_completo = caminho_completo[:-4]
            caminho_completo = caminho_completo + ".npy"
            
        self.Q = np.load(caminho_completo)
        print(f"Modelo SARSA carregado com sucesso de: {caminho_completo}")