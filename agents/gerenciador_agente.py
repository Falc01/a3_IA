import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class GerenciadorAgentePPO:
    """
    Classe responsável por centralizar a configuração, treinamento e 
    inferência do agente PPO adaptado para o cenário de Logística de Veículos.
    """
    def __init__(self, env: gym.Env = None, log_dir: str = "data/logs/", model_dir: str = "data/models/"):
        # O ambiente agora é opcional no início, permitindo carregar modelos apenas para inferência
        self.env = env
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.model = None
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def configurar_agente(self, kwargs_personalizados: dict = None, seed: int = 42) -> PPO:
        """
        Configura o PPO com os hiperparâmetros padrão da PoC, permitindo
        a sobrescrita de parâmetros para experimentos dinâmicos.
        """
        if self.env == None:
            raise ValueError("Para configurar um novo agente, um ambiente (env) precisa ser fornecido.")

        # Hiperparâmetros base (conforme especificação arquitetural)
        hiperparametros = {
            "policy": "MlpPolicy",
            "env": self.env,
            "learning_rate": 0.0003,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "ent_coef": 0.01,
            "verbose": 1,
            "seed": seed
        }
        
        # Permite alterar parâmetros dinamicamente sem mexer na estrutura da classe
        if kwargs_personalizados:
            hiperparametros.update(kwargs_personalizados)
        
        self.model = PPO(**hiperparametros)
        
        # Configura o Logger
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        self.model.set_logger(new_logger)
        
        return self.model

    def treinar(self, total_timesteps: int = 20000, callback: BaseCallback = None):
        if self.model is None:
            raise ValueError("O agente precisa ser configurado ou carregado antes do treino.")
        
        print(f"Iniciando treinamento logístico por {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Treinamento concluído com sucesso!")

    def salvar_modelo(self, nome_arquivo: str = "agente_ppo_logistica"):
        if self.model is None:
            raise ValueError("Não há modelo disponível para salvar.")
        
        # Remove a extensão .zip se o usuário a tiver colocado, evitando 'nome.zip.zip'
        if nome_arquivo.endswith(".zip"):
            nome_arquivo = nome_arquivo[:-4]
            
        caminho_completo = os.path.join(self.model_dir, nome_arquivo)
        self.model.save(caminho_completo)
        print(f"Modelo salvo com sucesso em: {caminho_completo}.zip")

    def carregar_modelo(self, nome_arquivo: str = "agente_ppo_logistica", env_atual: gym.Env = None):
        """
        Carrega os pesos do modelo. Permite associar um novo ambiente 
        caso o layout ou o grid tenham mudado no frontend.
        """
        if nome_arquivo.endswith(".zip"):
            nome_arquivo = nome_arquivo[:-4]
            
        caminho_completo = os.path.join(self.model_dir, nome_arquivo)
        
        # Se um novo env foi passado, prioriza ele; caso contrário usa o interno
        ambiente_alvo = env_atual if env_atual is not None else self.env
        
        self.model = PPO.load(caminho_completo, env=ambiente_alvo)
        if env_atual:
            self.env = env_atual
            
        print(f"Modelo carregado de: {caminho_completo}.zip")
        return self.model