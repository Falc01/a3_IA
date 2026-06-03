import gymnasium as gym
from agents.gerenciador_agente import GerenciadorAgentePPO

def inicializar_e_configurar_agente(env: gym.Env) -> GerenciadorAgentePPO:
    """
    Fábrica responsável por instanciar o Gerenciador e aplicar as
    configurações de hiperparâmetros otimizadas para a PoC de logística de rotas.
    """
    # 1. Instancia o gerenciador com os caminhos de dados padrão
    gerenciador = GerenciadorAgentePPO(
        env=env,
        log_dir="data/logs/",
        model_dir="data/models/"
    )
    
    # 2. Configura o modelo PPO interno (Aplica a MLP, learning rate de 0.0003 e ent_coef de 0.01)
    gerenciador.configurar_agente(seed=42)
    
    return gerenciador