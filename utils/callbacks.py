import numpy as np
import streamlit as st
from stable_baselines3.common.callbacks import BaseCallback

class StreamlitTrainingCallback(BaseCallback):
    """
    Callback customizado para interceptar as métricas de treino do PPO
    e renderizar o progresso em tempo real na interface do Streamlit.
    """
    def __init__(self, total_timesteps: int, barra_progresso, grafico_linha, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        # Componentes visuais do Streamlit passados por referência
        self.barra_progresso = barra_progresso
        self.grafico_linha = grafico_linha
        # Histórico local para plotagem
        self.historico_recompensas = []

    def _on_step(self) -> bool:
        # Calcular o percentual de progresso atual
        progresso = float(self.num_timesteps) / float(self.total_timesteps)
        progresso = min(progresso, 1.0)
        
        # Atualizar a barra de progresso no Frontend
        self.barra_progresso.progress(progresso, text=f"Treinando Agente: {self.num_timesteps}/{self.total_timesteps} passos")

        # Capturar a última recompensa do ambiente (se disponível)
        if "episode" in self.locals["infos"][0]:
            recompensa_episodio = self.locals["infos"][0]["episode"]["r"]
            self.historico_recompensas.append(recompensa_episodio)
            
            # Atualiza o gráfico de linha síncrono do Streamlit
            self.grafico_linha.line_chart(self.historico_recompensas)
            
        return True