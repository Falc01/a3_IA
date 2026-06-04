import numpy as np
import streamlit as st
from utils.plot_utils import build_reward_chart, build_steps_chart


class StreamlitCallback:
    """
    Callback customizado para interceptar o progresso do treino do SARSA
    e atualizar a barra de progresso, gráficos de recompensa e passos na tela do Streamlit.
    """
    def __init__(self, total_episodios: int, progress_bar, chart_placeholder, chart_steps_placeholder, log_placeholder, early_stop_enabled: bool = False) -> None:
        self.total_episodios = total_episodios
        self.progress_bar = progress_bar
        self.chart_ph = chart_placeholder
        self.chart_steps_ph = chart_steps_placeholder
        self.log_ph = log_placeholder
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self.early_stop_enabled = early_stop_enabled
        self.early_stopped = False

    def registrar_episodio(self, ep: int, total: int, reward: float, passos: int) -> None:
        self.rewards.append(reward)
        self.steps.append(passos)

        # Atualiza a interface a cada 25 episódios para evitar sobrecarga de WebSocket
        if ep % 25 == 0 or ep == total:
            progress = min(ep / total, 1.0)
            self.progress_bar.progress(progress, text=f"🔄 Treinando: {ep:,} / {total:,} episódios")
            
            # Atualiza os gráficos de recompensa e passos
            fig_reward = build_reward_chart(self.rewards)
            fig_steps = build_steps_chart(self.steps)
            self.chart_ph.plotly_chart(fig_reward, use_container_width=True, key=f"live_chart_{ep}")
            self.chart_steps_ph.plotly_chart(fig_steps, use_container_width=True, key=f"live_steps_{ep}")
            
            # Mostra estatísticas rápidas
            media_recente = np.mean(self.rewards[-50:]) if len(self.rewards) >= 50 else np.mean(self.rewards)
            passos_recente = np.mean(self.steps[-50:]) if len(self.steps) >= 50 else np.mean(self.steps)
            self.log_ph.markdown(
                f"**Métricas Atuais (Episódio {ep}):**\n"
                f"* Recompensa Recente Média: `{media_recente:.2f}`\n"
                f"* Passos Recentes Médios: `{passos_recente:.1f}`\n"
                f"* Melhor Recompensa Registrada: `{np.max(self.rewards):.2f}`"
            )

    def should_stop(self) -> bool:
        # Lógica de Early Stopping por estabilização no SARSA
        if self.early_stop_enabled and len(self.rewards) >= 120:
            recente = self.rewards[-50:]
            # Se o desvio padrão das recompensas nos últimos 50 episódios for menor que 0.5,
            # consideramos que o modelo convergiu e estabilizou
            if np.std(recente) < 0.5:
                self.early_stopped = True
                return True
        return False