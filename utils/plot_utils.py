import numpy as np
import plotly.graph_objects as go
from typing import Optional, Sequence


def build_grid_fig(
    w: int, h: int,
    start: Sequence[int], goal: Sequence[int],
    obstacles: Sequence[Sequence[int]],
    rota: Optional[Sequence[Sequence[int]]] = None,
    agent_pos: Optional[Sequence[int]] = None,
) -> go.Figure:
    """
    Gera uma figura Heatmap interativa do Plotly representando a grade de estados,
    os pontos de partida (A) e objetivo (B), os obstáculos e a rota percorrida pelo agente.
    """
    z = np.zeros((h, w))  # 0 = livre
    text = [["" for _ in range(w)] for _ in range(h)]
    colors = {
        "free": "#1e2130",
        "obstacle": "#374151",
        "start": "#4f46e5",
        "goal": "#16a34a",
        "path": "#7c3aed",
        "agent": "#f59e0b",
    }

    # Mapeamento do valor z para cores do gradiente discreto
    # 0=livre, 1=obstáculo, 2=start, 3=goal, 4=caminho, 5=agente
    COLOR_MAP = [
        colors["free"],
        colors["obstacle"],
        colors["start"],
        colors["goal"],
        colors["path"],
        colors["agent"],
    ]

    obs_set = {tuple(o) for o in obstacles}
    path_set = {tuple(p) for p in (rota or [])}

    for row in range(h):
        for col in range(w):
            pos = (col, row)
            if pos in obs_set:
                z[row][col] = 1
                text[row][col] = "🚧"
            elif pos == tuple(start):
                z[row][col] = 2
                text[row][col] = "A"
            elif pos == tuple(goal):
                z[row][col] = 3
                text[row][col] = "B"
            elif agent_pos and pos == tuple(agent_pos):
                z[row][col] = 5
                text[row][col] = "🤖"
            elif pos in path_set:
                z[row][col] = 4
                text[row][col] = "·"

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale=[[i / 5, c] for i, c in enumerate(COLOR_MAP)],
        showscale=False,
        zmin=0, zmax=5,
        xgap=2, ygap=2,
    ))

    # Criação das linhas divisórias da grade
    for i in range(w + 1):
        fig.add_shape(type="line", x0=i - 0.5, x1=i - 0.5, y0=-0.5, y1=h - 0.5,
                      line=dict(color="#2d3561", width=1))
    for j in range(h + 1):
        fig.add_shape(type="line", x0=-0.5, x1=w - 0.5, y0=j - 0.5, y1=j - 0.5,
                      line=dict(color="#2d3561", width=1))

    # Anotações de coordenadas apenas em grades pequenas (legibilidade)
    if w <= 12 and h <= 12:
        for row in range(h):
            for col in range(w):
                if z[row][col] == 0:
                    fig.add_annotation(
                        x=col, y=row,
                        text=f"<span style='color:#2d3561;font-size:9px'>{col},{row}</span>",
                        showarrow=False, font=dict(size=8),
                    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(320, min(560, h * 48)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.5, w - 0.5], constrain="domain"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.5, h - 0.5], scaleanchor="x", scaleratio=1, autorange="reversed"),
    )
    return fig


def build_reward_chart(rewards: list[float]) -> go.Figure:
    """
    Gera o gráfico de dispersão e curva de média móvel suavizada para as recompensas
    de treinamento obtidas por episódio.
    """
    ep = list(range(1, len(rewards) + 1))
    # Calcula a média móvel dinâmica de acordo com o tamanho do histórico
    window = max(1, len(rewards) // 10)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid").tolist()
    ep_smooth = ep[window - 1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=rewards, mode="lines",
                             line=dict(color="#4f46e5", width=1),
                             name="Recompensa", opacity=0.4))
    fig.add_trace(go.Scatter(x=ep_smooth, y=smoothed, mode="lines",
                             line=dict(color="#a78bfa", width=2.5),
                             name="Média móvel"))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
        margin=dict(l=30, r=10, t=30, b=30),
        height=240,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
        xaxis=dict(color="#64748b", title="Episódio", gridcolor="#2d3561"),
        yaxis=dict(color="#64748b", title="Recompensa", gridcolor="#2d3561"),
        title=dict(text="📈 Curva de Aprendizado (Recompensa por Episódio)",
                   font=dict(color="#c7d2fe", size=13)),
    )
    return fig


def build_steps_chart(steps: list[int]) -> go.Figure:
    """
    Gera o gráfico mostrando a quantidade de passos por episódio, provando a convergência
    para o menor caminho.
    """
    ep = list(range(1, len(steps) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=steps, mode="lines",
                             line=dict(color="#e11d48", width=1.5),
                             name="Passos"))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
        margin=dict(l=30, r=10, t=30, b=30),
        height=240,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
        xaxis=dict(color="#64748b", title="Episódio", gridcolor="#2d3561"),
        yaxis=dict(color="#64748b", title="Passos", gridcolor="#2d3561"),
        title=dict(text="🏃 Passos por Episódio (Convergência de Rota)",
                   font=dict(color="#fecdd3", size=13)),
    )
    return fig
