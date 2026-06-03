"""
Frontend Streamlit — Projeto de Busca de Rotas com PPO (RL)
Coloque este arquivo em:  frontend/app.py
Execute com:              streamlit run frontend/app.py
"""

import sys
import os

# Garante que o diretório raiz do projeto esteja no path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Importações do projeto ────────────────────────────────────────────────────
from envs.rota_env import RotaEnv, RotaConfig
from agents.gerenciador_agente import GerenciadorAgentePPO
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da Página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL Route Finder — PPO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #0e1117; }

    /* Cabeçalho hero */
    .hero-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 60%, #0f3460 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 28px 36px 20px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-box h1 { color: #e0e7ff; font-size: 2.2rem; margin: 0; }
    .hero-box p  { color: #94a3b8; font-size: 1rem; margin-top: 8px; }

    /* Cards de métricas */
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card .label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #a5b4fc; font-size: 1.8rem; font-weight: 700; }

    /* Seções */
    .section-title {
        color: #c7d2fe;
        font-size: 1.05rem;
        font-weight: 600;
        border-left: 3px solid #6366f1;
        padding-left: 10px;
        margin: 20px 0 12px;
    }

    /* Badges de estado da célula */
    .cell-legend {
        display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px;
    }
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: #1e2130; border: 1px solid #2d3561;
        border-radius: 20px; padding: 4px 12px;
        font-size: 0.8rem; color: #cbd5e1;
    }

    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13172b; }
    [data-testid="stSidebar"] h2 { color: #e0e7ff; }

    /* Info / warning boxes */
    .info-box {
        background: #1e2840; border: 1px solid #3b4ea8;
        border-radius: 10px; padding: 12px 16px;
        color: #93c5fd; font-size: 0.88rem;
    }
    .warn-box {
        background: #2a1e10; border: 1px solid #9a5a1e;
        border-radius: 10px; padding: 12px 16px;
        color: #fbbf24; font-size: 0.88rem;
    }
    .success-box {
        background: #0f2d1e; border: 1px solid #16803c;
        border-radius: 10px; padding: 12px 16px;
        color: #4ade80; font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State — inicialização
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "grid_w": 8,
        "grid_h": 8,
        "start": [0, 0],
        "goal": [7, 7],
        "obstacles": [],
        "cell_mode": "obstacle",   # "start" | "goal" | "obstacle" | "erase"
        "model": None,
        "trained": False,
        "training_rewards": [],
        "rota": [],
        "rota_info": {},
        "total_timesteps": 20000,
        "training_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Callback Streamlit
# ─────────────────────────────────────────────────────────────────────────────
class StreamlitCallback(BaseCallback):
    def __init__(self, total_timesteps, progress_bar, chart_placeholder, log_placeholder, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar
        self.chart_ph = chart_placeholder
        self.log_ph = log_placeholder
        self.rewards: list[float] = []

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.progress_bar.progress(progress, text=f"🔄 Treinando: {self.num_timesteps:,} / {self.total_timesteps:,} passos")

        if "episode" in self.locals["infos"][0]:
            r = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(float(r))
            st.session_state["training_rewards"] = self.rewards[:]

            if len(self.rewards) % 5 == 0 or len(self.rewards) <= 3:
                fig = _build_reward_chart(self.rewards)
                self.chart_ph.plotly_chart(fig, use_container_width=True, key=f"rt_{len(self.rewards)}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_grid_fig(
    w: int, h: int,
    start: list, goal: list,
    obstacles: list,
    rota: list | None = None,
    agent_pos: list | None = None,
) -> go.Figure:
    """Gera a figura Plotly da grade."""
    z = np.zeros((h, w))          # 0 = livre
    text = [["" for _ in range(w)] for _ in range(h)]
    colors = {
        "free": "#1e2130",
        "obstacle": "#374151",
        "start": "#4f46e5",
        "goal": "#16a34a",
        "path": "#7c3aed",
        "agent": "#f59e0b",
    }

    # Mapa de z para cor discreta
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

    # Grade de linhas (bordas)
    for i in range(w + 1):
        fig.add_shape(type="line", x0=i - 0.5, x1=i - 0.5, y0=-0.5, y1=h - 0.5,
                      line=dict(color="#2d3561", width=1))
    for j in range(h + 1):
        fig.add_shape(type="line", x0=-0.5, x1=w - 0.5, y0=j - 0.5, y1=j - 0.5,
                      line=dict(color="#2d3561", width=1))

    # Anotações de coordenada nas células (opcional — fica pesado em grids grandes)
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


def _build_reward_chart(rewards: list[float]) -> go.Figure:
    ep = list(range(1, len(rewards) + 1))
    # Média móvel
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


def _run_inference(model, config: RotaConfig) -> tuple[list, dict]:
    """Executa 1 episódio de inferência e devolve a rota + métricas."""
    env = RotaEnv(config)
    obs, _ = env.reset()
    path = [list(env._state.tolist())]
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < config.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        path.append(list(env._state.tolist()))
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    success = np.array_equal(env._state, config.goal)
    return path, {"steps": steps, "total_reward": total_reward, "success": success}


def _validate_config() -> str | None:
    """Retorna mensagem de erro ou None se config OK."""
    w, h = st.session_state.grid_w, st.session_state.grid_h
    s, g = st.session_state.start, st.session_state.goal
    obs = st.session_state.obstacles

    if s == g:
        return "⚠️ Início e destino não podem ser iguais."
    if not (0 <= s[0] < w and 0 <= s[1] < h):
        return f"⚠️ Posição de início {s} fora dos limites ({w}×{h})."
    if not (0 <= g[0] < w and 0 <= g[1] < h):
        return f"⚠️ Posição de destino {g} fora dos limites ({w}×{h})."
    obs_set = {tuple(o) for o in obs}
    if tuple(s) in obs_set:
        return "⚠️ O ponto de início está bloqueado por um obstáculo."
    if tuple(g) in obs_set:
        return "⚠️ O ponto de destino está bloqueado por um obstáculo."
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configurações
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configurações")
    st.divider()

    # Dimensões
    st.markdown('<div class="section-title">📐 Dimensões da Grade</div>', unsafe_allow_html=True)
    new_w = st.slider("Largura (colunas)", 4, 20, st.session_state.grid_w, key="sl_w")
    new_h = st.slider("Altura (linhas)", 4, 20, st.session_state.grid_h, key="sl_h")

    if new_w != st.session_state.grid_w or new_h != st.session_state.grid_h:
        st.session_state.grid_w = new_w
        st.session_state.grid_h = new_h
        # Resetar posições se saírem dos limites
        if st.session_state.start[0] >= new_w or st.session_state.start[1] >= new_h:
            st.session_state.start = [0, 0]
        if st.session_state.goal[0] >= new_w or st.session_state.goal[1] >= new_h:
            st.session_state.goal = [new_w - 1, new_h - 1]
        st.session_state.obstacles = [
            o for o in st.session_state.obstacles
            if o[0] < new_w and o[1] < new_h
        ]
        st.session_state.trained = False
        st.session_state.model = None
        st.session_state.rota = []

    st.divider()

    # Posições
    st.markdown('<div class="section-title">📍 Posições</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sx = st.number_input("Início X", 0, new_w - 1, st.session_state.start[0])
        sy = st.number_input("Início Y", 0, new_h - 1, st.session_state.start[1])
    with c2:
        gx = st.number_input("Destino X", 0, new_w - 1, st.session_state.goal[0])
        gy = st.number_input("Destino Y", 0, new_h - 1, st.session_state.goal[1])

    st.session_state.start = [int(sx), int(sy)]
    st.session_state.goal = [int(gx), int(gy)]

    st.divider()

    # Treinamento
    st.markdown('<div class="section-title">🧠 Treinamento PPO</div>', unsafe_allow_html=True)
    ts = st.select_slider(
        "Total de Timesteps",
        options=[5000, 10000, 20000, 50000, 100000],
        value=st.session_state.total_timesteps,
    )
    st.session_state.total_timesteps = ts

    st.divider()

    # Obstáculos
    st.markdown('<div class="section-title">🚧 Obstáculos (clique na grade)</div>', unsafe_allow_html=True)
    cell_mode = st.radio(
        "Modo de clique",
        ["obstacle", "erase", "start", "goal"],
        format_func=lambda x: {"obstacle": "🚧 Adicionar", "erase": "🧹 Apagar",
                                "start": "🔵 Mover início", "goal": "🟢 Mover destino"}[x],
        horizontal=False,
        index=["obstacle", "erase", "start", "goal"].index(st.session_state.cell_mode),
    )
    st.session_state.cell_mode = cell_mode

    if st.button("🗑️ Limpar todos os obstáculos"):
        st.session_state.obstacles = []
        st.session_state.rota = []
        st.rerun()

    # Export JSON
    st.divider()
    st.markdown('<div class="section-title">💾 Exportar Configuração</div>', unsafe_allow_html=True)
    config_dict = {
        "width": st.session_state.grid_w,
        "height": st.session_state.grid_h,
        "start": st.session_state.start,
        "goal": st.session_state.goal,
        "obstacles": st.session_state.obstacles,
    }
    st.download_button(
        "⬇️ Baixar config.json",
        data=json.dumps(config_dict, indent=2),
        file_name="rota_config.json",
        mime="application/json",
    )

    # Import JSON
    uploaded = st.file_uploader("📤 Carregar config.json", type="json")
    if uploaded:
        try:
            loaded = json.load(uploaded)
            st.session_state.grid_w = loaded.get("width", 8)
            st.session_state.grid_h = loaded.get("height", 8)
            st.session_state.start = loaded.get("start", [0, 0])
            st.session_state.goal = loaded.get("goal", [7, 7])
            st.session_state.obstacles = loaded.get("obstacles", [])
            st.session_state.trained = False
            st.session_state.model = None
            st.session_state.rota = []
            st.success("Configuração carregada!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CONTEÚDO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <h1>🤖 RL Route Finder — PPO</h1>
  <p>Configure a grade, defina obstáculos, treine o agente e visualize a rota encontrada.</p>
</div>
""", unsafe_allow_html=True)

# ── Métricas rápidas ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card"><div class="label">Grade</div>
    <div class="value">{st.session_state.grid_w}×{st.session_state.grid_h}</div></div>""",
    unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card"><div class="label">Obstáculos</div>
    <div class="value">{len(st.session_state.obstacles)}</div></div>""",
    unsafe_allow_html=True)
with m3:
    status = "✅ Treinado" if st.session_state.trained else "⏳ Aguardando"
    st.markdown(f"""<div class="metric-card"><div class="label">Agente PPO</div>
    <div class="value" style="font-size:1.1rem">{status}</div></div>""",
    unsafe_allow_html=True)
with m4:
    steps_info = st.session_state.rota_info.get("steps", "—")
    st.markdown(f"""<div class="metric-card"><div class="label">Passos na Rota</div>
    <div class="value">{steps_info}</div></div>""",
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_grade, tab_treino, tab_rota = st.tabs(["🗺️  Grade Interativa", "🧠  Treinar Agente", "🏁  Rota Final"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GRADE INTERATIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab_grade:
    st.markdown('<div class="section-title">Clique nas células para editar</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cell-legend">
      <span class="badge"><span style="color:#4f46e5">■</span> Início (A)</span>
      <span class="badge"><span style="color:#16a34a">■</span> Destino (B)</span>
      <span class="badge"><span style="color:#374151">■</span> Obstáculo</span>
      <span class="badge"><span style="color:#7c3aed">■</span> Rota PPO</span>
    </div>
    <br>
    """, unsafe_allow_html=True)

    err = _validate_config()
    if err:
        st.markdown(f'<div class="warn-box">{err}</div>', unsafe_allow_html=True)

    # Grade de botões clicáveis
    w = st.session_state.grid_w
    h = st.session_state.grid_h
    obs_set = {tuple(o) for o in st.session_state.obstacles}
    path_set = {tuple(p) for p in st.session_state.rota}

    EMOJI = {
        "free": "⬜",
        "obstacle": "🚧",
        "start": "🔵",
        "goal": "🟢",
        "path": "🟣",
    }

    for row in range(h):
        cols = st.columns(w)
        for col in range(w):
            pos = (col, row)
            if pos == tuple(st.session_state.start):
                icon = EMOJI["start"]
            elif pos == tuple(st.session_state.goal):
                icon = EMOJI["goal"]
            elif pos in obs_set:
                icon = EMOJI["obstacle"]
            elif pos in path_set:
                icon = EMOJI["path"]
            else:
                icon = EMOJI["free"]

            if cols[col].button(icon, key=f"cell_{col}_{row}", use_container_width=True):
                mode = st.session_state.cell_mode
                if mode == "start":
                    if pos != tuple(st.session_state.goal):
                        st.session_state.start = [col, row]
                        st.session_state.rota = []
                elif mode == "goal":
                    if pos != tuple(st.session_state.start):
                        st.session_state.goal = [col, row]
                        st.session_state.rota = []
                elif mode == "obstacle":
                    if pos not in (tuple(st.session_state.start), tuple(st.session_state.goal)):
                        if list(pos) not in st.session_state.obstacles:
                            st.session_state.obstacles.append(list(pos))
                        st.session_state.rota = []
                elif mode == "erase":
                    if list(pos) in st.session_state.obstacles:
                        st.session_state.obstacles.remove(list(pos))
                        st.session_state.rota = []
                st.rerun()

    # Visualização Plotly (somente leitura, mas mostra a rota)
    st.markdown('<div class="section-title">Visualização da Grade</div>', unsafe_allow_html=True)
    fig_grid = _build_grid_fig(
        w, h,
        st.session_state.start,
        st.session_state.goal,
        st.session_state.obstacles,
        rota=st.session_state.rota,
    )
    st.plotly_chart(fig_grid, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TREINAR AGENTE
# ══════════════════════════════════════════════════════════════════════════════
with tab_treino:
    st.markdown('<div class="section-title">Parâmetros do Treinamento PPO</div>', unsafe_allow_html=True)

    col_params, col_info = st.columns([2, 1])
    with col_params:
        hp_lr = st.select_slider("Learning Rate", [0.0001, 0.0003, 0.001, 0.003], value=0.0003)
        hp_ent = st.select_slider("Coeficiente de Entropia", [0.0, 0.005, 0.01, 0.05, 0.1], value=0.01)
        hp_nsteps = st.select_slider("n_steps", [128, 256, 512, 1024], value=512)
        hp_seed = st.number_input("Seed", 0, 9999, 42)

    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>💡 Dicas</b><br><br>
        <b>Learning Rate:</b> 0.0003 é estável para a maioria dos grids.<br><br>
        <b>Entropia:</b> Valores maiores = mais exploração. Útil para grids com labirintos.<br><br>
        <b>n_steps:</b> Passos coletados antes de atualizar a rede.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    err = _validate_config()
    if err:
        st.markdown(f'<div class="warn-box">{err}</div>', unsafe_allow_html=True)
    else:
        btn_train = st.button("🚀 Iniciar Treinamento", type="primary", use_container_width=True)

        if btn_train:
            st.session_state.training_rewards = []
            st.session_state.trained = False
            st.session_state.model = None
            st.session_state.rota = []

            # Monta configuração
            config = RotaConfig(
                width=st.session_state.grid_w,
                height=st.session_state.grid_h,
                start=tuple(st.session_state.start),
                goal=tuple(st.session_state.goal),
                obstacles=[tuple(o) for o in st.session_state.obstacles],
            )
            env = RotaEnv(config)

            gerenciador = GerenciadorAgentePPO(env=env)
            gerenciador.configurar_agente(
                kwargs_personalizados={
                    "learning_rate": hp_lr,
                    "ent_coef": hp_ent,
                    "n_steps": hp_nsteps,
                    "verbose": 0,
                },
                seed=int(hp_seed),
            )

            st.markdown('<div class="section-title">📊 Progresso do Treinamento</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0.0, text="Iniciando…")
            chart_ph = st.empty()
            log_ph = st.empty()

            callback = StreamlitCallback(
                total_timesteps=st.session_state.total_timesteps,
                progress_bar=progress_bar,
                chart_placeholder=chart_ph,
                log_placeholder=log_ph,
            )

            t0 = time.time()
            try:
                gerenciador.treinar(
                    total_timesteps=st.session_state.total_timesteps,
                    callback=callback,
                )
                elapsed = time.time() - t0
                st.session_state.model = gerenciador.model
                st.session_state.trained = True

                st.markdown(
                    f'<div class="success-box">✅ Treinamento concluído em {elapsed:.1f}s — '
                    f'{len(st.session_state.training_rewards)} episódios registrados.</div>',
                    unsafe_allow_html=True
                )
                progress_bar.progress(1.0, text="✅ Treinamento concluído!")

                # Exibir gráfico final
                if st.session_state.training_rewards:
                    st.plotly_chart(
                        _build_reward_chart(st.session_state.training_rewards),
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")

    # Histórico de recompensas (se já treinou antes)
    if st.session_state.training_rewards and not (err):
        with st.expander("📜 Ver histórico de recompensas", expanded=False):
            st.line_chart(st.session_state.training_rewards, height=200)
            rw = np.array(st.session_state.training_rewards)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Episódios", len(rw))
            c2.metric("Máxima", f"{rw.max():.1f}")
            c3.metric("Média (últimos 20%)", f"{rw[int(len(rw)*0.8):].mean():.1f}")
            c4.metric("Mínima", f"{rw.min():.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROTA FINAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_rota:
    if not st.session_state.trained or st.session_state.model is None:
        st.markdown(
            '<div class="warn-box">🧠 Nenhum agente treinado ainda. Vá à aba <b>Treinar Agente</b> e inicie o treinamento.</div>',
            unsafe_allow_html=True
        )
    else:
        btn_run = st.button("▶️ Executar Inferência (encontrar rota)", type="primary", use_container_width=True)

        if btn_run:
            config = RotaConfig(
                width=st.session_state.grid_w,
                height=st.session_state.grid_h,
                start=tuple(st.session_state.start),
                goal=tuple(st.session_state.goal),
                obstacles=[tuple(o) for o in st.session_state.obstacles],
            )
            with st.spinner("Calculando rota…"):
                rota, info = _run_inference(st.session_state.model, config)
            st.session_state.rota = rota
            st.session_state.rota_info = info

        if st.session_state.rota:
            info = st.session_state.rota_info
            success = info.get("success", False)

            if success:
                st.markdown(
                    f'<div class="success-box">🏁 Destino alcançado em <b>{info["steps"]}</b> passos! '
                    f'Recompensa acumulada: <b>{info["total_reward"]:.1f}</b></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="warn-box">⚠️ O agente não alcançou o destino no limite de passos ({info["steps"]}). '
                    f'Tente aumentar os timesteps de treinamento.</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="section-title">Rota Calculada pelo Agente</div>', unsafe_allow_html=True)

            fig_rota = _build_grid_fig(
                st.session_state.grid_w,
                st.session_state.grid_h,
                st.session_state.start,
                st.session_state.goal,
                st.session_state.obstacles,
                rota=st.session_state.rota,
            )
            st.plotly_chart(fig_rota, use_container_width=True)

            # Animação passo-a-passo
            with st.expander("🎬 Animação passo-a-passo", expanded=False):
                if st.button("▶️ Animar"):
                    anim_ph = st.empty()
                    for step_i, pos in enumerate(st.session_state.rota):
                        fig_anim = _build_grid_fig(
                            st.session_state.grid_w,
                            st.session_state.grid_h,
                            st.session_state.start,
                            st.session_state.goal,
                            st.session_state.obstacles,
                            rota=st.session_state.rota[:step_i + 1],
                            agent_pos=pos,
                        )
                        anim_ph.plotly_chart(fig_anim, use_container_width=True, key=f"anim_{step_i}")
                        time.sleep(0.15)

            # Tabela de passos
            with st.expander("📋 Sequência de movimentos", expanded=False):
                ACTION_NAMES = {0: "⬆️ Cima", 1: "⬇️ Baixo", 2: "⬅️ Esquerda", 3: "➡️ Direita"}
                rota = st.session_state.rota
                rows = []
                for i in range(1, len(rota)):
                    dx = rota[i][0] - rota[i-1][0]
                    dy = rota[i][1] - rota[i-1][1]
                    if dx == 0 and dy == -1: act = "⬆️ Cima"
                    elif dx == 0 and dy == 1: act = "⬇️ Baixo"
                    elif dx == -1 and dy == 0: act = "⬅️ Esquerda"
                    elif dx == 1 and dy == 0: act = "➡️ Direita"
                    else: act = "— (sem movimento)"
                    rows.append({"Passo": i, "De": str(rota[i-1]), "Para": str(rota[i]), "Ação": act})
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="color:#475569;text-align:center;font-size:0.8rem">'
    'RL Route Finder · PPO + Gymnasium · Streamlit Frontend</p>',
    unsafe_allow_html=True
)
