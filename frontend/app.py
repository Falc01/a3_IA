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

# ── Importações do projeto ────────────────────────────────────────────────────
from envs.rota_env import RotaEnv, RotaConfig
from agents.gerenciador_agente import GerenciadorAgenteSarsa
from utils.callbacks import StreamlitCallback
from utils.map_generator import is_solvable, generate_map_obstacles
from utils.plot_utils import build_grid_fig, build_reward_chart, build_steps_chart

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da Página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL Route Finder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    """Carrega o CSS customizado a partir do arquivo style.css"""
    css_path = os.path.join(ROOT_DIR, "frontend", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State — inicialização
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "grid_w": 8,
        "grid_h": 8,
        "start": [0, 0],
        "goal": [7, 7],
        "obstacles": None,
        "cell_mode": "obstacle",   # "start" | "goal" | "obstacle" | "erase"
        "model": None,
        "trained": False,
        "training_rewards": [],
        "training_steps": [],
        "rota": [],
        "rota_info": {},
        "total_episodes": 1000,
        "use_reward_shaping": True,
        "training_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Se obstacles for None, gera um mapa inicial solucionável
    if st.session_state["obstacles"] is None:
        st.session_state["obstacles"] = generate_map_obstacles(
            "Aleatório",
            st.session_state.grid_w,
            st.session_state.grid_h,
            st.session_state.start,
            st.session_state.goal,
            0.2
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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
    last_pos = env._state.tolist()

    while not done and steps < config.max_steps:
        action, _ = model.predict(int(obs), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        current_pos = env._state.tolist()
        if current_pos == last_pos:
            # Agente colidiu e ficou travado. Para a inferência para evitar loop repetitivo na UI
            break
            
        path.append(current_pos)
        last_pos = current_pos
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
_init_state()
_load_css()

with st.sidebar:
    st.markdown("## ⚙️ Configurações")
    st.divider()

    # Gerador de Mapas
    st.markdown('<div class="section-title">🎲 Gerador de Mapas</div>', unsafe_allow_html=True)
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

    # Configurações de Geração
    gen_type = st.selectbox(
        "Padrão de Obstáculos",
        ["Aleatório", "Paredes Alternadas", "Barreira Central"],
        key="gen_type"
    )
    
    gen_density = 0.2
    if gen_type == "Aleatório":
        gen_density = st.slider(
            "Densidade de Obstáculos",
            0.1, 0.5, 0.2, 0.05,
            key="gen_density"
        )
        
    if st.button("🎲 Gerar Novo Mapa", use_container_width=True):
        st.session_state.obstacles = generate_map_obstacles(
            gen_type,
            st.session_state.grid_w,
            st.session_state.grid_h,
            st.session_state.start,
            st.session_state.goal,
            gen_density
        )
        st.session_state.rota = []
        st.session_state.trained = False
        st.session_state.model = None
        st.rerun()

    st.divider()

    # Treinamento
    st.markdown('<div class="section-title">🧠 Treinamento SARSA</div>', unsafe_allow_html=True)
    episodes = st.number_input(
        "Total de Episódios",
        min_value=1,
        max_value=100000,
        value=int(st.session_state.total_episodes),
        step=100,
    )
    st.session_state.total_episodes = episodes
    early_stop_enabled = st.checkbox(
        "Habilitar Early Stopping",
        value=False,
        help="Interrompe o treinamento automaticamente quando as recompensas recentes estabilizam."
    )
    use_reward_shaping = st.checkbox(
        "Habilitar Heurística de Recompensa (BFS)",
        value=True,
        help="Usa a distância mais curta via BFS para guiar o robô. Desmarque para treinar de forma 'cega' (aprendizado mais lento)."
    )
    st.session_state.use_reward_shaping = use_reward_shaping

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

    # Export JSON
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
  <h1>🤖 RL Route Finder</h1>
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
    st.markdown(f"""<div class="metric-card"><div class="label">Agente SARSA</div>
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
    err = _validate_config()
    if err:
        st.markdown(f'<div class="warn-box">{err}</div>', unsafe_allow_html=True)

    # Visualização Plotly (somente leitura, mas mostra a rota)
    st.markdown('<div class="section-title">Visualização da Grade</div>', unsafe_allow_html=True)
    w = st.session_state.grid_w
    h = st.session_state.grid_h
    fig_grid = build_grid_fig(
        w, h,
        st.session_state.start,
        st.session_state.goal,
        st.session_state.obstacles,
        rota=st.session_state.rota,
    )
    st.plotly_chart(fig_grid, use_container_width=True, key="grid_map_tab1")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TREINAR AGENTE
# ══════════════════════════════════════════════════════════════════════════════
with tab_treino:
    st.markdown('<div class="section-title">Parâmetros do Treinamento SARSA</div>', unsafe_allow_html=True)

    col_params, col_info = st.columns([2, 1])
    with col_params:
        c1, c2 = st.columns(2)
        with c1:
            hp_lr = st.number_input("Learning Rate (Taxa de Aprendizado Alpha)", min_value=0.01, max_value=1.0, value=0.10, step=0.05, format="%.2f")
            hp_gamma = st.number_input("Gamma (Fator de Desconto)", min_value=0.0, max_value=1.0, value=0.95, step=0.05, format="%.2f")
            hp_seed = st.number_input("Seed", 0, 9999, 42)
        with c2:
            hp_epsilon = st.number_input("Epsilon Inicial (Exploração)", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.2f")
            hp_decay = st.number_input("Decaimento de Epsilon", min_value=0.80, max_value=1.0, value=0.99, step=0.01, format="%.2f")
            hp_epsilon_min = st.number_input("Epsilon Mínimo", min_value=0.0, max_value=0.5, value=0.01, step=0.01, format="%.2f")

    with col_info:
        st.markdown("""
        <div class="info-box" style="line-height: 1.5;">
        <b>💡 Guia Teórico & Hiperparâmetros SARSA</b><br><br>
        O <b>SARSA</b> é um algoritmo de Aprendizado por Reforço clássico (Tabular e <i>On-Policy</i>). Ele atualiza os valores da tabela Q com base no estado atual ($s$), ação atual ($a$), recompensa ($R$), próximo estado ($s'$) e a próxima ação ($a'$):
        <div style="background:#13172b; padding:8px; border-radius:6px; margin: 8px 0; border:1px solid #2d3561; font-family:monospace; text-align:center;">
        Q(s,a) &larr; Q(s,a) + &alpha; [R + &gamma; Q(s',a') - Q(s,a)]
        </div>
        <b>📚 Significado dos Parâmetros:</b><br><br>
        <b>• Alpha (&alpha;) - Aprendizado (Alpha):</b> Controla a velocidade de atualização. Um valor maior aprende rápido, mas pode instabilizar. Um valor menor (0.10) faz o aprendizado ser gradual e estável.<br><br>
        <b>• Gamma (&gamma;) - Desconto:</b> Mede a importância de recompensas futuras. Próximo de 1.0 (ex: 0.95) faz o agente planejar rotas de longo prazo (previdente). Próximo de 0 faz focar apenas em recompensas imediatas (oportunista).<br><br>
        <b>• Epsilon (&epsilon;) Inicial - Exploração:</b> Chance inicial de escolher uma ação aleatória em vez da melhor conhecida. Garante que o robô explore novos caminhos na grade no início do treino.<br><br>
        <b>• Decaimento de Epsilon:</b> Fator multiplicativo pelo qual o Epsilon é reduzido a cada episódio. Reduz gradualmente a exploração à medida que a Q-table se consolida, priorizando a rota otimizada.<br><br>
        <b>• Epsilon Mínimo:</b> Limite inferior para a exploração residual. Garante que o robô mantenha um nível mínimo de aleatoriedade saudável para testes.<br><br>
        <b>• Total de Episódios:</b> Quantidade de ciclos completos de treinamento. Mais ciclos dão tempo para que os valores de estados e ações convirjam perfeitamente.
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
                use_reward_shaping=st.session_state.use_reward_shaping
            )
            env = RotaEnv(config)

            gerenciador = GerenciadorAgenteSarsa(env=env)
            gerenciador.configurar_agente(seed=int(hp_seed))

            st.markdown('<div class="section-title">📊 Progresso do Treinamento</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0.0, text="Iniciando…")
            chart_ph = st.empty()
            chart_steps_ph = st.empty()
            log_ph = st.empty()

            callback = StreamlitCallback(
                total_episodios=st.session_state.total_episodes,
                progress_bar=progress_bar,
                chart_placeholder=chart_ph,
                chart_steps_placeholder=chart_steps_ph,
                log_placeholder=log_ph,
                early_stop_enabled=early_stop_enabled,
            )

            t0 = time.time()
            try:
                gerenciador.treinar(
                    total_episodios=st.session_state.total_episodes,
                    learning_rate=hp_lr,
                    gamma=hp_gamma,
                    epsilon_inicial=hp_epsilon,
                    epsilon_min=hp_epsilon_min,
                    decaimento_epsilon=hp_decay,
                    callback=callback,
                )
                elapsed = time.time() - t0
                st.session_state.training_rewards = callback.rewards
                st.session_state.training_steps = callback.steps
                st.session_state.model = gerenciador
                st.session_state.trained = True

                if callback.early_stopped:
                    st.info(f"⏱️ Parada Antecipada (Early Stopping) acionada! O aprendizado convergiu e se estabilizou no episódio {len(callback.rewards)}.")
                    progress_bar.progress(1.0, text="✅ Treinamento concluído (Early Stopping)!")
                else:
                    progress_bar.progress(1.0, text="✅ Treinamento concluído!")

                st.markdown(
                    f'<div class="success-box">✅ Treinamento finalizado em {elapsed:.1f}s — '
                    f'{len(callback.rewards)} episódios registrados.</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")

    # Histórico de recompensas persistente (se já treinou antes)
    if st.session_state.trained and st.session_state.training_rewards and not (err):
        st.markdown('<div class="section-title">📊 Resultados do Treinamento</div>', unsafe_allow_html=True)
        
        c_fig1, c_fig2 = st.columns(2)
        with c_fig1:
            st.plotly_chart(
                build_reward_chart(st.session_state.training_rewards),
                use_container_width=True,
                key="reward_chart_tab2_persistent",
            )
        with c_fig2:
            if st.session_state.training_steps:
                st.plotly_chart(
                    build_steps_chart(st.session_state.training_steps),
                    use_container_width=True,
                    key="steps_chart_tab2_persistent",
                )
        
        rw = np.array(st.session_state.training_rewards)
        st_steps = np.array(st.session_state.training_steps)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episódios", len(rw))
        c2.metric("Recompensa Máxima", f"{rw.max():.1f}")
        c3.metric("Média (últimos 20%)", f"{rw[int(len(rw)*0.8):].mean():.1f}")
        if len(st_steps) > 0:
            c4.metric("Menor Qtd Passos", int(st_steps.min()))
        else:
            c4.metric("Recompensa Mínima", f"{rw.min():.1f}")
            
        st.divider()
        
        # Mostrar a Q-table
        with st.expander("📊 Q-Table (Tabela Q de Estados e Ações)", expanded=False):
            gerenciador = st.session_state.model
            if gerenciador is not None:
                q_table = gerenciador.Q
                h = st.session_state.grid_h
                w = st.session_state.grid_w
                rows = []
                for s in range(len(q_table)):
                    cx = s // h
                    cy = s % h
                    rows.append({
                        "Estado": f"E{s} ({cx}, {cy})",
                        "cima": q_table[s, 0],
                        "direita": q_table[s, 1],
                        "baixo": q_table[s, 2],
                        "esquerda": q_table[s, 3]
                    })
                import pandas as pd
                df_q = pd.DataFrame(rows)
                st.dataframe(df_q, use_container_width=True, hide_index=True)


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

            fig_rota = build_grid_fig(
                st.session_state.grid_w,
                st.session_state.grid_h,
                st.session_state.start,
                st.session_state.goal,
                st.session_state.obstacles,
                rota=st.session_state.rota,
            )
            st.plotly_chart(fig_rota, use_container_width=True, key="route_map_tab3")

            # Animação passo-a-passo
            with st.expander("🎬 Animação passo-a-passo", expanded=False):
                if st.button("▶️ Animar"):
                    anim_ph = st.empty()
                    for step_i, pos in enumerate(st.session_state.rota):
                        fig_anim = build_grid_fig(
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
    'RL Route Finder · SARSA + Gymnasium · Streamlit Frontend</p>',
    unsafe_allow_html=True
)
