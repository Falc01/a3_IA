import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st

from envs.rota_env import RotaEnv
from agents.gerenciador_agente import GerenciadorAgentePPO
from utils.callbacks import StreamlitTrainingCallback


st.set_page_config(page_title="Rotas RL", layout="wide")
st.title("Planejamento de Rotas com PPO")

# --- Sidebar: Grid ---
st.sidebar.header("Grid")
largura = st.sidebar.slider("Largura", 3, 20, 5)
altura = st.sidebar.slider("Altura", 3, 20, 5)

st.sidebar.subheader("Origem")
col_ox, col_oy = st.sidebar.columns(2)
origem_x = col_ox.number_input("X", min_value=0, max_value=largura - 1, value=0, key="ox")
origem_y = col_oy.number_input("Y", min_value=0, max_value=altura - 1, value=0, key="oy")

st.sidebar.subheader("Destino")
col_dx, col_dy = st.sidebar.columns(2)
destino_x = col_dx.number_input("X", min_value=0, max_value=largura - 1, value=largura - 1, key="dx")
destino_y = col_dy.number_input("Y", min_value=0, max_value=altura - 1, value=altura - 1, key="dy")

# --- Sidebar: Obstaculos ---
st.sidebar.subheader("Obstaculos")
if "obstaculos" not in st.session_state:
    st.session_state.obstaculos = []

col_obsx, col_obsy = st.sidebar.columns(2)
obs_x = col_obsx.number_input("X", min_value=0, max_value=largura - 1, value=0, key="obs_x")
obs_y = col_obsy.number_input("Y", min_value=0, max_value=altura - 1, value=0, key="obs_y")

col_add, col_rm = st.sidebar.columns(2)
if col_add.button("Adicionar"):
    coord = (int(obs_x), int(obs_y))
    if coord != (int(origem_x), int(origem_y)) and coord != (int(destino_x), int(destino_y)):
        if coord not in st.session_state.obstaculos:
            st.session_state.obstaculos.append(coord)

if col_rm.button("Remover"):
    coord = (int(obs_x), int(obs_y))
    if coord in st.session_state.obstaculos:
        st.session_state.obstaculos.remove(coord)

if st.sidebar.button("Limpar Todos"):
    st.session_state.obstaculos = []

if st.session_state.obstaculos:
    st.sidebar.write("Lista:", st.session_state.obstaculos)

# --- Sidebar: Hiperparametros ---
st.sidebar.header("Hiperparametros PPO")
learning_rate = st.sidebar.select_slider("Learning Rate", [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.01], value=0.0003)
n_steps = st.sidebar.select_slider("n_steps", [128, 256, 512, 1024, 2048], value=512)
batch_size = st.sidebar.select_slider("Batch Size", [32, 64, 128, 256], value=64)
n_epochs = st.sidebar.slider("n_epochs", 3, 30, 10)
ent_coef = st.sidebar.select_slider("ent_coef", [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1], value=0.01)
total_timesteps = st.sidebar.select_slider("Timesteps", [5000, 10000, 20000, 50000, 100000], value=5000)


# --- Funcao de renderizacao do grid ---
def renderizar_grid(larg, alt, origem, destino, obstaculos, rota=None):
    rota_set = set(rota) if rota else set()
    celula_tam = max(28, min(48, 400 // max(larg, alt)))

    html = f"""
    <div style="display:flex; justify-content:center; padding:10px;">
    <table style="border-collapse:collapse; border:2px solid #333; border-radius:4px;">
    """
    for y in range(alt):
        html += "<tr>"
        for x in range(larg):
            cell = (x, y)
            if cell == tuple(origem):
                cor = "#66BB6A"
                texto = "A"
                font_cor = "#fff"
            elif cell == tuple(destino):
                cor = "#42A5F5"
                texto = "B"
                font_cor = "#fff"
            elif cell in set(obstaculos):
                cor = "#37474F"
                texto = ""
                font_cor = "#fff"
            elif cell in rota_set:
                cor = "#FFCA28"
                texto = ""
                font_cor = "#333"
            else:
                cor = "#FAFAFA"
                texto = ""
                font_cor = "#333"
            html += (
                f"<td style='width:{celula_tam}px; height:{celula_tam}px; "
                f"text-align:center; vertical-align:middle; "
                f"border:1px solid #BDBDBD; background:{cor}; "
                f"color:{font_cor}; font-weight:bold; font-size:14px;'>"
                f"{texto}</td>"
            )
        html += "</tr>"
    html += "</table></div>"
    return html


# --- Layout principal ---
col_grid, col_treino = st.columns([1, 1])

with col_grid:
    st.subheader("Grid")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(
        renderizar_grid(
            largura, altura,
            (int(origem_x), int(origem_y)),
            (int(destino_x), int(destino_y)),
            st.session_state.obstaculos,
            rota=st.session_state.get("rota", None)
        ),
        unsafe_allow_html=True
    )

    legenda = """
    <div style="display:flex; gap:16px; justify-content:center; margin-top:8px; font-size:13px;">
        <span><span style="display:inline-block;width:14px;height:14px;background:#66BB6A;border:1px solid #999;vertical-align:middle;"></span> Origem</span>
        <span><span style="display:inline-block;width:14px;height:14px;background:#42A5F5;border:1px solid #999;vertical-align:middle;"></span> Destino</span>
        <span><span style="display:inline-block;width:14px;height:14px;background:#37474F;border:1px solid #999;vertical-align:middle;"></span> Obstaculo</span>
        <span><span style="display:inline-block;width:14px;height:14px;background:#FFCA28;border:1px solid #999;vertical-align:middle;"></span> Rota</span>
    </div>
    """
    st.markdown(legenda, unsafe_allow_html=True)

    if st.session_state.get("rota"):
        with st.expander("Ver rota detalhada"):
            for i, pos in enumerate(st.session_state.rota):
                st.write(f"Passo {i}: {pos}")

with col_treino:
    st.subheader("Treinamento")

    if st.button("Treinar Agente", type="primary"):
        st.session_state.rota = None

        env = RotaEnv(
            largura=largura,
            altura=altura,
            origem=(int(origem_x), int(origem_y)),
            destino=(int(destino_x), int(destino_y)),
            obstaculos=st.session_state.obstaculos
        )

        kwargs_custom = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "ent_coef": ent_coef,
        }

        gerenciador = GerenciadorAgentePPO(env=env, log_dir="data/logs/", model_dir="data/models/")
        gerenciador.configurar_agente(kwargs_personalizados=kwargs_custom, seed=42)

        barra_progresso = st.progress(0, text="Treinando...")
        grafico_recompensa = st.line_chart([])

        callback = StreamlitTrainingCallback(
            total_timesteps=total_timesteps,
            barra_progresso=barra_progresso,
            grafico_linha=grafico_recompensa
        )

        gerenciador.treinar(total_timesteps=total_timesteps, callback=callback)
        gerenciador.salvar_modelo("agente_ppo_logistica")
        st.success("Modelo treinado e salvo com sucesso.")

        # Inferencia
        obs, _ = env.reset()
        rota = [tuple(obs.astype(int))]
        terminated = False
        truncated = False
        passos = 0
        max_passos = 2 * (largura * altura)

        while not terminated and not truncated and passos < max_passos:
            action, _ = gerenciador.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            rota.append(tuple(obs.astype(int)))
            passos += 1

        if terminated:
            st.success(f"Destino alcancado em {passos} passos.")
        else:
            st.warning("Nao convergiu. Aumente os timesteps ou ajuste hiperparametros.")

        st.session_state.rota = rota

        grid_placeholder.markdown(
            renderizar_grid(
                largura, altura,
                (int(origem_x), int(origem_y)),
                (int(destino_x), int(destino_y)),
                st.session_state.obstaculos,
                rota=rota
            ),
            unsafe_allow_html=True
        )
