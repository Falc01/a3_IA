# Projeto de Busca de Rotas com Aprendizado por Reforço (RL)

## 📌 Propósito do Projeto
Este projeto é uma Prova de Conceito (PoC) desenvolvida para validar o uso de Aprendizado por Reforço (Reinforcement Learning - RL), especificamente o algoritmo **DQN (Deep Q-Network)**, para encontrar caminhos e rotas otimizadas entre dois pontos (A e B) em um ambiente de grade/matriz bidimensional customizável.

## 🚀 Abordagem: Matriz Dinâmica e DQN
Para manter o MVP leve, altamente responsivo e de fácil validação, adotamos a abordagem de grade matricial pura via NumPy:
1.  **DQN (Deep Q-Network):** Escolhido por ser extremamente eficiente em ambientes de estado discreto e de tamanho controlado (como matrizes de navegação). O DQN utiliza redes neurais profundas para aproximar a função Q (intuição matemática da IA), aprendendo a desviar de obstáculos e otimizar a rota com base nas recompensas.
2.  **Customização via Frontend:** O tamanho da matriz (grid), a localização dos pontos de origem (A) e destino (B), bem como o posicionamento dos obstáculos, não são fixos (hardcoded). O usuário define tudo isso de forma dinâmica e visual através da interface gráfica antes de iniciar o treinamento da IA.

---

## 🏗️ Arquitetura do Projeto

Abaixo está a descrição da estrutura de pastas e seus propósitos:

### 1. `envs/`
*   **Propósito:** Definição do "mundo virtual" da grade.
*   **Função:** Contém a classe que herda de `gymnasium.Env` (ex: `rota_env.py`), definindo a matriz de estados, as ações discretas de movimento (Cima, Baixo, Esquerda, Direita) e a lógica de recompensas (penalidade por passo/colisão, recompensa por atingir o destino).
*   **Bibliotecas:** `gymnasium`, `numpy`.

### 2. `agents/`
*   **Propósito:** Configuração do cérebro da IA.
*   **Função:** Scripts para inicializar e configurar o algoritmo DQN e seus hiperparâmetros (taxa de aprendizado, tamanho do buffer de repetição, política MLP).
*   **Bibliotecas:** `stable-baselines3`.

### 3. `config/`
*   **Propósito:** Centralização das configurações do cenário.
*   **Função:** Salvar temporariamente o mapa desenhado pelo usuário no frontend (em formato JSON) com as dimensões da matriz, obstáculos e posições de início/fim para que o ambiente de treinamento possa lê-lo.

### 4. `scripts/`
*   **Propósito:** Execução de tarefas em segundo plano.
*   **Função:**
    *   `train.py`: Script para carregar a configuração da matriz, instanciar o ambiente e treinar o agente DQN.
    *   `evaluate.py`: Script para testar o agente treinado e extrair a sequência exata de coordenadas do caminho encontrado.

### 5. `frontend/`
*   **Propósito:** Interface visual do usuário.
*   **Função:** Dashboard em Streamlit (`app.py`) onde o usuário define a matriz, desenha os obstáculos, clica para treinar o agente em tempo real e visualiza a rota calculada desenhada graficamente na tela.
*   **Bibliotecas:** `streamlit`.

### 6. `utils/`
*   **Propósito:** Scripts auxiliares.
*   **Função:** Funções utilitárias de suporte para renderização dos gráficos do mapa e do trajeto.
*   **Bibliotecas:** `matplotlib` ou `plotly`.

### 7. `data/`
*   **Propósito:** Armazenamento persistente de arquivos do ciclo de vida da IA.
*   **Função:** Salvar os logs do TensorBoard (`logs/`) e os pesos do modelo treinado (`models/agente_dqn.zip`).

---

## 📚 Documentação Técnica

*   **Modelagem de IA e Arquitetura:** Para entender o ambiente Gymnasium, a lógica de recompensas e a configuração do DQN, leia o guia de [Arquitetura e Aprendizado por Reforço](file:///c:/Users/joaof/Downloads/Unifacs/inteligencia_artificial/a3_IA/docs/architecture_and_rl.md).
*   **Alocação de Tarefas:** Para ver qual integrante da equipe é responsável por cada parte do código, consulte o [Guia de Contribuição](file:///c:/Users/joaof/Downloads/Unifacs/inteligencia_artificial/a3_IA/CONTRIBUTING.md) na raiz do projeto.


