# Especificação de Arquitetura e Modelagem do Aprendizado por Reforço (RL)

Este documento detalha a modelagem matemática do ambiente Gymnasium, as configurações do agente DQN (Deep Q-Network) e como o fluxo de execução se integra ao frontend em Streamlit.

---

## 🌍 1. Modelagem do Ambiente Gymnasium (`envs/rota_env.py`)

O ambiente herda de `gymnasium.Env` e simula a grade bidimensional (grid) onde o agente navega.

### Espaço de Observação (Observation Space)
O estado representa a posição atual do agente no grid.
*   **Tipo:** `Box(low=0, high=max_dim, shape=(2,), dtype=np.float32)`
*   **Vetor de Estado:** `[x_atual, y_atual]` (coordenadas do agente).

### Espaço de Ações (Action Space)
O agente possui 4 movimentos discretos possíveis:
*   `0`: Mover para Cima (eixo Y diminui: `y - 1`)
*   `1`: Mover para Baixo (eixo Y aumenta: `y + 1`)
*   `2`: Mover para a Esquerda (eixo X diminui: `x - 1`)
*   `3`: Mover para a Direita (eixo X aumenta: `x + 1`)

### Função de Recompensa (Reward Function)
A política de recompensas guiará o comportamento do agente para atingir o objetivo com eficiência:
*   **Sucesso (Destino alcançado):** `+100` (fim do episódio: `terminated = True`).
*   **Colisão (Obstáculo ou parede):** `-10` (o agente é penalizado e permanece na mesma posição).
*   **Custo de Passo:** `-1` (aplicado a cada movimento válido para forçar o agente a encontrar o caminho mais curto).

### Condição de Parada Adicional (Truncation)
*   Para evitar loops infinitos caso a IA fique dando voltas no início do treino, o episódio é interrompido (**truncated**) caso atinja um limite máximo de passos calculado como `2 * (largura * altura)` do grid.

---

## 🧠 2. Configuração do Agente DQN (`stable-baselines3`)

Como a grade dinâmica é um espaço de estado bidimensional e discreto de tamanho pequeno a médio (ex: $10 \times 10$), o algoritmo **DQN (Deep Q-Network)** é ideal para este problema. Ele aproxima a tabela de valores Q utilizando uma rede neural (Multi-Layer Perceptron - MLP).

### Hiperparâmetros Otimizados para MVP
Para que o treinamento ocorra de forma fluida durante a demonstração no frontend (tempo de execução inferior a 15 segundos):
*   **Política:** `MlpPolicy` (rede neural densa padrão).
*   **Learning Rate (Taxa de Aprendizado):** `0.001` (convergência rápida).
*   **Buffer Size (Memória de Replay):** `10000` (otimizado para o tamanho do espaço de estados).
*   **Exploration Fraction:** `0.2` (a IA explora caminhos aleatórios nos primeiros 20% do treino e passa a agir de forma ótima nos 80% restantes).
*   **Total Timesteps:** `15000` a `20000` passos são suficientes para convergência em grades de até $10 \times 10$.

---

## 🔄 3. Integração do Fluxo com o Frontend (Streamlit)

O Streamlit atua como o regente do ciclo de vida da simulação. O fluxo funciona da seguinte forma:

```
[ Usuário no Frontend ]
      │ (Ajusta Grid, Início, Fim e Obstáculos)
      ▼
[ Salva config/config.json ]
      │
      ▼
[ Inicia scripts/train.py ] ──► (Instancia RotaEnv & Treina DQN)
      │
      ▼
[ Salva data/models/agente_dqn.zip ]
      │
      ▼
[ Roda scripts/evaluate.py ] ──► (Gera lista de coordenadas da rota ideal)
      │
      ▼
[ Renderiza Rota no Frontend ]
```
