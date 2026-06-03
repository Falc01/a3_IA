<<<<<<< HEAD
# Especificação de Arquitetura e Modelagem do Aprendizado por Reforço (RL)

Este documento detalha a modelagem matemática do ambiente Gymnasium, as configurações do agente PPO (Proximal Policy Optimization) e como o fluxo de execução se integra diretamente ao frontend em Streamlit.

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
*   Para evitar loops infinitos, o episódio é interrompido (**truncated**) caso atinja um limite máximo de passos calculado como `2 * (largura * altura)` do grid.

---

## 🧠 2. Configuração do Agente PPO (`stable-baselines3`)

Como a grade dinâmica é configurável pelo usuário e pode variar de tamanho (ex: de $5 \times 5$ até $20 \times 20$), adotamos o algoritmo **PPO (Proximal Policy Optimization)**. Ele é extremamente estável para ambientes dinâmicos e evita o sobreajuste (overfitting) a um tamanho específico de grid.

### Hiperparâmetros Otimizados para o MVP
*   **Política:** `MlpPolicy` (rede neural densa padrão).
*   **Learning Rate (Taxa de Aprendizado):** `0.0003` (padrão estável do PPO).
*   **n_steps:** `512` (número de passos coletados antes de atualizar a rede neural. Um valor menor ajuda a treinar mais rápido em grids pequenos).
*   **batch_size:** `64` (tamanho do lote de processamento da rede).
*   **n_epochs:** `10` (número de épocas de otimização a cada atualização).
*   **ent_coef (Coeficiente de Entropia):** `0.01` (estimula a "curiosidade" da IA, garantindo que ela continue explorando o grid mesmo se o usuário adicionar obstáculos no caminho direto).

---

## 🔄 3. Integração Direta no Frontend (Streamlit)

Para garantir uma interface fluida, em tempo real e sem a necessidade de rodar processos lentos de terminal em segundo plano, **o modelo de IA e o ambiente Gymnasium são importados diretamente dentro do código do Streamlit (`frontend/app.py`)**.

### Fluxo de Execução Embutido

1.  **Interface de Configuração:** O usuário configura os parâmetros do grid (largura, altura, início, fim e obstáculos) através de controles gráficos.
2.  **Inicialização:** O Streamlit instancia a classe `RotaEnv` diretamente na memória do Python.
3.  **Treinamento Síncrono:** Ao clicar em "Treinar", o Streamlit inicia o loop de treinamento do PPO.
4.  **Monitoramento com Callbacks:**
    *   Utilizamos a classe `BaseCallback` do `stable-baselines3` herdada em um callback personalizado.
    *   A cada passo do treino, o callback captura o progresso e as recompensas obtidas.
    *   Esses dados são enviados em tempo real para um componente de progresso (`st.progress`) e um gráfico de linha dinâmico (`st.line_chart`) no painel do usuário, mostrando a IA convergindo.
5.  **Inferência Visual:** Após o treino (que leva poucos segundos), o modelo é executado no ambiente por um episódio e a rota final calculada é desenhada na grade colorida.
=======
# Especificação de Arquitetura e Modelagem do Aprendizado por Reforço (RL)

Este documento detalha a modelagem matemática do ambiente Gymnasium, as configurações do agente PPO (Proximal Policy Optimization) e como o fluxo de execução se integra diretamente ao frontend em Streamlit.

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
*   Para evitar loops infinitos, o episódio é interrompido (**truncated**) caso atinja um limite máximo de passos calculado como `2 * (largura * altura)` do grid.

---

## 🧠 2. Configuração do Agente PPO (`stable-baselines3`)

Como a grade dinâmica é configurável pelo usuário e pode variar de tamanho (ex: de $5 \times 5$ até $20 \times 20$), adotamos o algoritmo **PPO (Proximal Policy Optimization)**. Ele é extremamente estável para ambientes dinâmicos e evita o sobreajuste (overfitting) a um tamanho específico de grid.

### Hiperparâmetros Otimizados para o MVP
*   **Política:** `MlpPolicy` (rede neural densa padrão).
*   **Learning Rate (Taxa de Aprendizado):** `0.0003` (padrão estável do PPO).
*   **n_steps:** `512` (número de passos coletados antes de atualizar a rede neural. Um valor menor ajuda a treinar mais rápido em grids pequenos).
*   **batch_size:** `64` (tamanho do lote de processamento da rede).
*   **n_epochs:** `10` (número de épocas de otimização a cada atualização).
*   **ent_coef (Coeficiente de Entropia):** `0.01` (estimula a "curiosidade" da IA, garantindo que ela continue explorando o grid mesmo se o usuário adicionar obstáculos no caminho direto).

---

## 🔄 3. Integração Direta no Frontend (Streamlit)

Para garantir uma interface fluida, em tempo real e sem a necessidade de rodar processos lentos de terminal em segundo plano, **o modelo de IA e o ambiente Gymnasium são importados diretamente dentro do código do Streamlit (`frontend/app.py`)**.

### Fluxo de Execução Embutido

1.  **Interface de Configuração:** O usuário configura os parâmetros do grid (largura, altura, início, fim e obstáculos) através de controles gráficos.
2.  **Inicialização:** O Streamlit instancia a classe `RotaEnv` diretamente na memória do Python.
3.  **Treinamento Síncrono:** Ao clicar em "Treinar", o Streamlit inicia o loop de treinamento do PPO.
4.  **Monitoramento com Callbacks:**
    *   Utilizamos a classe `BaseCallback` do `stable-baselines3` herdada em um callback personalizado.
    *   A cada passo do treino, o callback captura o progresso e as recompensas obtidas.
    *   Esses dados são enviados em tempo real para um componente de progresso (`st.progress`) e um gráfico de linha dinâmico (`st.line_chart`) no painel do usuário, mostrando a IA convergindo.
5.  **Inferência Visual:** Após o treino (que leva poucos segundos), o modelo é executado no ambiente por um episódio e a rota final calculada é desenhada na grade colorida.
>>>>>>> a814cfcc1169d475372cedb01931d2acc50ee120
