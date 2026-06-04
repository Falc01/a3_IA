# Especificação de Arquitetura e Modelagem do Aprendizado por Reforço (RL)

Este documento detalha a modelagem matemática do ambiente Gymnasium, as configurações do agente SARSA Tabular e como o fluxo de execução se integra diretamente ao frontend em Streamlit.

---

## 🌍 1. Modelagem do Ambiente Gymnasium (`envs/rota_env.py`)

O ambiente herda de `gymnasium.Env` e simula a grade bidimensional (grid) onde o agente navega.

### Espaço de Observação (Observation Space)
Como o agente utiliza uma tabela Q clássica (Tabular), mapeamos a posição bidimensional do agente para um índice numérico discreto.
*   **Tipo:** `Discrete(width * height)`
*   **Mapeamento de Estado ($S$):** Para a posição $(x, y)$ em um grid de altura $H$, a observação retornada é calculada por:
    $$S = x \times H + y$$
    Esse mapeamento transforma a posição 2D do grid em um índice de linha 1D na Q-table.

### Espaço de Ações (Action Space)
O agente possui 4 movimentos discretos possíveis:
*   `0`: Mover para Cima (eixo Y diminui: `y - 1`)
*   `1`: Mover para a Direita (eixo X aumenta: `x + 1`)
*   `2`: Mover para Baixo (eixo Y aumenta: `y + 1`)
*   `3`: Mover para a Esquerda (eixo X diminui: `x - 1`)

### Função de Recompensa (Reward Function)
A política de recompensas guiará o comportamento do agente para atingir o objetivo com eficiência:
*   **Sucesso (Destino alcançado):** `+100` (fim do episódio: `terminated = True`).
*   **Colisão (Obstáculo ou parede):** `-10` (o agente é penalizado e permanece na mesma posição).
*   **Custo de Passo:** `-1` (aplicado a cada movimento válido para forçar o agente a encontrar o caminho mais curto).
*   **Heurística de Recompensa (BFS - Opcional):** Para acelerar o aprendizado, adicionamos um potencial baseado na distância mais curta do estado atual ao destino calculado via algoritmo BFS. A recompensa adicional é dada por $+0.5$ se o agente se aproximar do destino e $-0.5$ se se afastar.

### Condição de Parada Adicional (Truncation)
*   Para evitar loops infinitos de exploração, o episódio é interrompido (**truncated**) caso atinja um limite máximo de passos calculado como `2 * (largura * altura)` do grid.

---

## 🧠 2. Configuração do Agente SARSA Tabular (`agents/gerenciador_agente.py`)

Em substituição a bibliotecas externas pesadas como o `stable-baselines3` (PPO), implementamos o algoritmo **SARSA (State-Action-Reward-State-Action)** puro do zero em NumPy. O SARSA é um algoritmo *On-Policy* clássico, adequado para fins acadêmicos e extremamente eficiente para grades discretas.

### Regra de Atualização
A Q-table $Q(S, A)$ é uma matriz de dimensões $(W \times H, 4)$. A cada passo do ambiente, o agente atualiza o valor da tabela através da equação:
$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

### Hiperparâmetros Controlados
*   **Alpha ($\alpha$):** Taxa de aprendizado (padrão `0.10`). Controla o peso das novas experiências na Q-table.
*   **Gamma ($\gamma$):** Fator de desconto (padrão `0.95`). Modula o peso das recompensas de longo prazo.
*   **Epsilon Inicial ($\epsilon$):** Taxa inicial de exploração na política $\epsilon$-greedy (padrão `0.30`).
*   **Decaimento de Epsilon:** Fator multiplicativo aplicado por episódio para reduzir a exploração à medida que o agente aprende (padrão `0.99`).
*   **Epsilon Mínimo:** Piso residual de exploração (padrão `0.01`).

---

## 🔄 3. Integração no Frontend (Streamlit)

Para garantir uma interface responsiva, com treinamento instantâneo, **o modelo de IA e o ambiente Gymnasium são executados diretamente na memória do Streamlit (`frontend/app.py`)**.

### Fluxo de Execução
1.  **Interface de Configuração:** O usuário desenha obstáculos, altera dimensões do grid e configura posições de Início/Destino na barra lateral.
2.  **Ajuste de Parâmetros:** O usuário configura os hiperparâmetros clássicos do SARSA Tabular na aba "Treinar Agente".
3.  **Treinamento Síncrono:** Ao clicar em "Iniciar Treinamento", o Streamlit roda o loop de episódios do SARSA. 
4.  **Monitoramento via Callbacks:**
    *   Um callback captura a recompensa e quantidade de passos por episódio.
    *   Os gráficos de recompensa acumulada e passos são plotados em tempo real na tela.
    *   Uma visualização interativa do DataFrame com os valores numéricos da Q-Table é exposta para fins didáticos.
5.  **Inferência Visual:** O usuário pode rodar a inferência para ver a rota traçada pelo agente ou assistir a uma animação passo a passo do robô navegando.
