# Guia de Contribuição e Alocação da Equipe - PoC Rota RL

Este documento registra a divisão de tarefas e responsabilidades da equipe de desenvolvimento para a Prova de Conceito (PoC) do Sistema de Rotas com Aprendizado por Reforço.

Como este é um projeto de MVP/PoC focado em agilidade, não utilizaremos uma estratégia complexa de branching no Git. Todo o desenvolvimento será feito diretamente na branch principal (`main`) ou em branches simples temporárias, prezando pela comunicação contínua entre os integrantes.

---

## 📋 Alocação de Tarefas (6 Desenvolvedores)

Abaixo está a divisão de tarefas baseada no planejamento técnico da PoC:

### 1. 🌍 Ambiente e Recompensas (Pair Programming)
*   **Foco e Responsabilidades:**
    *   Criar o ambiente de simulação da grade/matriz dinâmica (`envs/rota_env.py`).
    *   Implementar a conversão da coordenada bidimensional para observações discretas 1D no Gymnasium para compatibilidade com o SARSA Tabular.
    *   Implementar a lógica de recompensas e penalidades do Gymnasium (recompensas por alcançar o objetivo, penalidade por colisão com obstáculos, custo negativo por passo para forçar a rota mais curta e potencial BFS de proximidade opcional).
*   **Bibliotecas Utilizadas:** `gymnasium`, `numpy`.
*   **Responsáveis:**
    *   👤 *Daniel*
    *   👤 *Perrone*

---

### 2. 🧠 Treinamento e IA (Pair Programming)
*   **Foco e Responsabilidades:**
    *   Implementar do zero em NumPy o algoritmo clássico **SARSA Tabular (On-Policy)** no arquivo `agents/gerenciador_agente.py`.
    *   Modelar a estrutura da tabela Q (Q-Table) e aplicar a regra clássica de atualização de estados e ações.
    *   Garantir a seleção de ações baseada na política $\epsilon$-greedy com decaimento geométrico por episódio.
    *   Garantir o salvamento dos pesos do modelo treinado em formato `.npy` (NumPy arrays).
*   **Bibliotecas Utilizadas:** `numpy`
*   **Responsáveis:**
    *   👤 *Spinola*
    *   👤 *Adaime*

---

### 3. 🏗️ Arquitetura e Integração (Individual)
*   **Foco e Responsabilidades:**
    *   Garantir a integridade da arquitetura modular do projeto.
    *   Garantir que a classe do ambiente matricial (`envs/`) se comunique perfeitamente com o agente SARSA (`agents/`) e que ambos possam ser importados sem atrito diretamente no frontend.
    *   Organizar a estrutura de diretórios, gerenciar as dependências (`requirements.txt`) e realizar refatorações no código.
*   **Responsável:**
    *   👤 *Kawan*

---

### 4. 🖥️ Frontend e Demonstração (Individual)
*   **Foco e Responsabilidades:**
    *   Criar o dashboard web interativo em Streamlit (`frontend/app.py`).
    *   Permitir que o usuário defina o tamanho da matriz, configure a origem, o destino e adicione/remova obstáculos de forma dinâmica e amigável.
    *   Importar o ambiente e o agente SARSA, acionar o treinamento em tempo real, exibindo os gráficos de recompensa acumulada, quantidade de passos por episódio, tabela Q em tempo real e a rota resultante passo a passo.
    *   Disponibilizar controle numérico e ajustes finos para os hiperparâmetros clássicos do SARSA ($\alpha$, $\gamma$, $\epsilon$, decaimento, semente, etc.).
*   **Bibliotecas Utilizadas:** `streamlit`, `plotly` (para renderização de grade e plots).
*   **Responsáveis:**
    *   👤 *Aurea*
