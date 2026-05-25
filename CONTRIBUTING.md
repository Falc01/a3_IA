# Guia de Contribuição e Alocação da Equipe - PoC Rota RL

Este documento registra a divisão de tarefas e responsabilidades da equipe de desenvolvimento para a Prova de Conceito (PoC) do Sistema de Rotas com Aprendizado por Reforço.

Como este é um projeto de MVP/PoC focado em agilidade, não utilizaremos uma estratégia complexa de branching no Git. Todo o desenvolvimento será feito diretamente na branch principal (`main`) ou em branches simples temporárias, prezando pela comunicação contínua entre os integrantes.

---

## 📋 Alocação de Tarefas (6 Desenvolvedores)

Abaixo está a divisão de tarefas baseada no planejamento técnico da PoC:

### 1. 🌍 Ambiente e Recompensas (Pair Programming)
*   **Foco e Responsabilidades:**
    *   Criar o ambiente de simulação da grade/matriz dinâmica (`envs/rota_env.py`).
    *   Implementar os limites do grid dinâmico com base no arquivo de configuração do usuário.
    *   Implementar a lógica de recompensas e penalidades do Gymnasium (recompensas por alcançar o objetivo, penalidade por colisão com obstáculos e custo negativo por passo para forçar a rota mais curta).
*   **Bibliotecas Utilizadas:** `gymnasium`, `numpy`.
*   **Responsáveis:**
    *   👤 *[Nome do Desenvolvedor 1]*
    *   👤 *[Nome do Desenvolvedor 2]*

---

### 2. 🧠 Treinamento e IA (Pair Programming)
*   **Foco e Responsabilidades:**
    *   Configurar o algoritmo **DQN (Deep Q-Network)**, ideal para mapear estados discretos (coordenadas da matriz) in ações.
    *   Ajustar hiperparâmetros de treinamento (learning rate, buffer size, exploration fraction) para garantir que a IA convirja rapidamente (em menos de 10-15 segundos) no frontend.
    *   Desenvolver o script de treinamento (`scripts/train.py`) e garantir o salvamento do modelo em arquivo `.zip`.
*   **Bibliotecas Utilizadas:** `stable-baselines3`
*   **Responsáveis:**
    *   👤 *[Nome do Desenvolvedor 3]*
    *   👤 *[Nome do Desenvolvedor 4]*

---

### 3. 🏗️ Arquitetura e Integração (Individual)
*   **Foco e Responsabilidades:**
    *   Garantir a integridade da arquitetura modular do projeto.
    *   Garantir que a classe do ambiente matricial (`envs/`) se comunique perfeitamente com a IA (`agents/`) e com os scripts de execução (`scripts/`).
    *   Organizar a estrutura de diretórios, gerenciar as dependências (`requirements.txt`) e realizar refatorações no código.
*   **Responsável:**
    *   👤 *[Nome do Desenvolvedor 5]*

---

### 4. 🖥️ Frontend e Demonstração (Individual)
*   **Foco e Responsabilidades:**
    *   Criar o dashboard web interativo em Streamlit (`frontend/app.py`).
    *   Permitir que o usuário defina o tamanho da matriz, configure a origem, o destino e adicione/remova obstáculos de forma dinâmica e amigável.
    *   Interagir com o script de treino e renderizar o mapa do grid e a rota resultante passo a passo.
*   **Bibliotecas Utilizadas:** `streamlit`, `matplotlib` (para plotagem da grade).
*   **Responsáveis:**
    *   👤 *[Nome do Desenvolvedor 6]*
