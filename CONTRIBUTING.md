<<<<<<< HEAD
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
    *   👤 *Daniel*
    *   👤 *Perrone*

---

### 2. 🧠 Treinamento e IA (Pair Programming)
*   **Foco e Responsabilidades:**
    *   Configurar o algoritmo **PPO (Proximal Policy Optimization)**, ideal para lidar com a variação dinâmica de tamanhos de matrizes e obstáculos configurados via frontend.
    *   Ajustar os hiperparâmetros do PPO (taxa de aprendizado, n_steps, batch_size e ent_coef para exploração/entropia) para garantir que a IA aprenda rapidamente (em menos de 10-15 segundos) no frontend.
    *   Criar a classe de Callback personalizada (`BaseCallback`) para enviar as métricas de treino em tempo real para o painel do Streamlit.
    *   Garantir o salvamento dos pesos do modelo treinado em formato `.zip`.
*   **Bibliotecas Utilizadas:** `stable-baselines3`
*   **Responsáveis:**
    *   👤 *Spinola*
    *   👤 *Adaime*

---

### 3. 🏗️ Arquitetura e Integração (Individual)
*   **Foco e Responsabilidades:**
    *   Garantir a integridade da arquitetura modular do projeto.
    *   Garantir que a classe do ambiente matricial (`envs/`) se comunique perfeitamente com a IA (`agents/`) e que ambos possam ser importados sem atrito diretamente no frontend.
    *   Organizar a estrutura de diretórios, gerenciar as dependências (`requirements.txt`) e realizar refatorações no código.
*   **Responsável:**
    *   👤 *Kawan*

---

### 4. 🖥️ Frontend e Demonstração (Individual)
*   **Foco e Responsabilidades:**
    *   Criar o dashboard web interativo em Streamlit (`frontend/app.py`).
    *   Permitir que o usuário defina o tamanho da matriz, configure a origem, o destino e adicione/remova obstáculos de forma dinâmica e amigável.
    *   Importar o ambiente e o agente PPO, e acionar o treinamento sob demanda, exibindo a barra de progresso, o gráfico de recompensa em tempo real e a rota resultante passo a passo.
    *   Permitir que o usuario possa mudar os hiperparâmetros do modelo, assim fazendo ele testar o projeto por completo
*   **Bibliotecas Utilizadas:** `streamlit`, `matplotlib` (para plotagem da grade).
*   **Responsáveis:**
    *   👤 *Aurea*
=======
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
    *   👤 *Daniel*
    *   👤 *Perrone*

---

### 2. 🧠 Treinamento e IA (Pair Programming)
*   **Foco e Responsabilidades:**
    *   Configurar o algoritmo **PPO (Proximal Policy Optimization)**, ideal para lidar com a variação dinâmica de tamanhos de matrizes e obstáculos configurados via frontend.
    *   Ajustar os hiperparâmetros do PPO (taxa de aprendizado, n_steps, batch_size e ent_coef para exploração/entropia) para garantir que a IA aprenda rapidamente (em menos de 10-15 segundos) no frontend.
    *   Criar a classe de Callback personalizada (`BaseCallback`) para enviar as métricas de treino em tempo real para o painel do Streamlit.
    *   Garantir o salvamento dos pesos do modelo treinado em formato `.zip`.
*   **Bibliotecas Utilizadas:** `stable-baselines3`
*   **Responsáveis:**
    *   👤 *Spinola*
    *   👤 *Adaime*

---

### 3. 🏗️ Arquitetura e Integração (Individual)
*   **Foco e Responsabilidades:**
    *   Garantir a integridade da arquitetura modular do projeto.
    *   Garantir que a classe do ambiente matricial (`envs/`) se comunique perfeitamente com a IA (`agents/`) e que ambos possam ser importados sem atrito diretamente no frontend.
    *   Organizar a estrutura de diretórios, gerenciar as dependências (`requirements.txt`) e realizar refatorações no código.
*   **Responsável:**
    *   👤 *Kawan*

---

### 4. 🖥️ Frontend e Demonstração (Individual)
*   **Foco e Responsabilidades:**
    *   Criar o dashboard web interativo em Streamlit (`frontend/app.py`).
    *   Permitir que o usuário defina o tamanho da matriz, configure a origem, o destino e adicione/remova obstáculos de forma dinâmica e amigável.
    *   Importar o ambiente e o agente PPO, e acionar o treinamento sob demanda, exibindo a barra de progresso, o gráfico de recompensa em tempo real e a rota resultante passo a passo.
    *   Permitir que o usuario possa mudar os hiperparâmetros do modelo, assim fazendo ele testar o projeto por completo
*   **Bibliotecas Utilizadas:** `streamlit`, `matplotlib` (para plotagem da grade).
*   **Responsáveis:**
    *   👤 *Aurea*
>>>>>>> a814cfcc1169d475372cedb01931d2acc50ee120
