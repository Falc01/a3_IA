# Projeto de Busca de Rotas com Aprendizado por Reforço (RL)

## 📌 Propósito do Projeto
Este projeto é uma Prova de Conceito (PoC) desenvolvida para validar o uso de Aprendizado por Reforço (Reinforcement Learning - RL), especificamente o algoritmo **PPO (Proximal Policy Optimization)**, para encontrar caminhos e rotas otimizadas entre dois pontos (A e B) em um ambiente de grade/matriz bidimensional customizável.

## 🚀 Abordagem: Matriz Dinâmica e PPO
Para manter o MVP leve, altamente responsivo e de fácil validação, adotamos a abordagem de grade matricial pura via NumPy:
1.  **PPO (Proximal Policy Optimization):** Escolhido por sua extrema estabilidade e robustez a mudanças no tamanho da matriz e obstáculos configurados dinamicamente. O PPO utiliza regularização de entropia (uma métrica de "curiosidade") que evita que a IA fique presa em loops ou vicie em um único caminho, permitindo redescobrir trajetos alternativos de forma rápida.
2.  **Customização e Integração no Frontend:** O usuário define dinamicamente via Streamlit o tamanho da matriz, obstáculos e posições A/B. Para maior performance e monitoramento em tempo real, o ambiente e o agente de RL são importados e executados diretamente dentro da interface gráfica (Streamlit), utilizando Callbacks (`BaseCallback`) para plotar o progresso do aprendizado do agente em tempo real.

---

## 🏗️ Arquitetura do Projeto

Abaixo está a descrição da estrutura de pastas e seus propósitos:

### 1. `envs/`
*   **Propósito:** Definição do "mundo virtual" da grade.
*   **Função:** Contém a classe que herda de `gymnasium.Env` (ex: `rota_env.py`), definindo a matriz de estados, as ações discretas de movimento (Cima, Baixo, Esquerda, Direita) e a lógica de recompensas (penalidade por passo/colisão, recompensa por atingir o destino).
*   **Bibliotecas:** `gymnasium`, `numpy`.

### 2. `agents/`
*   **Propósito:** Configuração do cérebro da IA.
*   **Função:** Scripts para inicializar e configurar o algoritmo PPO e seus hiperparâmetros (taxa de aprendizado, entropia, batch size).
*   **Bibliotecas:** `stable-baselines3`.

### 3. `config/`
*   **Propósito:** Centralização das configurações do cenário.
*   **Função:** Armazenar temporariamente a matriz desenhada pelo usuário (em formato JSON) com as dimensões e obstáculos para replicação e fins de logging.

### 4. `scripts/`
*   **Propósito:** Scripts utilitários de CLI.
*   **Função:**
    *   `train.py`: Script alternativo em linha de comando para rodar o treino isolado do PPO.
    *   `evaluate.py`: Script CLI para testar e validar o carregamento de modelos salvos.

### 5. `frontend/`
*   **Propósito:** Interface visual do usuário.
*   **Função:** Dashboard em Streamlit (`app.py`) onde o usuário define a matriz, desenha os obstáculos, clica para treinar o agente em tempo real e visualiza a rota calculada e a curva de aprendizado (recompensa) subindo dinamicamente.
*   **Bibliotecas:** `streamlit`.

### 6. `utils/`
*   **Propósito:** Scripts auxiliares.
*   **Função:** Funções utilitárias de suporte para renderização dos gráficos do mapa e do trajeto.
*   **Bibliotecas:** `matplotlib` ou `plotly`.

### 7. `data/`
*   **Propósito:** Armazenamento persistente de arquivos do ciclo de vida da IA.
*   **Função:** Salvar os logs do TensorBoard (`logs/`) e os pesos do modelo treinado (`models/agente_ppo.zip`).

---

## 📚 Documentação Técnica

*   **Modelagem de IA e Arquitetura:** Para entender o ambiente Gymnasium, a lógica de recompensas e a configuração detalhada do PPO, leia o guia de [Arquitetura e Aprendizado por Reforço](./docs/architecture_and_rl.md).
*   **Alocação de Tarefas:** Para ver qual integrante da equipe é responsável por cada parte do código, consulte o [Guia de Contribuição](./CONTRIBUTING.md) na raiz do projeto.

---

## 🛠️ Como Configurar e Executar (Passo a Passo)

Para rodar este projeto em sua máquina local, siga as instruções abaixo no terminal (na raiz do projeto):

### 1. Criar o Ambiente Virtual (venv)
```powershell
python -m venv venv
```

### 2. Ativar o Ambiente Virtual
*   **No Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\activate
    ```
*   **No Linux/macOS:**
    ```bash
    source venv/bin/activate
    ```

*(Você verá um prefixo `(venv)` no seu terminal indicando que o ambiente está ativo).*

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar o Frontend (Streamlit)
```bash
streamlit run frontend/app.py
```

