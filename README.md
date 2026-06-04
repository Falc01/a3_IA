# Projeto de Busca de Rotas com Aprendizado por Reforço (RL)

## 📌 Propósito do Projeto
Este projeto é uma Prova de Conceito (PoC) desenvolvida para demonstrar o uso de Aprendizado por Reforço (Reinforcement Learning - RL) em ambientes discretos. Implementamos o algoritmo clássico **SARSA Tabular (On-Policy)** para encontrar caminhos e rotas otimizadas entre dois pontos (A e B) em um ambiente de grade/matriz bidimensional customizável.

## 🚀 Abordagem: Matriz Dinâmica e SARSA Tabular
Para manter o projeto didático, leve, altamente responsivo e de fácil validação, adotamos a abordagem de grade matricial pura via NumPy e Gymnasium:
1.  **SARSA Tabular:** Implementado puramente do zero em NumPy, livre de dependências de redes neurais pesadas como o PyTorch (Stable Baselines3). O algoritmo armazena uma matriz `Q` (Q-Table) que aprende a associar cada célula discreta do grid (estado) à melhor ação de movimentação (Cima, Direita, Baixo, Esquerda).
2.  **Customização e Integração no Frontend:** O usuário define dinamicamente via Streamlit o tamanho da matriz, obstáculos e posições A/B. Para maior performance e monitoramento em tempo real, o ambiente e o agente de RL são importados e executados diretamente dentro da interface gráfica (Streamlit), utilizando Callbacks para plotar o progresso do aprendizado do agente (recompensas e passos por episódio) em tempo real, além de expor a própria tabela Q final.

---

## 🏗️ Arquitetura do Projeto

Abaixo está a descrição da estrutura de pastas e seus propósitos:

### 1. `envs/`
*   **Propósito:** Definição do "mundo virtual" da grade.
*   **Função:** Contém a classe que herda de `gymnasium.Env` (`rota_env.py`), definindo a matriz de estados discretos ($S = x \times H + y$), as ações discretas de movimento e a lógica de recompensas (custo de passo, colisão com obstáculos e recompensa de objetivo).
*   **Bibliotecas:** `gymnasium`, `numpy`.

### 2. `agents/`
*   **Propósito:** Configuração do cérebro da IA.
*   **Função:** Contém o arquivo `gerenciador_agente.py` que abriga a lógica do algoritmo SARSA Tabular (tabela Q, seleção $\epsilon$-greedy e decaimento temporal, treinamento, inferência, e exportação/carregamento dos pesos em arquivos `.npy`).

### 3. `frontend/`
*   **Propósito:** Interface visual do usuário.
*   **Função:** Dashboard web moderno em Streamlit (`app.py` e `style.css`) onde o usuário define a matriz, desenha obstáculos, configura hiperparâmetros (Alpha, Gamma, Epsilon), inicia o treinamento em tempo real, analisa as curvas de aprendizado, visualiza a Q-table e executa animações da rota resultante.
*   **Bibliotecas:** `streamlit`, `plotly` (para plotagem interativa da grade e gráficos).

### 4. `utils/`
*   **Propósito:** Scripts auxiliares.
*   **Função:**
    *   `map_generator.py`: Lógica para gerar novos padrões de obstáculos (Aleatório, Paredes Alternadas, Barreira Central) e verificar solvabilidade via busca em largura (BFS).
    *   `plot_utils.py`: Funções para renderizar a grade do mapa via Plotly e construir os gráficos dinâmicos de curva de aprendizado (recompensa e passos por episódio).
    *   `callbacks.py`: Gerenciamento e coleta do progresso das rodadas de treinamento para exibição síncrona na tela.

### 5. `tests/`
*   **Propósito:** Garantia de qualidade e testes automatizados.
*   **Função:** Suítes de testes unitários para validar a consistência do ambiente de rotas e geradores de mapa.

---

## 📚 Documentação Técnica

*   **Modelagem de IA e Arquitetura:** Para entender o mapeamento de estados discretos, a lógica de recompensas e a equação de atualização clássica do SARSA, leia o guia de [Arquitetura e Aprendizado por Reforço](./docs/architecture_and_rl.md).
*   **Alocação de Tarefas:** Para ver qual integrante da equipe é responsável por cada parte do código, consulte o [Guia de Contribuição](./CONTRIBUTING.md) na raiz do projeto.

---

## 🛠️ Como Configurar e Executar (Passo a Passo)

Para rodar este projeto em sua máquina local, siga as instruções abaixo no terminal (na raiz do projeto):

### 1. Criar o Ambiente Virtual (venv)
```powershell
python -m venv .venv
```

### 2. Ativar o Ambiente Virtual
*   **No Windows (PowerShell):**
    ```powershell
    .\.venv\Scripts\activate
    ```
*   **No Linux/macOS:**
    ```bash
    source .venv/bin/activate
    ```

*(Você verá um prefixo `(.venv)` no seu terminal indicando que o ambiente está ativo).*

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar o Frontend (Streamlit)
```bash
streamlit run frontend/app.py
```
