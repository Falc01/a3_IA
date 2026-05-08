# Projeto de Busca de Rotas com Aprendizado por Reforço (RL)

## Propósito do Projeto
Este projeto é uma Prova de Conceito (PoC) desenvolvida para validar a utilização de agentes de Inteligência Artificial, especificamente Aprendizado por Reforço (Reinforcement Learning - RL), para encontrar rotas otimizadas entre dois pontos (A e B) em um ambiente de grade (grid).

## Arquitetura do Projeto

Abaixo está a descrição da estrutura de pastas, seus propósitos, funções e as bibliotecas principais que serão utilizadas em cada uma:

### 1. `data/`
*   **Propósito:** Armazenamento de arquivos persistentes gerados durante o ciclo de vida da IA.
*   **Função:** 
    *   `models/`: Armazena os pesos dos modelos treinados (arquivos `.zip`).
    *   `logs/`: Armazena métricas de treinamento para visualização (ex: TensorBoard).
*   **Bibliotecas:** Nenhuma específica para armazenamento, mas gerenciada pelo `stable-baselines3`.

### 2. `envs/`
*   **Propósito:** Definição do mundo virtual onde o agente atua.
*   **Função:** Contém a classe que herda de `gymnasium.Env`, definindo o mapa, os estados, as ações possíveis e o sistema de recompensas.
*   **Bibliotecas:** `gymnasium`, `numpy`.

### 3. `agents/`
*   **Propósito:** Configuração e inicialização do "cérebro" da IA.
*   **Função:** Scripts para configurar hiperparâmetros e instanciar algoritmos de RL (como PPO ou DQN).
*   **Bibliotecas:** `stable-baselines3`.

### 4. `config/`
*   **Propósito:** Centralização de parâmetros globais.
*   **Função:** Armazenar arquivos de configuração em formato **JSON** que definem o tamanho do mapa, número de episódios de treino, etc.
*   **Bibliotecas:** `json` (nativa do Python).

### 5. `scripts/`
*   **Propósito:** Execução de tarefas práticas.
*   **Função:** 
    *   `train.py`: Script principal para iniciar o treinamento.
    *   `evaluate.py`: Script para carregar um modelo e testar sua performance em cenários reais.
*   **Bibliotecas:** `stable-baselines3`, `gymnasium`.

### 6. `frontend/`
*   **Propósito:** Interface de visualização para o usuário final.
*   **Função:** Criar um dashboard web onde o usuário pode ver o agente se movendo no mapa em tempo real ou após o treino.
*   **Bibliotecas:** `streamlit`.

### 7. `utils/`
*   **Propósito:** Funções de suporte reutilizáveis.
*   **Função:** Cálculos matemáticos, manipulação de strings, helpers de plotagem e outras utilidades.
*   **Bibliotecas:** `numpy`, `matplotlib` (opcional).

---

## Como Executar
*(Instruções futuras serão adicionadas aqui após a implementação dos scripts)*
