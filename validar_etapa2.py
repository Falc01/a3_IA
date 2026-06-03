import gymnasium as gym
from agents.ppo_agent import inicializar_e_configurar_agente

print("🤖 Iniciando teste de fumaça da Etapa 2...")

try:
    # 1. Criamos um ambiente padrão do Gymnasium para testes rápidos
    env_teste = gym.make("Pendulum-v1") 
    
    # 2. Inicializamos o agente usando a nossa fábrica de IA
    gerenciador = inicializar_e_configurar_agente(env_teste)
    print("✅ Sucesso: O Agente PPO foi configurado e instanciado perfeitamente!")
    
    # 3. Testamos se o loop de treinamento síncrono responde (1 passo para ser instantâneo)
    gerenciador.treinar(total_timesteps=1)
    print("✅ Sucesso: O loop de treinamento está ativo!")
    
    # 4. Testamos a persistência do modelo
    gerenciador.salvar_modelo(nome_arquivo="teste_checkpoint")
    print("✅ Sucesso: Sistema de salvamento de arquivos operacional!")
    
    print("\n🎉 ETAPA 2 CONCLUÍDA COM SUCESSO! O cérebro da IA está pronto para o Frontend.")

except Exception as e:
    print(f"\n❌ Erro encontrado na Etapa 2: {e}")