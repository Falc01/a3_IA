import sys
from pathlib import Path

# === CONFIGURAÇÃO DO CAMINHO (MUITO IMPORTANTE) ===
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

print(f"🔍 Caminho raiz adicionado: {root_dir}")

# Agora importa
from envs.rota_env import RotaEnv, RotaConfig


def test_rota_env():
    print("\n🧪 Iniciando testes do RotaEnv...\n")
    
    # ==================== OBSTÁCULOS ====================
    obstacles = [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [6, 3],
        [3, 6]
    ]
    
    config = RotaConfig(obstacles=obstacles)
    env = RotaEnv(config=config)        # ← Solução 1 aplicada
    # ===================================================
    
    obs, info = env.reset()
    
    print(f"Start: {info.get('start', 'N/A')} | Goal: {info['goal']}")
    print(f"Obstáculos: {len(obstacles)} adicionados\n")
    
    total_reward = 0.0
    steps = 0
    
    # Caminho exemplo (pode precisar ser ajustado por causa dos obstáculos)
    actions = [3] * 7 + [1] * 7  
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        env.render()
        
        print(f"Step {steps:2d} | Ação: {action} | Recompensa: {reward:6.1f} | Total: {total_reward:6.1f}")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Teste Finalizado!")
    print(f"   Passos: {steps} | Recompensa Total: {total_reward:.1f}")
    print(f"   Chegou no objetivo: {terminated}\n")
    
    print("🎉 Teste concluído com sucesso!")


if __name__ == "__main__":
    test_rota_env()
import sys
from pathlib import Path

# === CONFIGURAÇÃO DO CAMINHO (MUITO IMPORTANTE) ===
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

print(f"🔍 Caminho raiz adicionado: {root_dir}")

# Agora importa
from envs.rota_env import RotaEnv, RotaConfig


def test_rota_env():
    print("\n🧪 Iniciando testes do RotaEnv...\n")
    
    # ==================== OBSTÁCULOS ====================
    obstacles = [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [6, 3],
        [3, 6]
    ]
    
    config = RotaConfig(obstacles=obstacles)
    env = RotaEnv(config=config)        # ← Solução 1 aplicada
    # ===================================================
    
    obs, info = env.reset()
    
    print(f"Start: {info.get('start', 'N/A')} | Goal: {info['goal']}")
    print(f"Obstáculos: {len(obstacles)} adicionados\n")
    
    total_reward = 0.0
    steps = 0
    
    # Caminho exemplo (pode precisar ser ajustado por causa dos obstáculos)
    actions = [3] * 7 + [1] * 7  
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        env.render()
        
        print(f"Step {steps:2d} | Ação: {action} | Recompensa: {reward:6.1f} | Total: {total_reward:6.1f}")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Teste Finalizado!")
    print(f"   Passos: {steps} | Recompensa Total: {total_reward:.1f}")
    print(f"   Chegou no objetivo: {terminated}\n")
    
    print("🎉 Teste concluído com sucesso!")


if __name__ == "__main__":
    test_rota_env()