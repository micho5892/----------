import numpy as np
import time

# python -m 強化学習モデル.random_agent_test

# 作成したGymnasium環境をインポート（ファイル名を lbm_heatsink_env.py とした場合）
from .lbm_heatsink_env import LBMHeatSinkEnv

def run_random_agent_test():
    print("=== Starting Random Agent Test (Plan A: Sphere Subtraction) ===")
    
    # 1. 環境のインスタンス化
    # ※内部でTaichiの初期化と、シミュレータ(LBMSimulator)の準備が行われます
    env = LBMHeatSinkEnv(mode="plan_a")
    
    # 2. 環境のリセット
    print("Resetting environment... (Running initial LBM warm-up)")
    start_time = time.time()
    obs, info = env.reset()
    print(f"Reset completed in {time.time() - start_time:.2f} seconds.")
    print(f"Initial Observation Shape (C, X, Y, Z): {obs.shape}")
    
    # テストするステップ数（何回連続で削るか）
    # ※発散しやすいLBMで、20回も連続で形を変えてエラーなく走れば、システムはかなり安定しています
    num_steps = 20
    
    for step in range(num_steps):
        if step == 0:
            env.runner.save_state_vti("rl_state_initial.vti")
            continue
        print(f"\n--- Step {step + 1}/{num_steps} ---")
        
        # 3. アクションのサンプリング
        # action_space.sample() は [-1.0, 1.0] の範囲で4次元(X, Y, Z, R)のランダムな値を生成します
        action = env.action_space.sample()
        
        print(f"Sampled Action (Normalized):")
        print(f"  X: {action[0]:.3f}, Y: {action[1]:.3f}, Z: {action[2]:.3f}, Radius: {action[3]:.3f}")
        
        # 4. 環境を1ステップ進める（削る ＆ ウォームスタートでシミュレーション）
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        
        # 5. 結果の確認
        print(f"Step computation time: {step_time:.3f} s")
        print(f"Reward: {reward:.5f}")
        print(f"Info: {info}")
        
        # NaN（計算の発散）のチェック
        if np.isnan(obs).any():
            print("❌ ERROR: Observation contains NaN. LBM simulation diverged (exploded).")
            break
            
        if terminated or truncated:
            print("Episode finished early (Terminated/Truncated).")
            break

    # 6. ランダムに削った結果と、風の流れをすべてVTKに保存
    env.runner.save_state_vti("rl_state.vti")

    print("\n=== Random Agent Test Completed ===")

if __name__ == "__main__":
    run_random_agent_test()