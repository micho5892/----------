# ==========================================================
# train_agent.py
# 強化学習(PPO)を用いてヒートシンク形状を最適化する学習スクリプト
# ==========================================================
import os
import argparse
import time

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from .rl_run_optimize_config import RLParameterOptimizer
from .lbm_heatsink_env import LBMHeatSinkEnv

# python -m 強化学習モデル.train_agent

# ==========================================================
# カスタム 3D-CNN フィーチャーエクストラクタ
# ==========================================================
class Custom3DCNN(BaseFeaturesExtractor):
    """
    (5, 64, 64, 128) のような巨大な3D観測テンソルを入力として受け取り、
    空間的な特徴（風の通り道、熱の滞留など）を抽出して圧縮するネットワーク。
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # 3D Convolutionで次元を徐々に落とす
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # ダミーデータを通してFlatten後の次元数を自動計算
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        # 抽出した特徴を指定の次元(features_dim)のベクトルに変換
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# ==========================================================
# メイン学習ループ
# ==========================================================
def train_agent(
    nx=64, ny=64, nz=128, 
    U_inlet_p=0.1, 
    total_timesteps=2000, 
    model_save_path="models/ppo_heatsink"
):
    print("=== Starting RL Training (PPO) ===")
    
    # 1. 保存用ディレクトリの作成
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    tensorboard_log = "logs/heatsink_tb/"
    os.makedirs(tensorboard_log, exist_ok=True)

    # 2. シミュレーション環境の準備（ランダムテストと同じ手順）
    optimizer = RLParameterOptimizer(nx=nx, U_inlet_p=U_inlet_p, L_domain=0.1, fluid="Air")
    success = optimizer.run(verbose=False)
    if not success:
        print("⚠️ 最適化に失敗しましたが、ベストエフォートで続行します。")

    overrides = optimizer.get_sim_overrides()

    print("\nInitializing Environment...")
    env = LBMHeatSinkEnv(
        mode="plan_a",
        nx=nx, ny=ny, nz=nz,
        sim_config_overrides=overrides,
        warmup_time_sum_scale=0.1,
    )

    # 3. PPOモデルの構築
    # policy_kwargs で上で定義したカスタム3D-CNNを指定する
    policy_kwargs = dict(
        features_extractor_class=Custom3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    print("\nInitializing PPO Model (Custom 3D-CNN)...")
    # PPOのハイパーパラメータ（最初は学習しやすい設定にしています）
    model = PPO(
        "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=100,         # 1エピソード(100ステップ)ごとに学習を更新
        batch_size=25,       # バッチサイズ
        gamma=0.99,          # 割引率
        verbose=1,
        tensorboard_log=tensorboard_log,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # チェックポイントコールバック（500ステップごとにモデルをバックアップ保存）
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path="models/checkpoints/",
        name_prefix="ppo_heatsink"
    )

    # 4. 学習の開始
    print(f"\nTraining for {total_timesteps} timesteps...")
    start_time = time.time()
    
    # 学習実行！
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.2f} seconds.")

    # 5. モデルの保存
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")

    # 6. 最後に、学習後の賢いAIで1回テストプレイをしてVTIを保存
    print("\nRunning a test episode with the trained model...")
    obs, _ = env.reset()
    for i in range(50): # 50回削らせてみる
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    env.runner.save_state_vti("rl_state_trained_result.vti")
    print("Test episode complete. Saved 'rl_state_trained_result.vti'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--U-inlet-p", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000, help="学習する合計ステップ数")
    args = parser.parse_args()

    train_agent(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        U_inlet_p=args.U_inlet_p,
        total_timesteps=args.timesteps
    )