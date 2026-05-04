# ==========================================================
# random_agent_test.py
# ==========================================================
import argparse
import sys
import time

import numpy as np

from .lbm_heatsink_env import LBMHeatSinkEnv
from .rl_run_optimize_config import RLParameterOptimizer

def run_random_agent_test(
    nx: int = 128,
    ny: int = 128,
    nz: int = 256,
    U_inlet_p: float = 0.05,
    warmup_time_sum_scale: float = 0.1,
):
    print("=== Starting Random Agent Test (Plan A: Sphere Subtraction) ===")

    # 1. パラメータ最適化エンジン（部品）の初期化と実行
    # 今後、流体を "Water" にしたり、Reの範囲を変えたりする場合はここで kwargs で渡すだけです
    optimizer = RLParameterOptimizer(
        nx=nx,
        U_inlet_p=U_inlet_p,
        L_domain=0.1,
        fluid="Air",
    )
    
    print("run_optimize を実行中（reset 前に入口 u_lbm 等を決定）…")
    success = optimizer.run(verbose=True)
    if not success:
        print("警告: 最適化に失敗しました。パラメータ範囲などを確認してください。")
        # 失敗しても続行することはできますが、物理的に無理な状態になる可能性があります

    # 最適化結果からシミュレータ用のオーバーライド辞書を取得
    overrides = optimizer.get_sim_overrides()

    # 2. 強化学習環境の構築
    env = LBMHeatSinkEnv(
        mode="plan_a",
        nx=nx,
        ny=ny,
        nz=nz,
        sim_config_overrides=overrides,
        warmup_time_sum_scale=warmup_time_sum_scale,
    )

    # 3. 環境のリセット
    # ここで Runner.compute_warmup_steps_from_characteristic_times が走り、
    # 物理法則に基づいた最適なウォームアップステップ数が自動算出され実行されます。
    print("\nResetting environment... (ウォームアップは基準時間から自動算出)")
    start_time = time.time()
    obs, info = env.reset()
    print(f"Reset completed in {time.time() - start_time:.2f} seconds.")
    print(f"Initial Observation Shape (C, X, Y, Z): {obs.shape}")
    
    meta = info.get("reset_warmup")
    if meta is not None:
        print(
            f"ウォームアップ: steps={meta['warmup_steps']}, "
            f"scale={meta['time_sum_scale']}, "
            f"t_adv={meta['t_adv_s']}, t_th={meta['t_th_s']}, dt={meta['dt']}"
        )

    # 4. ランダムアクションによるステップ実行テスト
    num_steps = 20

    for step in range(num_steps):
        if step == 0:
            env.runner.save_state_vti("rl_state_initial.vti")
            
        print(f"\n--- Step {step + 1}/{num_steps} ---")

        action = env.action_space.sample()

        print("Sampled Action (Normalized):")
        print(
            f"  X: {action[0]:.3f}, Y: {action[1]:.3f}, "
            f"Z: {action[2]:.3f}, Radius: {action[3]:.3f}"
        )

        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start

        print(f"Step computation time: {step_time:.3f} s")
        print(f"Reward: {reward:.5f}")
        print(f"Info: {info}")

        if np.isnan(obs).any():
            print("❌ ERROR: Observation contains NaN. LBM simulation diverged (exploded).")
            break

        if terminated or truncated:
            print("Episode finished early (Terminated/Truncated).")
            break

    # テスト終了後に最終状態を VTI として保存
    env.runner.save_state_vti("rl_state_final.vti")
    print("\n=== Random Agent Test Completed ===")


def _parse_args():
    p = argparse.ArgumentParser(description="RL ヒートシンク環境のランダム行動テスト")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=128)
    p.add_argument("--U-inlet-p", type=float, default=0.1, help="物理的な入口流速 [m/s]")
    p.add_argument(
        "--warmup-time-sum-scale",
        type=float,
        default=3.0,
        help="(t_adv + t_th) に掛ける倍率（既定 3）",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_random_agent_test(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        U_inlet_p=args.U_inlet_p,
        warmup_time_sum_scale=args.warmup_time_sum_scale,
    )