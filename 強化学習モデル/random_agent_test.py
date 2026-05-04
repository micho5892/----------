import argparse
import sys
import time
from pathlib import Path

import numpy as np

# python -m 強化学習モデル.random_agent_test

_LBM_UI = Path(__file__).resolve().parents[1] / "lbm_ui_designer"
if str(_LBM_UI) not in sys.path:
    sys.path.insert(0, str(_LBM_UI))

from lbm_param_optimize import run_optimize  # type: ignore[import-not-found]

from .lbm_heatsink_env import LBMHeatSinkEnv
from .rl_run_optimize_config import build_rl_heatsink_optimize_config


def primary_to_sim_overrides(primary: dict) -> dict:
    """run_optimize の primary を SimConfig 用に変換（入口 LBM 速度・代表物理流速・流体物性）。"""
    u_lbm = float(primary["u_lbm"])
    nu = float(primary["nu"])
    kf = float(primary["k_f"])
    rho_f = float(primary["rho_f"])
    cp_f = float(primary["Cp_f"])
    L_dom = float(primary["L_domain"])
    U_p = float(primary["U"])
    fluid_dp = {"nu": nu, "k": kf, "rho": rho_f, "Cp": cp_f}
    return {
        "Lx_p": L_dom,
        "U_inlet_p": U_p,
        "boundary_conditions": {
            20: {
                "type": "inlet",
                "velocity": [0.0, 0.0, -u_lbm],
                "temperature": 0.0,
            },
        },
        "domain_properties": {
            0: fluid_dp,
            20: fluid_dp,
            21: fluid_dp,
        },
    }


def run_random_agent_test(
    nx: int = 64,
    ny: int = 64,
    nz: int = 128,
    warmup_time_sum_scale: float = 3.0,
):
    print("=== Starting Random Agent Test (Plan A: Sphere Subtraction) ===")

    opt_cfg = build_rl_heatsink_optimize_config(nx, L_domain=0.1)
    print("run_optimize を実行中（reset 前に入口 u_lbm 等を決定）…")
    opt_out = run_optimize(opt_cfg)
    if not opt_out.get("success"):
        print(f"警告: run_optimize success=False — {opt_out.get('message')}")
    primary = opt_out["primary"]
    print(
        f"  最適化後 primary: u_lbm={primary['u_lbm']:.6f}, "
        f"U={primary['U']:.6f}, L_ref={primary['L_ref']:.6f} m"
    )

    overrides = primary_to_sim_overrides(primary)
    env = LBMHeatSinkEnv(
        mode="plan_a",
        nx=nx,
        ny=ny,
        nz=nz,
        sim_config_overrides=overrides,
        warmup_time_sum_scale=warmup_time_sum_scale,
    )

    print("Resetting environment... (ウォームアップは基準時間から自動算出)")
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

    num_steps = 20

    for step in range(num_steps):
        if step == 0:
            env.runner.save_state_vti("rl_state_initial.vti")
            continue
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

    env.runner.save_state_vti("rl_state.vti")

    print("\n=== Random Agent Test Completed ===")


def _parse_args():
    p = argparse.ArgumentParser(description="RL ヒートシンク環境のランダム行動テスト")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=128)
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
        warmup_time_sum_scale=args.warmup_time_sum_scale,
    )
