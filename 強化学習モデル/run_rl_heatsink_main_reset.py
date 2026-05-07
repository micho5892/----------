"""
LBMHeatSinkEnv.reset() と同じ格子・物性・境界・ウォームアップステップ数を、
refactoredv2.7/main.py の run_simulation() だけで実行する（形状は geometry.build_rl_heatsink）。

例（プロジェクトルートで）:
  conda run -n lbm-sim python -m 強化学習モデル.run_rl_heatsink_main_reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REF = _PROJECT_ROOT / "refactoredv2.7"


def rl_heatsink_sim_config_kwargs(nx: int, ny: int, nz: int) -> dict:
    """HeatSinkSimulationRunner / LBMHeatSinkEnv と同じ SimConfig 引数（benchmark_name は run_simulation が設定）。"""
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "boundary_conditions": {
            20: {
                "type": "inlet",
                "velocity": [0.0, 0.0, -0.05],
                "temperature": 0.0,
            },
            21: {"type": "outlet"},
            10: {"type": "isothermal_wall", "temperature": 1.0},
        },
        "domain_properties": {
            0: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
            20: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
            21: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
            10: {"nu": 0.0, "k": 400.0, "rho": 8960.0, "Cp": 385.0},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_simulation で RL ヒートシンクの reset 相当（ウォームアップのみ）を実行"
    )
    base_size = 128
    nx = ny = 1
    nz = 2
    nx *= base_size
    ny *= base_size
    nz *= base_size
    parser.add_argument("--nx", type=int, default=nx)
    parser.add_argument("--ny", type=int, default=ny)
    parser.add_argument("--nz", type=int, default=nz)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10000,
        help="LBMHeatSinkEnv.reset の warmup_steps と同じ LBM ステップ数",
    )
    parser.add_argument(
        "--artifact-parent",
        type=str,
        default="",
        help="空なら <リポジトリ>/results を使用",
    )
    parser.add_argument("--arch", type=str, default="gpu", choices=("gpu", "cpu"))
    args = parser.parse_args()

    if str(_REF) not in sys.path:
        sys.path.insert(0, str(_REF))

    from config import SimConfig
    from main import run_simulation

    cfg_kw = rl_heatsink_sim_config_kwargs(args.nx, args.ny, args.nz)
    cfg_pre = SimConfig(**cfg_kw)
    max_time_p = float(args.warmup_steps) * float(cfg_pre.dt)

    artifact_parent = args.artifact_parent.strip()
    if not artifact_parent:
        artifact_parent = str(_PROJECT_ROOT / "results")

    paths_out: dict = {}
    run_simulation(
        arch=args.arch,
        benchmark="rl_heatsink",
        max_time_p=max_time_p,
        steady_detection=False,
        visualization_mode="none",
        ramp_time_p=1.0,
        artifact_parent=artifact_parent,
        paths_out=paths_out,
        U_inlet_p=0.05,
        **cfg_kw,
    )

    print("--- run_simulation (reset 相当) 完了 ---")
    print("out_dir:", paths_out.get("out_dir"))
    print("fields npz:", paths_out.get("npz_path"))
    print("log:", paths_out.get("log_path"))


if __name__ == "__main__":
    main()
