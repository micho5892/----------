"""
IBM 平板上下壁 + Z 方向入口/出口の検証用ランチャー。

実行例（プロジェクトルートから）:
  conda run -n lbm-sim python refactoredv2.7/run_benchmark_ibm_y_wall_channel.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from main import run_simulation
from lbm_logger import configure_logging, get_logger

_log = get_logger(__name__)

WALL_LBM = 10


def run_ibm_y_wall_channel_short():
    nx, ny, nz = 64, 48, 32
    Lx_p = 0.1
    u_lbm = 0.08

    run_simulation(
        benchmark="ibm_channel_y_walls",
        fp_dtype="float32",
        steady_detection=False,
        nx=nx,
        ny=ny,
        nz=nz,
        Lx_p=Lx_p,
        U_inlet_p=1.0,
        max_time_p=5.0,
        ramp_time_p=0.5,
        vis_interval=50,
        vti_export_interval=0,
        particles_inject_per_step=0,
        data_export_interval=0,
        domain_properties={
            0: {"nu": 0.02, "k": 0.0, "rho": 1.0, "Cp": 1.0},
            20: {"nu": 0.02, "k": 0.0, "rho": 1.0, "Cp": 1.0},
            21: {"nu": 0.02, "k": 0.0, "rho": 1.0, "Cp": 1.0},
            10: {"nu": 0.0, "k": 0.0, "rho": 1.0, "Cp": 1.0},
        },
        boundary_conditions={
            20: {"type": "inlet", "velocity": [0.0, 0.0, -u_lbm], "temperature": 0.0},
            21: {"type": "outlet"},
        },
        physics_models={
            "immersed_boundary": {
                "objects": [
                    {"shape": "y_plate", "side": "lower", "wall_thickness_lbm": WALL_LBM, "type": "fixed"},
                    {"shape": "y_plate", "side": "upper", "wall_thickness_lbm": WALL_LBM, "type": "fixed"},
                ]
            }
        },
    )


if __name__ == "__main__":
    configure_logging(os.path.join(_THIS_DIR, "ibm_y_wall_channel_run.log"))
    _log.info("Starting IBM y-wall channel validation run")
    run_ibm_y_wall_channel_short()
