"""LBMHeatSink + lbm_param_optimize.run_optimize 用の既定 YAML 相当 dict を生成する。"""

from __future__ import annotations


def build_rl_heatsink_optimize_config(
    nx: int,
    *,
    L_domain: float = 0.1,
    fluid: str = "Air",
    solid: str = "Copper",
    temperature_K: float = 300.0,
    pressure_Pa: float = 101_325.0,
    fix_cr: bool = True,
    regularization: float = 1e-5,
    maxiter: int = 30000,
) -> dict:
    """
    代表長さ L_ref: 初期ヒートシンクブロックの x 方向辺長（高さではない）。
    margin_x = nx//8 のとき (nx - 2*margin_x)/nx * L_domain = 0.75 * L_domain（nx が 8 の倍数のとき）。
    """
    margin = nx // 8
    span_cells = max(1, nx - 2 * margin)
    L_ref = float(span_cells) / float(nx) * float(L_domain)

    return {
        "fluid": fluid,
        "temperature_K": temperature_K,
        "pressure_Pa": pressure_Pa,
        "solid": solid,
        "fix_cr": fix_cr,
        "fixed": {
            "nx": float(nx),
            "L_domain": float(L_domain),
            "L_ref": L_ref,
            "nu": True,
            "k_f": True,
            "rho_f": True,
            "Cp_f": True,
            "k_s": True,
            "rho_s": True,
            "Cp_s": True,
            "beta_f": True,
            # SimConfig.U_inlet_p（代表物理流速 [m/s]）と揃える
            "U": 0.1,
        },
        "ranges": {
            "Re": {"min": 200.0, "max": 800.0},
            "tau_f マージン": {"min": 0.04, "max": 2.0},
            "tau_gf マージン": {"min": 0.04, "max": 2.0},
            "tau_gs マージン": {"min": 0.05, "max": 2.0},
        },
        "regularization": regularization,
        "maxiter": maxiter,
    }
