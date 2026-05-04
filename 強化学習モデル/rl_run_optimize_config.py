# ==========================================================
# rl_run_optimize_config.py
# ==========================================================
"""
強化学習環境(LBMHeatSinkEnv)に渡すためのパラメータ最適化および
オーバーライド辞書生成をカプセル化したモジュール。
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

_LBM_UI = Path(__file__).resolve().parents[1] / "lbm_ui_designer"
if str(_LBM_UI) not in sys.path:
    sys.path.insert(0, str(_LBM_UI))

try:
    from lbm_param_optimize import run_optimize  # type: ignore[import-not-found]
except ImportError:
    run_optimize = None


class RLParameterOptimizer:
    """
    強化学習用のシミュレーション環境(SimConfig)に渡すパラメータを、
    物理・LBMの制約(Re, tau等)を満たすように最適化し、
    設定を上書きするオーバーライド辞書(overrides)を生成するクラス。
    """
    def __init__(self, nx: int, **kwargs):
        self.nx = nx
        self.L_domain = float(kwargs.get("L_domain", 0.1))
        self.U_inlet_p = float(kwargs.get("U_inlet_p", 0.05))
        self.fluid = str(kwargs.get("fluid", "Air"))
        self.solid = str(kwargs.get("solid", "Copper"))
        self.temperature_K = float(kwargs.get("temperature_K", 300.0))
        self.pressure_Pa = float(kwargs.get("pressure_Pa", 101325.0))
        
        self.fix_cr = bool(kwargs.get("fix_cr", False))
        self.regularization = float(kwargs.get("regularization", 1e-5))
        self.maxiter = int(kwargs.get("maxiter", 30000))
        
        # ==========================================
        # ★修正: Re を ranges から削除し、銅の高い熱拡散率に合わせて上限を広げる
        # ==========================================
        self.ranges = kwargs.get("ranges", {
            "tau_f マージン": {"min": 0.01, "max": 2.0},
            "tau_gf マージン": {"min": 0.01, "max": 2.0},
            "tau_gs マージン": {"min": 0.01, "max": 20.0}, # 銅は熱が伝わりやすいため余裕を持たせる
        })
        self.targets = kwargs.get("targets", {})

        self._primary_result = None
        self._optimize_report = None

    def _build_config_dict(self) -> dict:
        margin = self.nx // 8
        span_cells = max(1, self.nx - 2 * margin)
        L_ref = float(span_cells) / float(self.nx) * self.L_domain

        return {
            "fluid": self.fluid,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "solid": self.solid,
            "fix_cr": self.fix_cr,
            "fixed": {
                "nx": float(self.nx),
                "L_domain": self.L_domain,
                "L_ref": L_ref,
                "nu": True,
                "k_f": True,
                "rho_f": True,
                "Cp_f": True,
                "k_s": True,
                "rho_s": True,
                "Cp_s": True,
                "beta_f": True,
                "U": self.U_inlet_p,
            },
            "ranges": self.ranges,
            "targets": self.targets,
            "regularization": self.regularization,
            "maxiter": self.maxiter,
        }

    def run(self, verbose: bool = True) -> bool:
        if run_optimize is None:
            raise RuntimeError("lbm_param_optimize.run_optimize が利用できません。パスを確認してください。")
            
        cfg = self._build_config_dict()
        if verbose:
            print("[RLParameterOptimizer] Running optimization with config:")
            pprint(cfg)
            
        out = run_optimize(cfg)
        self._optimize_report = out
        
        # ==========================================
        # ★修正: 失敗時でも「途中まで計算したベストエフォートな値」を保持させる
        # ==========================================
        self._primary_result = out.get("primary", None)
        
        if out.get("success"):
            if verbose:
                print(f"[RLParameterOptimizer] Optimization Successful.")
                print(f"  u_lbm (LBM Velocity) = {self._primary_result['u_lbm']:.6f}")
                print(f"  tau_f margin = {out['state']['tau_f マージン']:.6f}")
                print(f"  Re = {out['state']['Re']:.2f}")
            return True
        else:
            if verbose:
                print(f"[RLParameterOptimizer] ⚠️ Optimization Failed: {out.get('message')}")
                print("[RLParameterOptimizer] 失敗しましたが、得られたパラメータで続行を試みます。")
            return False

    def get_sim_overrides(self) -> dict:
        if self._primary_result is None:
            raise ValueError("最適化結果がありません。設定ファイルや流体物性を確認してください。")
            
        primary = self._primary_result
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
            "u_lbm_inlet": u_lbm,
            "boundary_conditions": {
                20: {
                    "type": "inlet",
                    "velocity":[0.0, 0.0, -u_lbm],
                    "temperature": 0.0,
                },
            },
            "domain_properties": {
                0: fluid_dp,
                20: fluid_dp,
                21: fluid_dp,
            },
        }

    def get_full_state(self) -> dict | None:
        if self._optimize_report:
            return self._optimize_report.get("state")
        return None