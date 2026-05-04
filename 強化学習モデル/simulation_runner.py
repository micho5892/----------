import math

import numpy as np
import taichi as ti

from .core_imports import (
    Analytics,
    BoundaryManager,
    LBMSimulator,
    PhysicsManager,
    SimConfig,
    SimulationContext,
)
from .shape_modifier import ShapeModifier

# context.py と同一のセル ID（NumPy 側の判定用）
_CELL_FLUID_A = 0
_CELL_SOLID = 10
_CELL_INLET = 20
_CELL_OUTLET = 21


def _merge_sim_config_kwargs(base: dict, overrides: dict | None) -> dict:
    """SimConfig に渡す kwargs を浅い／部分的にマージする。"""
    if not overrides:
        return dict(base)
    out = dict(base)
    for key, val in overrides.items():
        if key == "boundary_conditions" and isinstance(val, dict):
            merged = dict(out.get("boundary_conditions") or {})
            merged.update(val)
            out["boundary_conditions"] = merged
        elif key == "domain_properties" and isinstance(val, dict):
            base_dp = out.get("domain_properties") or {}
            merged = {
                int(k): (dict(v) if isinstance(v, dict) else v)
                for k, v in base_dp.items()
            }
            for cid, props in val.items():
                ic = int(cid)
                if isinstance(props, dict) and isinstance(merged.get(ic), dict):
                    inner = dict(merged[ic])
                    inner.update(props)
                    merged[ic] = inner
                elif isinstance(props, dict):
                    merged[ic] = dict(props)
                else:
                    merged[ic] = props
            out["domain_properties"] = merged
        else:
            out[key] = val
    return out


class HeatSinkSimulationRunner:
    """
    強化学習や可視化スクリプトから呼び出される、ヒートシンクシミュレーションの統合管理クラス
    """
    @staticmethod
    def default_sim_config_kwargs(nx: int, ny: int, nz: int) -> dict:
        return {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "benchmark_name": "rl_heatsink",
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

    def __init__(
        self,
        nx=64,
        ny=64,
        nz=128,
        mode="plan_a",
        sim_config_overrides: dict | None = None,
    ):
        self.mode = mode
        
        # Taichiの初期化 (ここで1回だけ呼ばれる)
        ti.init(arch=ti.gpu, device_memory_fraction=1.0)
        
        # 1. コンフィグの設定
        cfg_kwargs = _merge_sim_config_kwargs(
            self.default_sim_config_kwargs(nx, ny, nz),
            sim_config_overrides,
        )
        self.cfg = SimConfig(**cfg_kwargs)
        
        # 2. 各種コンポーネントの初期化
        self.ctx = SimulationContext(self.cfg.nx, self.cfg.ny, self.cfg.nz, self.cfg.n_particles, self.cfg.fp_dtype)
        self.ctx.set_materials(self.cfg.get_materials_dict())
        self.ctx.set_g_thermal_wall_tables_from_config(self.cfg)

        self.modifier = ShapeModifier(self.ctx, self.cfg)
        self.sim = LBMSimulator(self.cfg)
        self.analytics = Analytics(self.sim.d3q19, self.cfg)
        self.bc_manager = BoundaryManager(self.sim.d3q19, self.cfg)
        self.physics_manager = PhysicsManager(self.sim.d3q19, self.cfg)

        # main.py の run_simulation と同じソフトスタート・時刻管理
        self._current_time_p = 0.0
        self.ramp_time_p = 1.0
        self.last_reset_warmup_meta: dict | None = None

    def compute_warmup_steps_from_characteristic_times(
        self, time_sum_scale: float
    ) -> tuple[int, dict]:
        """
        reset 用ウォームアップステップ数を、ユーザー指定の基準時間の和から求める。

        - 物理的基準時間 t_adv = (観測領域内の流体セル体積) / (入口面積 × |入口物理流速|)
        - 熱的基準時間 t_th = (固体ヒートシンクの流体隣接面積) / 流体の熱拡散率 alpha
        - 物理時間 T = time_sum_scale * (t_adv + t_th)、ステップ数 = ceil(T / dt)
        """
        cid = self.ctx.cell_id.to_numpy()
        nx, ny, nz = int(self.cfg.nx), int(self.cfg.ny), int(self.cfg.nz)
        dx = float(self.cfg.dx)
        dt = float(self.cfg.dt)

        fluid_ids = (_CELL_FLUID_A, _CELL_INLET, _CELL_OUTLET)
        n_fluid = int(np.isin(cid, fluid_ids).sum())
        V_phys = n_fluid * (dx**3)

        n_inlet = int((cid == _CELL_INLET).sum())
        A_in = n_inlet * (dx**2)

        v_lbm = np.array(
            self.cfg.boundary_conditions[20]["velocity"], dtype=np.float64
        )
        v_lbm_mag = float(np.linalg.norm(v_lbm))
        u_ref = max(float(self.cfg.u_lbm_inlet), 1e-30)
        U_p = float(self.cfg.U_inlet_p)
        v_phys_mag = abs(v_lbm_mag) * U_p / u_ref
        Q = A_in * v_phys_mag

        if Q < 1e-40:
            t_adv = 0.0
        else:
            t_adv = V_phys / Q

        fp = self.cfg.domain_properties.get(0, {})
        kf = float(fp.get("k", 0.026))
        rhof = float(fp.get("rho", 1.2))
        cpf = float(fp.get("Cp", 1005.0))
        alpha = kf / max(rhof * cpf, 1e-30)

        # 固体セル (_CELL_SOLID) と流体隣接面の総面積（各面を一度だけ数える）
        fluid_m = np.isin(cid, fluid_ids)
        solid_m = cid == _CELL_SOLID
        n_face = 0
        n_face += int(np.logical_and(solid_m[:-1, :, :], fluid_m[1:, :, :]).sum())
        n_face += int(np.logical_and(solid_m[1:, :, :], fluid_m[:-1, :, :]).sum())
        n_face += int(np.logical_and(solid_m[:, :-1, :], fluid_m[:, 1:, :]).sum())
        n_face += int(np.logical_and(solid_m[:, 1:, :], fluid_m[:, :-1, :]).sum())
        n_face += int(np.logical_and(solid_m[:, :, :-1], fluid_m[:, :, 1:]).sum())
        n_face += int(np.logical_and(solid_m[:, :, 1:], fluid_m[:, :, :-1]).sum())
        A_hs = float(n_face) * (dx**2)

        if alpha < 1e-40:
            t_th = float("inf")
        else:
            t_th = A_hs / alpha

        t_sum = t_adv + t_th
        if not math.isfinite(t_sum):
            steps = 1
        else:
            T_phys = float(time_sum_scale) * t_sum
            steps = max(1, int(math.ceil(T_phys / max(dt, 1e-30))))

        meta = {
            "time_sum_scale": float(time_sum_scale),
            "dx": dx,
            "dt": dt,
            "n_fluid_cells": n_fluid,
            "V_fluid_m3": V_phys,
            "n_inlet_cells": n_inlet,
            "A_inlet_m2": A_in,
            "v_phys_mag_m_s": v_phys_mag,
            "Q_m3_s": Q,
            "t_adv_s": t_adv if math.isfinite(t_adv) else None,
            "A_heat_surface_m2": A_hs,
            "alpha_fluid_m2_s": alpha,
            "t_th_s": t_th if math.isfinite(t_th) else None,
            "warmup_steps": steps,
        }
        return steps, meta

    def reset_simulation(
        self,
        warmup_steps: int | None = None,
        *,
        warmup_time_sum_scale: float = 3.0,
    ):
        """
        シミュレーションを初期状態（ソリッドブロック）に戻し、暖機運転を行う。

        warmup_steps が None のとき、warmup_time_sum_scale と幾何・物性から自動算出する。
        """
        self._current_time_p = 0.0
        self.modifier.build_initial_block()
        if warmup_steps is None:
            warmup_steps, meta = self.compute_warmup_steps_from_characteristic_times(
                warmup_time_sum_scale
            )
            self.last_reset_warmup_meta = meta
        else:
            self.last_reset_warmup_meta = None
        self.sim.init_fields(self.ctx)
        self.run_steps(int(warmup_steps))

    def modify_shape_sphere(self, cx, cy, cz, radius):
        """案A: 指定した球の範囲を削る"""
        self.modifier.subtract_sphere(cx, cy, cz, radius)

    def run_steps(self, num_steps):
        """指定回数だけLBMのステップを進める（main.py のループと同じ順序・ランプ）"""
        for _ in range(num_steps):
            self.physics_manager.apply_all(self.ctx, self._current_time_p)

            if self.ramp_time_p > 0.0 and self._current_time_p < self.ramp_time_p:
                ramp_factor = 0.5 * (
                    1.0 - math.cos(math.pi * self._current_time_p / self.ramp_time_p)
                )
            else:
                ramp_factor = 1.0

            sponge_amp = float(self.cfg.sponge_strength_amp(self._current_time_p))

            self.sim.collide_and_stream(self.ctx, ramp_factor, sponge_amp)
            self.bc_manager.apply_all_before_macro(self.ctx, ramp_factor)
            self.sim.update_macro(self.ctx)
            self.bc_manager.apply_all_after_macro(self.ctx)
            self.sim.move_particles(self.ctx, self.cfg.particles_inject_per_step)

            self._current_time_p += self.cfg.dt

    def get_observation_arrays(self):
        """観測用テンソル（速度、温度、SDF）を取得する"""
        v_np = self.ctx.v.to_numpy()
        temp_np = self.ctx.temp.to_numpy()
        sdf_np = self.ctx.sdf.to_numpy()
        
        obs = np.zeros((5, self.cfg.nx, self.cfg.ny, self.cfg.nz), dtype=np.float32)
        obs[0:3, :, :, :] = np.transpose(v_np, (3, 0, 1, 2))
        obs[3, :, :, :] = temp_np
        obs[4, :, :, :] = sdf_np
        return obs

    def get_metrics(self):
        """現在の評価指標（Nu数、圧力損失）を計算して返す"""
        temp_np = self.ctx.temp.to_numpy()
        v_np = self.ctx.v.to_numpy()
        rho_np = self.ctx.rho.to_numpy()
        
        # 熱量計算 (側面四方から出ていく熱)
        heat_x0 = np.sum(temp_np[0, :, :] * np.maximum(-v_np[0, :, :, 0], 0))
        heat_x1 = np.sum(temp_np[-1, :, :] * np.maximum(v_np[-1, :, :, 0], 0))
        heat_y0 = np.sum(temp_np[:, 0, :] * np.maximum(-v_np[:, 0, :, 1], 0))
        heat_y1 = np.sum(temp_np[:, -1, :] * np.maximum(v_np[:, -1, :, 1], 0))
        total_heat_flux = float(heat_x0 + heat_x1 + heat_y0 + heat_y1)
        
        # 圧力損失計算 (天井と側面四方の密度差)
        rho_in = np.mean(rho_np[:, :, -1])
        rho_out_faces = np.concatenate([
            rho_np[0, :, :].flatten(), rho_np[-1, :, :].flatten(),
            rho_np[:, 0, :].flatten(), rho_np[:, -1, :].flatten()
        ])
        rho_out = np.mean(rho_out_faces)
        pressure_drop = float(rho_in - rho_out)
        
        return total_heat_flux, pressure_drop

    def save_state_vti(self, filepath="rl_state.vti"):
        """
        強化学習用：形状(phi/sdf)、cell_id、密度、流場(v/temp)をまとめたVTIを保存する
        """
        import pyvista as pv
        
        # 1. 必要なデータを numpy に変換
        phi_np = self.ctx.phi.to_numpy()
        sdf_np = self.ctx.sdf.to_numpy()
        temp_np = self.ctx.temp.to_numpy()
        v_np = self.ctx.v.to_numpy()
        cell_id_np = self.ctx.cell_id.to_numpy()
        rho_np = self.ctx.rho.to_numpy()
        
        # 2. PyVistaのUniformGridを作成
        grid = pv.ImageData()
        grid.dimensions = np.array(phi_np.shape) + 1
        grid.spacing = (self.cfg.dx, self.cfg.dx, self.cfg.dx)
        grid.origin = (0, 0, 0)
        
        # 3. データをFortran order(一次元化)して追加
        # (PyVista/VTK はデータをこの形式で受け取る必要があります)
        grid.cell_data["Phi (Solid Fraction)"] = phi_np.flatten(order="F")
        grid.cell_data["SDF (Distance)"] = sdf_np.flatten(order="F")
        grid.cell_data["Cell ID"] = cell_id_np.flatten(order="F")
        grid.cell_data["Density"] = rho_np.flatten(order="F")
        grid.cell_data["Temperature"] = temp_np.flatten(order="F")
        
        # 速度は3次元ベクトルとして保存
        grid.cell_data["Velocity"] = v_np.reshape(-1, 3, order="F")
        
        # 4. 保存
        grid.save(filepath)
        print(f"RL state (Shape & Flow) saved to: {filepath}")

