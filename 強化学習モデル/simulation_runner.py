import numpy as np
import taichi as ti

from .core_imports import (
    Analytics,
    BoundaryManager,
    LBMSimulator,
    SimConfig,
    SimulationContext,
)
from .shape_modifier import ShapeModifier

class HeatSinkSimulationRunner:
    """
    強化学習や可視化スクリプトから呼び出される、ヒートシンクシミュレーションの統合管理クラス
    """
    def __init__(self, nx=64, ny=64, nz=128, mode="plan_a"):
        self.mode = mode
        
        # Taichiの初期化 (ここで1回だけ呼ばれる)
        ti.init(arch=ti.gpu, device_memory_fraction=1.0)
        
        # 1. コンフィグの設定
        self.cfg = SimConfig(
            nx=nx, ny=ny, nz=nz,
            benchmark_name="rl_heatsink",
            boundary_conditions={
                20: {"type": "inlet", "velocity":[0.0, 0.0, -0.05], "temperature": 0.0},
                21: {"type": "outlet"},
                10: {"type": "isothermal_wall", "temperature": 1.0},
            },
            domain_properties={
                0:  {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
                20: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0}, 
                21: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0}, 
                10: {"nu": 0.0,    "k": 400.0, "rho": 8960.0, "Cp": 385.0},
            }
        )
        
        # 2. 各種コンポーネントの初期化
        self.ctx = SimulationContext(self.cfg.nx, self.cfg.ny, self.cfg.nz, self.cfg.n_particles, self.cfg.fp_dtype)
        self.ctx.set_materials(self.cfg.get_materials_dict())
        self.ctx.set_g_thermal_wall_tables_from_config(self.cfg)

        self.modifier = ShapeModifier(self.ctx, self.cfg)
        self.sim = LBMSimulator(self.cfg)
        self.analytics = Analytics(self.sim.d3q19, self.cfg)
        self.bc_manager = BoundaryManager(self.sim.d3q19, self.cfg)

    def reset_simulation(self, warmup_steps=100):
        """シミュレーションを初期状態（ソリッドブロック）に戻し、暖機運転を行う"""
        self.modifier.build_initial_block()
        self.sim.init_fields(self.ctx)
        self.run_steps(warmup_steps)

    def modify_shape_sphere(self, cx, cy, cz, radius):
        """案A: 指定した球の範囲を削る"""
        self.modifier.subtract_sphere(cx, cy, cz, radius)

    def run_steps(self, num_steps):
        """指定回数だけLBMのステップを進める"""
        for _ in range(num_steps):
            self.sim.collide_and_stream(self.ctx, 1.0, 1.0)
            self.bc_manager.apply_all_before_macro(self.ctx, 1.0)
            self.sim.update_macro(self.ctx)
            self.bc_manager.apply_all_after_macro(self.ctx)

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
        強化学習用：現在の形状(phi/sdf)と流場(v/temp)を全てまとめたVTIを保存する
        """
        import pyvista as pv
        
        # 1. 必要なデータを numpy に変換
        phi_np = self.ctx.phi.to_numpy()
        sdf_np = self.ctx.sdf.to_numpy()
        temp_np = self.ctx.temp.to_numpy()
        v_np = self.ctx.v.to_numpy()
        
        # 2. PyVistaのUniformGridを作成
        grid = pv.ImageData()
        grid.dimensions = np.array(phi_np.shape) + 1
        grid.spacing = (self.cfg.dx, self.cfg.dx, self.cfg.dx)
        grid.origin = (0, 0, 0)
        
        # 3. データをFortran order(一次元化)して追加
        # (PyVista/VTK はデータをこの形式で受け取る必要があります)
        grid.cell_data["Phi (Solid Fraction)"] = phi_np.flatten(order="F")
        grid.cell_data["SDF (Distance)"] = sdf_np.flatten(order="F")
        grid.cell_data["Temperature"] = temp_np.flatten(order="F")
        
        # 速度は3次元ベクトルとして保存
        grid.cell_data["Velocity"] = v_np.reshape(-1, 3, order="F")
        
        # 4. 保存
        grid.save(filepath)
        print(f"RL state (Shape & Flow) saved to: {filepath}")

