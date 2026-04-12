# ==========================================================
# config.py — SimConfig、materials_dict・境界条件の ID ベース管理
# ==========================================================
import math
import taichi as ti
from context import FLUID_A, INLET, OUTLET, SOLID, SOLID_HEAT_SOURCE
from lbm_logger import get_logger

log = get_logger(__name__)

# 浮動小数点型（run_simulation 内で cfg.fp_dtype に合わせて上書きされる）
TI_FLOAT = ti.f32


class SimConfig:
    def __init__(self, **kwargs):
        # float32 / float16 の指定（Taichi のフィールド・カーネルで使用）
        fp = kwargs.get('fp_dtype', 'float32')
        self.fp_dtype = ti.f32 if fp == 'float32' else ti.f16

        self.nx = kwargs.get('nx', 128)
        self.ny = kwargs.get('ny', 128)
        self.nz = kwargs.get('nz', 32)
        self.benchmark_name = kwargs.get('benchmark_name', 'unknown')
        self.Lx_p = kwargs.get('Lx_p', 0.1)
        self.dx = self.Lx_p / self.nx
        self.sponge_thickness = kwargs.get('sponge_thickness', 0.0)
        # スポンジ「強度」の時間減衰（セル厚 sponge_thickness は固定のまま）
        # duration<=0 なら従来どおり常に全強度。>0 なら [start, start+duration] で振幅 1→0（半周期コサイン）
        self.sponge_strength_decay_start_p = float(kwargs.get('sponge_strength_decay_start_p', 0.0))
        self.sponge_strength_decay_duration_p = float(kwargs.get('sponge_strength_decay_duration_p', 0.0))

        self.steps = kwargs.get('steps', 600)
        self.vis_interval = kwargs.get('vis_interval', 20)
        self.filename = kwargs.get('filename', 'output.gif')

        self.n_particles = kwargs.get('n_particles', 10000)
        self.particles_inject_per_step = kwargs.get('particles_inject_per_step', max(1, self.n_particles // 50))
        self.vti_export_interval = kwargs.get('vti_export_interval', 0)
        self.vti_path_template = kwargs.get('vti_path_template', 'results/step_{:06d}.vti')

        # === フェーズ1.5: 汎用境界条件辞書の取得 ===
        self.boundary_conditions = kwargs.get('boundary_conditions', {})
        self.physics_models = kwargs.get('physics_models', {})

        # 代表流速の自動取得（inlet設定の中から【最大の速度】を探す）
        self.u_lbm_inlet = 0.1
        self.U_inlet_p = kwargs.get('U_inlet_p', 1.0)
        max_u_lbm = 0.0

        # ▼ 追加: 周期境界フラグ 
        self.periodic_x = kwargs.get('periodic_x', False)
        self.periodic_y = kwargs.get('periodic_y', False)
        self.periodic_z = kwargs.get('periodic_z', False)

        # 等温壁の温度場 g: 隣接流体から非平衡成分を補外する NEEM（無効時は純粋平衡分布のみ）
        self.neem_isothermal_wall = kwargs.get('neem_isothermal_wall', True)
        
        for bc_id, bc_info in self.boundary_conditions.items():
            if bc_info.get("type") == "inlet":
                v = bc_info.get("velocity", [0.0, 0.0, 0.0])
                v_mag = max(abs(v[0]), abs(v[1]), abs(v[2]))
                if v_mag > max_u_lbm:
                    max_u_lbm = float(v_mag)
                    
        if max_u_lbm > 0.0:
            self.u_lbm_inlet = max_u_lbm

        self.dt = self.dx * self.u_lbm_inlet / self.U_inlet_p

        # === 複数領域の独立した物性辞書 ===
        self.domain_properties = kwargs.get('domain_properties', {})
        self.T_inlet_p = kwargs.get("T_inlet_p", 0.0)
        self.T_wall_p = kwargs.get("T_wall_p", 1.0)
        default_delta_t = self.T_wall_p - self.T_inlet_p
        if abs(default_delta_t) < 1e-12:
            default_delta_t = 1.0
        self.delta_T_ref = kwargs.get("delta_T_ref", default_delta_t)

        # 回転円筒の角速度
        # - `omega_cylinder` は LBM格子単位（旧来互換）
        # - `omega_cylinder_phys` を指定した場合は `omega_lbm = omega_phys * dt` へ変換して `omega_cylinder` に反映
        self.omega_cylinder_phys = kwargs.get('omega_cylinder_phys', None)
        self.omega_cylinder = kwargs.get('omega_cylinder', 0.0)
        if self.omega_cylinder_phys is not None:
            self.omega_cylinder = float(self.omega_cylinder_phys) * float(self.dt)
        self.cylinder_center = kwargs.get('cylinder_center',[self.nx * 0.5, self.ny * 0.5, self.nz * 0.75])

        axis_str = kwargs.get('rotation_axis', 'z').lower()
        if axis_str == 'x':
            self.rot_axis_id = 0
        elif axis_str == 'y':
            self.rot_axis_id = 1
        else:
            self.rot_axis_id = 2  # デフォルトはZ軸

        log.debug(
            "SimConfig: benchmark=%s grid=%sx%sx%s",
            self.benchmark_name,
            self.nx,
            self.ny,
            self.nz,
        )

    def sponge_strength_amp(self, time_p: float) -> float:
        """collide_and_stream 内スポンジ粘性ブーストの全体倍率 [0, 1]。"""
        dur = self.sponge_strength_decay_duration_p
        if dur <= 0.0:
            return 1.0
        t = float(time_p) - self.sponge_strength_decay_start_p
        if t <= 0.0:
            return 1.0
        if t >= dur:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * t / dur))

    def get_materials_dict(self):
        """設定された各IDの物性から、独立して tau_f, tau_g を計算してテーブルに登録する"""
        mat_dict = {}
        
        for cid, props in self.domain_properties.items():
            # 各領域ごとの物性値を取得 (指定がなければ水のデフォルト値)
            nu = props.get("nu", 1.0e-5)
            k_val = props.get("k", 0.6)
            rho = props.get("rho", 1000.0)
            Cp = props.get("Cp", 4180.0)
            
            # --- 速度の緩和時間 (tau_f) の計算 ---
            nu_lbm = nu * self.dt / (self.dx**2)
            tau_f = 3.0 * nu_lbm + 0.5

            # --- 熱の緩和時間 (tau_g) の計算 ---
            alpha = k_val / (rho * Cp)
            alpha_lbm = alpha * self.dt / (self.dx**2)
            tau_g = 3.0 * alpha_lbm + 0.5

            is_fluid_flag = 0 if nu <= 1e-12 else 1
            
            # （念のため）境界条件の type に "wall" が含まれていれば確実に固体(0)にする
            if cid in self.boundary_conditions:
                bc_type = self.boundary_conditions[cid].get("type", "")
                if "wall" in bc_type:
                    is_fluid_flag = 0
            
            mat_dict[cid] = (tau_f, tau_g, is_fluid_flag)
            
        return mat_dict