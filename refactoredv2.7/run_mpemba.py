import os
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from main import run_simulation, run_optimize
from geometry import GeometryBuilder
import physics
from config import SimConfig, RenderConfig, ParticleConfig
# =====================================================================
# 1. モンキーパッチ: カスタム物理モデルを外部から注入できるようにする
# =====================================================================
_original_init = physics.PhysicsManager.__init__

def _patched_init(self, d3q19, cfg):
    _original_init(self, d3q19, cfg)
    # SimConfig に custom_physics_models があれば追加で読み込む
    extras = getattr(cfg, "custom_physics_models", None)
    if extras:
        for model in extras:
            self.models.append(model)

physics.PhysicsManager.__init__ = _patched_init

# =====================================================================
# 2. カスタム形状ビルダーの追加
# =====================================================================
@staticmethod
@ti.kernel
def _build_mpemba_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
    wall = 4
    
    # 容器のサイズと位置 (中央に配置)
    cup_width = int(nx * 0.4)
    cup_height = int(nz * 0.4)
    cup_thick = 2
    
    cx_min = (nx - cup_width) // 2
    cx_max = cx_min + cup_width
    cz_min = wall + 10 # 下にも空気が通るように床から浮かせる
    cz_max = cz_min + cup_height
    
    for i, j, k in cell_id:
        cell_id[i, j, k] = 0 # デフォルトは空気 (ID: 0)
        sdf[i, j, k] = 100.0
        
        # 冷凍庫の壁 (ID: 10)
        if i < wall or i >= nx - wall or k < wall:
            cell_id[i, j, k] = 10
            sdf[i, j, k] = 0.0
            
        # 天井を開放境界にして完全密閉による非物理的な挙動を回避 (ID: 21)
        elif k >= nz - 1:
            cell_id[i, j, k] = 21
            sdf[i, j, k] = 100.0
            
        else:
            # 容器周辺
            if cx_min <= i <= cx_max and cz_min <= k <= cz_max:
                # 密閉された容器の壁 (ID: 2)
                if i < cx_min + cup_thick or i > cx_max - cup_thick or k < cz_min + cup_thick or k > cz_max - cup_thick:
                    cell_id[i, j, k] = 2 
                    sdf[i, j, k] = 0.0
                else:
                    # 容器の内部の液体 (ID: 1)
                    cell_id[i, j, k] = 1 
                    sdf[i, j, k] = 100.0

def build_mpemba(self, ctx):
    self._build_mpemba_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

# GeometryBuilder クラスに無理やり追加
GeometryBuilder._build_mpemba_kernel = _build_mpemba_kernel
GeometryBuilder.build_mpemba = build_mpemba

# =====================================================================
# 3. 水の温度を追跡＆初期化するカスタム物理モニター
# =====================================================================
@ti.data_oriented
class MpembaMonitor(physics.PhysicsModel):
    def __init__(self, init_temp):
        self.init_temp = init_temp
        self.initialized = False
        self.history = []
        self.time_history =[]
        self.step_counter = 0

        # run_case では run_simulation() より先に Monitor が生成されるため、
        # ti.field は Taichi 初期化後（apply初回）に遅延生成する。
        self.sum_temp = None
        self.count = None

    def _ensure_fields_initialized(self):
        if self.sum_temp is None:
            self.sum_temp = ti.field(dtype=ti.f32, shape=())
        if self.count is None:
            self.count = ti.field(dtype=ti.i32, shape=())

    def apply(self, ctx, current_time):
        self._ensure_fields_initialized()

        if not self.initialized:
            self._init_temp_kernel(ctx)
            self.initialized = True
        
        # 毎ステップ計算すると重いので 10ステップに1回記録
        if self.step_counter % 10 == 0:
            self.sum_temp[None] = 0.0
            self.count[None] = 0
            self._calc_avg_kernel(ctx)
            
            avg_t = self.sum_temp[None] / max(1, self.count[None])
            self.history.append(avg_t)
            self.time_history.append(current_time)
            
        self.step_counter += 1

    @ti.kernel
    def _init_temp_kernel(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            cid = ctx.cell_id[i, j, k]
            if cid == 1:
                ctx.temp[i, j, k] = self.init_temp
            elif cid == 2:
                # 容器も水と同じ初期温度にする（過渡応答を正確に見るため）
                ctx.temp[i, j, k] = self.init_temp
            elif cid == 0 or cid == 10 or cid == 21:
                # 空気、冷凍庫の壁、開放境界は 0.0
                ctx.temp[i, j, k] = 0.0

    @ti.kernel
    def _calc_avg_kernel(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == 1: # 水のみをカウント
                ti.atomic_add(self.sum_temp[None], ctx.temp[i, j, k])
                ti.atomic_add(self.count[None], 1)

# =====================================================================
# 4. シミュレーション実行関数
# =====================================================================
def run_case(case_name, init_temp, target_Ra=1e6):
    nx = nz = 128
    ny = 4
    L_domain = 0.1
    dx = L_domain / nx
    
    # 水の領域のサイズを代表長とする
    cup_width = int(nx * 0.4)
    cup_thick = 2
    L_eff = dx * (cup_width - 2 * cup_thick)
    
    # 水(ID=1)をターゲットにLBMの安定パラメータを最適化
    config_opt = {
        "fluid": "Water",
        "temperature_K": 300,
        "pressure_Pa": 101325,
        "solid": "Copper",
        "fix_cr": False,
        "fixed": {
            "nx": nx,             
            "L_domain": L_domain, 
            "L_ref": L_eff,
            "u_lbm": 0.05,
            "U": 1.0,
        },
        "ranges": {
            # "Ra": {"min": target_Ra-1e4, "max": target_Ra+1e4},
            # "Pr": {"min": 6.9, "max": 7.1},
        },
        "targets": {
            "Ra": {"value": target_Ra, "weight": 1.0},
            "Pr": {"value": 7.0, "weight": 1.0},
        },
        "target_regularization": 1.0,
        "regularization": 1.0e-5,
        "maxiter": 300000,
    }

    print(f"\n--- Optimizing parameters for {case_name} ---")
    result = run_optimize(config_opt)
    if not result["success"]:
        print(f"[ERROR] {case_name}: パラメータ最適化に失敗")
        return None, None
        
    state = result["state"]
    nu_w = state["nu"]
    k_w = state["k_f"]
    rho_w = state["rho_f"]
    Cp_w = state["Cp_f"]
    alpha_w = k_w / (rho_w * Cp_w)
    
    g_phys = 9.81
    delta_T = 1.0
    beta_p = (target_Ra * nu_w * alpha_w) / (g_phys * delta_T * (L_eff**3))
    
    # --- 流体と固体の物性を派生 ---
    # 空気 (ID 0) の物性: 安定性のため極端な密度比は避けつつ Pr=0.7 を実現
    nu_a = nu_w * 2.0
    alpha_a = nu_a / 0.7 
    rho_a = rho_w * 0.01
    Cp_a = Cp_w * 1.0
    k_a = alpha_a * rho_a * Cp_a
    
    # 容器 (ID 2) の物性 (固体): 冷気が伝わりやすいように高熱伝導率
    k_cup = k_w * 50.0  
    rho_cup = rho_w * 2.0
    Cp_cup = Cp_w * 0.5
    
    monitor = MpembaMonitor(init_temp)
    
    # 熱拡散のみの場合の緩和時間 t_c を基準に、対流による冷却が完了する時間を設定
    t_c = (L_eff**2) / alpha_w
    max_time_p = t_c * 0.8 
    max_time_p = 10.0 # デバッグ用に10秒に固定
    print(f"[{case_name}] Running up to {max_time_p:.3f} s")
    
    artifact_parent = os.path.join("results", "mpemba_effect")
    paths_out: dict = {}

    render_cfg = RenderConfig(
        output_format="mp4",
        vti_export_interval=0,
    )
    particle_cfg = ParticleConfig(
        n_particles=0,
        particles_inject_per_step=0,
    )

    cfg = SimConfig(
        benchmark_name="mpemba",
        fp_dtype="float32",
        steady_detection=False,
        state=state,
        artifact_parent=artifact_parent,
        paths_out=paths_out,
        nx=nx,
        ny=ny,
        nz=nz,
        Lx_p=L_domain,
        periodic_y=True,
        U_inlet_p=state["U"],
        u_lbm_inlet=float(state["u_lbm"]),
        visualization_mode="none",
        target_video_fps=60.0,
        max_time_p=max_time_p,
        ramp_time_p=0.0,
        sponge_thickness=0.0,
        domain_properties={
            0: {"nu": nu_a, "k": k_a, "rho": rho_a, "Cp": Cp_a},
            1: {"nu": nu_w, "k": k_w, "rho": rho_w, "Cp": Cp_w},
            2: {"nu": 0.0, "k": k_cup, "rho": rho_cup, "Cp": Cp_cup},
            10: {"nu": 0.0, "k": k_w, "rho": rho_w, "Cp": Cp_w},
        },
        physics_models={
            "boussinesq": {"g_vec": [0.0, 0.0, -g_phys], "beta": beta_p, "T_ref": 0.0}
        },
        boundary_conditions={
            10: {"type": "isothermal_wall", "temperature": 0.0},
            21: {"type": "outlet"},
        },
        custom_physics_models=[monitor],
        render=render_cfg,
        particles=particle_cfg,
    )
    run_simulation(cfg)
    
    return np.array(monitor.time_history), np.array(monitor.history)

# =====================================================================
# 5. 実行と結果の比較プロット
# =====================================================================
if __name__ == "__main__":
    # Case 1: 常温の水 (初期温度 0.5)
    t_warm, T_warm = run_case("Warm_Water", init_temp=0.5, target_Ra=1e6)
    
    # Case 2: 高温の水 (初期温度 1.0)
    t_hot, T_hot = run_case("Hot_Water", init_temp=1.0, target_Ra=1e6)
    
    if t_warm is not None and t_hot is not None:
        plt.figure(figsize=(10, 6))
        
        plt.plot(t_warm, T_warm, label="Warm Water (Init T=0.5)", color="blue", linewidth=2)
        plt.plot(t_hot, T_hot, label="Hot Water (Init T=1.0)", color="red", linewidth=2)
        
        # 冷却目標ライン (凍結開始温度のイメージ)
        target_temp = 0.1
        plt.axhline(target_temp, color='gray', linestyle='--', label=f'Target Temp = {target_temp}')
        
        plt.xlabel("Physical Time [s]", fontsize=12)
        plt.ylabel("Average Water Temperature (Dimensionless)", fontsize=12)
        plt.title("Cooling Curves: Investigating the Mpemba Effect via Natural Convection", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plot_path = os.path.join("results", "mpemba_effect", "cooling_curves_comparison.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        print(f"\n[SUCCESS] Cooling curve comparison plot saved to: {plot_path}")
        
        # 逆転（交差）があったかの簡易判定
        if np.min(T_hot) < target_temp and np.min(T_warm) < target_temp:
            idx_hot = np.where(T_hot <= target_temp)[0][0]
            idx_warm = np.where(T_warm <= target_temp)[0][0]
            time_hot = t_hot[idx_hot]
            time_warm = t_warm[idx_warm]
            print(f"Time to reach T={target_temp} -> Warm: {time_warm:.3f}s, Hot: {time_hot:.3f}s")
            if time_hot < time_warm:
                print(">>> Mpemba Effect Observed! Hot water cooled faster.")
            else:
                print(">>> Normal cooling observed. Warm water cooled faster.")