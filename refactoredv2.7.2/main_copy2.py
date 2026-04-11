# ==========================================================
# main.py — run_simulation・メインループ（ベンチマーク検証対応版）
# ==========================================================
import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import taichi as ti
from datetime import datetime
import math 

ti.init(arch=ti.gpu)

from context import SimulationContext, INLET, OUTLET
from config import SimConfig
from geometry import GeometryBuilder
from boundary import BoundaryConditionManager
from solver import LBMSimulator
from export import build_vis_frame, export_gif_frames
from analytics import Analytics

try:
    from vtk_export import export_step
except ImportError:
    export_step = None

try:
    from diagnostics import log_field_diagnostics
except ImportError:
    log_field_diagnostics = None


def run_simulation(**kwargs):
    # 実行するベンチマークの種類を取得 ('parallel_plates', 'cavity', 'cylinder')
    benchmark = kwargs.pop("benchmark", "cylinder")

    out_dir = kwargs.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = os.path.join(out_dir, f"{benchmark}_{timestamp}.gif")
    if "filename" not in kwargs:
        kwargs["filename"] = default_filename

    cfg = SimConfig(**kwargs)
    ctx = SimulationContext(cfg.nx, cfg.ny, cfg.nz, cfg.n_particles)

    materials = cfg.get_materials_dict()
    ctx.set_materials(materials)

    # ==========================================================
    # 1. ベンチマークに応じた形状と境界の設定
    # ==========================================================
    geo = GeometryBuilder()
    
    if benchmark == "parallel_plates":
        geo.build_parallel_plates(ctx)
        geo.set_inlet_outlet(ctx)
    elif benchmark == "cavity":
        geo.build_lid_driven_cavity(ctx)
        # キャビティは密閉空間なので set_inlet_outlet は呼ばない
    elif benchmark == "cylinder":
        geo.build_benchmark_cylinder(ctx)
        geo.set_inlet_outlet(ctx)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark}")

    sim = LBMSimulator(cfg)
    bc = BoundaryConditionManager(sim.d3q19, cfg)
    analytics = Analytics(sim.d3q19, cfg)
    sim.init_fields(ctx)

    frames =[]
    print(f"--- Starting Benchmark: {benchmark.upper()} ---")
    print(f"Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}, steps={cfg.steps}")
    print(f"[Properties] nu: {cfg.nu_p:.2e}, k: {cfg.k_p:.3f}, BC: {'T' if cfg.bc_type=='T' else 'q'}")
    instance_attributes = [attr for attr in dir(cfg) if not attr.startswith('__') and not callable(getattr(cfg, attr))]
    for instance_attribute in instance_attributes:
        instance_value = getattr(cfg, instance_attribute)
        print(instance_attribute,": ",instance_value)
    vti_enabled = getattr(cfg, "vti_export_interval", 0) > 0 and export_step is not None
    if vti_enabled:
        vti_dir = os.path.dirname(cfg.vti_path_template)
        if vti_dir:
            os.makedirs(vti_dir, exist_ok=True)



    # ...(中略)...

    # ==========================================================
    # 2. メインループ
    # ==========================================================
    ramp_steps = 3000 # 【追加】ソフトスタートにかけるステップ数

    for step in range(cfg.steps):
        sim.collide(ctx)
        sim.stream_and_bc(ctx)
        
        # 【追加】ソフトスタート係数（0.0 から 1.0 へコサインカーブで滑らかに上昇）
        if step < ramp_steps:
            ramp_factor = 0.5 * (1.0 - math.cos(math.pi * step / ramp_steps))
        else:
            ramp_factor = 1.0
        
        for target_id, vel in cfg.bc_velocity_by_id.items():
            current_vel = vel * ramp_factor # 速度を徐々に上げる
            bc.apply_velocity_bc(ctx, target_id, current_vel, rho_val=1.0, temp_val=0.0)
            
        for target_id in cfg.bc_outlet_ids:
            bc.apply_outlet_bc(ctx, target_id)

        sim.update_macro(ctx)

        for target_id, flux in cfg.bc_heat_flux_by_id.items():
            bc.apply_heat_flux_bc(ctx, target_id, flux)

        sim.move_particles(ctx, cfg.particles_inject_per_step)

        # 診断ログ
        if log_field_diagnostics is not None:
            if step <= 2000 and step % 100 == 0:
                log_field_diagnostics(ctx, step, log_distributions=(1200 <= step <= 1600 and step % 100 == 0))

        # ==========================================================
        # 3. ベンチマークに応じた評価とログ出力
        # ==========================================================
        if step % cfg.vis_interval == 0:
            if benchmark == "parallel_plates":
                # 平行平板: ヌセルト数(Nu)の理論値との誤差を検証
                k_target = int(cfg.nz * 0.1) # 出口付近の十分に発達した位置
                local_nu = analytics.get_local_Nu(ctx, k_target)
                target_nu = 7.54 if cfg.bc_type == 'T' else 8.23
                p_error = abs(local_nu - target_nu) / target_nu * 100
                print(f"Step {step:5d} | Nu_local: {local_nu:.3f} | Target Nu: {target_nu} | Error: {p_error:.2f} %")

            elif benchmark == "cavity":
                # キャビティ: 中心点の速度モニタリングによる定常状態への収束確認
                # 中心 (nx/2, ny/2, nz/2) の速度をサンプリング
                # 注意: GPUからデータを取ってくるため ti.kernel を使わず簡単にサンプリング(numpy経由)
                v_np = ctx.v.to_numpy()
                cx, cy, cz = cfg.nx//2, cfg.ny//2, cfg.nz//2
                u_center = v_np[cx, cy, cz, 0] # X方向速度
                w_center = v_np[cx, cy, cz, 2] # Z方向速度
                vel_mag = (u_center**2 + w_center**2)**0.5
                print(f"Step {step:5d} | Center Vel(u,w): ({u_center:.4f}, {w_center:.4f}) | Vel Mag: {vel_mag:.6f}")

            elif benchmark == "cylinder":
                # 円柱: 後流域における横方向(X方向)の速度変動によるカルマン渦の観測
                # 後方の点 (nx/2 + nx*0.1, ny/2, nz*0.25) のX方向速度を取得
                v_np = ctx.v.to_numpy()
                ox, oy, oz = int(cfg.nx*0.6), cfg.ny//2, int(cfg.nz*0.25)
                u_wake = v_np[ox, oy, oz, 0]
                print(f"Step {step:5d} | Wake Cross-Vel (X): {u_wake:+.5f}  (変動していればカルマン渦発生)")

            # GIF用フレーム生成
            canvas = build_vis_frame(ctx, cfg)
            frames.append(canvas)

        if vti_enabled and step % cfg.vti_export_interval == 0:
            export_step(ctx, step, cfg.vti_path_template, dx=cfg.dx)

    export_gif_frames(frames, cfg.filename, fps=12)
    print(f"Finished Benchmark: {benchmark}. Saved as {cfg.filename}")
    return cfg.filename


if __name__ == "__main__":
    # ==========================================================
    # 実行プリセット (コメントアウトを切り替えてテスト)
    # ==========================================================

    """
    # 【検証1】 平行平板チャネル (Nu=7.54の確認)
    run_simulation(
        benchmark="parallel_plates",
        nx=64, ny=64, nz=512,
        Lx_p=0.05, U_inlet_p=0.05, nu_p=2.5e-5, k_p=0.026,
        steps=50000, vis_interval=100, bc_type="T",
        bc_velocity_by_id={INLET:[0.0, 0.0, -0.05]},
        bc_outlet_ids=[OUTLET],
        vti_export_interval=0,
        particles_inject_per_step=0
    )
    """

    """
    # 【検証2】 リッド・ドリブン・キャビティ (Re=400)
    run_simulation(
        benchmark="cavity",
        nx=128, ny=4, nz=128, # Y方向は薄くして擬似2D
        Lx_p=0.1, U_inlet_p=0.1, nu_p=2.5e-5, k_p=0.026,
        steps=10000, vis_interval=200, bc_type="T",
        bc_velocity_by_id={20:[0.1, 0.0, 0.0], 21: [0.0, 0.0, 0.0]}, # 上壁移動, 他静止
        bc_outlet_ids=[], # 密閉空間
        vti_export_interval=0,
        particles_inject_per_step=0
    )
    """

    
    # 【検証3】 円柱周り流れ・カルマン渦 (Re=150)
    run_simulation(
        benchmark="cylinder",
        nx=128, ny=8, nz=512,
        Lx_p=0.1, U_inlet_p=0.1, nu_p=6.66e-6, k_p=0.026,
        steps=15000, vis_interval=200, bc_type="T",
        bc_velocity_by_id={INLET:[0.0, 0.0, -0.1]},
        bc_outlet_ids=[OUTLET],
        vti_export_interval=0,
        particles_inject_per_step=200
    )
    