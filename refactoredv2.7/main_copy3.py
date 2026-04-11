# ==========================================================
# main.py — run_simulation・メインループ（物理時間・定常判定・ETA搭載版）
# ==========================================================
import os
import sys
import time
import math
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import taichi as ti
from datetime import datetime
import json
import numpy as np

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


def format_eta(seconds):
    """秒数を h m s の見やすいフォーマットに変換する"""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0: return f"{h}h {m}m {s}s"
    elif m > 0: return f"{m}m {s}s"
    else: return f"{s}s"

class SteadyStateDetector:
    """流れの定常化（および周期定常化）を自動検知するクラス"""
    def __init__(self, window_time_p, tolerance, extra_time_p, U_ref):
        self.window_time_p = window_time_p
        self.tolerance = tolerance
        self.extra_time_p = extra_time_p
        self.U_ref = U_ref
        self.history =[]
        self.is_steady = False
        self.steady_detected_time_p = None

    def update(self, current_time_p, val):
        self.history.append((current_time_p, val))
        
        # ウィンドウ2つ分より古いデータを破棄
        cutoff_time = current_time_p - 2.0 * self.window_time_p
        while self.history and self.history[0][0] < cutoff_time:
            self.history.pop(0)

        # 履歴が十分(約2ウィンドウ分)蓄積されたか
        if not self.is_steady and len(self.history) > 10:
            dt_hist = self.history[-1][0] - self.history[0][0]
            if dt_hist >= 1.9 * self.window_time_p:
                mid_time = current_time_p - self.window_time_p
                # 前半と後半のウィンドウに分割
                half1 =[v for t, v in self.history if t < mid_time]
                half2 =[v for t, v in self.history if t >= mid_time]
                
                if half1 and half2:
                    mean1, mean2 = sum(half1) / len(half1), sum(half2) / len(half2)
                    amp1, amp2 = max(half1) - min(half1), max(half2) - min(half2)

                    # 代表速度(U_ref)に対する相対変化率で判定
                    mean_diff = abs(mean1 - mean2) / (self.U_ref + 1e-8)
                    amp_diff = abs(amp1 - amp2) / (self.U_ref + 1e-8)

                    if mean_diff < self.tolerance and amp_diff < self.tolerance:
                        self.is_steady = True
                        self.steady_detected_time_p = current_time_p
                        return True # 初回検知フラグ
        return False

    def should_stop(self, current_time_p):
        if self.is_steady and self.steady_detected_time_p is not None:
            if current_time_p - self.steady_detected_time_p >= self.extra_time_p:
                return True
        return False


def run_simulation(**kwargs):
    benchmark = kwargs.pop("benchmark", "cylinder")

    # === 時間・終了判定の制御パラメータ ===
    max_time_p = kwargs.pop("max_time_p", 10.0)             # 最大物理時間 [s]
    ramp_time_p = kwargs.pop("ramp_time_p", 1.0)            # ソフトスタート時間 [s]
    steady_window_p = kwargs.pop("steady_window_p", 1.0)    # 定常判定の比較ウィンドウ [s]
    steady_tolerance = kwargs.pop("steady_tolerance", 0.01) # 定常判定の許容誤差 (U_inlet比)
    steady_extra_p = kwargs.pop("steady_extra_p", 2.0)      # 定常検知後、回し続ける時間 [s]

    out_dir = kwargs.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kwargs["filename"] = kwargs.get("filename", os.path.join(out_dir, f"{benchmark}_{timestamp}.gif"))

    cfg = SimConfig(**kwargs)
    ctx = SimulationContext(cfg.nx, cfg.ny, cfg.nz, cfg.n_particles)
    ctx.set_materials(cfg.get_materials_dict())

    geo = GeometryBuilder()
    if benchmark == "parallel_plates":
        geo.build_parallel_plates(ctx)
        geo.set_inlet_outlet(ctx)
    elif benchmark == "cavity":
        geo.build_lid_driven_cavity(ctx)
    elif benchmark == "cylinder":
        geo.build_benchmark_cylinder(ctx)
        geo.set_inlet_outlet(ctx)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    sim = LBMSimulator(cfg)
    bc = BoundaryConditionManager(sim.d3q19, cfg)
    analytics = Analytics(sim.d3q19, cfg)
    sim.init_fields(ctx)

    detector = SteadyStateDetector(steady_window_p, steady_tolerance, steady_extra_p, cfg.U_inlet_p)

    frames =[]
    print(f"\n--- Starting Benchmark: {benchmark.upper()} ---")
    print(f"Grid: {cfg.nx}x{cfg.ny}x{cfg.nz} | Max Time: {max_time_p:.2f} s | dt: {cfg.dt:.6e} s")
    print(f"[Properties] nu: {cfg.nu_p:.2e}, k: {cfg.k_p:.3f}, BC: {'T' if cfg.bc_type=='T' else 'q'}\n")
    
    instance_attributes = [attr for attr in dir(cfg) if not attr.startswith('__') and not callable(getattr(cfg, attr))]
    for instance_attribute in instance_attributes:
        instance_value = getattr(cfg, instance_attribute)
        print(instance_attribute,": ",instance_value)

    vti_enabled = getattr(cfg, "vti_export_interval", 0) > 0 and export_step is not None

    step = 0
    current_time_p = 0.0
    start_wall_time = time.time()

    while current_time_p < max_time_p:
        
        # ==================================================
        # 【追加】Phase 2: 結合係数 G のソフトスタート (キャビテーション誘発)
        # 最初の数秒は単相流で定常化させ、波が消えてから G を 0 から -5.0 へ滑らかに下げる
        # ==================================================
        phase_start_time = 4.0 # キャビテーション開始時間 (例: 4秒後から)
        phase_ramp_time  = 2.0 # 相分離を強める期間 (例: 2秒かけてGを下げる)
        G_target = -5.0        # Shan-Chenモデルで明確な気液分離が起きる臨界値
        
        if current_time_p < phase_start_time:
            G_current = 0.0
        elif current_time_p < phase_start_time + phase_ramp_time:
            progress = (current_time_p - phase_start_time) / phase_ramp_time
            G_current = G_target * 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            G_current = G_target
            
        # 衝突の直前に力を計算し、流速をシフトさせる
        sim.calc_interaction_force(ctx, G_current)
        # ==================================================

        sim.collide(ctx)
        sim.stream_and_bc(ctx)
        
        # ソフトスタート係数
        if ramp_time_p > 0.0 and current_time_p < ramp_time_p:
            ramp_factor = 0.5 * (1.0 - math.cos(math.pi * current_time_p / ramp_time_p))
        else:
            ramp_factor = 1.0
        
        for target_id, vel in cfg.bc_velocity_by_id.items():
            bc.apply_velocity_bc(ctx, target_id, vel * ramp_factor, rho_val=1.0, temp_val=0.0)
        for target_id in cfg.bc_outlet_ids:
            bc.apply_outlet_bc(ctx, target_id)
            
        sim.update_macro(ctx)

        for target_id, flux in cfg.bc_heat_flux_by_id.items():
            bc.apply_heat_flux_bc(ctx, target_id, flux)

        sim.move_particles(ctx, cfg.particles_inject_per_step)

        # 解析・ログ出力・ETA・終了判定
        if step % cfg.vis_interval == 0:
            monitor_val = 0.0
            monitor_name = ""

            if benchmark == "parallel_plates":
                k_target = int(cfg.nz * 0.1)
                monitor_val = analytics.get_local_Nu(ctx, k_target)
                monitor_name = f"Nu({k_target})"
            elif benchmark == "cavity":
                v_np = ctx.v.to_numpy()
                cx, cy, cz = cfg.nx//2, cfg.ny//2, cfg.nz//2
                monitor_val = (v_np[cx, cy, cz, 0]**2 + v_np[cx, cy, cz, 2]**2)**0.5
                monitor_name = "Center|V|"
            elif benchmark == "cylinder":
                v_np = ctx.v.to_numpy()
                ox, oy, oz = int(cfg.nx*0.6), cfg.ny//2, int(cfg.nz*0.25)
                monitor_val = v_np[ox, oy, oz, 0]
                monitor_name = "Wake(X)V"

            just_steady = detector.update(current_time_p, monitor_val)
            if just_steady:
                print(f"\n[SUCCESS] Steady state (or periodic) detected at t = {current_time_p:.3f} s!")
                print(f"[INFO] Simulation will continue for another {steady_extra_p:.3f} s to record data.\n")

            # ETA (残り予測時間) の計算
            elapsed_wall = time.time() - start_wall_time
            target_time_p = max_time_p
            if detector.is_steady:
                target_time_p = min(max_time_p, detector.steady_detected_time_p + detector.extra_time_p)
                
            rem_time_p = target_time_p - current_time_p
            eta_sec = (elapsed_wall / (current_time_p + 1e-12)) * rem_time_p
            eta_str = format_eta(eta_sec)

            print(f"Step {step:6d} | t: {current_time_p:.3f}s / {target_time_p:.3f}s | ETA: {eta_str} | {monitor_name}: {monitor_val:+.4f}")

            canvas = build_vis_frame(ctx, cfg)
            frames.append(canvas)

        if vti_enabled and step % cfg.vti_export_interval == 0:
            export_step(ctx, step, cfg.vti_path_template, dx=cfg.dx)

        if detector.should_stop(current_time_p):
            print(f"\n[INFO] Target extra time reached. Stopping simulation gracefully at t = {current_time_p:.3f} s.")
            break

        current_time_p += cfg.dt
        step += 1

    # ==========================================================
    # 【追加】定常状態の最終データ（物性値＆フィールド）を保存
    # ==========================================================
    print("\n[INFO] Saving final steady-state fields and properties...")
    
    # 1. メタデータ（物性値や計算パラメータ）の保存 (JSON形式)
    metadata = {
        "benchmark": benchmark,
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
        "dx": cfg.dx, "dt": cfg.dt,
        "Lx_p": cfg.Lx_p, "U_inlet_p": cfg.U_inlet_p,
        "nu_p": cfg.nu_p, "k_p": cfg.k_p, "Pr": cfg.Pr,
        "T_inlet_p": cfg.T_inlet_p, "T_wall_p": cfg.T_wall_p,
        "rho_p": cfg.rho_p, "Cp_p": cfg.Cp_p,
        "final_time_p": current_time_p,
        "final_step": step
    }
    meta_path = os.path.join(out_dir, f"{benchmark}_{timestamp}_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    # 2. 3Dフィールドデータ（速度・温度・密度・セルID）の保存 (NumPy圧縮形式: .npz)
    npz_path = os.path.join(out_dir, f"{benchmark}_{timestamp}_fields.npz")
    np.savez_compressed(
        npz_path,
        v=ctx.v.to_numpy(),       # 速度ベクトル (nx, ny, nz, 3)
        temp=ctx.temp.to_numpy(), # 温度 (nx, ny, nz)
        rho=ctx.rho.to_numpy(),   # 密度 (nx, ny, nz)
        cell_id=ctx.cell_id.to_numpy() # 領域ID (nx, ny, nz)
    )
    
    # 3. 最終状態をParaView確認用のVTIとしても強制出力
    if export_step is not None:
        final_vti_path = os.path.join(out_dir, f"{benchmark}_{timestamp}_final.vti")
        # 既存のvti出力関数を流用して最終フレームを別名で保存
        export_step(ctx, step, final_vti_path, dx=cfg.dx)

    print(f" -> Metadata saved to: {meta_path}")
    print(f" -> Fields saved to: {npz_path}")
    print(f" -> Final VTI saved to: {final_vti_path}\n")
    # ==========================================================

    export_gif_frames(frames, cfg.filename, fps=12)
    print(f"Finished Benchmark: {benchmark}. Saved as {cfg.filename}")
    return cfg.filename



if __name__ == "__main__":
    # ==========================================================
    # 実行プリセット (物理時間ベース)
    # ==========================================================

    """
    # 【検証1】 平行平板チャネル (Nu=7.54の確認)
    run_simulation(
        benchmark="parallel_plates",
        nx=64, ny=64, nz=512,
        Lx_p=0.05, U_inlet_p=0.05, nu_p=2.5e-5, k_p=0.026,
        max_time_p=30.0,            # タイムアウト時間 (s)
        ramp_time_p=1.0,            # ソフトスタート時間 (s)
        steady_window_p=2.5,        # 判定ウィンドウ
        steady_tolerance=0.01,      # 許容誤差 (U_inlet比)
        steady_extra_p=2.0,         # 定常検知後に回す時間
        vis_interval=100, bc_type="q",
        bc_velocity_by_id={INLET:[0.0, 0.0, -0.05]},
        bc_outlet_ids=[OUTLET],
        vti_export_interval=0, particles_inject_per_step=0
    )

    
    # 【検証2】 リッド・ドリブン・キャビティ (Re=400)
    run_simulation(
        benchmark="cavity",
        nx=128, ny=4, nz=128,
        Lx_p=0.1, U_inlet_p=0.1, nu_p=2.5e-5, k_p=0.026,
        max_time_p=30.0, ramp_time_p=0.0, 
        steady_window_p=2.0, steady_tolerance=0.005, steady_extra_p=1.0,
        vis_interval=200, bc_type="T",
        bc_velocity_by_id={20:[0.1, 0.0, 0.0], 21:[0.0, 0.0, 0.0]}, 
        bc_outlet_ids=[], vti_export_interval=0, particles_inject_per_step=0
    )
    """


    # 【検証3】 円柱周り流れ・カルマン渦 (Re=150)
    run_simulation(
        benchmark="cylinder",
        nx=128, ny=8, nz=512,
        Lx_p=0.1, U_inlet_p=0.1, nu_p=6.66e-6, k_p=0.026,
        max_time_p=30.0, ramp_time_p=2.0,
        steady_window_p=1.5,        # 約2.5周期分を比較
        steady_tolerance=0.02,      # 2%以下の変動になったら定常化
        steady_extra_p=4.0,         # 綺麗な渦を長めに録画する
        vis_interval=100, bc_type="T",
        bc_velocity_by_id={INLET:[0.0, 0.0, -0.1]},
        bc_outlet_ids=[OUTLET],
        vti_export_interval=0, particles_inject_per_step=200
    )
    