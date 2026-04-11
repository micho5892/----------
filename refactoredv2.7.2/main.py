# ==========================================================
# main.py — run_simulation・メインループ（ロガー搭載版）
# ==========================================================
import os
import sys
import time
import math
import logging
import shutil
import importlib
from datetime import datetime
import json
import numpy as np
from pprint import pprint, pformat




#
# lbm_ui_designer から最適化関数を呼び出すためのパス追加
# （refactoredv2.5/main.py から見たプロジェクトルート: 1階層上）
#
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LBM_UI_DIR = os.path.join(_PROJECT_ROOT, "lbm_ui_designer")
if _LBM_UI_DIR not in sys.path:
    sys.path.insert(0, _LBM_UI_DIR)

run_optimize = None
try:
    lbm_opt = importlib.import_module("lbm_param_optimize")
    run_optimize = getattr(lbm_opt, "run_optimize", None)
    
except Exception:
    # sim の実行自体を止めない（optimizer 側の依存が無い場合があるため）
    run_optimize = None


def format_eta(seconds):
    """秒数を h m s の見やすいフォーマットに変換する"""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0: return f"{h}h {m}m {s}s"
    elif m > 0: return f"{m}m {s}s"
    else: return f"{s}s"

# ==========================================================
# ▼ 追加: ロガーのセットアップ関数
# ==========================================================
def setup_logger(log_file_path):
    logger = logging.getLogger("LBM_Sim")
    logger.setLevel(logging.DEBUG)
    
    # 既存のハンドラがあればクリア
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # ファイルハンドラ (DEBUGレベルまで全て出力)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # コンソールハンドラ (コンソールはINFOレベル以上をスッキリ表示)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO) 
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

class SteadyStateDetector:
    """流れの定常化（および周期定常化）を自動検知するクラス"""
    def __init__(self, window_time_p, tolerance, extra_time_p, U_ref, dt):
        self.window_time_p = window_time_p
        self.tolerance = tolerance
        self.extra_time_p = extra_time_p
        self.U_ref = U_ref
        self.dt = dt
        self.history =[]
        self.is_steady = False
        self.steady_detected_time_p = None

    def update(self, current_time_p, val):
        self.history.append((current_time_p, val))
        cutoff_time = current_time_p - 2.0 * self.window_time_p
        while self.history and self.history[0][0] < cutoff_time:
            self.history.pop(0)

        if not self.is_steady and len(self.history) > 10:
            dt_hist = self.history[-1][0] - self.history[0][0]
            if dt_hist >= 1.9 * self.window_time_p:
                mid_time = current_time_p - self.window_time_p
                half1 =[v for t, v in self.history if t < mid_time]
                half2 =[v for t, v in self.history if t >= mid_time]
                
                if half1 and half2:
                    mean1, mean2 = sum(half1) / len(half1), sum(half2) / len(half2)
                    amp1, amp2 = max(half1) - min(half1), max(half2) - min(half2)
                    mean_diff = abs(mean1 - mean2) / (self.U_ref * self.dt + 1e-8)
                    amp_diff = abs(amp1 - amp2) / (self.U_ref * self.dt + 1e-8)

                    if mean_diff < self.tolerance and amp_diff < self.tolerance:
                        self.is_steady = True
                        self.steady_detected_time_p = current_time_p
                        return True
        return False

    def should_stop(self, current_time_p):
        if self.is_steady and self.steady_detected_time_p is not None:
            if current_time_p - self.steady_detected_time_p >= self.extra_time_p:
                return True
        return False


def run_simulation(**kwargs):
    import taichi as ti

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    # Taichi は run_simulation 内で初回のみ初期化（arch は最初の呼び出しで確定）
    _ti_initialized = False

    from context import SimulationContext, INLET, OUTLET
    from config import SimConfig
    from geometry import GeometryBuilder
    from boundary import BoundaryManager
    from solver import LBMSimulator
    from export import build_vis_frame, export_gif_frames
    from analytics import Analytics
    from physics import PhysicsManager
    from data_exporter import DataExporter

    try:
        from vtk_export import export_step
    except ImportError:
        export_step = None


    # global _ti_initialized
    if not _ti_initialized:
        arch = kwargs.pop("arch", "gpu")
        ti.init(
            arch=ti.cpu if arch == "cpu" else ti.gpu,
            device_memory_fraction=1.0
        )
        _ti_initialized = True

    benchmark = kwargs.pop("benchmark", "cylinder")
    state = kwargs.pop("state", None)
    kwargs["benchmark_name"] = benchmark

    # kwargsからエクスポート設定を取得
    data_export_interval = kwargs.pop("data_export_interval", 0)
    data_export_start_p = kwargs.pop("data_export_start_p", 5.0)

    max_time_p = kwargs.pop("max_time_p", 10.0)             
    ramp_time_p = kwargs.pop("ramp_time_p", 1.0)            
    steady_detection = kwargs.pop("steady_detection", True)  # True: 定常検知で早期終了, False: max_time_p までのみ実行
    steady_window_p = kwargs.pop("steady_window_p", 1.0)    
    steady_tolerance = kwargs.pop("steady_tolerance", 0.001) 
    steady_extra_p = kwargs.pop("steady_extra_p", 2.0)      

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = kwargs.get("out_dir", f"results/{benchmark}_{timestamp}")
    vti_dir = kwargs.get("vti_dir", f"results/{benchmark}_{timestamp}/vti")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vti_dir, exist_ok=True)
    
    # ファイルのパス設定
    gif_path = kwargs.get("filename", os.path.join(out_dir, f"{benchmark}_{timestamp}.gif"))
    kwargs["filename"] = gif_path
    log_path = os.path.join(out_dir, f"{benchmark}_{timestamp}.log")
    
    # ロガーの起動
    logger = setup_logger(log_path)

    cfg = SimConfig(**kwargs)
    import config as config_mod
    config_mod.TI_FLOAT = cfg.fp_dtype
    ctx = SimulationContext(cfg.nx, cfg.ny, cfg.nz, cfg.n_particles, cfg.fp_dtype)
    ctx.set_materials(cfg.get_materials_dict())

    # DataExporterの初期化 (保存間隔が設定されていれば)
    exporter = None
    if data_export_interval > 0:
        export_dir = os.path.join(out_dir, "training_data/run_01")
        exporter = DataExporter(ctx, cfg, output_dir=export_dir)

    geo = GeometryBuilder()
    
    # "benchmark" の文字列から、呼び出すメソッド名を自動生成 (例: "build_rotating_hollow_cylinder")
    build_method_name = f"build_{benchmark}"
    
    # GeometryBuilderの中に該当するメソッドが存在するかチェックして実行
    if hasattr(geo, build_method_name):
        build_method = getattr(geo, build_method_name)
        build_method(ctx)
    else:
        raise ValueError(f"❌ エラー: '{build_method_name}' という形状関数が geometry.py に見つかりません！")

    sim = LBMSimulator(cfg)
    bc_manager = BoundaryManager(sim.d3q19, cfg)
    physics_manager = PhysicsManager(sim.d3q19, cfg)
    analytics = Analytics(sim.d3q19, cfg)
    sim.init_fields(ctx)
    logger.info(
        f"Nu model: {analytics.nu_model_name} | L_ref={analytics.nu_l_ref_p:.6e} | k_ref_mode={analytics.nu_k_ref_mode}"
    )

    if steady_detection:
        # 速度用の判定器 (基準は入口流速)
        detector_v = SteadyStateDetector(steady_window_p, steady_tolerance, steady_extra_p, cfg.U_inlet_p, cfg.dt)
        # 温度用の判定器 (基準は入口の最大温度差: 約1.0)
        detector_t = SteadyStateDetector(steady_window_p, steady_tolerance, steady_extra_p, 1.0, cfg.dt)
    else:
        detector_v = None
        detector_t = None

    # 両方が定常になった瞬間の時間を記録する変数
    global_steady_time_p = None

    frames =[]
    logger.info(f"--- Starting Benchmark: {benchmark.upper()} ---")
    logger.info(f"State: {pformat(state)}")
    logger.info(f"Grid: {cfg.nx}x{cfg.ny}x{cfg.nz} | Max Time: {max_time_p:.2f} s | dt: {cfg.dt:.6e} s")
    if steady_detection:
        logger.info("Steady-state detection: ON (will stop after steady + extra time, or at max_time_p)")
    else:
        logger.info("Steady-state detection: OFF (will run until max_time_p only)")
    
    instance_attributes =[attr for attr in dir(cfg) if not attr.startswith('__') and not callable(getattr(cfg, attr))]
    for instance_attribute in instance_attributes:
        instance_value = getattr(cfg, instance_attribute)
        logger.info(f"{instance_attribute}: {instance_value}")

    vti_enabled = getattr(cfg, "vti_export_interval", 0) > 0 and export_step is not None

    step = 0
    current_time_p = 0.0
    start_wall_time = time.time()

    while current_time_p < max_time_p:
        if ramp_time_p > 0.0 and current_time_p < ramp_time_p:
            ramp_factor = 0.5 * (1.0 - math.cos(math.pi * current_time_p / ramp_time_p))
        else:
            ramp_factor = 1.0

        physics_manager.apply_all(ctx, current_time_p)

        # 回転壁にも soft-start を掛けるため ramp_factor を渡す
        sim.collide_and_stream(ctx, ramp_factor)
        
        bc_manager.apply_all_before_macro(ctx, ramp_factor)
        sim.update_macro(ctx)
        bc_manager.apply_all_after_macro(ctx)
        sim.move_particles(ctx, cfg.particles_inject_per_step)

        if step % cfg.vis_interval == 0:
            v_np = ctx.v.to_numpy()
            temp_np = ctx.temp.to_numpy()
            rho_np = ctx.rho.to_numpy() # 追加
            
            # 異常値の座標を取得
            max_temp_idx = np.unravel_index(np.argmax(temp_np), temp_np.shape)
            min_rho_idx = np.unravel_index(np.argmin(rho_np), rho_np.shape)
            v_mag = np.linalg.norm(v_np, axis=-1)
            
            # デバッグログ（ターミナルには出ず、.logファイルにのみ記録されます。見たい場合は logger.info に変更）
            logger.debug(f"V_max: {v_mag.max():.4e}, V_mean: {v_mag.mean():.4e}")
            logger.debug(f"T_max: {temp_np.max():.4f}, T_min: {temp_np.min():.4f}, T_mean: {temp_np.mean():.4f}")
            logger.debug(f"T_max: {temp_np.max():.4f} at {max_temp_idx}, T_min: {temp_np.min():.4f}")
            logger.debug(f"Rho_min: {rho_np.min():.4f} at {min_rho_idx}")
            if np.isnan(v_mag).any() or np.isnan(temp_np).any():
                logger.error("NaN (計算発散) が発生しています！シミュレーションを強制終了します。")
                break # 発散したらループを抜ける

            monitor_val_v = 0.0
            monitor_val_t = 0.0
            monitor_name = ""

            if benchmark == "parallel_plates":
                k_target = int(cfg.nz * 0.1)
                monitor_val_v = analytics.get_local_Nu(ctx, k_target)
                monitor_val_t = temp_np[cfg.nx//2, cfg.ny//2, k_target]
                monitor_name = f"Nu={monitor_val_v:.2f} | T={monitor_val_t:.3f}"
            elif benchmark == "cavity":
                cx, cy, cz = cfg.nx//2, cfg.ny//2, cfg.nz//2
                monitor_val_v = (v_np[cx, cy, cz, 0]**2 + v_np[cx, cy, cz, 2]**2)**0.5
                monitor_val_t = temp_np[cx, cy, cz]
                monitor_name = f"V_cen={monitor_val_v:.4f} | T_cen={monitor_val_t:.3f}"
            elif benchmark == "cylinder":
                ox, oy, oz = int(cfg.nx*0.6), cfg.ny//2, int(cfg.nz*0.25)
                monitor_val_v = v_np[ox, oy, oz, 0]
                monitor_val_t = temp_np[ox, oy, oz]
                monitor_name = f"V_wake={monitor_val_v:.4f} | T_wake={monitor_val_t:.3f}"
            elif benchmark == "heat_exchanger":
                global_hash = analytics.compute_global_hash(ctx)
                monitor_val_v = global_hash[0]  # 全運動エネルギー (速度のハッシュ)
                monitor_val_t = global_hash[1]  # 全熱エネルギー (温度のハッシュ)

                monitor_name = f"KE={monitor_val_v:.2e} | TE={monitor_val_t:.2e}"
            elif benchmark == "rotating_cylinder":
                pass


            if detector_v is not None and detector_t is not None:
                detector_v.update(current_time_p, monitor_val_v)
                detector_t.update(current_time_p, monitor_val_t)
                
                # 速度と温度の【両方】が定常状態に達したかをチェック
                if detector_v.is_steady and detector_t.is_steady and global_steady_time_p is None:
                    global_steady_time_p = current_time_p
                    logger.info(f"[SUCCESS] Flow AND Thermal steady state detected at t = {current_time_p:.3f} s!")
                    logger.info(f"[INFO] Simulation will continue for another {steady_extra_p:.3f} s to record data.")

            elapsed_wall = time.time() - start_wall_time
            target_time_p = max_time_p
            if global_steady_time_p is not None:
                target_time_p = min(max_time_p, global_steady_time_p + steady_extra_p)
                
            rem_time_p = target_time_p - current_time_p
            eta_sec = (elapsed_wall / (current_time_p + 1e-12)) * rem_time_p
            eta_str = format_eta(eta_sec)

            # ステータスの表示 (速度と温度の判定状況を V:OK T:Wait のように表示)
            status_v = "OK" if (detector_v and detector_v.is_steady) else "Wait"
            status_t = "OK" if (detector_t and detector_t.is_steady) else "Wait"
            status_str = f"[V:{status_v} T:{status_t}]" if detector_v else "[MaxTime]"

            # ==========================================================
            # ★追加：カルマン渦検証用のポイントプローブ (時系列データの保存)
            # ==========================================================
            if benchmark == "benchmark_cylinder":
                # 円柱(X中央, Z=75%)から、直径(nx*0.1)の約2倍だけ風下(Zのマイナス方向)に離れた位置
                probe_x = int(cfg.nx * 0.6)  # 中心から少し横にズラす(渦の中心を通るように)
                probe_y = cfg.ny // 2
                probe_z = int(cfg.nz * 0.75 - cfg.nx * 0.2)
                
                probe_vx = v_np[probe_x, probe_y, probe_z, 0] # X方向の横揺れ速度
                
                probe_file = os.path.join(out_dir, "probe.csv")
                if not os.path.exists(probe_file):
                    with open(probe_file, "w") as f:
                        f.write("time,vx\n")
                with open(probe_file, "a") as f:
                    f.write(f"{current_time_p},{probe_vx:.6e}\n")

            logger.info(f"Step {step:6d} | t: {current_time_p:.3f}s / {target_time_p:.3f}s | ETA: {eta_str} | {status_str} {monitor_name}")

            canvas = build_vis_frame(ctx, cfg)
            frames.append(canvas)


        if exporter is not None:
            # 指定された時間が経過してから（流れが発達してから）保存を開始
            if current_time_p >= data_export_start_p:
                if step % data_export_interval == 0:
                    exporter.export_snapshot(step, current_time_p)

        if vti_enabled and step % cfg.vti_export_interval == 0:
            vti_path = os.path.join(vti_dir, 'step_{:06d}.vti')
            export_step(ctx, step, vti_path, dx=cfg.dx)

        if global_steady_time_p is not None:
            if current_time_p - global_steady_time_p >= steady_extra_p:
                logger.info(f"[INFO] Target extra time reached. Stopping simulation gracefully at t = {current_time_p:.3f} s.")
                break

        

        current_time_p += cfg.dt
        step += 1

    logger.info("Saving final steady-state fields and properties...")
    
    metadata = {
        "benchmark": benchmark,
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
        "dx": cfg.dx, "dt": cfg.dt,
        "Lx_p": cfg.Lx_p, "U_inlet_p": cfg.U_inlet_p,
        "domain_properties": cfg.domain_properties, 
        "final_time_p": current_time_p,
        "final_step": step
    }
    meta_path = os.path.join(out_dir, f"{benchmark}_{timestamp}_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    npz_path = os.path.join(out_dir, f"{benchmark}_{timestamp}_fields.npz")
    np.savez_compressed(
        npz_path,
        v=ctx.v.to_numpy(),
        temp=ctx.temp.to_numpy(),
        rho=ctx.rho.to_numpy(),
        cell_id=ctx.cell_id.to_numpy()
    )
    
    if export_step is not None:
        final_vti_path = os.path.join(out_dir, f"{benchmark}_{timestamp}_final.vti")
        export_step(ctx, step, final_vti_path, dx=cfg.dx)

    logger.info(f" -> Metadata saved to: {meta_path}")
    logger.info(f" -> Fields saved to: {npz_path}")
    if export_step is not None:
        logger.info(f" -> Final VTI saved to: {final_vti_path}")

    export_gif_frames(frames, cfg.filename, fps=int(1/(cfg.vis_interval*cfg.dt)))
    logger.info(f"Finished Benchmark: {benchmark}. Saved as {cfg.filename}")

    # 出力ディレクトリ全体を zip 圧縮
    try:
        base_name = os.path.abspath(out_dir)
        zip_path = shutil.make_archive(base_name, 'zip', root_dir=out_dir)
        logger.info(f" -> Output directory zipped to: {zip_path}")
    except Exception as e:
        logger.error(f"Failed to create zip archive for {out_dir}: {e}")

    return current_time_p < max_time_p


if __name__ == "__main__":

    """
    run_simulation(
        benchmark="heat_exchanger",
        fp_dtype="float32",
        steady_detection=True,
        nx=166, ny=166, nz=664,
        Lx_p=1.0000e-01, 
        
        U_inlet_p=7.8749e-01,
        max_time_p=30.0, ramp_time_p=2.0,
        vis_interval=100, vti_export_interval=500,
        
        # ▼ Auto-Designed Similarity Parameters
        # Target Re=5000.0, Pr_sim=1.00, Cr=2912.2894
        domain_properties={
            0:  {"nu": 1.57e-05, "k": 0.026, "rho": 1.2, "Cp": 1423.31},
            22: {"nu": 1.57e-05, "k": 0.026, "rho": 1.2, "Cp": 1423.31},
            23: {"nu": 1.57e-05, "k": 0.026, "rho": 1.2, "Cp": 1423.31},
            
            2:  {"nu": 1.57e-05, "k": 0.026, "rho": 1.2, "Cp": 1423.31},
            24: {"nu": 1.57e-05, "k": 0.026, "rho": 1.2, "Cp": 1423.31},
            25: {"nu": 1.57e-05, "k": 0.026, "rho": 1.2, "Cp": 1423.31},
            
            10: {"nu": 0.0, "k": 400.000, "rho": 8960.0, "Cp": 544.51}
        },
        
        boundary_conditions={
            22: {"type": "inlet",  "velocity":[0.0, 0.0,  0.1000], "temperature": 1.0},
            24: {"type": "inlet",  "velocity":[0.0, 0.0, -0.0500], "temperature": 0.0},
            23: {"type": "outlet"},
            25: {"type": "outlet"},
            10: {"type": "adiabatic_wall"}, 
        },
        flow_type="counter"
    )
    """

    """
    run_simulation(
        benchmark="heat_exchanger",
        fp_dtype="float32",
        steady_detection=True,
        nx=161, ny=161, nz=644,  # 推奨アスペクト比 1:1:4
        Lx_p=0.1,

        U_inlet_p=3.00000e-01,
        max_time_p=30.0, ramp_time_p=2.0,
        vis_interval=100, vti_export_interval=500,

        # =========================================================
        # ▼ Auto-Designed Similarity Parameters
        # Target Re=3000.0, Pr_sim_in=2.58
        # Result conversion: Nu_real = Nu_sim * (6.97 / 2.58)^(1/3)
        # =========================================================
        domain_properties={
            # 【Internal Fluid: Water】 ID: 0(Mid), 22(In), 23(Out)
            0:  {"nu": 1.00e-06, "k": 0.6, "rho": 1000.0, "Cp": 1545.6},        
            22: {"nu": 1.00e-06, "k": 0.6, "rho": 1000.0, "Cp": 1545.6},        
            23: {"nu": 1.00e-06, "k": 0.6, "rho": 1000.0, "Cp": 1545.6},        

            # 【External Fluid: Water】 ID: 2(Mid), 24(In), 25(Out)
            2:  {"nu": 1.00e-06, "k": 0.6, "rho": 1000.0, "Cp": 772.8},
            24: {"nu": 1.00e-06, "k": 0.6, "rho": 1000.0, "Cp": 772.8},
            25: {"nu": 1.00e-06, "k": 0.6, "rho": 1000.0, "Cp": 772.8},

            # 【Solid Tube: Copper】 ID: 10
            10: {"nu": 0.0, "k": 400.0, "rho": 8960.0, "Cp": 1533.3}
        },

        boundary_conditions={
            22: {"type": "inlet",  "velocity":[0.0, 0.0,  0.16], "temperature": 1.0},
            24: {"type": "inlet",  "velocity":[0.0, 0.0, -0.08], "temperature": 0.0},
            23: {"type": "outlet"},
            25: {"type": "outlet"},
            10: {"type": "adiabatic_wall"},
        },
        flow_type="counter"
    )

    """
    
    """
    # 【検証5】 マグヌス効果 (回転円柱とカルマン渦の干渉)
    run_simulation(
        benchmark="rotating_cylinder",
        fp_dtype="float32",
        steady_detection=False,
        nx=128, ny=8, nz=512,  # 2D的な流れを観察するためY方向は薄く
        Lx_p=0.1, 
        
        U_inlet_p=0.1,     
        rho_p=1.2,          # 空気を想定
        Cp_p=1005.0,
        
        max_time_p=30.0, ramp_time_p=1.0,steady_window_p=4.0, steady_tolerance=0.001, steady_extra_p=4.0,
        vis_interval=100, vti_export_interval=0, particles_inject_per_step=200,
        
        # ▼ 回転の設定 (omega = 0.015 は、壁の表面速度が風の速度とほぼ同じになる黄金比)
        omega_cylinder=0.0, 
        cylinder_center=[128 * 0.5, 4.0, 512 * 0.75],
        
        domain_properties={
            0:  {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
            20: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0}, # INLET
            21: {"nu": 1.5e-5, "k": 0.026, "rho": 1.2, "Cp": 1005.0}, # OUTLET
            30: {"nu": 0.0,    "k": 10.0,  "rho": 8000.0, "Cp": 500.0}, # ROTATING_WALL
        },
        
        boundary_conditions={
            # 上空から下(Z=0)に向かって風が吹く
            20: {"type": "inlet",  "velocity":[0.0, 0.0, -0.1], "temperature": 0},
            21: {"type": "outlet"},
            30: {"type": "isothermal_wall", "temperature": 1.5}, 
        }
    )
    
    
    
    # 【検証6】 回転する中空円筒 (Z軸回転)
    run_simulation(
        benchmark="rotating_hollow_cylinder",
        fp_dtype="float32",
        steady_detection=False,
        nx=166, ny=166, nz=664,
        Lx_p=0.1, 
        
        U_inlet_p=7.8434e-02,     # ★ 0.1 だと壁への衝突で発散しやすいため、最初は 0.05 を推奨
        
        max_time_p=15.0, ramp_time_p=2.0,
        vis_interval=100, vti_export_interval=300, particles_inject_per_step=300,
        
        omega_cylinder_phys=0.02, 
        cylinder_center=[166 * 0.5, 166 * 0.5, 664 * 0.35],
        rotation_axis='z',
        
        domain_properties={
            0:  {"nu": 1.57e-05, "k": 0.385, "rho": 1.2, "Cp": 29637.26},
            10: {"nu": 0.0, "k": 1120.000, "rho": 8960.0, "Cp": 11338.08},      # ★追加: 外側の静止壁(風洞)
            20: {"nu": 1.57e-05, "k": 0.385, "rho": 1.2, "Cp": 29637.26}, 
            21: {"nu": 1.57e-05, "k": 0.385, "rho": 1.2, "Cp": 29637.26}, 
            30: {"nu": 0.0, "k": 1120.000, "rho": 8960.0, "Cp": 11338.08}, 
        },
        
        boundary_conditions={
            10: {"type": "adiabatic_wall"}, # ★追加: 外壁の境界条件
            20: {"type": "inlet",  "velocity":[0.0, 0.0, 0.1], "temperature": 1}, # ★ U_inlet_p に合わせて 0.05 に変更
            21: {"type": "outlet"},
            30: {"type": "adiabatic_wall"}, 
        }
    )
    """
    def test_rotating_hollow_cylinder():
        # ====================================================================
        # ⑦ 開放系の回転中空円筒 (Rotating Hollow Cylinder)
        # 穴の向きと風の向きが同じ(Z軸)。外壁を持たない開放系での、
        # 回転による遠心力と、エッジからの複雑な剥離渦を学習します。
        # ====================================================================
        for i in[0.03, 0.025, 0.02]:
            print(f"\n=== Optimizing for Rotating Hollow Cylinder (tau_margin = {i}) ===")
            
            config_hollow = {
                "fluid": "Air",
                "temperature_K": 300,
                "pressure_Pa": 101325,
                "solid": "Copper",
                "fix_cr": True,

                "fixed":{
                    "nx": 224,
                    "nu": True,
                    "k_f": True,
                    "rho_f": True,
                    "k_s": True,
                    "L_domain": 0.1,
                    # ★円筒の外径(134.4セル)を物理長に換算
                    "L_ref": 0.06, 
                    "u_lbm": 0.10, # 回転速度が上乗せされるため、安全な0.10に抑える
                    "U": i*10
                },

                "ranges":{
                    "Re":{
                        "min": 10,     # 代表長さが大きいのでReも巨大になります
                        "max": 4000
                    },
                    "tau_f マージン":{
                        "min": 0.015, 
                        "max": 2.00
                    }, 
                    "tau_gf マージン":{
                        "min": 0.015,
                        "max": 2.00
                    }, 
                    "tau_gs マージン":{
                        "min": 0.015,
                        "max": 2.00
                    },
                },
                "targets": {
                    # "Re": {"value": 100.0, "weight": 0.5},
                    "tau_f マージン": {"value": 0.03, "weight": 1.0},
                },
                "target_regularization": 1, # 目標値への適合を促す
                "regularization": 1.0e-3,
                "maxiter": 30000,
            }

            result_hollow = run_optimize(config_hollow)

            if result_hollow["success"] == True:
                try:
                    result_run = run_simulation(
                        benchmark="ai_training_rotating_hollow",
                        fp_dtype="float32",
                        steady_detection=False,
                        state = result_hollow["state"],
                        
                        nx=int(result_hollow["state"]["nx"]), ny=112, nz=728,  
                        Lx_p=0.1,                
                        U_inlet_p=result_hollow["state"]["U"],
                
                        # ▼ 回転速度 (壁の表面速度が風速の20%程度になるように設定)
                        omega_cylinder_phys=3.14, 
                        cylinder_center=[112.0, 56.0, 728 * 0.75], # 形状定義に合わせた中心
                        rotation_axis='z', # Z軸周りの回転
                        
                        max_time_p=15.0, 
                        ramp_time_p=2.0,

                        vis_interval=100, 
                        vti_export_interval=500, particles_inject_per_step=0,
                        
                        data_export_interval=0,
                        data_export_start_p=0,
                        
                        domain_properties={
                            0:  {"nu": result_hollow["state"]["nu"], "k": result_hollow["state"]["k_f"], "rho": result_hollow["state"]["rho_f"], "Cp": result_hollow["state"]["Cp_f"]},
                            20: {"nu": result_hollow["state"]["nu"], "k": result_hollow["state"]["k_f"], "rho": result_hollow["state"]["rho_f"], "Cp": result_hollow["state"]["Cp_f"]}, 
                            21: {"nu": result_hollow["state"]["nu"], "k": result_hollow["state"]["k_f"], "rho": result_hollow["state"]["rho_f"], "Cp": result_hollow["state"]["Cp_f"]}, 
                            30: {"nu": 0.0, "k": result_hollow["state"]["k_s"], "rho": result_hollow["state"]["rho_s"], "Cp": result_hollow["state"]["Cp_s"]}, 
                        },
                        
                        boundary_conditions={
                            20: {"type": "inlet",  "velocity":[0.0, 0.0, -result_hollow["state"]["u_lbm"]], "temperature": 0.0},
                            21: {"type": "outlet"},
                            30: {"type": "isothermal_wall", "temperature": 1.0}, # 回転する熱源
                        }
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                if False == result_run:
                    break 
            else:
                print(f"Failed to optimize parameters for tau_margin={i}")
 
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
    

    
    # 【検証4】 二重管熱交換器 (向流・並流の切り替え可能)
    # 内部流体と外部流体はひとまず水相当。物性は nu_p, k_p 等で調整。
    # 【改訂版】 汎用性を高めた run_simulation の引数設計
    run_simulation(
        benchmark="heat_exchanger",
        fp_dtype="float32",
        nx=64, ny=64, nz=256,
        Lx_p=0.1, 
        
        # 1. 物性の分離（流体と固体で異なる熱伝導率を設定可能にする）
        nu_fluid=1.0e-5,    # 流体の動粘度
        k_fluid=0.02,       # 流体の熱伝導率
        k_solid=0.50,       # 固体の熱伝導率（チューブ壁。流体より熱を通しやすい例）
        
        # 2. シミュレーション制御
        max_time_p=30.0, ramp_time_p=2.0,
        steady_window_p=1.5, steady_tolerance=0.01, steady_extra_p=2.0,
        vis_interval=100,
        
        # 3. 領域(ID)ごとの役割と境界条件の明示的定義（ディクショナリ化）
        # これにより、将来 bc_type="q" などのグローバル変数を廃止できます
        boundary_conditions={
            # 内部流体入口 (Velocity & Temperature Dirichlet)
            22: {"type": "inlet",  "velocity":[0.0, 0.0,  0.05], "temperature": 1.0},
            
            # 外部流体入口 (Velocity & Temperature Dirichlet)
            24: {"type": "inlet",  "velocity": [0.0, 0.0, -0.05], "temperature": 0.0},
            
            # 出口 (Neumann / Zero-gradient)
            23: {"type": "outlet"},
            25: {"type": "outlet"},
            
            # 外部ケーシング壁（断熱壁）
            10: {"type": "adiabatic_wall"}, 

            # 【追加テスト】外壁(10)を「温度1.5に保ち続けるヒーター」にする
            # 10: {"type": "isothermal_wall", "temperature": 1.5},

            # もし一定の熱を加え続けるならこう書きます：
            # 10: {"type": "constant_heat_flux", "q": 0.005}, 
            
            # ※ 内部チューブ（SOLID）は共役熱伝達としてLBMソルバが自然に解くため、
            #    境界条件の指定は不要（あるいは type: "conjugate" と明示）
        },

        # ▼ 向流 / 並流 の切り替えフラグ（geometry構築用） ▼
        flow_type="counter", 
        
        vti_export_interval=500, particles_inject_per_step=100
        
    )

    
    もし「Shan-Chenの多相流効果」をオンにしたい場合
    run_simulation(
        benchmark="heat_exchanger",
        # ... 既存の設定 ...
        
        physics_models={
            # プラグインとして多相流を追加！
            # "shan_chen": {"G_target": -5.0, "phase_start_time": 4.0}
            
            # 将来的に、自然対流を追加したくなったら…
            # "boussinesq": {"g_vec":[0.0, -9.8, 0.0], "beta": 0.003, "T_ref": 0.0}
        },
    )
    """
    
    """
    # バックステップ流れは Re=500〜1500あたりで強烈な乱流に遷移します
    for i in[ 0.02, 0.025, 0.03]:
        
        print(f"\n=== Optimizing for Re = {i} ===")
        
        config = {
            "fluid": "Air",
            "temperature_K": 300,
            "pressure_Pa": 101325,
            "solid": "Copper",
            "fix_cr": True,

            "fixed":{
                "nx": 224,
                "nu": True,
                "k_f": True,
                "rho_f": True,
                "k_s": True,
                "L_domain": 0.1,
                # ★段差の高さ 107セルの物理長を指定
                "L_ref": 0.04776785714285714, 
                "u_lbm": 0.15,
            },

            "ranges":{
                "Re":{
                    "min": 500,        
                    "max": 2000
                },
                "tau_f マージン":{
                    "min": i, # 代表長さが大きいので余裕でクリアできます
                    "max": 2.00
                }, 
                "tau_gf マージン":{
                    "min": i,
                    "max": 2.00
                }, 
                "tau_gs マージン":{
                    "min": 0.05,
                    "max": 2.00
                },
            },
            "targets": {
                "Re": 500.0,  # Reを近づけたい
            },
            "target_regularization": 1, # 目標値への適合を促す
            "regularization": 1.0e-3,
            "maxiter": 30000,
        }
        

        result = run_optimize(config)
        if result["success"] == True:

            # 【AI事前学習用】 限界メッシュ・高解像度ハイブリッドデータ生成
            if False == run_simulation(
                benchmark="ai_training_backstep",
                fp_dtype="float32",
                steady_detection=False,
                
                nx = result["state"]["nx"], ny=112, nz=728,  
                Lx_p=0.1,                
                
                # ★修正1：LBMの音速の壁を超えない安全な流速（マッハ数 0.26）
                U_inlet_p = result["state"]["U"],
                
                max_time_p=15.0, 
                ramp_time_p=2.0,
                
                vis_interval=100, 
                vti_export_interval=500, particles_inject_per_step=0,
                
                data_export_interval=200,
                data_export_start_p=5.0,
                
                # ★修正2：流速を落とした分、nu と k を下げて Re=600 / Pr=0.71 を厳密に維持
                domain_properties={
                    0:  {"nu": result["state"]["nu"], "k": result["state"]["k_f"], "rho": result["state"]["rho_f"], "Cp": result["state"]["Cp_f"]},
                    20: {"nu": result["state"]["nu"], "k": result["state"]["k_f"], "rho": result["state"]["rho_f"], "Cp": result["state"]["Cp_f"]}, 
                    21: {"nu": result["state"]["nu"], "k": result["state"]["k_f"], "rho": result["state"]["rho_f"], "Cp": result["state"]["Cp_f"]}, 
                    10: {"nu": 0.0,     "k": result["state"]["k_s"],    "rho": result["state"]["rho_s"], "Cp": result["state"]["Cp_s"]}, 
                },
                
                boundary_conditions={
                    # ★修正3：境界条件の速度も U_inlet_p に合わせる
                    20: {"type": "inlet",  "velocity":[0.0, 0.0, -result["state"]["u_lbm"]], "temperature": 0.0},
                    21: {"type": "outlet"},
                    10: {"type": "isothermal_wall", "temperature": 1.0}, 
                }
            ):
                break
        else:
            print(f"Failed to optimize parameters for {i}")

    """

    def run_rotating_cylinder():
        # 回転円柱は Re=200〜600 程度で非常に美しい非対称な渦(マグヌス効果)を出します
        for i in [0.03, 0.035, 0.04]:
            print(f"\n=== Optimizing for Re = {i} ===")
            
            config = {
                "fluid": "Air",
                "temperature_K": 300,
                "pressure_Pa": 101325,
                "solid": "Copper",
                "fix_cr": True,

                "fixed":{
                    "nx": 224,
                    "nu": True,
                    "k_f": True,
                    "rho_f": True,
                    "k_s": True,
                    "L_domain": 0.1,
                    # ★直径 48セルの物理長
                    "L_ref": 0.0214285714285714, 
                    "u_lbm": 0.10,   # 回転速度が乗るため、0.10 に抑えて安全を確保
                },

                "ranges":{
                    "Re":{
                        "min": 200,        
                        "max": 1000
                    },
                    "tau_f マージン":{
                        "min": i, 
                        "max": 2.00
                    }, 
                    "tau_gf マージン":{
                        "min": i,
                        "max": 2.00
                    }, 
                    "tau_gs マージン":{
                        "min": 0.05,
                        "max": 2.00
                    },
                },
                "targets": {
                    "Re": {"value": 400.0, "weight": 0.5},
                    "tau_f マージン": {"value": i, "weight": 1.0},
                },
                "target_regularization": 1, # 目標値への適合を促す
                "regularization": 1.0e-3,
                "maxiter": 30000,
            }
            

            result = run_optimize(config)

            if result["success"] == True:

                # 【AI事前学習用】 限界メッシュ・高解像度ハイブリッドデータ生成
                if False == run_simulation(
                    benchmark="ai_training_rotating",
                    fp_dtype="float32",
                    steady_detection=False,
                    state = result["state"],
                    
                    nx = result["state"]["nx"], ny=112, nz=728,  
                    Lx_p=0.1,                
                    sponge_thickness = 40.0,
                    
                    # ★修正1：LBMの音速の壁を超えない安全な流速（マッハ数 0.26）
                    U_inlet_p = result["state"]["U"],

                    omega_cylinder_phys=3.14, 
                    cylinder_center=[112.0, 56.0, 546.0],
                    rotation_axis='y',
            
                    max_time_p=15.0, 
                    ramp_time_p=2.0,

                    vis_interval=100, 
                    vti_export_interval=1000, particles_inject_per_step=0,
                    
                    data_export_interval=200,
                    data_export_start_p=5.0,
                    
                    # ★修正2：流速を落とした分、nu と k を下げて Re=600 / Pr=0.71 を厳密に維持
                    domain_properties={
                        0:  {"nu": result["state"]["nu"], "k": result["state"]["k_f"], "rho": result["state"]["rho_f"], "Cp": result["state"]["Cp_f"]},
                        20: {"nu": result["state"]["nu"], "k": result["state"]["k_f"], "rho": result["state"]["rho_f"], "Cp": result["state"]["Cp_f"]}, 
                        21: {"nu": result["state"]["nu"], "k": result["state"]["k_f"], "rho": result["state"]["rho_f"], "Cp": result["state"]["Cp_f"]}, 
                        30: {"nu": 0.0,     "k": result["state"]["k_s"],    "rho": result["state"]["rho_s"], "Cp": result["state"]["Cp_s"]}, 
                    },
                    
                    boundary_conditions={
                        # ★修正3：境界条件の速度も U_inlet_p に合わせる
                        20: {"type": "inlet",  "velocity":[0.0, 0.0, -result["state"]["u_lbm"]], "temperature": 0.0},
                        21: {"type": "outlet"},
                        30: {"type": "isothermal_wall", "temperature": 1.0}, 
                    },

                ):
                    break
            else:
                print(f"Failed to optimize parameters for {i}")

    """
    # ====================================================================
    # ⑤ 傾いた平板 (Inclined Plate) の自動探索シミュレーション
    # 鋭利な角からの剥離と、迎角による翼端渦の生成を学習します。
    # 代表長さが短い(30セル)ため、マージンは厳しめ(小さめ)からスタートします。
    # ====================================================================
    for i in[0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
        print(f"\n=== Optimizing for Inclined Plate (tau_margin = {i}) ===")
        
        config_plate = {
            "fluid": "Air",
            "temperature_K": 300,
            "pressure_Pa": 101325,
            "solid": "Copper",
            "fix_cr": True,

            "fixed":{
                "nx": 224,
                "nu": True,
                "k_f": True,
                "rho_f": True,
                "k_s": True,
                "L_domain": 0.1,
                # ★投影面積 (高さ30セル相当)
                "L_ref": 0.01339285714285714, 
                "u_lbm": 0.12, # 鋭角で増速するため安全な0.12
            },

            "ranges":{
                "Re":{
                    "min": 10,        
                    "max": 1000
                },
                "tau_f マージン":{
                    "min": 0.01, 
                    "max": 2.00
                }, 
                "tau_gf マージン":{
                    "min": 0.01,
                    "max": 2.00
                }, 
                "tau_gs マージン":{
                    "min": 0.05,
                    "max": 2.00
                },
            },
            "targets": {
                "Re": {"value": 100.0, "weight": 0.5},
                "tau_f マージン": {"value": i, "weight": 1.0},
            },
            "target_regularization": 1, # 目標値への適合を促す
            "regularization": 1.0e-3,
            "maxiter": 30000,
        }

        result_plate = run_optimize(config_plate)
        pprint(result_plate)


        if result_plate["success"] == True:
            # 限界メッシュ・高解像度データ生成
            if False == run_simulation(
                benchmark="ai_training_inclined_plate",
                fp_dtype="float32",
                steady_detection=False,
                
                nx=int(result_plate["state"]["nx"]), ny=112, nz=728,  
                Lx_p=0.1,                
                U_inlet_p=result_plate["state"]["U"],
        
                max_time_p=15.0, 
                ramp_time_p=2.0,

                vis_interval=100, 
                vti_export_interval=500, particles_inject_per_step=0,
                
                data_export_interval=200,
                data_export_start_p=5.0,
                
                domain_properties={
                    0:  {"nu": result_plate["state"]["nu"], "k": result_plate["state"]["k_f"], "rho": result_plate["state"]["rho_f"], "Cp": result_plate["state"]["Cp_f"]},
                    20: {"nu": result_plate["state"]["nu"], "k": result_plate["state"]["k_f"], "rho": result_plate["state"]["rho_f"], "Cp": result_plate["state"]["Cp_f"]}, 
                    21: {"nu": result_plate["state"]["nu"], "k": result_plate["state"]["k_f"], "rho": result_plate["state"]["rho_f"], "Cp": result_plate["state"]["Cp_f"]}, 
                    10: {"nu": 0.0, "k": result_plate["state"]["k_s"], "rho": result_plate["state"]["rho_s"], "Cp": result_plate["state"]["Cp_s"]}, 
                },
                
                boundary_conditions={
                    20: {"type": "inlet",  "velocity":[0.0, 0.0, -result_plate["state"]["u_lbm"]], "temperature": 0.0},
                    21: {"type": "outlet"},
                    10: {"type": "isothermal_wall", "temperature": 1.0}, # 板を熱源とする
                }
            ):
                break # 完走したら次の形状へ
        else:
            print(f"Failed to optimize parameters for tau_margin={i}")

    """
    def run_mixed_convection():
        # ====================================================================
        # ⑥ 混合対流 (Mixed Convection) の自動探索シミュレーション
        # 浮力（Boussinesq近似）による自然対流と、強烈な温度勾配を学習します。
        # 発散しやすいため、マージンは少し大きめからスタートします。
        # ====================================================================
        for i in [0.03, 0.04, 0.05, 0.06, 0.07]:

            print(f"\n=== Optimizing for Mixed Convection (tau_margin = {i}) ===")
            
            config_conv = {
                "fluid": "Air",
                "temperature_K": 300,
                "pressure_Pa": 101325,
                "solid": "Copper",
                "fix_cr": True,

                "fixed":{
                    "nx": 224,
                    "nu": True,
                    "k_f": True,
                    "rho_f": True,
                    "k_s": True,
                    "L_domain": 0.1,
                    # ★チャネル幅 (214セル相当)
                    "L_ref": 0.09553571428571428, 
                    "u_lbm": 0.05,   # 浮力による対流を主役にするため、風は極めて弱くする

                },

                "ranges":{
                    "Re":{
                        "min": 10,      # ゆっくりな風なのでReは小さめ
                        "max": 600
                    },
                    "tau_f マージン":{
                        "min": i, 
                        "max": 2.00
                    }, 
                    "tau_gf マージン":{
                        "min": i,
                        "max": 2.00
                    }, 
                    "tau_gs マージン":{
                        "min": 0.05,
                        "max": 2.00
                    },
                },
                "regularization": 1.0e-3,
                "maxiter": 30000,
            }

            result_conv = run_optimize(config_conv)

            if result_conv["success"] == True:
                # 限界メッシュ・高解像度データ生成
                if False == run_simulation(
                    benchmark="ai_training_mixed_convection",
                    fp_dtype="float32",
                    steady_detection=False,
                    state=result_conv["state"],
                    
                    nx=int(result_conv["state"]["nx"]), ny=112, nz=728,  
                    Lx_p=0.1,                
                    U_inlet_p=result_conv["state"]["U"],
            
                    max_time_p=15.0, 
                    ramp_time_p=2.0,

                    vis_interval=100, 
                    vti_export_interval=500, particles_inject_per_step=0,
                    
                    data_export_interval=200,
                    data_export_start_p=5.0,
                    
                    domain_properties={
                        0:  {"nu": result_conv["state"]["nu"], "k": result_conv["state"]["k_f"], "rho": result_conv["state"]["rho_f"], "Cp": result_conv["state"]["Cp_f"]},
                        20: {"nu": result_conv["state"]["nu"], "k": result_conv["state"]["k_f"], "rho": result_conv["state"]["rho_f"], "Cp": result_conv["state"]["Cp_f"]}, 
                        21: {"nu": result_conv["state"]["nu"], "k": result_conv["state"]["k_f"], "rho": result_conv["state"]["rho_f"], "Cp": result_conv["state"]["Cp_f"]}, 
                        10: {"nu": 0.0, "k": result_conv["state"]["k_s"], "rho": result_conv["state"]["rho_s"], "Cp": result_conv["state"]["Cp_s"]}, 
                        11: {"nu": 0.0, "k": result_conv["state"]["k_s"], "rho": result_conv["state"]["rho_s"], "Cp": result_conv["state"]["Cp_s"]}, # 左壁用に追加
                    },
                    
                    # ★追加：Boussinesq近似による浮力プラグイン
                    physics_models={
                        "boussinesq": {"g_vec":[0.0, 0.0, 9.8], "beta": 0.2, "T_ref": 0.0}
                    },
                    
                    boundary_conditions={
                        # ★修正2：弱い風を下向き(-Z方向)に吹き下ろす。
                        # (もし完全な無風の自然対流にしたい場合は [0.0, 0.0, 0.0] にする)
                        20: {"type": "inlet",  "velocity":[0.0, 0.0, -result_conv["state"]["u_lbm"]], "temperature": 0.0},
                        21: {"type": "outlet"},
                        11: {"type": "isothermal_wall", "temperature": 1.0}, # 左壁 (熱源: ここから上へ昇る)
                        10: {"type": "isothermal_wall", "temperature": 0.0}, # 右壁 (冷源: ここから下へ沈む)
                    }
                ):
                    break # 完走したら終了
            else:
                print(f"Failed to optimize parameters for tau_margin={i}")

    run_rotating_cylinder()