# ==========================================================
# main.py — run_simulation・メインループ（ロガー搭載版）
# ==========================================================
import os
import sys
import time
import math
import shutil
import importlib
from datetime import datetime
import json
import numpy as np
from pprint import pprint, pformat

from lbm_logger import configure_logging, get_logger


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

class AsymptoticSteadyDetector:
    """
    系の物理量が漸近値（最終的な安定値）に向かう傾向を線形回帰で予測し、
    定常化までの「残り時間（ETA）」を物理ベースで算出する高度な判定器。
    """
    def __init__(self, window_time_p, dt_sample, tolerance):
        # サンプリング間隔とウィンドウサイズの設定
        self.dt_sample = dt_sample
        # 振動（カルマン渦など）を吸収するため、ウィンドウ内のデータ数は最低でも20は確保
        self.window_size = max(20, int(window_time_p / dt_sample))
        self.tolerance = tolerance
        self.history =[]
        self.is_steady = False

    def update(self, val):
        self.history.append(val)
        # 移動平均のためのバッファも含めて過去履歴を保持
        if len(self.history) > self.window_size * 2:
            self.history.pop(0)

        if len(self.history) < self.window_size * 2:
            return False, "Analyzing..."

        # 1. 移動平均（SMA）をかけて微小な振動ノイズや渦の周期変動を取り除く
        window = self.window_size
        smoothed = np.convolve(self.history, np.ones(window)/window, mode='valid')
        
        # 2. Y(現在値) と dY/dt(変化率) の配列を作成
        Y = smoothed[:-1]
        dY = (smoothed[1:] - smoothed[:-1]) / self.dt_sample

        # 3. 最小二乗法（線形回帰）: dY/dt = a * Y + b
        cov_matrix = np.cov(Y, dY)
        var_Y = cov_matrix[0, 0]
        cov_Y_dY = cov_matrix[0, 1]

        # 分散が極めてゼロに近い場合（すでに完全に変化がない場合）
        if var_Y < 1e-20:
            self.is_steady = True
            return True, "Steady"

        a = cov_Y_dY / var_Y
        b = np.mean(dY) - a * np.mean(Y)

        # 4. 判定ロジック
        if a >= -1e-6:
            # a が正またはゼロに近い = まだ直線的に成長中で予測不可能
            self.is_steady = False
            return False, "Developing..."

        # 漸近予測が有効な場合
        tau = -1.0 / a            # 時定数（収束の速さ）
        Y_inf = -b / a            # 最終的に落ち着く予測値 (漸近値)
        Y_current = smoothed[-1]  # 現在の平滑化値

        # 現在値から漸近値までの残り距離
        diff = abs(Y_current - Y_inf)
        # 許容されるブレ幅（目標値の tolerance %）
        threshold = max(abs(Y_inf) * self.tolerance, 1e-12)

        if diff <= threshold:
            # 許容誤差内に飛び込んだら定常とみなす
            self.is_steady = True
            return True, "Steady"
        
        # 5. 到達までの残り時間（ETA）の逆算
        # 指数関数減衰の式: diff(t) = diff_0 * exp(-t/tau) を t について解く
        ratio = diff / threshold
        eta_p = tau * np.log(ratio)

        return False, f"ETA:{eta_p:.1f}s"

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
    from diagnostics import log_parallel_plates_transport_diagnostics

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
    state = kwargs.pop("state", {})
    kwargs["benchmark_name"] = benchmark

    # kwargsからエクスポート設定を取得
    data_export_interval = kwargs.pop("data_export_interval", 0)
    data_export_start_p = kwargs.pop("data_export_start_p", 5.0)
            
    ramp_time_p = kwargs.pop("ramp_time_p", 1.0)            
    steady_detection = kwargs.pop("steady_detection", True)  # True: 定常検知で早期終了, False: max_time_p までのみ実行
    steady_window_p = kwargs.pop("steady_window_p", 1.0)    
    steady_tolerance = kwargs.pop("steady_tolerance", 0.001) 
    steady_extra_p = kwargs.pop("steady_extra_p", 2.0)      
    max_time_p = kwargs.pop("max_time_p", 10.0) 

    # 出力先: artifact_parent を指定すると「その直下に 1 実行ごとのサブフォルダ」を作成する。
    # 未指定かつ out_dir も無いときの既定の親は results（従来と同様に results/<benchmark>_<timestamp>/）。
    # paths_out に dict を渡すと、確定した out_dir / vti_dir / npz 相当の情報を書き戻す。
    artifact_parent = kwargs.pop("artifact_parent", None)
    paths_out = kwargs.pop("paths_out", None)
    explicit_out_dir = kwargs.pop("out_dir", None)
    explicit_vti_dir = kwargs.pop("vti_dir", None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = f"{benchmark}_{timestamp}"

    if artifact_parent is not None:
        ap = str(artifact_parent).strip()
        if not ap:
            ap = "results"
        os.makedirs(ap, exist_ok=True)
        out_dir = os.path.join(ap, run_subdir)
    elif explicit_out_dir is not None:
        out_dir = explicit_out_dir
    else:
        out_dir = os.path.join("results", run_subdir)

    os.makedirs(out_dir, exist_ok=True)

    if explicit_vti_dir is not None:
        vti_dir = explicit_vti_dir
    else:
        vti_dir = os.path.join(out_dir, "vti")
    os.makedirs(vti_dir, exist_ok=True)
    
    # ファイルのパス設定
    gif_path = kwargs.get("filename", os.path.join(out_dir, f"{benchmark}_{timestamp}.gif"))
    kwargs["filename"] = gif_path
    log_path = os.path.join(out_dir, f"{benchmark}_{timestamp}.log")
    
    # ロガー（lbm_logger: 実行フォルダの .log とコンソールへ）
    configure_logging(log_path)
    logger = get_logger(__name__)
    logger.info("Output run directory: %s", os.path.abspath(out_dir))

    cfg = SimConfig(**kwargs)


    # IBM の力ログなど、実行フォルダ（.log / .npz と同階層）に出す
    cfg.out_dir = os.path.abspath(out_dir)
    import config as config_mod
    config_mod.TI_FLOAT = cfg.fp_dtype
    ctx = SimulationContext(cfg.nx, cfg.ny, cfg.nz, cfg.n_particles, cfg.fp_dtype)
    ctx.set_materials(cfg.get_materials_dict())

    ctx.set_g_thermal_wall_tables_from_config(cfg)

    # DataExporterの初期化 (保存間隔が設定されていれば)
    exporter = None
    if data_export_interval > 0:
        export_dir = os.path.join(out_dir, "training_data/run_01")
        exporter = DataExporter(ctx, cfg, output_dir=export_dir)

    geo = GeometryBuilder()
    
    # "benchmark" の文字列から、呼び出すメソッド名を自動生成 (例: "build_rotating_hollow_cylinder")
    build_method_name = f"build_{benchmark}"

    # 互換フォールバック（旧名 -> 新名）
    build_alias = {
        "build_cylinder": "build_benchmark_cylinder",
    }

    # GeometryBuilderの中に該当するメソッドが存在するかチェックして実行
    selected_method = build_method_name
    if not hasattr(geo, selected_method):
        alias_name = build_alias.get(build_method_name, None)
        if alias_name is not None and hasattr(geo, alias_name):
            logger.warning(
                "Geometry method '%s' が見つからないため '%s' を使用します。",
                build_method_name,
                alias_name,
            )
            selected_method = alias_name

    if hasattr(geo, selected_method):
        build_method = getattr(geo, selected_method)
        build_method(ctx)
    else:
        import geometry as geometry_module
        available = sorted(
            [name for name in dir(geo) if name.startswith("build_")]
        )
        raise ValueError(
            "❌ エラー: "
            f"'{build_method_name}' という形状関数が geometry.py に見つかりません！ "
            f"(loaded={geometry_module.__file__}, available={available})"
        )

    sim = LBMSimulator(cfg)
    bc_manager = BoundaryManager(sim.d3q19, cfg)
    physics_manager = PhysicsManager(sim.d3q19, cfg)
    analytics = Analytics(sim.d3q19, cfg)
    ibm_runtime = None
    for _model in physics_manager.models:
        if hasattr(_model, "ibm"):
            ibm_runtime = _model.ibm
            break
    sim.init_fields(ctx)
    if benchmark in ("parallel_plates", "parallel_plates_ibm"):
        logger.info(
            "Nu model: %s | L_ref=%.6e | k_ref_mode=%s",
            analytics.nu_model_name,
            analytics.nu_l_ref_p,
            analytics.nu_k_ref_mode,
        )

    if steady_detection:
        dt_sample = cfg.dt * cfg.vis_interval  # updateを呼ぶ実時間間隔
        detector_v = AsymptoticSteadyDetector(steady_window_p, dt_sample, steady_tolerance)
        detector_t = AsymptoticSteadyDetector(steady_window_p, dt_sample, steady_tolerance)
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

            if benchmark in ("parallel_plates", "parallel_plates_ibm"):
                k_target = int(cfg.nz * 0.15)
                local_nu = analytics.get_local_Nu(
                    ctx, k_target, ibm=ibm_runtime, log_thermal_slice=True
                )
                local_t = temp_np[cfg.nx//2, cfg.ny//2, k_target]
                
                # ▼ 判定には局所値ではなく、領域全体の全エネルギーハッシュを使う
                global_hash = analytics.compute_global_hash(ctx)
                monitor_val_v = global_hash[0]
                monitor_val_t = global_hash[1]
                
                # ログの表示用にはNu数と温度を残す
                monitor_name = f"Nu={local_nu:.2f} | T={local_t:.3f}"
                log_parallel_plates_transport_diagnostics(
                    ctx, cfg, k_target=k_target, logger=logger, step=step
                )
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
                is_v_steady, msg_v = detector_v.update(monitor_val_v)
                is_t_steady, msg_t = detector_t.update(monitor_val_t)
                joint_steady = is_v_steady and is_t_steady

                if joint_steady and global_steady_time_p is None:
                    global_steady_time_p = current_time_p
                    logger.info(f"[SUCCESS] Flow AND Thermal steady state detected at t = {current_time_p:.3f} s!")
                    logger.info(f"[INFO] Simulation will continue for another {steady_extra_p:.3f} s to record data.")
                elif (not joint_steady) and global_steady_time_p is not None:
                    logger.info(
                        f"[INFO] Joint steady lost (Developing again) at t = {current_time_p:.3f} s. "
                        f"ETA target reset to max_time_p = {max_time_p:.3f} s; after next joint steady, "
                        f"will wait {steady_extra_p:.3f} s again."
                    )
                    global_steady_time_p = None

                status_str = f"[V:{msg_v} T:{msg_t}]"
            else:
                status_str = "[MaxTime]"

            elapsed_wall = time.time() - start_wall_time
            target_time_p = max_time_p
            if global_steady_time_p is not None:
                target_time_p = min(max_time_p, global_steady_time_p + steady_extra_p)

            rem_time_p = target_time_p - current_time_p
            eta_sec = (elapsed_wall / (current_time_p + 1e-12)) * rem_time_p
            eta_str = format_eta(eta_sec)

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

        # 回転壁にも soft-start を掛けるため ramp_factor を渡す
        sponge_amp = float(cfg.sponge_strength_amp(current_time_p))
        sim.collide_and_stream(ctx, ramp_factor, sponge_amp)

        bc_manager.apply_all_before_macro(ctx, ramp_factor)
        sim.update_macro(ctx)
        bc_manager.apply_all_after_macro(ctx)
        sim.move_particles(ctx, cfg.particles_inject_per_step)

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

    if isinstance(paths_out, dict):
        paths_out["out_dir"] = out_dir
        paths_out["vti_dir"] = vti_dir
        paths_out["gif_path"] = cfg.filename
        paths_out["npz_path"] = npz_path
        paths_out["meta_path"] = meta_path
        paths_out["run_subdir"] = run_subdir
        paths_out["log_path"] = log_path
        paths_out["ibm_forces_csv"] = os.path.join(out_dir, "ibm_forces.csv")

    return current_time_p < max_time_p


if __name__ == "__main__":
    pass