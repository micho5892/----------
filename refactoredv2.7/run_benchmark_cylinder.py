import os
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# LBMシミュレータ本体とオプティマイザのインポート
from main import run_simulation, run_optimize
from lbm_logger import get_logger, configure_logging

_log = get_logger(__name__)

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

def analyze_and_plot(force_csv, target_re, U_inlet_lbm, D_lbm, ny_lbm, out_dir):
    """
    保存されたCSVから抗力(C_D)と揚力(C_L)を計算し、
    FFTを用いてストルーハル数(St)を特定・可視化する。
    """
    _log.info("Analyzing IBM force data from: %s", force_csv)
    data = np.loadtxt(force_csv, delimiter=',', skiprows=1)
    
    steps = data[:, 0]
    fx_lbm = data[:, 1] # X方向の力 (揚力)
    fz_lbm = data[:, 3] # Z方向の力 (抗力: Z方向から風が吹いているため)

    # 最初の50%は過渡状態(渦が安定する前)として捨てる
    start_idx = int(len(steps) * 0.5)
    steps_steady = steps[start_idx:]
    fx_steady = fx_lbm[start_idx:]
    fz_steady = fz_lbm[start_idx:]

    # LBM単位系での係数計算 (C_D, C_L)
    rho_lbm = 1.0 # LBMでの基準密度
    A_lbm = D_lbm * ny_lbm # 前面投影面積(LBM単位)
    q_lbm = 0.5 * rho_lbm * (U_inlet_lbm**2) * A_lbm

    cl = fx_steady / q_lbm
    cd = np.abs(fz_steady) / q_lbm  # 風下方向なので絶対値をとる

    cd_mean = np.mean(cd)
    cl_amp = (np.max(cl) - np.min(cl)) / 2.0

    # FFTによる St数 の計算 (サンプリング間隔の特定)
    dt_record_lbm = steps_steady[1] - steps_steady[0]
    N = len(cl)
    
    # 直流成分を除去してFFT
    yf = fft(cl - np.mean(cl))
    xf = fftfreq(N, dt_record_lbm)[:N//2]
    
    dominant_idx = np.argmax(np.abs(yf[:N//2]))
    f_lbm = xf[dominant_idx]
    
    # St = f * D / U
    st_fft = f_lbm * D_lbm / U_inlet_lbm

    _log.info("==================================================")
    _log.info(f" IBM Cylinder Benchmark (Re = {target_re}) Results")
    _log.info("==================================================")
    _log.info(f" Mean Drag Coefficient (C_D) : {cd_mean:.4f}  (Target: ~1.34)")
    _log.info(f" Lift Coefficient Amp  (C_L) : ±{cl_amp:.4f}  (Target: ~±0.3)")
    _log.info(f" Strouhal Number       (St)  : {st_fft:.4f}  (Target: ~0.165)")

    # グラフの描画
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 係数の時系列推移
    ax1.plot(steps_steady, cd, label='Drag $C_D$', color='red')
    ax1.plot(steps_steady, cl, label='Lift $C_L$', color='blue')
    ax1.set_title(f'Force Coefficients (Re={target_re})')
    ax1.set_xlabel('LBM Steps')
    ax1.set_ylabel('Coefficient')
    ax1.grid(True, linestyle='--')
    ax1.legend()
    
    # FFTスペクトル
    ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'g-')
    ax2.axvline(x=f_lbm, color='orange', linestyle='--', label=f'Peak St = {st_fft:.3f}')
    ax2.set_title('Frequency Spectrum of Lift ($C_L$)')
    ax2.set_xlabel('Frequency (LBM inverse steps)')
    ax2.set_xlim([0, max(f_lbm * 4.0, 0.01)])
    ax2.grid(True, linestyle='--')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"IBM_Cylinder_Validation_Re{int(target_re)}.png")
    plt.savefig(plot_path, dpi=300)
    _log.info("Validation plot saved to: %s", plot_path)
    
    # plt.show() # バッチ実行時に止まらないようにコメントアウト

def run_ibm_cylinder_benchmark(target_re=100.0):
    
    _log.info(f"\n=== Optimizing for IBM Cylinder (Re = {target_re}) ===")
    
    # 空間のベース設定
    nx_base = 100
    ny_base = 4    # 2D的に解くために薄く設定
    nz_base = 300
    Lx_p_val = 0.1
    D_phys = Lx_p_val * 0.2  # 直径は流路幅(X)の20%

    nx_base = 150  # 100から150に拡大 (隣の渦との干渉を防ぐため)
    ny_base = 4    
    nz_base = 400  # 出口までの距離も少し伸ばす
    Lx_p_val = 0.15 # nxに比例して拡大
    D_phys = 0.02   # 円柱の物理直径はそのままキープ (20セル相当)

    nx_base = 200          # 空間の幅をさらに広げる
    ny_base = 4
    nz_base = 400
    Lx_p_val = 0.2         # 空間幅に合わせて物理長も拡大
    D_phys = Lx_p_val * 0.1 # 円柱直径は流路幅の 10% (これでブロック率が10%に下がる)
    
    # オプティマイザの設定
    config_opt = {
        "fluid": "Air",
        "temperature_K": 300,
        "pressure_Pa": 101325,
        "solid": "Copper", # 今回は剛体として動かないので熱伝導等は不問
        "fix_cr": True,

        "fixed":{
            "nx": nx_base,
            "nu": True,
            "k_f": True,
            "rho_f": True,
            "k_s": True,
            "L_domain": Lx_p_val,
            "L_ref": D_phys, # 代表長さを円柱の直径とする
            "u_lbm": 0.05,   # マッハ数を低く抑えるための安全な設定
        },

        "ranges":{
            "Re":{
                "min": target_re * 0.8,
                "max": target_re * 1.2
            },
            "tau_f マージン": {"min": 0.01, "max": 2.00}, 
            "tau_gf マージン": {"min": 0.01, "max": 2.00}, 
            "tau_gs マージン": {"min": 0.01, "max": 2.00},
        },
        "targets": {
            "Re": {"value": target_re, "weight": 1.0},
        },
        "target_regularization": 1,
        "regularization": 1.0e-3,
        "maxiter": 30000,
    }

    result = run_optimize(config_opt)
    
    if not result.get("success", False):
        _log.error("Failed to optimize parameters for Re=%s", target_re)
        return

    st = result["state"]
    artifact_parent = os.path.join("results", f"ibm_cylinder_Re{int(target_re)}")
    run_paths: dict = {}
    _log.info("Optimization successful. Actual Re: %.2f", st["Re"])
    _log.info(
        "Simulation artifacts will be under: %s/<run_subdir>/ (same folder as .log and .npz)",
        os.path.abspath(artifact_parent),
    )

    # 円柱の配置位置の計算 (物理座標系)
    # Zの大きい方(nz-1)から小さい方(0)へ流れる設定なので、上流側(Zが大きい方)に配置
    Lz_p = Lx_p_val * (nz_base / nx_base)
    cylinder_z_p = Lz_p * 0.75 

    try:
        run_simulation(
            benchmark="empty_domain",  # geometry.py の空チャネル
            artifact_parent=artifact_parent,
            paths_out=run_paths,
            fp_dtype="float32",
            steady_detection=False, # 波形を取得するため定常検知はオフ
            state=st, # オプティマイザの計算結果を渡す
            periodic_x=True,
            periodic_y=True,
            periodic_z=False,
            
            nx=int(st["nx"]), ny=ny_base, nz=nz_base,  
            Lx_p=st["L_domain"],                
            U_inlet_p=st["U"],
            
            # カルマン渦の周期を数回とれる程度の十分な時間
            max_time_p=300.0, 
            ramp_time_p=2,
            
            vis_interval=100, 
            vti_export_interval=0, particles_inject_per_step=0,
            data_export_interval=0,
            
            sponge_thickness=40.0,
            sponge_strength_decay_start_p=3.0,
            sponge_strength_decay_duration_p=5.0,
            
            domain_properties={
                0:  {"nu": st["nu"], "k": st["k_f"], "rho": st["rho_f"], "Cp": st["Cp_f"]},
                20: {"nu": st["nu"], "k": st["k_f"], "rho": st["rho_f"], "Cp": st["Cp_f"]}, 
                21: {"nu": st["nu"], "k": st["k_f"], "rho": st["rho_f"], "Cp": st["Cp_f"]}, 
            },
            
            boundary_conditions={
                # 上(Z=nz-1)から下(Z=0)へ風を吹かせる
                20: {"type": "inlet",  "velocity":[0.0, 0.0, -st["u_lbm"]], "temperature": 0.0},
                21: {"type": "outlet"},
            },
            
            # ▼ IBM プラグインの呼び出し ▼
            physics_models={
                "immersed_boundary": {
                    "objects": [
                        {
                            "shape": "cylinder",
                            "radius_p": D_phys / 2.0,
                            "center_p": [Lx_p_val * 0.51, 0.0, cylinder_z_p],
                            "type": "fixed",  # その場で固定
                            "temperature": 1.0  # 無次元の目標温度（境界の無次元温度と整合させる）
                            # Thermal IBM: 無次元目標温度を指定すると ctx.S_g へ熱源が載る（例 "temperature": 1.0）
                        }
                    ]
                }
            }
        )
    except Exception as e:
        _log.error("Simulation error: %s", e)
        return

    # シミュレーション終了後、記録された CSV（実行フォルダ＝.log と同階層）から St を解析
    sim_out_dir = run_paths.get("out_dir")
    csv_path = run_paths.get("ibm_forces_csv")
    if not sim_out_dir or not csv_path:
        _log.error("paths_out に out_dir / ibm_forces_csv が入っていません。run_simulation の戻りを確認してください。")
        return
    if not os.path.isfile(csv_path):
        _log.error("Force CSV not found at %s. Simulation might have failed.", csv_path)
        return
    _log.info("Post-processing IBM forces from: %s", csv_path)
    _log.info("Plots and validation PNG will be written under: %s", sim_out_dir)
    dx_val = st["dx"]
    D_lbm_val = D_phys / dx_val
    analyze_and_plot(csv_path, target_re, st["u_lbm"], D_lbm_val, ny_base, sim_out_dir)

if __name__ == "__main__":
    # トップレベルで共通ロガーをファイルにも出力するように設定
    configure_logging("ibm_cylinder_run.log")
    
    for re_target in [100.0]:
        run_ibm_cylinder_benchmark(target_re=re_target)