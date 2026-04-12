import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib

try:
    from scipy.linalg import eigh
except ImportError:
    eigh = None

# シミュレータ本体からのインポート
from main import run_simulation
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LBM_UI_DIR = os.path.join(_PROJECT_ROOT, "lbm_ui_designer")
if _LBM_UI_DIR not in sys.path:
    sys.path.insert(0, _LBM_UI_DIR)

run_optimize = None
try:
    lbm_opt = importlib.import_module("lbm_param_optimize")
    get_fluid_properties_coolprop = getattr(lbm_opt, "get_fluid_properties_coolprop", None)
    
except Exception:
    # sim の実行自体を止めない（optimizer 側の依存が無い場合があるため）
    run_optimize = None

# 解析解用の物理パラメータ（平均速度 U_mean に整合する圧力勾配を内部で決定）
# run_channel_benchmark 内で CoolProp（またはフォールバック）と同期する
MU_PHYS = 1.2e-4   # 動粘性係数 μ [Pa·s]（ポアズイユ式にそのまま入れる）
RHO_FLUID = 1.2    # 流体密度 [kg/m³]（domain_properties の流体と一致）
H_PHYS = 0.1       # 有効間隔 h（平板間の距離）


# CoolProp 未使用時の空気代表値（300K 付近のおおよそ）
_AIR_FALLBACK_FP = {
    "rho_f": 1.2,
    "nu": 1.5e-5,
    "k_f": 0.026,
    "Cp_f": 1005.0,
}


def fluid_transport_numbers(*, nu, k, rho, cp):
    """
    シミュレータの get_materials_dict と同じ定義で輸送係数をまとめる。

    - 運動粘度 ν [m²/s]
    - 熱拡散率 α = k/(ρ Cp) [m²/s]  … config.py の alpha と一致
    - プラントル数 Pr = ν/α
    - 動粘性 μ = ν ρ [Pa·s] … plot_poiseuille_validation の MU_PHYS と一致させる用
    """
    alpha = float(k) / (float(rho) * float(cp))
    pr = float(nu) / alpha if alpha > 1e-30 else float("nan")
    return {
        "nu": float(nu),
        "k": float(k),
        "rho": float(rho),
        "Cp": float(cp),
        "alpha": alpha,
        "Pr": pr,
        "mu_dynamic": float(nu) * float(rho),
    }


def print_benchmark_fluid_intent(
    fp, *, u_inlet_p, lx_p, nx, ny, wall_thickness_cells=10
):
    """
    ベンチマーク意図としての ν と k/(ρCp) が自己矛盾なく入っているかを表示する。
    Graetz 横断の厳密曲線は Pr を含まないが、発達長さ・Nu 議論では Pr が効く。
    """
    from wall_metrics import channel_hydraulic_diameter_p

    z = fluid_transport_numbers(
        nu=fp["nu"], k=fp["k_f"], rho=fp["rho_f"], cp=fp["Cp_f"]
    )
    dx = float(lx_p) / float(nx)
    d_h = float(channel_hydraulic_diameter_p(nx, ny, lx_p, wall_thickness_cells=wall_thickness_cells))
    re_dh = float(u_inlet_p) * d_h / z["nu"] if z["nu"] > 1e-30 else float("nan")
    print("[benchmark fluid] 物性（流体セル ID 0,20,21 と domain_properties で一致すること）:")
    print(f"    rho = {z['rho']:.6g} kg/m3,  Cp = {z['Cp']:.6g} J/(kg·K)")
    print(f"    nu  = {z['nu']:.6e} m2/s,   k = {z['k']:.6e} W/(m·K)")
    print(f"    alpha = k/(rho*Cp) = {z['alpha']:.6e} m2/s  （SimConfig.get_materials_dict と同式）")
    print(f"    Pr = nu/alpha = {z['Pr']:.6f}")
    print(f"    mu = nu*rho = {z['mu_dynamic']:.6e} Pa·s  （解析ポアズイユの MU_PHYS に使用）")
    print(f"    dx = Lx_p/nx = {dx:.6e} m,  D_h (parallel plates) ≈ {d_h:.6e} m")
    print(f"    U_inlet_p = {u_inlet_p:.6g} m/s  →  Re_Dh = U*D_h/nu ≈ {re_dh:.4f}")


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """台形則で ∫ y dx。NumPy 2.x の trapz 削除・trapezoid 名変更に依存しない。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def poiseuille_velocity(y_phys, mu, h, u_mean):
    """
    平行平板間の定常ポアズイユ流（壁 y=0, h で u=0）。

    u(y) = (1/(2μ)) (dp/dx) y (y-h) 、
    断面平均が u_mean となるよう dp/dx = -12μ u_mean / h^2 を用いる。
    """
    dp_dx = -12.0 * mu * u_mean / (h ** 2)
    return (1.0 / (2.0 * mu)) * dp_dx * y_phys * (y_phys - h)


def graetz_parallel_plates_theta_first_mode(
    eta: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """
    平行平板 Poiseuille 流に対する「熱的に十分発達」領域の無次元温度の第1モード θ(η)。

    η = y/h ∈ [0,1]、無次元速度 u(η)/ū = 6 η(1-η) のとき、等温壁 θ(0)=θ(1)=0 を満たす
    Graetz 型の固有値問題

        d²θ/dη² + λ (u(η)/ū) θ = 0

    の最小正の固有値 λ に対応する固有関数（符号は中心で正になるよう取る）。

    normalize=True のとき、返す配列を離散 mean(θ) で 1 にスケールする。
    normalize=False のときは生の θ（プロット側で LBM と同じ離散平均で割る用）。
    """
    eta = np.asarray(eta, dtype=np.float64)
    n = eta.size
    if n < 4:
        out = np.ones_like(eta, dtype=np.float64)
        return out if normalize else out

    if eigh is None:
        # scipy 無し: 形状のみの粗い近似（壁で 0、中心で最大の放物線型）
        theta = (eta * (1.0 - eta)).astype(np.float64)
        if normalize:
            theta = theta / (np.mean(theta) + 1e-30)
        return theta

    h = float(eta[-1] - eta[0]) / float(n - 1)
    u_nd = 6.0 * eta * (1.0 - eta)  # ū = 1
    m = n - 2
    # 内部点の二階差分（Dirichlet 0）: φ'' + λ u φ = 0
    A = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        A[i, i] = -2.0 / (h ** 2)
        if i > 0:
            A[i, i - 1] = 1.0 / (h ** 2)
        if i < m - 1:
            A[i, i + 1] = 1.0 / (h ** 2)
    u_int = u_nd[1:-1]
    M = np.diag(u_int)
    # (-A) φ = λ M φ 、最小 λ > 0 が第1モード
    lam, vecs = eigh(-A, M, check_finite=False)
    pos = lam > 1e-12
    if not np.any(pos):
        theta = (eta * (1.0 - eta)).astype(np.float64)
        if normalize:
            theta = theta / (np.mean(theta) + 1e-30)
        return theta
    idx = int(np.flatnonzero(pos)[0])
    phi_int = vecs[:, idx].copy()
    if np.max(np.abs(phi_int)) > 1e-15 and phi_int[np.argmax(np.abs(phi_int))] < 0:
        phi_int = -phi_int
    phi = np.zeros(n, dtype=np.float64)
    phi[1:-1] = phi_int
    if normalize:
        phi = phi / (np.mean(phi) + 1e-30)
    return phi


def extract_channel_profiles_from_npz(npz_path, nx, ny, nz):
    """保存された結果ファイルから、発達した下流のプロファイルを抽出する"""
    data = np.load(npz_path)
    v_field = data['v']       # (nx, ny, nz, 3)
    temp_field = data['temp'] # (nx, ny, nz)
    
    # 空間のX方向中央、Z方向の十分に下流（風が発達しきった領域）をカット
    x_mid = nx // 2
    z_target = int(nz * 0.15)  # 主流方向に全長の 15% 相当の下流位置（main.py parallel_plates の k_target と一致）
    
    # Y方向（壁と壁の間）のプロファイルを取得
    w_profile = v_field[x_mid, :, z_target, 2] # Z方向(主流)の速度
    t_profile = temp_field[x_mid, :, z_target] # 温度
    
    # geometry.py の parallel_plates の定義に合わせて、流体領域のみを抽出
    # j < 10 と j >= ny-10 が固体壁なので、流体は 10 から ny-11 まで
    wall_thickness = 10
    fluid_y_start = wall_thickness
    fluid_y_end = ny - wall_thickness
    
    w_fluid = w_profile[fluid_y_start:fluid_y_end]
    t_fluid = t_profile[fluid_y_start:fluid_y_end]
    
    # 無次元Y座標 (-1.0 〜 1.0) を作成 (流路の中心を 0.0 とする)
    num_fluid_cells = fluid_y_end - fluid_y_start
    y_coords = np.linspace(-1.0, 1.0, num_fluid_cells)
    
    return y_coords, w_fluid, t_fluid

def plot_poiseuille_validation(y_coords, w_fluid, t_fluid, U_inlet, out_dir, T_wall=1.0):
    """解析解（厳密解）と比較するグラフを描画して保存する"""
    # 解析解（こちらは U_inlet をベースにしてOK）
    y_phys = (H_PHYS / 2.0) * (y_coords + 1.0)
    w_analytical = poiseuille_velocity(y_phys, MU_PHYS, H_PHYS, U_inlet)
    w_ana_normalized = w_analytical / U_inlet
    
    # ▼変更：LBMの結果は「その断面の実際の平均流速」で割って無次元化する
    U_local_mean = np.mean(-w_fluid)
    w_normalized = (-w_fluid) / U_local_mean

    # 無次元高さ y_coords ∈ [-1,1] → η = y/h = (2y/H + 1)/2 = (y_coords + 1)/2
    # t_fluid は extract で固体壁帯を除いた流体セルのみ → ここでの平均も流体のみ
    eta = 0.5 * (y_coords + 1.0)
    theta_raw = graetz_parallel_plates_theta_first_mode(eta, normalize=False)
    denom_ana = float(np.mean(theta_raw))
    theta_analytical = theta_raw / (denom_ana + 1e-30)

    delta_T = np.asarray(t_fluid, dtype=np.float64) - float(T_wall)
    denom_lbm = float(np.mean(delta_T))
    if abs(denom_lbm) < 1e-12:
        denom_lbm = -1e-12
    t_normalized = delta_T / denom_lbm

    # 連続 ∫ と離散 mean の差（η が等間隔のとき台形則は端点重みのみ変わる）
    eta_span = float(eta[-1] - eta[0])
    if eta.size > 1 and eta_span > 1e-30:
        int_theta = _trapezoid_integral(theta_raw, eta)
        mean_continuous_uniform = int_theta / eta_span
    else:
        mean_continuous_uniform = denom_ana
    print(
        "[validation thermal] fluid-only mean(T - T_w) (LBM, discrete):",
        f"{denom_lbm:.8g}",
    )
    print(
        "[validation thermal] mean(theta_raw) analytical (same eta grid, discrete):",
        f"{denom_ana:.8g}",
    )
    print(
        "[validation thermal] ∫theta_raw d_eta / (eta1-eta0) (trapezoid) vs discrete mean:",
        f"{mean_continuous_uniform:.8g} vs {denom_ana:.8g}",
        f"(diff {mean_continuous_uniform - denom_ana:.8g})",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 1. Velocity Profile (速度の放物線) ---
    ax1.plot(w_ana_normalized, y_coords, '-', color='black', linewidth=2, label='Analytical (Poiseuille)')
    ax1.plot(w_normalized, y_coords, 'o', color='blue', markerfacecolor='none', label='LBM (Present)')
    ax1.set_title('Velocity Profile (Poiseuille Flow)')
    ax1.set_xlabel('V_z / U_mean')
    ax1.set_ylabel('2y / H (Dimensionless Height)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_xlim([0.0, 1.6])
    ax1.set_ylim([-1.0, 1.0])

    # --- 2. Temperature Profile（熱発達後の Graetz 第1モードと比較） ---
    ax2.plot(theta_analytical, y_coords, '-', color='black', linewidth=2, label='Analytical (Graetz, 1st mode)')
    ax2.plot(t_normalized, y_coords, 'o', color='red', markerfacecolor='none', label='LBM (Present)')
    ax2.set_title('Temperature Profile (vs Graetz fully developed)')
    ax2.set_xlabel(r'$(T - T_w) / \overline{(T - T_w)}$')
    ax2.set_ylabel('2y / H (Dimensionless Height)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_xlim([0.0, 2.0])
    ax2.set_ylim([-1.0, 1.0])

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "Poiseuille_Thermal_Validation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Validation plot saved to: {plot_path}")
    plt.show()

def run_channel_benchmark(artifact_parent=None):
    """
    artifact_parent: この実行の成果物の「親フォルダ」。
        未指定時は results/validation_poiseuille 。その直下に parallel_plates_<日時>/ が 1 本の実行ごとに作成される。
    """
    print(f"==================================================")
    print(f" Starting Poiseuille Flow & Heat Transfer Validation")
    print(f"==================================================")
    
    # --- パラメータの設定 ---
    fp = get_fluid_properties_coolprop("Air", 300.0, 101325.0)  # "Water"
    if fp is None:
        print(
            "[WARN] CoolProp が使えないため、空気の代表物性にフォールバックします。"
            "（Pr ≈ 0.71 付近の意図は維持されますが、数値は文献値と完全一致しません）"
        )
        fp = dict(_AIR_FALLBACK_FP)
    nx, ny, nz = 32, 64, 512  # Z方向に長くして、風が発達しきる距離を稼ぐ
    Lx_p = 0.1
    U_inlet_p = 0.005  # 綺麗な層流を保つため、流速は少し遅め（断面平均として解析解と対応）
    global RHO_FLUID, MU_PHYS
    RHO_FLUID = fp["rho_f"]
    MU_PHYS = fp["nu"] * RHO_FLUID
    print_benchmark_fluid_intent(fp, u_inlet_p=U_inlet_p, lx_p=Lx_p, nx=nx, ny=ny)
    if artifact_parent is None:
        artifact_parent = os.path.join("results", "validation_poiseuille")
    run_paths = {}

    # --- シミュレーションの実行 ---
    run_simulation(
        benchmark="parallel_plates", # geometry.py にある平行平板
        artifact_parent=artifact_parent,
        paths_out=run_paths,
        fp_dtype="float32",
        steady_detection=False,       
        steady_window_p=2.0,         
        steady_tolerance=0.001,      
        steady_extra_p=0.5,          
        # ▼ 追加: X方向を無限幅(周期境界)にし、Y方向を有限にする
        periodic_x=True,
        periodic_y=False,
        periodic_z=False,
        # neem_isothermal_wall=False,
        
        nx=nx, ny=ny, nz=nz,
        Lx_p=Lx_p,
        U_inlet_p=U_inlet_p,
        
        max_time_p=90.0,  
        ramp_time_p=1.0,  # 衝撃波を防ぐためソフトスタート
        
        vis_interval=100, 
        vti_export_interval=0, particles_inject_per_step=0,
        
        sponge_thickness=0.0, # 出口での反射を防ぐためスポンジ層をON
        
        # 空間に存在する全IDの物性を定義 (ゼロ割防止)
        # （これで Pr≒0.71 となり、熱的助走区間が短くなって下流でNu=7.54に漸近します）
        domain_properties={
            0:  {"nu": fp["nu"], "k": fp["k_f"], "rho": fp["rho_f"], "Cp": fp["Cp_f"]}, # 流体
            10: {"nu": 0.0,  "k": 400.0, "rho": 8960.0, "Cp": 385.0},  # 上下の壁
            20: {"nu": fp["nu"], "k": fp["k_f"], "rho": fp["rho_f"], "Cp": fp["Cp_f"]}, # 入口
            21: {"nu": fp["nu"], "k": fp["k_f"], "rho": fp["rho_f"], "Cp": fp["Cp_f"]}, # 出口
        },
        
        boundary_conditions={
            # 入口から冷たい風(T=0.0)を入れる
            20: {"type": "inlet",  "velocity":[0.0, 0.0, -U_inlet_p], "temperature": 0.0},
            21: {"type": "outlet"},
            # 上下の壁(10)を熱源(T=1.0)にして、熱がどう伝わるかを見る
            10: {"type": "isothermal_wall", "temperature": 1.0},
        }
    )

    out_dir = run_paths.get("out_dir")
    if not out_dir or not os.path.isdir(out_dir):
        print("[ERROR] run_simulation did not return a valid output directory (paths_out).")
        return

    npz_path = run_paths.get("npz_path")
    if not npz_path or not os.path.isfile(npz_path):
        npz_files = [f for f in os.listdir(out_dir) if f.endswith("_fields.npz")]
        if not npz_files:
            print("[ERROR] No .npz field data found.")
            return
        npz_files.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)), reverse=True)
        npz_path = os.path.join(out_dir, npz_files[0])
    
    # プロファイルの抽出とプロット
    y_coords, w_prof, t_prof = extract_channel_profiles_from_npz(npz_path, nx, ny, nz)
    plot_poiseuille_validation(y_coords, w_prof, t_prof, U_inlet_p, out_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="平行平板ポアズイユ＋熱伝達検証。成果物は artifact_parent 直下に 1 実行ごとのフォルダへ保存。"
    )
    parser.add_argument(
        "artifact_parent",
        nargs="?",
        default=os.path.join("results", "validation_poiseuille"),
        help="実行ごとのサブフォルダを作成する親ディレクトリ（既定: results/validation_poiseuille）",
    )
    args = parser.parse_args()
    run_channel_benchmark(artifact_parent=args.artifact_parent)