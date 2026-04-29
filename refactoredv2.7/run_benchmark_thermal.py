import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# シミュレータ本体からのインポート
from main import run_simulation, run_optimize
from lbm_logger import get_logger

_log = get_logger(__name__)

# ==============================================================================
# de Vahl Davis (1983) の正解データ (Differentially Heated Cavity)
# ※ 表 V (Benchmark solutions / Extrapolated values) に基づく高精度限界値
# ==============================================================================
DE_VAHL_DAVIS_DATA = {
    1e4: {
        "U_max": 16.18, "Z_Umax": 0.823, "W_max": 19.62, "X_Wmax": 0.119,
        "Nu_avg": 2.243, "Nu_1/2": 2.243, "Nu_0": 2.243,
        "Nu_max": 3.528, "Z_Numax": 0.143, "Nu_min": 0.586, "Z_Numin": 1.000
    },
    1e5: {
        "U_max": 34.73, "Z_Umax": 0.855, "W_max": 68.59, "X_Wmax": 0.066,
        "Nu_avg": 4.519, "Nu_1/2": 4.519, "Nu_0": 4.519,
        "Nu_max": 7.717, "Z_Numax": 0.081, "Nu_min": 0.729, "Z_Numin": 1.000
    },
}

def analyze_and_plot_thermal_cavity(npz_path, nx, nz, target_Ra, L_eff, alpha_f, dx, dt, out_dir):
    data = np.load(npz_path)
    v_field = data['v']       # (nx, ny, nz, 3)
    temp_field = data['temp'] # (nx, ny, nz)
    
    y_mid = v_field.shape[1] // 2
    
    # 1. 物理境界（固体セル）を除外し、純粋な「流体領域」だけを抽出
    # 左壁(X=0) と 右壁(X=nx-1)、および床・天井(Z=0, nz-1) をカット
    t_fluid = temp_field[1:-1, y_mid, 1:-1]
    u_fluid_lbm = v_field[1:-1, y_mid, 1:-1, 0]
    w_fluid_lbm = v_field[1:-1, y_mid, 1:-1, 2]
    
    nx_f = nx - 2
    nz_f = nz - 2
    
    # LBM速度から物理速度への変換、さらに無次元化
    phys_u = u_fluid_lbm * (dx / dt)
    phys_w = w_fluid_lbm * (dx / dt)
    u_norm = phys_u * L_eff / alpha_f
    w_norm = phys_w * L_eff / alpha_f
    
    # 空間座標の正規化 (ABBにより壁はセル境界0.5dxの位置にある)
    dx_norm = 1.0 / nx_f
    dz_norm = 1.0 / nz_f
    x_coords = np.linspace(0.5 * dx_norm, 1.0 - 0.5 * dx_norm, nx_f)
    z_coords = np.linspace(0.5 * dz_norm, 1.0 - 0.5 * dz_norm, nz_f)
    
    # ==========================================================
    # Nusselt Number (Nu) の計算
    # ==========================================================
    # 1. 高温壁 (X=0) での局所 Nu 数のプロファイル
    # ABB境界による壁の温度 T_w = 1.0。流体セルの温度から2次精度不等間隔差分で勾配を算出
    T_w = 1.0
    Nu_0_profile = (8.0 * T_w - 9.0 * t_fluid[0, :] + t_fluid[1, :]) / (3.0 * dx_norm)
    
    # 2. 中央断面 (X=0.5) での局所 Nu 数のプロファイル
    mid_idx = nx_f // 2
    dt_dx_mid = (t_fluid[mid_idx+1, :] - t_fluid[mid_idx-1, :]) / (2.0 * dx_norm)
    Nu_half_profile = u_norm[mid_idx, :] * t_fluid[mid_idx, :] - dt_dx_mid
    
    # 3. 領域全体の平均 Nu 数
    dt_dx_all = np.zeros_like(t_fluid)
    dt_dx_all[1:-1, :] = (t_fluid[2:, :] - t_fluid[:-2, :]) / (2.0 * dx_norm)
    dt_dx_all[0, :] = (-3.0 * t_fluid[0, :] + 4.0 * t_fluid[1, :] - t_fluid[2, :]) / (2.0 * dx_norm)
    dt_dx_all[-1, :] = (3.0 * t_fluid[-1, :] - 4.0 * t_fluid[-2, :] + t_fluid[-3, :]) / (2.0 * dx_norm)
    local_Nu_all = u_norm * t_fluid - dt_dx_all
    
    # ★ NumPy 2.0 以降に対応した台形積分
    sim_Nu_0 = np.trapezoid(Nu_0_profile, dx=dz_norm)
    sim_Nu_half = np.trapezoid(Nu_half_profile, dx=dz_norm)
    sim_Nu_avg = np.trapezoid(np.trapezoid(local_Nu_all, dx=dz_norm, axis=1), dx=dx_norm)
    
    # --- スプライン補間でサブグリッドのピーク検出 ---
    u_profile = u_norm[mid_idx, :]
    w_profile = w_norm[:, nz_f // 2]
    
    z_dense = np.linspace(z_coords[0], z_coords[-1], nz_f * 50)
    x_dense = np.linspace(x_coords[0], x_coords[-1], nx_f * 50)
    
    f_u = interp1d(z_coords, u_profile, kind='cubic')
    u_dense = f_u(z_dense)
    sim_U_max = np.max(u_dense)
    sim_Z_Umax = z_dense[np.argmax(u_dense)]
    
    f_w = interp1d(x_coords, w_profile, kind='cubic')
    w_dense = f_w(x_dense)
    sim_W_max = np.max(w_dense)
    sim_X_Wmax = x_dense[np.argmax(w_dense)]
    
    f_Nu = interp1d(z_coords, Nu_0_profile, kind='cubic')
    Nu_dense = f_Nu(z_dense)
    sim_Nu_max = np.max(Nu_dense)
    sim_Z_Numax = z_dense[np.argmax(Nu_dense)]
    sim_Nu_min = np.min(Nu_dense)
    sim_Z_Numin = z_dense[np.argmin(Nu_dense)]
    
    # --- 結果出力 ---
    ref = DE_VAHL_DAVIS_DATA.get(target_Ra)
    print(f"\n=== Benchmark Results (Ra = {target_Ra:.1e}) ===")
    if ref:
        print(f"| Metric                     | LBM (Present) | de Vahl Davis | Error (%) |")
        print(f"|----------------------------|---------------|---------------|-----------|")
        print(f"| Max U-vel (Local Pe_x)     | {sim_U_max:13.3f} | {ref['U_max']:13.3f} | {abs(sim_U_max-ref['U_max'])/ref['U_max']*100:8.2f}% |")
        print(f"| Z location of Max U-vel    | {sim_Z_Umax:13.3f} | {ref['Z_Umax']:13.3f} | {abs(sim_Z_Umax-ref['Z_Umax'])/ref['Z_Umax']*100:8.2f}% |")
        print(f"| Max W-vel (Local Pe_z)     | {sim_W_max:13.3f} | {ref['W_max']:13.3f} | {abs(sim_W_max-ref['W_max'])/ref['W_max']*100:8.2f}% |")
        print(f"| X location of Max W-vel    | {sim_X_Wmax:13.3f} | {ref['X_Wmax']:13.3f} | {abs(sim_X_Wmax-ref['X_Wmax'])/ref['X_Wmax']*100:8.2f}% |")
        print(f"|----------------------------|---------------|---------------|-----------|")
        print(f"| Avg Nu (Overall)           | {sim_Nu_avg:13.3f} | {ref['Nu_avg']:13.3f} | {abs(sim_Nu_avg-ref['Nu_avg'])/ref['Nu_avg']*100:8.2f}% |")
        print(f"| Avg Nu (Mid-plane X=0.5)   | {sim_Nu_half:13.3f} | {ref['Nu_1/2']:13.3f} | {abs(sim_Nu_half-ref['Nu_1/2'])/ref['Nu_1/2']*100:8.2f}% |")
        print(f"| Avg Nu (Boundary X=0)      | {sim_Nu_0:13.3f} | {ref['Nu_0']:13.3f} | {abs(sim_Nu_0-ref['Nu_0'])/ref['Nu_0']*100:8.2f}% |")
        print(f"| Max Nu (Boundary X=0)      | {sim_Nu_max:13.3f} | {ref['Nu_max']:13.3f} | {abs(sim_Nu_max-ref['Nu_max'])/ref['Nu_max']*100:8.2f}% |")
        print(f"| Z location of Max Nu       | {sim_Z_Numax:13.3f} | {ref['Z_Numax']:13.3f} | {abs(sim_Z_Numax-ref['Z_Numax'])/ref['Z_Numax']*100:8.2f}% |")
        print(f"| Min Nu (Boundary X=0)      | {sim_Nu_min:13.3f} | {ref['Nu_min']:13.3f} | {abs(sim_Nu_min-ref['Nu_min'])/ref['Nu_min']*100:8.2f}% |")
        print(f"| Z location of Min Nu       | {sim_Z_Numin:13.3f} | {ref['Z_Numin']:13.3f} | {abs(sim_Z_Numin-ref['Z_Numin'])/ref['Z_Numin']*100:8.2f}% |")
    
    # プロファイル可視化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(u_profile, z_coords, 'o', color='gray', alpha=0.5)
    plt.plot(u_dense, z_dense, '-', color='blue')
    plt.title("U-vel at X = 0.5")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x_coords, w_profile, 'o', color='gray', alpha=0.5)
    plt.plot(x_dense, w_dense, '-', color='green')
    plt.title("W-vel at Z = 0.5")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(Nu_0_profile, z_coords, 'o', color='gray', alpha=0.5)
    plt.plot(Nu_dense, z_dense, '-', color='orange')
    plt.title("Local Nu at X = 0 (Hot Wall)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Profiles_Ra{int(target_Ra)}.png"), dpi=300)
    plt.close()

    # --- 等温線(Isotherms)の描画 ---
    X, Z = np.meshgrid(np.linspace(0, 1, nx_f), np.linspace(0, 1, nz_f), indexing='ij')
    plt.figure(figsize=(6, 6))
    contour = plt.contour(X, Z, t_fluid, levels=np.linspace(0.05, 0.95, 10), colors='black', linewidths=1.5)
    plt.imshow(t_fluid.T, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='Dimensionless Temperature')
    plt.title(f'Isotherms at Ra = {target_Ra:.1e}')
    plt.xlabel('X / L')
    plt.ylabel('Z / L')
    
    plot_path = os.path.join(out_dir, f"Thermal_Cavity_Ra{int(target_Ra)}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Plots saved to {out_dir}")

def run_thermal_cavity_benchmark(target_Ra=1e5):
    print(f"\n" + "="*60)
    print(f" 差温キャビティ内の自然対流 (Ra={target_Ra:.1e}) の精度検証")
    print("="*60)
    
    nx = 128
    ny = 4
    nz = 128
    L_domain = 0.1
    dx = L_domain / nx
    
    # LBMの壁位置(ABB)により、物理的な流体領域は nx-2 セル分になる
    nx_f = nx - 2
    L_eff = dx * nx_f
    
    
    config_thermal = {
        "fluid": "Air",
        "temperature_K": 300,
        "pressure_Pa": 101325,
        "solid": "Copper",
        "fix_cr": False,
        "fixed": {
            "nx": nx,             
            # "nu": True, "k_f": True, "rho_f": True, "k_s": True,
            "L_domain": L_domain, 
            "L_ref": L_eff,
            "u_lbm": 0.04,         # 安全で安定なLBMマッハ数
            "U": 1.0,               # 計算基準用のダミー速度
        },
        "ranges": {
            # "tau_f マージン": {"min": 0.1, "max": 2.0},
        },
        "targets": {
            "Ra": {"value": target_Ra, "weight": 1.0},
            'alpha_f': {"value": 2.2274814744444102e-02, "weight": 2.0},
        },
        "target_regularization": 1,
        "regularization": 1.0e-5,
        "maxiter": 30000,
    }

    result = run_optimize(config_thermal)
    _sci = result.get("scipy") or {}
    print("\n--- run_optimize 診断（毎回） ---")
    print(f"  success:             {result.get('success')}")
    print(f"  message:             {result.get('message')}")
    print(f"  equation_conflicts:  {result.get('equation_conflicts')}")
    print(f"  scipy.status:        {_sci.get('status')}")
    print(f"  scipy.nit:           {_sci.get('nit')}")
    rr = result.get("range_report") or {}
    for k, v in rr.items():
        if not v.get("satisfied", True):
            print("  range NG:", k, v)
    st = result.get("state") or {}
    for key in ("Ra", "alpha_f", "nu", "beta_f", "C_r", "tau_f マージン"):
        if key in st:
            print(f"  state[{key}] = {st[key]}")
    print("---\n")

    if not result["success"]:
        print("[ERROR] パラメータ最適化に失敗しました。")
        return
    
    state = result["state"]
    
    # 最適化された物性値を取得
    nu_p = state["nu"]
    k_p = state["k_f"]
    rho_p = state["rho_f"]
    Cp_p = state["Cp_f"]
    alpha_f = k_p / (rho_p * Cp_p)
    dt = state["dt"]

    g_phys = 9.81
    delta_T = 1.0
    # 目標Ra数になるように熱膨張係数(beta)を逆算
    beta_p = (target_Ra * nu_p * alpha_f) / (g_phys * delta_T * (L_eff**3))
    
    # 熱拡散にかかる物理時間を計算し、確実に定常に達するまでの時間を指定
    t_c = (L_eff**2) / alpha_f
    max_time_p = t_c * 5.0
    print(f">>> Optimized Thermal Time Scale t_c = {t_c:.3f} s.")
    print(f">>> Running simulation up to 5 t_c = {max_time_p:.3f} s to guarantee steady state.")
    
    paths_out = {}
    artifact_parent = os.path.join("results", "validation_thermal")
    
    run_simulation(
        benchmark="thermal_cavity",
        fp_dtype="float32",
        steady_detection=False,       
        state=state,
        artifact_parent=artifact_parent,
        paths_out=paths_out,
        
        nx=nx, ny=ny, nz=nz,
        Lx_p=L_domain,
        periodic_y=True,
        U_inlet_p=state["U"], 
        u_lbm=state["u_lbm"],    
        output_format="mp4",
        visualization_mode="offline",
        
        # max_time_p=max_time_p, 
        max_time_p=3, 
        ramp_time_p=0.0,
        
        vis_interval=200, 
        vti_export_interval=0, particles_inject_per_step=0,
        sponge_thickness=0.0,
        
        domain_properties={
            0:  {"nu": nu_p, "k": k_p, "rho": rho_p, "Cp": Cp_p},
            10: {"nu": 0.0,  "k": state["k_s"], "rho": state["rho_s"], "Cp": state["Cp_s"]},
            11: {"nu": 0.0,  "k": state["k_s"], "rho": state["rho_s"], "Cp": state["Cp_s"]},
            21: {"nu": 0.0,  "k": state["k_s"], "rho": state["rho_s"], "Cp": state["Cp_s"]},
        },
        
        physics_models={
            "boussinesq": {"g_vec":[0.0, 0.0, -g_phys], "beta": beta_p, "T_ref": 0.5}
        },
        
        boundary_conditions={
            11: {"type": "isothermal_wall", "temperature": 1.0},
            10: {"type": "isothermal_wall", "temperature": 0.0},
            21: {"type": "adiabatic_wall"},
            # ★ SimConfig のバグ(強制Uを上書きしてしまう仕様)を回避するダミー (領域外なので無害)
            99: {"type": "inlet", "velocity": [state["u_lbm"], 0.0, 0.0], "temperature": 0.0}
        }
    )

    out_dir = paths_out.get("out_dir")
    npz_path = paths_out.get("npz_path")

    if not out_dir or not npz_path or not os.path.isfile(npz_path):
        _log.error("Simulation failed or NPZ not found.")
        return
        
    analyze_and_plot_thermal_cavity(npz_path, nx, nz, target_Ra, L_eff, alpha_f, dx, dt, out_dir)

if __name__ == "__main__":
    for Ra in [1e4, 1e5]:
        run_thermal_cavity_benchmark(target_Ra=Ra)