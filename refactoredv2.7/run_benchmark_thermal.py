import os
import numpy as np
import matplotlib.pyplot as plt

# シミュレータ本体からのインポート
from main import run_simulation

# ==============================================================================
# de Vahl Davis (1983) の正解データ (Differentially Heated Cavity)
# Ra (レイリー数) ごとの無次元最大流速とその位置
# ==============================================================================
DE_VAHL_DAVIS_DATA = {
    1e4: {"U_max": 16.18, "Z_Umax": 0.823, "W_max": 19.61, "X_Wmax": 0.119},
    1e5: {"U_max": 34.73, "Z_Umax": 0.855, "W_max": 68.59, "X_Wmax": 0.066},
}

# ▼ 引数に dx, dt を追加
def analyze_and_plot_thermal_cavity(npz_path, nx, nz, target_Ra, L_domain, alpha_f, dx, dt, out_dir):
    data = np.load(npz_path)
    v_field = data['v']       # (nx, ny, nz, 3)
    temp_field = data['temp'] # (nx, ny, nz)
    
    y_mid = v_field.shape[1] // 2
    
    # 中央断面のデータ抽出 (LBM単位系)
    u_2d = v_field[:, y_mid, :, 0] 
    w_2d = v_field[:, y_mid, :, 2] 
    t_2d = temp_field[:, y_mid, :] 
    
    # ▼ 追加: LBM格子速度から物理速度(m/s)への変換
    phys_u_2d = u_2d * (dx / dt)
    phys_w_2d = w_2d * (dx / dt)
    
    # ▼ 修正: 物理速度を使って論文の定義通りに無次元化 (U* = U_phys * L / alpha)
    u_norm = phys_u_2d * L_domain / alpha_f
    w_norm = phys_w_2d * L_domain / alpha_f
    
    
    # 縦の中心線上 (X=0.5) の U速度プロファイル
    u_profile = u_norm[nx // 2, :]
    # 横の中心線上 (Z=0.5) の W速度プロファイル
    w_profile = w_norm[:, nz // 2]
    
    # シミュレーション結果の最大値を取得
    sim_U_max = np.max(np.abs(u_profile))
    sim_Z_Umax = np.argmax(np.abs(u_profile)) / float(nz - 1)
    
    sim_W_max = np.max(np.abs(w_profile))
    sim_X_Wmax = np.argmax(np.abs(w_profile)) / float(nx - 1)
    
    # --- 理論値との比較プリント ---
    ref = DE_VAHL_DAVIS_DATA.get(target_Ra)
    print(f"\n=== Benchmark Results (Ra = {target_Ra:.1e}) ===")
    if ref:
        print(f"| Metric       | LBM (Present) | de Vahl Davis | Error (%) |")
        print(f"|--------------|---------------|---------------|-----------|")
        print(f"| Max U-vel    | {sim_U_max:13.2f} | {ref['U_max']:13.2f} | {abs(sim_U_max-ref['U_max'])/ref['U_max']*100:8.2f}% |")
        print(f"| Z pos of U   | {sim_Z_Umax:13.3f} | {ref['Z_Umax']:13.3f} | {abs(sim_Z_Umax-ref['Z_Umax'])/ref['Z_Umax']*100:8.2f}% |")
        print(f"| Max W-vel    | {sim_W_max:13.2f} | {ref['W_max']:13.2f} | {abs(sim_W_max-ref['W_max'])/ref['W_max']*100:8.2f}% |")
        print(f"| X pos of W   | {sim_X_Wmax:13.3f} | {ref['X_Wmax']:13.3f} | {abs(sim_X_Wmax-ref['X_Wmax'])/ref['X_Wmax']*100:8.2f}% |")
    
    # --- 等温線(Isotherms)の描画 ---
    X, Z = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nz), indexing='ij')
    
    plt.figure(figsize=(6, 6))
    contour = plt.contour(X, Z, t_2d, levels=np.linspace(0.05, 0.95, 10), colors='black', linewidths=1.5)
    plt.imshow(t_2d.T, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='Dimensionless Temperature')
    plt.title(f'Isotherms at Ra = {target_Ra:.1e}')
    plt.xlabel('X / L')
    plt.ylabel('Z / L')
    
    plot_path = os.path.join(out_dir, f"Thermal_Cavity_Ra{int(target_Ra)}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Isotherm plot saved to: {plot_path}")
    plt.show()


def run_thermal_cavity_benchmark(target_Ra=1e5):
    print(f"==================================================")
    print(f" Starting Thermal Cavity Validation (Ra={target_Ra:.1e})")
    print(f"==================================================")
    
    # Y は薄いスライス: 周期境界で横方向端の人工反射を減らし、2D 差温キャビティに近づける
    nx, ny, nz = 128, 4, 128
    L_domain = 0.1
    g_phys = 9.81
    delta_T = 1.0
    
    # 空気ベースの物性値
    nu_p = 1.5e-5
    Pr = 0.71
    alpha_f = nu_p / Pr
    k_p = alpha_f * 1.2 * 1005.0 # k = alpha * rho * Cp
    
    # レイリー数から逆算して、必要な熱膨張率(beta)を算出
    # Ra = (g * beta * dT * L^3) / (nu * alpha)
    beta_p = (target_Ra * nu_p * alpha_f) / (g_phys * delta_T * (L_domain**3))
    
    out_dir = f"results/validation_thermal_Ra{int(target_Ra)}"
    
    run_simulation(
        benchmark="thermal_cavity",
        out_dir=out_dir,
        fp_dtype="float32",
        steady_detection=False,       
        steady_window_p=3.0,         
        steady_tolerance=0.0000,      
        steady_extra_p=0.5,          
        
        nx=nx, ny=ny, nz=nz,
        Lx_p=L_domain,
        periodic_y=True,
        U_inlet_p=1.0, # ダミー(計算用)
        u_lbm=0.05,    # マッハ数エラーを避けるため低めに設定
        
        max_time_p=20.0, 
        ramp_time_p=0.0,
        
        vis_interval=200, 
        vti_export_interval=0, particles_inject_per_step=0,
        sponge_thickness=0.0, # 密閉空間なのでスポンジはオフ！
        
        domain_properties={
            0:  {"nu": nu_p, "k": k_p, "rho": 1.2, "Cp": 1005.0},
            10: {"nu": 0.0,  "k": 400.0, "rho": 8960.0, "Cp": 385.0}, # 冷源
            11: {"nu": 0.0,  "k": 400.0, "rho": 8960.0, "Cp": 385.0}, # 熱源
            21: {"nu": 0.0,  "k": 400.0, "rho": 8960.0, "Cp": 385.0}, # 上下壁
        },
        
        # ★ Z軸マイナス方向(下向き)に重力を設定
        physics_models={
            "boussinesq": {"g_vec":[0.0, 0.0, -g_phys], "beta": beta_p, "T_ref": 0.5}
        },
        
        # 左 i=0 が ID=11（高温）、右 i=nx-1 が ID=10（低温）。g 移流の ABB はこの辞書の温度を参照する。
        boundary_conditions={
            11: {"type": "isothermal_wall", "temperature": 1.0},
            10: {"type": "isothermal_wall", "temperature": 0.0},
            21: {"type": "adiabatic_wall"},
        }
    )

    npz_files =[f for f in os.listdir(out_dir) if f.endswith('_fields.npz')]
    if not npz_files:
        return
        
    npz_path = os.path.join(out_dir, npz_files[0])
    analyze_and_plot_thermal_cavity(npz_path, nx, nz, target_Ra, L_domain, alpha_f, out_dir)

if __name__ == "__main__":
    # まずは Ra = 10万 でテスト
    run_thermal_cavity_benchmark(target_Ra=1e5)