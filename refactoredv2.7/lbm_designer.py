# ==========================================================
# lbm_auto_designer.py — 物理物性データベース & LBM自動設計ツール
# ==========================================================
import math

# 1. 物理物性データベース (現実の数値)
# nu: 動粘度[m^2/s], k: 熱伝導率 [W/m K], rho: 密度 [kg/m^3], Cp: 比熱 [J/kg K]
MATERIALS = {
    "Water":     {"nu": 1.0e-6,  "k": 0.6,   "rho": 1000.0, "Cp": 4180.0},
    "Air":       {"nu": 1.5e-5,  "k": 0.026, "rho": 1.2,    "Cp": 1005.0},
    "EngineOil": {"nu": 5.0e-5,  "k": 0.15,  "rho": 850.0,  "Cp": 2000.0},
    "Copper":    {"nu": 0.0,     "k": 400.0, "rho": 8960.0, "Cp": 385.0},
    "Stainless": {"nu": 0.0,     "k": 16.0,  "rho": 8000.0, "Cp": 500.0},
    "Aluminum":  {"nu": 0.0,     "k": 237.0, "rho": 2700.0, "Cp": 900.0},
}

def design_heat_exchanger(
    fluid_in_name="Water",
    fluid_out_name="Water",
    solid_name="Copper",
    target_Re=500.0,      # 目標とする内部流のレイノルズ数
    nx=128,               # 空間解像度 (x方向)
    Lx_p=0.1,             # 内部流チューブの内径 [m] (代表長さ)
    u_lbm_in=0.1,         # 内部流の LBM 流速 (0.05 ~ 0.15推奨)
    u_out_ratio=0.5,      # 外部流の流速比 (内部流に対して)
    tau_min=0.52,         # ギリギリを攻めるための限界tau
    tau_solid_target=0.8  # 固体のtauは安定しやすい 0.8 付近を狙う
):
    print("=" * 65)
    print(" LBM Auto Designer: Similarity Law & Parameter Generator")
    print("=" * 65)

    # 物性値の取得
    f_in  = MATERIALS[fluid_in_name]
    f_out = MATERIALS[fluid_out_name]
    sol   = MATERIALS[solid_name]

    # --- 1. 流動の逆算 (Re -> U_inlet) ---
    D_ratio = 0.4
    D_p = Lx_p * D_ratio      # 内部流チューブの内径 [m] (物理スケール)
    N_cells = nx * D_ratio    # 代表長さに対応するセル数
    tube_thickness_p = Lx_p*2/nx   # チューブの厚み [m] (物理スケール)

    # --- 1. 流動の逆算 (Re -> U_inlet) ---
    U_inlet_p = (target_Re * f_in["nu"]) / D_p
    dx = Lx_p / nx
    dt = dx * u_lbm_in / U_inlet_p

    # セルレイノルズ数と流速のtau
    Re_delta = target_Re / N_cells
    tau_f_in = 0.5 + 3.0 * (u_lbm_in / Re_delta)
    
    # --- 2. 相似則の適用 (Prの緩和と仮想Cpの計算) ---
    def calc_virtual_Cp(mat, u_lbm_target, is_fluid=True):
        alpha_real = mat["k"] / (mat["rho"] * mat["Cp"])
        if is_fluid:
            Pr_real = mat["nu"] / alpha_real
            # 安定限界(tau_min)から許容される最大のPr数を逆算
            Pr_max = (3.0 * nx * u_lbm_target) / (target_Re * (tau_min - 0.5))
            Pr_sim = min(Pr_real, Pr_max)
            alpha_sim = mat["nu"] / Pr_sim
            tau_g = 0.5 + 3.0 * (u_lbm_target * nx) / (target_Re * Pr_sim)
        else:
            Pr_real, Pr_sim = None, None
            # 固体はターゲットtau(0.8など)になるように直接alphaを逆算
            alpha_sim = (tau_solid_target - 0.5) * (dx**2) / (3.0 * dt)
            tau_g = tau_solid_target
            
        Cp_sim = mat["k"] / (mat["rho"] * alpha_sim)
        return Cp_sim, Pr_real, Pr_sim, tau_g

    Cp_sim_in, Pr_real_in, Pr_sim_in, tau_g_in = calc_virtual_Cp(f_in, u_lbm_in, True)
    
    # 外部流の計算 (Reは内部流を基準にして換算)
    u_lbm_out = u_lbm_in * u_out_ratio
    Re_out_sim = target_Re * u_out_ratio * (f_in["nu"] / f_out["nu"]) # 粘性違いを加味した外部流Re
    tau_f_out = 0.5 + 3.0 * (u_lbm_out * nx) / Re_out_sim
    Cp_sim_out, Pr_real_out, Pr_sim_out, tau_g_out = calc_virtual_Cp(f_out, u_lbm_out, True)
    
    # 固体の計算
    Cp_sim_sol, _, _, tau_g_sol = calc_virtual_Cp(sol, u_lbm_in, False)

    # --- 3. その他の無次元数 (Pe, Bi) ---
    Pe_in = target_Re * Pr_sim_in
    # 内部流のNu数概算 (層流 Nu=4.36, 乱流 Nu=0.023*Re^0.8*Pr^0.4)
    Nu_est = 4.36 if target_Re < 2300 else 0.023 * (target_Re**0.8) * (Pr_sim_in**0.4)
    # 熱伝達率 h = Nu * k / L_x
    # Biot数 Bi = h * t / k_s
    Bi = Nu_est * (f_in["k"] / sol["k"]) * (tube_thickness_p / Lx_p)

    # --- レポート出力 ---
    print(f"\n[1. Target Conditions]")
    print(f"  Shape          : Tube Heat Exchanger")
    print(f"  Materials      : In={fluid_in_name}, Out={fluid_out_name}, Tube={solid_name}")
    print(f"  Target Re (In) : {target_Re:.1f}")
    print(f"  Physical U_in  : {U_inlet_p:.6f} [m/s]")

    print(f"\n[2. Dimensionless Numbers (Similarity Applied)]")
    print(f"  Internal Flow  : Re = {target_Re:.1f} | Pr_real = {Pr_real_in:.2f} -> Pr_sim = {Pr_sim_in:.2f} | Pe = {Pe_in:.1f}")
    print(f"  External Flow  : Re = {Re_out_sim:.1f} | Pr_real = {Pr_real_out:.2f} -> Pr_sim = {Pr_sim_out:.2f}")
    print(f"  Est. Nu (In)   : {Nu_est:.2f}")
    print(f"  Biot Number    : {Bi:.6f} (<< 0.1 means uniform wall temp)")

    print(f"\n[3. LBM Stability (tau values)]")
    print(f"  tau_f_in       : {tau_f_in:.4f}")
    print(f"  tau_f_out      : {tau_f_out:.4f}")
    print(f"  tau_g_in       : {tau_g_in:.4f} (Safety Limit: {tau_min})")
    print(f"  tau_g_out      : {tau_g_out:.4f}")
    print(f"  tau_g_solid    : {tau_g_sol:.4f}")
    
    if tau_f_in < tau_min or tau_f_out < tau_min:
        print("\n❌ WARNING: tau_f is below safety limit! Increase 'nx' or decrease 'target_Re'.")
        return

    # --- run_simulation のコード生成 ---
    print("\n" + "=" * 65)
    print(" COPY & PASTE THIS INTO YOUR main.py")
    print("=" * 65)
    
    code = f"""    run_simulation(
        benchmark="heat_exchanger",
        fp_dtype="float32",
        steady_detection=True,
        nx={nx}, ny={nx}, nz={nx * 4},  # 推奨アスペクト比 1:1:4
        Lx_p={Lx_p}, 
        
        U_inlet_p={U_inlet_p:.6e},
        max_time_p=30.0, ramp_time_p=2.0,
        vis_interval=100, vti_export_interval=500,
        
        # =========================================================
        # ▼ Auto-Designed Similarity Parameters
        # Target Re={target_Re}, Pr_sim_in={Pr_sim_in:.2f}
        # Result conversion: Nu_real = Nu_sim * ({Pr_real_in:.2f} / {Pr_sim_in:.2f})^(1/3)
        # =========================================================
        domain_properties={{
            # 【Internal Fluid: {fluid_in_name}】 ID: 0(Mid), 22(In), 23(Out)
            0:  {{"nu": {f_in["nu"]:.2e}, "k": {f_in["k"]}, "rho": {f_in["rho"]}, "Cp": {Cp_sim_in:.1f}}},
            22: {{"nu": {f_in["nu"]:.2e}, "k": {f_in["k"]}, "rho": {f_in["rho"]}, "Cp": {Cp_sim_in:.1f}}},
            23: {{"nu": {f_in["nu"]:.2e}, "k": {f_in["k"]}, "rho": {f_in["rho"]}, "Cp": {Cp_sim_in:.1f}}},
            
            # 【External Fluid: {fluid_out_name}】 ID: 2(Mid), 24(In), 25(Out)
            2:  {{"nu": {f_out["nu"]:.2e}, "k": {f_out["k"]}, "rho": {f_out["rho"]}, "Cp": {Cp_sim_out:.1f}}},
            24: {{"nu": {f_out["nu"]:.2e}, "k": {f_out["k"]}, "rho": {f_out["rho"]}, "Cp": {Cp_sim_out:.1f}}},
            25: {{"nu": {f_out["nu"]:.2e}, "k": {f_out["k"]}, "rho": {f_out["rho"]}, "Cp": {Cp_sim_out:.1f}}},
            
            # 【Solid Tube: {solid_name}】 ID: 10
            10: {{"nu": 0.0, "k": {sol["k"]}, "rho": {sol["rho"]}, "Cp": {Cp_sim_sol:.1f}}}
        }},
        
        boundary_conditions={{
            22: {{"type": "inlet",  "velocity":[0.0, 0.0,  {u_lbm_in}], "temperature": 1.0}},
            24: {{"type": "inlet",  "velocity":[0.0, 0.0, -{u_lbm_out}], "temperature": 0.0}},
            23: {{"type": "outlet"}},
            25: {{"type": "outlet"}},
            10: {{"type": "adiabatic_wall"}}, 
        }},
        flow_type="counter"
    )"""
    print(code)
    print("=" * 65)


if __name__ == "__main__":
    # ▼ ここでシミュレーションしたい条件を指示するだけ！
    design_heat_exchanger(
        fluid_in_name="Water",
        fluid_out_name="Water",
        solid_name="Copper",
        target_Re=5000.0,     # 解きたいレイノルズ数
        nx=int(2**(7+1/3)),              # PCの限界に合わせたメッシュ
        u_lbm_in=0.15,        # LBMの基準流速
        tau_min=0.51         # 発散しないための安全マージン
    )