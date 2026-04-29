import os
import sys

# 必要なモジュールのインポート
from run_benchmark_channel import extract_channel_profiles_from_npz, plot_poiseuille_validation
from run_benchmark_cylinder import analyze_and_plot
from main import run_simulation, run_optimize
from lbm_logger import get_logger, configure_logging

_log = get_logger(__name__)



def verify_parallel_plates(target_re=100.0):
    print("\n" + "="*60)
    print(" 1. 平行平板チャネル層流 (Nu=7.54) と 厳密解の検証")
    print("="*60)
    
    # 1. パラメータの最適化 (Re=100の層流をターゲット)
    config_channel = {
        "fluid": "Air",
        "temperature_K": 300,
        "pressure_Pa": 101325,
        "solid": "Copper",
        "fix_cr": True,
        "fixed": {
            "nx": 64,              # 流路の解像度
            "nu": True, "k_f": True, "rho_f": True, "k_s": True,
            "L_domain": 0.05,      # 物理長 [m]
            "L_ref": 0.05,         # 代表長さ（チャネル幅）[m]
            "u_lbm": 0.05,         # LBM空間での最大流速 (マッハ数制限をクリア)
        },
        "ranges": {
            "Re": {"min": 50, "max": 2000},
            "tau_f マージン": {"min": 0.02, "max": 2.0},
            "tau_gf マージン": {"min": 0.02, "max": 2.0},
        },
        "targets": {
            "Re": {"value": target_re, "weight": 1.0},
        },
        "target_regularization": 1,
        "regularization": 1.0e-3,
        "maxiter": 30000,
    }

    result = run_optimize(config_channel)
    if not result["success"]:
        print("[ERROR] 平行平板のパラメータ最適化に失敗しました。")
        return

    state = result["state"]
    nz=result["nx"] * 8 
    max_time_p = nx * nz / result["alpha_f"]
    paths_out = {}
    nx = int(state["nx"])
    artifact_parent = os.path.join("results", "validation_poiseuille")
    # 2. シミュレーションの実行
    run_simulation(
        benchmark="parallel_plates",  # ジオメトリビルダーで定義済みの名前
        fp_dtype="float32",
        steady_detection=True,        # 定常状態を自動検知して終了
        steady_window_p=2.5,
        steady_tolerance=0.005,
        steady_extra_p = 3.0,
        state=state,
        artifact_parent=artifact_parent,
        periodic_x=True,
        periodic_y=True,
        periodic_z=False,
    
        nx=nx, ny=nx, nz=nx * 8,      # 流れが十分に発達する長さに設定
        Lx_p=state["L_domain"],
        U_inlet_p=state["U"],
        
        max_time_p=max_time_p, 
        ramp_time_p=2.0,
        visualization_mode="offline",
        vis_interval=100, vti_export_interval=0,

        sponge_thickness=40.0,
        sponge_strength_decay_start_p=4.0,
        sponge_strength_decay_duration_p=5.0,
        
        # 最適化された物性値を各IDへマッピング
        domain_properties={
            0:  {"nu": state["nu"], "k": state["k_f"], "rho": state["rho_f"], "Cp": state["Cp_f"]},
            10: {"nu": 0.0,         "k": state["k_s"], "rho": state["rho_s"], "Cp": state["Cp_s"]}, # 壁
            20: {"nu": state["nu"], "k": state["k_f"], "rho": state["rho_f"], "Cp": state["Cp_f"]}, # INLET
            21: {"nu": state["nu"], "k": state["k_f"], "rho": state["rho_f"], "Cp": state["Cp_f"]}, # OUTLET
        },
        boundary_conditions={
            20: {"type": "inlet", "velocity": [0.0, 0.0, -state["u_lbm"]], "temperature": 0.0},
            21: {"type": "outlet"},
            10: {"type": "isothermal_wall", "temperature": 1.0}, # 等温壁（NEEM適用）
        },
        paths_out=paths_out
    )

    # 3. 厳密解との比較プロット
    out_dir = paths_out.get("out_dir")
    npz_path = paths_out.get("npz_path")
    if out_dir and npz_path and os.path.isfile(npz_path):
        print("\n>>> 解析解（ポアズイユ流れ・温度分布）との比較グラフを生成します...")
        y_coords, w_prof, t_prof = extract_channel_profiles_from_npz(npz_path, nx, nx, nx * 8)
        plot_poiseuille_validation(y_coords, w_prof, t_prof, state["U"], out_dir)
        print(f">>> グラフを {out_dir} に保存しました。Nu数が約7.54になるかログを確認してください。")

def verify_karman_vortex(target_re=150.0):
    print("\n" + "="*60)
    print(f" 2. 円柱周りのカルマン測 (Re={int(target_re)}) の精度検証")
    print("="*60)
    
    # 1. パラメータの最適化 (Re=150で綺麗なカルマン渦が発生)
    config_cyl = {
        "fluid": "Air",
        "temperature_K": 300,
        "pressure_Pa": 101325,
        "solid": "Copper",
        "fix_cr": True,
        "fixed": {
            "nx": 128,             
            "nu": True, "k_f": True, "rho_f": True, "k_s": True,
            "L_domain": 0.1, 
            "L_ref": 0.015,        # 円柱の直径 [m]
            "u_lbm": 0.05,         # LBM速度
        },
        "ranges": {
            # "Re": {"min": 100, "max": 200},
            "tau_f マージン": {"min": 0.02, "max": 2.0},
        },
        "targets": {
            "Re": {"value": target_re, "weight": 1.0},
        },
        "target_regularization": 1,
        "regularization": 1.0e-3,
        "maxiter": 30000,
    }

    result = run_optimize(config_cyl)
    if not result["success"]:
        print("[ERROR] カルマン渦のパラメータ最適化に失敗しました。")
        return

    state = result["state"]
    nx = int(state["nx"])
    paths_out = {}
    artifact_parent = os.path.join("results", "validation_karman_vortex")
    # 2. シミュレーションの実行
    # ※IBMのVPMとIterative Forcing（フェーズ3までの実装）をフル稼働させます
    run_simulation(
        benchmark="cylinder",         # 領域は単なる直方体 (境界はINLET/OUTLET)
        fp_dtype="float32",
        steady_detection=False,       # カルマン渦は振動するため定常検知はOFF
        state=state,
        
        artifact_parent=artifact_parent,
        periodic_x=False,
        periodic_y=True,
        periodic_z=False,

        nx=nx, ny=8, nz=nx * 4,       # 2D的な流れを見るためにnyは薄く
        Lx_p=state["L_domain"],
        U_inlet_p=state["U"],
        
        max_time_p=20.0,              # カルマン渦が成長するまでの時間
        ramp_time_p=2.0,              # 衝撃波を防ぐソフトスタート
        visualization_mode="offline",
        vis_interval=100, 
        vti_export_interval=0, 
        particles_inject_per_step=200, # 渦を可視化するためのパーティクル

        sponge_thickness=40.0,
        sponge_strength_decay_start_p=3.0,
        sponge_strength_decay_duration_p=5.0,
        
        domain_properties={
            0:  {"nu": state["nu"], "k": state["k_f"], "rho": state["rho_f"], "Cp": state["Cp_f"]},
            10: {"nu": 0.0,         "k": state["k_s"], "rho": state["rho_s"], "Cp": state["Cp_s"]},
            20: {"nu": state["nu"], "k": state["k_f"], "rho": state["rho_f"], "Cp": state["Cp_f"]},
            21: {"nu": state["nu"], "k": state["k_f"], "rho": state["rho_f"], "Cp": state["Cp_f"]},
        },
        boundary_conditions={
            20: {"type": "inlet", "velocity": [0.0, 0.0, -state["u_lbm"]], "temperature": 0.0},
            21: {"type": "outlet"},
        },
        
        # ▼ IBM (Immersed Boundary Method) プラグインの設定 ▼
        physics_models={
            "immersed_boundary": {
                "phase1_epsilon_lbm": 1.5,
                "phase1_num_iterations": 3,   # 速度の反復強制 (Spurious flow除去)
                "phase2_num_iterations": 3,   # 熱の反復強制
                "phase2_enable_iterative_thermal": False,
                "phase2_heat_relax": 1.0,
                "objects": [
                    {
                        "shape": "cylinder",
                        "radius_p": state["L_ref"] / 2.0, # 直径から半径へ
                        "center_p": [state["L_domain"] * 0.51, 0.0, state["L_domain"] * 4 * 0.85],
                        "type": "fixed",
                        "temperature": 1.0
                    }
                ]
            }
        },
        paths_out=paths_out
    )
    
    out_dir = paths_out.get("out_dir")
    print(f"\n>>> シミュレーション完了。結果は {out_dir} に保存されました。")
    print(">>> 渦の挙動は出力された GIFアニメーション を確認してください。")
    print(">>> (main.py の設定により、probe.csv にX方向速度の振動履歴が記録されています)")

    sim_out_dir = paths_out.get("out_dir")
    csv_path = paths_out.get("ibm_forces_csv")
    if not sim_out_dir or not csv_path:
        _log.error("paths_out に out_dir / ibm_forces_csv が入っていません。run_simulation の戻りを確認してください。")
        return
    if not os.path.isfile(csv_path):
        _log.error("Force CSV not found at %s. Simulation might have failed.", csv_path)
        return
    _log.info("Post-processing IBM forces from: %s", csv_path)
    _log.info("Plots and validation PNG will be written under: %s", sim_out_dir)
    dx_val = state["dx"]
    D_lbm_val = state["L_ref"] / dx_val
    analyze_and_plot(csv_path, target_re, state["u_lbm"], D_lbm_val, 8, sim_out_dir)

if __name__ == "__main__":
    # 1. Nu=7.54 と速度・温度プロファイルの検証
    # for target_Re in [200.0, 300.0, 400.0, 500.0, 700.0, 900.0, 1000.0]:
    #     verify_parallel_plates(target_Re)
    
    # 2. カルマン渦の検証
    for target_re in [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]:
        verify_karman_vortex(target_re)