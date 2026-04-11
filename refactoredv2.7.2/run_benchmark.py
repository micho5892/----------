import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# シミュレータ本体からのインポート
from main import run_simulation
from config import SimConfig

# ==============================================================================
# Ghia et al. (1982) 参照データ: プロジェクトの ベンチマークテストデータ/*.csv
# CavityFlow_y.csv … 縦中心線 (x=0.5) の u(y)
# CavityFlow_x.csv … 横中心線 (y=0.5) の v(x)
# ヘッダ列: grid pt., y|x, Re=100, 400, 1000, 3200, 5000, 7500, 10000
# ==============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_BENCHMARK_DATA_DIR = os.path.join(_PROJECT_ROOT, "ベンチマークテストデータ")

# CSV 上のレイノルズ数列（0-based 列インデックス: 先頭2列は grid / 座標）
_GHIA_RE_TO_COL = {100: 2, 400: 3, 1000: 4, 3200: 5, 5000: 6, 7500: 7, 10000: 8}


def _read_ghia_csv_column(csv_path: str, value_col: int):
    """1列目が座標(y または x)、value_col が対象 Re の速度値。座標昇順で返す。"""
    coords, vals = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) <= value_col:
                continue
            coords.append(float(row[1]))
            vals.append(float(row[value_col]))
    order = np.argsort(coords)
    return np.asarray(coords, dtype=np.float64)[order], np.asarray(vals, dtype=np.float64)[order]


def load_ghia_reference_data(target_re: float):
    """
    ベンチマークテストデータの CSV から Ghia 参照曲線を読み込む。
    対応 Re が無い場合は None。
    """
    r = int(round(float(target_re)))
    if r not in _GHIA_RE_TO_COL:
        return None
    col = _GHIA_RE_TO_COL[r]
    y_path = os.path.join(_BENCHMARK_DATA_DIR, "CavityFlow_y.csv")
    x_path = os.path.join(_BENCHMARK_DATA_DIR, "CavityFlow_x.csv")
    if not os.path.isfile(y_path) or not os.path.isfile(x_path):
        return None
    y, u = _read_ghia_csv_column(y_path, col)
    x, v = _read_ghia_csv_column(x_path, col)
    return {"y": y, "u": u, "x": x, "v": v}


def extract_profiles_from_npz(npz_path, nx, nz):
    """保存された結果ファイルから、中央断面の速度プロファイルを抽出する"""
    data = np.load(npz_path)
    v_field = data['v']  # (nx, ny, nz, 3)
    
    # 2D的なキャビティ流れを想定し、Y方向(奥行き)の中央スライスを取得
    y_mid = v_field.shape[1] // 2
    x_mid = nx // 2
    z_mid = nz // 2
    
    # X=0.5 (縦の中心線) 上の、X方向速度 (U)
    # 配列は (nx, ny, nz, 3) なので、x_mid に固定し、z方向に沿って取得
    u_profile = v_field[x_mid, y_mid, :, 0]
    
    # Z=0.5 (横の中心線) 上の、Z方向速度 (W) ※LBMのZ軸はGhiaのY軸に相当
    w_profile = v_field[:, y_mid, z_mid, 2]
    
    # 座標の正規化 (0.0 〜 1.0)
    z_coords = np.linspace(0, 1.0, nz)
    x_coords = np.linspace(0, 1.0, nx)
    
    return x_coords, z_coords, u_profile, w_profile

def plot_ghia_comparison(x_coords, z_coords, u_profile, w_profile, target_re, U_inlet, out_dir):
    """Ghiaデータと比較するグラフを描画して保存する"""
    ghia = load_ghia_reference_data(target_re)
    if ghia is None:
        print(
            f"[Warning] Ghia CSV for Re={int(target_re)} not found "
            f"(or missing { _BENCHMARK_DATA_DIR }). Skipping plot."
        )
        return
    
    # LBMの速度を、壁の速度(U_inlet)で割って無次元化 (0.0 ~ 1.0)
    u_normalized = u_profile / U_inlet
    w_normalized = w_profile / U_inlet

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 1. U-velocity along vertical line (x = 0.5) ---
    ax1.plot(u_normalized, z_coords, '-', label='LBM (Present)', color='blue')
    ax1.plot(ghia['u'], ghia['y'], 'o', label='Ghia et al. (1982)', color='red', markerfacecolor='none')
    ax1.set_title(f'U-Velocity along vertical centerline (Re={int(target_re)})')
    ax1.set_xlabel('U / U_wall')
    ax1.set_ylabel('Y / L')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_xlim([-0.5, 1.1])
    ax1.set_ylim([0.0, 1.0])

    # --- 2. V-velocity along horizontal line (y = 0.5) ---
    ax2.plot(x_coords, w_normalized, '-', label='LBM (Present)', color='green')
    ax2.plot(ghia['x'], ghia['v'], 'o', label='Ghia et al. (1982)', color='red', markerfacecolor='none')
    ax2.set_title(f'V-Velocity along horizontal centerline (Re={int(target_re)})')
    ax2.set_xlabel('X / L')
    ax2.set_ylabel('V / U_wall')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([-0.6, 0.6])

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"Ghia_Validation_Re{int(target_re)}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Validation plot saved to: {plot_path}")
    plt.show()

def run_cavity_benchmark(target_re=100.0):
    print(f"==================================================")
    print(f" Starting Lid-Driven Cavity Validation (Re={target_re})")
    print(f"==================================================")
    
    # --- パラメータの設定 ---
    nx, ny, nz = 128, 4, 128 # 2D的な流れなのでYは薄くする
    L_domain = 0.1
    U_inlet_p = 0.1
    u_lbm = 0.1
    
    # Re = U * L / nu から動粘度を逆算
    nu_p = (U_inlet_p * L_domain) / target_re

    out_dir = f"results/validation_cavity/Re{int(target_re)}"
    
    # --- シミュレーションの実行 ---
    # (※定常状態になるまで自動で回す設定)
    run_simulation(
        benchmark="lid_driven_cavity",
        out_dir=out_dir,
        fp_dtype="float32",
        steady_detection=False,       # 定常になったら自動ストップ
        steady_window_p=2.0,         # 2秒間の変動を監視
        steady_tolerance=0.0001,      # 変動が0.5%以下になれば定常とみなす
        steady_extra_p=1.0,          # 定常判定後、念のため0.5秒回して終了
        
        nx=nx, ny=ny, nz=nz,
        Lx_p=L_domain,              
        U_inlet_p=U_inlet_p,
        
        max_time_p=30.0,  # タイムアウト時間
        ramp_time_p=0.0,  # キャビティはフタを急発進させるのでソフトスタートなし
        
        vis_interval=200, 
        vti_export_interval=0, particles_inject_per_step=0,
        
        domain_properties={
            0:  {"nu": nu_p, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
            20: {"nu": 0.0, "k": 400.0, "rho": 8960.0, "Cp": 385.0}, 
            21: {"nu": 0.0, "k": 400.0, "rho": 8960.0, "Cp": 385.0}, 
        },
        
        boundary_conditions={
            # ID 20 (上面のフタ) が X方向にスライドして動く
            20: {"type": "adiabatic_wall"},
            21: {"type": "adiabatic_wall"},
        }
    )

    # --- 実行終了後、最新の .npz ファイルを探して解析 ---
    # np.savez_compressed によって out_dir 内に _fields.npz が作られているはず
    npz_files = [f for f in os.listdir(out_dir) if f.endswith('_fields.npz')]
    if not npz_files:
        print("[ERROR] No .npz field data found in the output directory.")
        return
        
    npz_path = os.path.join(out_dir, npz_files[0])
    
    # プロファイルの抽出とプロット
    x_coords, z_coords, u_prof, w_prof = extract_profiles_from_npz(npz_path, nx, nz)
    plot_ghia_comparison(x_coords, z_coords, u_prof, w_prof, target_re, U_inlet_p, out_dir)

if __name__ == "__main__":
    for re_target in[100.0, 400.0, 1000.0]:
        run_cavity_benchmark(target_re=re_target)