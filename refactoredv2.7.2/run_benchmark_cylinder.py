import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# シミュレータ本体からのインポート
from main import run_simulation

# ==============================================================================
# Roshko (1954) 等に基づく円柱周りのSt数の経験式 (層流 Re=50〜200)
# ==============================================================================
def theoretical_strouhal(re):
    if re < 47:
        return 0.0 # 臨界Re未満では渦は放出されない
    return 0.212 - 4.5 / re

def analyze_and_plot_strouhal(csv_path, target_re, U_inlet, D_phys, out_dir):
    # プローブデータの読み込み
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    time = data[:, 0]
    vx = data[:, 1]
    
    # 初期の過渡状態(渦が安定するまで)を切り捨てる (後半の60%だけを使用)
    start_idx = int(len(time) * 0.4)
    time_steady = time[start_idx:]
    vx_steady = vx[start_idx:]
    
    # -----------------------------------------------------
    # 1. ピーク検出(Time Domain)による周波数計算
    # -----------------------------------------------------
    peaks, _ = find_peaks(vx_steady)
    if len(peaks) < 2:
        print("[ERROR] Not enough peaks found. Simulation time might be too short.")
        return
        
    # ピーク間の平均時間から周期(T)と周波数(f)を求める
    peak_times = time_steady[peaks]
    avg_period = np.mean(np.diff(peak_times))
    f_time = 1.0 / avg_period
    st_time = f_time * D_phys / U_inlet
    
    # -----------------------------------------------------
    # 2. FFT(Frequency Domain)による周波数計算
    # -----------------------------------------------------
    dt = time_steady[1] - time_steady[0]
    N = len(vx_steady)
    yf = fft(vx_steady - np.mean(vx_steady)) # 直流成分を除去してFFT
    xf = fftfreq(N, dt)[:N//2]
    
    # 最大パワーを持つ周波数を抽出
    dominant_idx = np.argmax(np.abs(yf[:N//2]))
    f_fft = xf[dominant_idx]
    st_fft = f_fft * D_phys / U_inlet
    
    # 理論値との比較
    st_theory = theoretical_strouhal(target_re)
    
    print(f"\n=== Strouhal Number Benchmark (Re = {target_re}) ===")
    print(f" Theoretical St  : {st_theory:.4f}")
    print(f" LBM St (Peaks)  : {st_time:.4f} (Error: {abs(st_time-st_theory)/st_theory*100:.2f}%)")
    print(f" LBM St (FFT)    : {st_fft:.4f} (Error: {abs(st_fft-st_theory)/st_theory*100:.2f}%)")

    # -----------------------------------------------------
    # グラフの描画
    # -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 時系列波形
    ax1.plot(time_steady, vx_steady, 'b-')
    ax1.plot(time_steady[peaks], vx_steady[peaks], 'ro')
    ax1.set_title('Cross-stream Velocity (Probe)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Vx [m/s]')
    ax1.grid(True, linestyle='--')
    
    # FFTスペクトル
    ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'g-')
    ax2.axvline(x=f_fft, color='r', linestyle='--', label=f'Dominant f = {f_fft:.2f} Hz')
    ax2.set_title('Frequency Spectrum (FFT)')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim([0, f_fft * 4.0]) # ピークの4倍あたりまで表示
    ax2.grid(True, linestyle='--')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"Strouhal_Validation_Re{int(target_re)}.png")
    plt.savefig(plot_path, dpi=300)
    print(f" -> Validation plot saved to: {plot_path}")
    plt.show()

def run_cylinder_benchmark(target_re=100.0):
    print(f"==================================================")
    print(f" Starting Karman Vortex Validation (Re={target_re})")
    print(f"==================================================")
    
    # --- 2D的なカルマン渦を綺麗に出すため、Y方向は薄く設定 ---
    nx, ny, nz = 128, 4, 512 
    Lx_p = 0.1
    D_phys = Lx_p * 0.1  # geometry.py で radius = nx*0.05 なので直径は nx*0.1
    U_inlet_p = 0.2
    
    # Re = U * D / nu から動粘度を逆算
    nu_p = (U_inlet_p * D_phys) / target_re
    
    out_dir = f"results/validation_cylinder_Re{int(target_re)}"
    
    run_simulation(
        benchmark="benchmark_cylinder",
        out_dir=out_dir,
        fp_dtype="float32",
        
        # 渦の周期を測るため、定常化検知はOFFにして一定時間(15秒)回し切る
        steady_detection=False,
        
        nx=nx, ny=ny, nz=nz,
        Lx_p=Lx_p,              
        U_inlet_p=U_inlet_p,
        u_lbm=0.1,  
        
        max_time_p=15.0, 
        ramp_time_p=1.0, 
        
        vis_interval=50, # FFTのサンプリングレートを高めるため細かく保存
        vti_export_interval=0, particles_inject_per_step=0,
        
        sponge_thickness=40.0, # Z=0(出口)の音響波反射を防ぐためON
        
        domain_properties={
            0:  {"nu": nu_p, "k": 0.026, "rho": 1.2, "Cp": 1005.0},
            10: {"nu": 0.0,  "k": 400.0, "rho": 8960.0, "Cp": 385.0}, 
            20: {"nu": nu_p, "k": 0.026, "rho": 1.2, "Cp": 1005.0}, 
            21: {"nu": nu_p, "k": 0.026, "rho": 1.2, "Cp": 1005.0}, 
        },
        
        boundary_conditions={
            # 上から下(-Z方向)へ風を吹かせる
            20: {"type": "inlet",  "velocity":[0.0, 0.0, -U_inlet_p], "temperature": 0.0},
            21: {"type": "outlet"},
            10: {"type": "adiabatic_wall"}, 
        }
    )

    csv_path = os.path.join(out_dir, "probe.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Probe data not found at {csv_path}")
        return
        
    analyze_and_plot_strouhal(csv_path, target_re, U_inlet_p, D_phys, out_dir)

if __name__ == "__main__":
    # Re=100 と Re=150 の2パターンで検証
    for re_target in[100.0, 150.0]:
        run_cylinder_benchmark(target_re=re_target)