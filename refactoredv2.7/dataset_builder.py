import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm # pip install tqdm (進捗バー用)
import traceback

class DatasetBuilder:
    def __init__(self, input_dir, output_dir, pool_size=2):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pool_size = pool_size # 2x2x2 で空間を粗くする
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # GPUが使えるならGPUで高速処理
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(f"[DatasetBuilder] Using device: {self.device}")

    def filter_3d(self, tensor):
        """3Dの平均プーリング（空間を粗くする）"""
        return F.avg_pool3d(tensor, kernel_size=self.pool_size, stride=self.pool_size)

    def compute_gradients(self, u):
        """中心差分で速度場の空間勾配 (du_i/dx_j) を計算"""
        # u shape: (1, 3, nx, ny, nz)
        grad_x = (torch.roll(u, shifts=-1, dims=2) - torch.roll(u, shifts=1, dims=2)) / 2.0
        grad_y = (torch.roll(u, shifts=-1, dims=3) - torch.roll(u, shifts=1, dims=3)) / 2.0
        grad_z = (torch.roll(u, shifts=-1, dims=4) - torch.roll(u, shifts=1, dims=4)) / 2.0
        
        # 境界での差分誤差を簡易的にマスク（端は0にする）
        grad_x[:, :, 0, :, :] = 0; grad_x[:, :, -1, :, :] = 0
        grad_y[:, :, :, 0, :] = 0; grad_y[:, :, :, -1, :] = 0
        grad_z[:, :, :, :, 0] = 0; grad_z[:, :, :, :, -1] = 0
        
        return grad_x, grad_y, grad_z

    def process_snapshot(self, npz_path, save_id):
        # 1. データのロード
        data = np.load(npz_path)
        v_np = data['v'] # (nx, ny, nz, 3)
        cell_id_np = data['cell_id'] # (nx, ny, nz)
        
        # PyTorchテンソルに変換し、(Batch, Channel, X, Y, Z) の形にする
        v_tensor = torch.tensor(v_np, dtype=torch.float32, device=self.device).permute(3, 0, 1, 2).unsqueeze(0)
        
        # --- Step 1: マクロ量の粗視化 ---
        u_bar = self.filter_3d(v_tensor) # 入力データ X (粗い速度場)
        
        # 流体・固体のマスクも粗視化 (0.5未満なら流体、それ以上なら固体壁境界とみなす)
        mask_tensor = torch.tensor(cell_id_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        mask_bar = self.filter_3d(mask_tensor)
        is_fluid = (mask_bar < 0.5).float() # 流体領域フラグ
        
        # --- Step 2: SGS応力の計算 (tau_ij = bar{u_i * u_j} - bar{u}_i * bar{u}_j) ---
        tau_sgs = torch.zeros((1, 6, u_bar.shape[2], u_bar.shape[3], u_bar.shape[4]), device=self.device)
        
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                # 高解像度で掛け算してから平均
                uu_bar = self.filter_3d(v_tensor[:, i:i+1, ...] * v_tensor[:, j:j+1, ...])
                # 平均したものを掛け算
                bar_u_bar_u = u_bar[:, i:i+1, ...] * u_bar[:, j:j+1, ...]
                
                tau_sgs[:, idx, ...] = uu_bar - bar_u_bar_u
                idx += 1
                
        # --- Step 3: ひずみ速度テンソル S_ij の計算 ---
        grad_x, grad_y, grad_z = self.compute_gradients(u_bar)
        
        S_ij = torch.zeros_like(tau_sgs)
        S_ij[:, 0, ...] = grad_x[:, 0, ...] # S_xx
        S_ij[:, 1, ...] = 0.5 * (grad_x[:, 1, ...] + grad_y[:, 0, ...]) # S_xy
        S_ij[:, 2, ...] = 0.5 * (grad_x[:, 2, ...] + grad_z[:, 0, ...]) # S_xz
        S_ij[:, 3, ...] = grad_y[:, 1, ...] # S_yy
        S_ij[:, 4, ...] = 0.5 * (grad_y[:, 2, ...] + grad_z[:, 1, ...]) # S_yz
        S_ij[:, 5, ...] = grad_z[:, 2, ...] # S_zz

        # --- Step 4: 理想の渦動粘度 nu_t と tau マップの逆算 (Lillyの動的モデル近似) ---
        # nu_t = - (tau_sgs * S_ij) / (2 * S_ij * S_ij)
        num = -(tau_sgs[:, 0]*S_ij[:, 0] + tau_sgs[:, 3]*S_ij[:, 3] + tau_sgs[:, 5]*S_ij[:, 5] +
                2.0*(tau_sgs[:, 1]*S_ij[:, 1] + tau_sgs[:, 2]*S_ij[:, 2] + tau_sgs[:, 4]*S_ij[:, 4]))
        
        den = 2.0 * (S_ij[:, 0]**2 + S_ij[:, 3]**2 + S_ij[:, 5]**2 +
                     2.0*(S_ij[:, 1]**2 + S_ij[:, 2]**2 + S_ij[:, 4]**2)) + 1e-12 # ゼロ割防止
        
        nu_t = num / den
        
        # 物理的に負の粘性（逆カスケード）はLBMを即座に破壊するため、0以上でクリップ
        nu_t = torch.relu(nu_t)
        
        # LBMのタウマージン(tau - 0.5)への変換 (tau_turbulent = 3 * nu_t)
        # ※ここでは単位をスケーリングせず、そのまま正解ラベルとして扱う
        ideal_tau_margin = 3.0 * nu_t * is_fluid[:, 0, ...] 
        
        # 異常値のカット (最大でも0.1(tau=0.6)程度に抑える)
        ideal_tau_margin = torch.clamp(ideal_tau_margin, min=0.0, max=0.1)

        # --- Step 5: AI入力用データ (X) と 正解ラベル (Y) の保存 ---
        # 入力X:[Ux, Uy, Uz, 固体マスク] の4チャンネル
        X = torch.cat([u_bar, mask_bar], dim=1).squeeze(0).cpu().numpy() # (4, x, y, z)
        Y = ideal_tau_margin.squeeze(0).cpu().numpy() # (1, x, y, z) または (x, y, z)

        # 圧縮保存 (.npz または PyTorchの .pt)
        save_path = os.path.join(self.output_dir, f"dataset_{save_id:05d}.npz")
        np.savez_compressed(save_path, X=X, Y=Y)

def main():
    # 変換したいベンチマーク結果のディレクトリを指定
    base_dir = "results/ai_training_01_20260327_035553/training_data/run_01"
    output_dir = "ai_dataset/processed_ai_training_01_20260327_035553"

    base_dir = "C:/Users/hainy/Downloads/ai_training_backstep_20260328_030901/training_data/run_01"
    output_dir = "ai_dataset/processed_ai_training_backstep_20260328_030901"

    # base_dir = "C:/Users/hainy/Downloads/ai_training_inclined_plate_20260330_181853_training_data/training_data/run_01"
    # output_dir = "ai_dataset/processed_ai_training_inclined_plate_20260330_181853"

    # base_dir = "results/ai_training_rotating_20260401_022615/training_data/run_01"
    # output_dir = "ai_dataset/processed_ai_training_rotating_20260401_022615"

    # base_dir = "results/ai_training_mixed_convection_20260405_154305/training_data/run_01"
    # output_dir = "ai_dataset/processed_ai_training_mixed_convection_20260405_154305"

    builder = DatasetBuilder(input_dir=base_dir, output_dir=output_dir, pool_size=2)
    
    # スナップショットのリストを取得
    npz_files = sorted(glob.glob(os.path.join(base_dir, "snapshot_*.npz")))
    print(f"Found {len(npz_files)} snapshots.")
    
    for i, file_path in enumerate(tqdm(npz_files)):
        try:
            builder.process_snapshot(file_path, save_id=i)
        except Exception as e:
            print(f"\n[ERROR] ファイル '{file_path}' の処理中にエラーが発生しました！")
            traceback.print_exc()
            break # エラーが起きたら強制終了

if __name__ == "__main__":
    main()