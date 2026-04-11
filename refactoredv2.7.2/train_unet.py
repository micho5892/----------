import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================================
# 0. 学習設定（切り替えしやすいように集約）
# ==========================================================
USE_LOG_TARGET = True
LOG_EPS = 0.0  # 必要なら >0 にして log1p 前にオフセット可
USE_WEIGHTED_LOSS = False
WEIGHT_ALPHA = 0.0
LINEAR_TARGET_SCALE = 1000.0  # 対数/線形どちらでもターゲット振幅調整に使う

def inverse_target_transform(y_tensor):
    """学習空間 -> 物理量空間 へ戻す（推論時にも利用可）。"""
    if USE_LOG_TARGET:
        y_scaled = torch.clamp(torch.expm1(y_tensor) - LOG_EPS, min=0.0)
        return y_scaled / LINEAR_TARGET_SCALE
    return y_tensor / LINEAR_TARGET_SCALE

# ==========================================================
# 1. 3D U-Net モデルの定義 (軽量・VRAM節約設計)
# ==========================================================
class DoubleConv3D(nn.Module):
    """3D畳み込み -> 正規化 -> 活性化関数 を2回繰り返すブロック"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True) 
        )
    def forward(self, x):
        return self.conv(x)

class SGS_UNet3D(nn.Module):
    """
    Sub-Grid Scale 応力 (tau_margin) 推論用 3D U-Net
    入力: 4チャンネル (Ux, Uy, Uz, 固体マスク)
    出力: 1チャンネル (tau_turbulent)
    """
    def __init__(self, in_channels=4, out_channels=1, base_c=16):
        super().__init__()
        # ダウンサンプリング (エンコーダ)
        self.inc = DoubleConv3D(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_c, base_c*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_c*2, base_c*4))
        
        # アップサンプリング (デコーダ: U-Net特有のスキップ接続を含む)
        self.up1 = nn.ConvTranspose3d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(base_c*4, base_c*2) # スキップ接続でチャンネルが倍になるため base_c*4
        
        self.up2 = nn.ConvTranspose3d(base_c*2, base_c, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(base_c*2, base_c)
        
        self.outc = nn.Conv3d(base_c, out_channels, kernel_size=1)
        
        # tauは必ずプラスなのでReLUで0以下をカット
        # self.relu = nn.ReLU(inplace=True) # 正解データを対数変換するときはReLUを外す

    def forward(self, x):
        # エンコーダ
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # デコーダ
        y = self.up1(x3)
        y = torch.cat([y, x2], dim=1) # スキップ接続
        y = self.conv_up1(y)
        
        y = self.up2(y)
        y = torch.cat([y, x1], dim=1) # スキップ接続
        y = self.conv_up2(y)
        
        out = self.outc(y)
        # return self.relu(out)
        return out

# ==========================================================
# 2. データセットとデータローダー (ランダムクロップ機能付き)
# ==========================================================
class CFDDataset(Dataset):
    def __init__(self, npz_files, crop_size=(64, 32, 64), use_log_target=True, log_eps=0.0):
        self.files = npz_files
        self.crop_size = crop_size
        self.use_log_target = use_log_target
        self.log_eps = log_eps
        
    def __len__(self):
        return len(self.files)
    
# train_unet.py 内の CFDDataset クラス

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = data['X'] # (4, nx, ny, nz)
        Y = data['Y'] # (1, nx, ny, nz)

        # =======================================================
        # ▼ 追加：Yが3次元(チャンネルがない)なら、先頭に(1,)を追加して4次元にする
        # =======================================================
        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=0) # (1, nx, ny, nz) になる！

        
        cx, cy, cz = self.crop_size
        channels, sx, sy, sz = X.shape
        
        # ==========================================================
        # ★ 修正：スポンジ層の除外（Z方向の安全マージン）
        # ==========================================================
        # 元の空間で40セルのスポンジ層があった。
        # dataset_builderで pool_size=2 で半分に縮小されているため、
        # 縮小後の空間では 20セル分 が猛毒領域となる。
        # 安全を見て、Z方向の両端「25セル分」は絶対にクロップしないようにする。
        margin_z = 25 

        # ==========================================================
        # ★追加：Target-Aware Sampling (乱流狙い撃ちクロップ)
        # ==========================================================
        max_retries = 10  # 最大10回までサイコロを振り直す
        
        for _ in range(max_retries):
            # X と Y は物理的に正しい壁面境界層なので、端から端までフルに使う
            ix = np.random.randint(0, sx - cx + 1)
            iy = np.random.randint(0, sy - cy + 1)
            
            # Z だけは、端っこ(スポンジ層)を避けた中央の「純粋な物理空間」からのみ切り抜く
            # (sz - margin_z) が右端の限界、margin_z が左端の限界
            iz = np.random.randint(margin_z, sz - cz - margin_z + 1)
            
            Y_crop = Y[:, ix:ix+cx, iy:iy+cy, iz:iz+cz]

            # 対数変換学習ではダイナミックレンジ圧縮で偏りが緩和されるため、
            # target-aware 判定をスキップしてランダムクロップを採用する。
            if self.use_log_target:
                break
            
            # 切り抜いた箱の中に「激しい渦(Yが0.01以上：スケール後で10以上)」があるか？
            if Y_crop.max() > 0.01:
                # 激しい渦が見つかったら、即座に採用してループを抜ける！
                break
                
            # もしハズレ(全部ゼロ付近)でも、一定確率(例えば20%)でそのまま採用する。
            # (※AIが「ゼロの場所はちゃんとゼロにする」ことを忘れないようにするため)
            if np.random.rand() < 0.2:
                break
                
        # 最終的に決まった座標で X も切り抜く
        X_crop = X[:, ix:ix+cx, iy:iy+cy, iz:iz+cz]

        if self.use_log_target:
            # log(1 + scale*y + eps): 小さい物理量でも学習しやすくする
            Y_crop = np.log1p(Y_crop * LINEAR_TARGET_SCALE + self.log_eps)

        return torch.tensor(X_crop, dtype=torch.float32), torch.tensor(Y_crop, dtype=torch.float32)

# ==========================================================
# 3. メイン学習ループ
# ==========================================================
def main():
    # データセットがあるフォルダを指定（全種類を含めるならワイルドカードを使う）
    # ai_dataset フォルダの直下にある「すべてのサブフォルダ内」の .npz を読み込む
    dataset_dir = "ai_dataset" 
    npz_files = sorted(glob.glob(os.path.join(dataset_dir, "*/*.npz")))
    
    print(f"Found {len(npz_files)} training samples.")
    
    if len(npz_files) == 0:
        print("エラー: データが見つかりません。")
        return

    # ハイパーパラメータの設定
    # batch_size = 4        # VRAMがキツイ場合は 2 や 1 に下げてください
    batch_size = 8        # パラメータ調整のため、batch_sizeを1にしてみる
    num_epochs = 30       # 学習回数
    learning_rate = 1e-3  # 学習率
    
    # データの80%を学習用、20%を検証用に分割
    train_size = int(0.8 * len(npz_files))
    train_files = npz_files[:train_size]
    val_files = npz_files[train_size:]
    
    train_dataset = CFDDataset(
        train_files, crop_size=(64, 32, 64),
        use_log_target=USE_LOG_TARGET, log_eps=LOG_EPS
    )
    val_dataset = CFDDataset(
        val_files, crop_size=(64, 32, 64),
        use_log_target=USE_LOG_TARGET, log_eps=LOG_EPS
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # デバイスの設定 (CUDAが使えるか確認)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデル、損失関数、オプティマイザの初期化
    model = SGS_UNet3D().to(device)
    criterion = nn.SmoothL1Loss(reduction='none') 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 学習ループ
    
    print("--- Starting Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mae_phys = 0.0
        
        max_pred_val = 0.0 
        max_true_val = 0.0
        min_pred_val = float('inf')
        min_true_val = float('inf')

        alpha = WEIGHT_ALPHA if USE_WEIGHTED_LOSS else 0.0
        
        for batch_X, batch_Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            target_Y = batch_Y if USE_LOG_TARGET else (batch_Y * LINEAR_TARGET_SCALE)
            
            fluid_mask = (batch_X[:, 3:4, :, :, :] < 0.5).float()
            
            optimizer.zero_grad()
            predictions = model(batch_X)

            raw_loss = criterion(predictions, target_Y)
            
            # 生のLossを計算 (SmoothL1)
            # raw_loss = criterion(predictions, target_Y)
            
            # =======================================================
            # ★特効薬3：値に応じた「重み（Weight）」の導入
            # =======================================================
            # 正解データ(target_Y)が大きい場所ほど、Lossを何倍にも増幅する。
            # 例: target_Yが 0 なら 重み 1.0 (等倍)
            #     target_Yが 100 なら 重み 11.0 (11倍のペナルティ！)
            # 係数(0.1)は、様子を見て 0.5 や 1.0 に上げても良いです。
            weight_map = 1.0 + alpha * target_Y
            
            # 重みを掛け合わせた上で、流体領域のLossの平均をとる
            loss = (raw_loss * weight_map * fluid_mask).sum() / (fluid_mask.sum() + 1e-8)
            # =======================================================
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)

            pred_phys = inverse_target_transform(predictions)
            true_phys = inverse_target_transform(target_Y)
            abs_err_phys = torch.abs(pred_phys - true_phys)
            mae_phys = (abs_err_phys * fluid_mask).sum() / (fluid_mask.sum() + 1e-8)
            train_mae_phys += mae_phys.item() * batch_X.size(0)
            
            max_pred_val = max(max_pred_val, pred_phys.max().item())
            max_true_val = max(max_true_val, true_phys.max().item())

            min_pred_val = min(min_pred_val, pred_phys.min().item())
            min_true_val = min(min_true_val, true_phys.min().item())
            
        train_loss /= len(train_loader.dataset)
        train_mae_phys /= len(train_loader.dataset)
        
        # 検証(Validation)フェーズ
        model.eval()
        val_loss = 0.0
        val_mae_phys = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                target_Y = batch_Y if USE_LOG_TARGET else (batch_Y * LINEAR_TARGET_SCALE)
                fluid_mask = (batch_X[:, 3:4, :, :, :] < 0.5).float()
                
                predictions = model(batch_X)
                raw_loss = criterion(predictions, target_Y)
                
                # 検証用にも同じ重みを適用
                weight_map = 1.0 + alpha * target_Y
                loss = (raw_loss * weight_map * fluid_mask).sum() / (fluid_mask.sum() + 1e-8)
                
                val_loss += loss.item() * batch_X.size(0)

                pred_phys = inverse_target_transform(predictions)
                true_phys = inverse_target_transform(target_Y)
                abs_err_phys = torch.abs(pred_phys - true_phys)
                mae_phys = (abs_err_phys * fluid_mask).sum() / (fluid_mask.sum() + 1e-8)
                val_mae_phys += mae_phys.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae_phys /= len(val_loader.dataset)
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss(transformed): {train_loss:.4e} | Val Loss(transformed): {val_loss:.4e} "
            f"| Train MAE(physical): {train_mae_phys:.4e} | Val MAE(physical): {val_mae_phys:.4e}"
        )
        print(
            f"  -> Physical Target Max: {max_true_val:.4f} | Physical AI Prediction Max: {max_pred_val:.4f} "
            f"| Physical Target Min: {min_true_val:.4f} | Physical AI Prediction Min: {min_pred_val:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "sgs_unet_model.pth")
            print("  -> Saved best model!")

if __name__ == "__main__":
    main()