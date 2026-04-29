# ==========================================================
# export.py — 可視化フレーム構築・GIF出力（Matplotlib リッチ版）
# ==========================================================
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from context import SOLID, SOLID_HEAT_SOURCE
from lbm_logger import get_logger
import io

log = get_logger(__name__)

def build_vis_frame(ctx, cfg, current_time_p=0.0, step=0):
    """Matplotlib を用いてリッチな可視化フレーム（RGB配列）を生成する"""
    
    # NumPy配列へ変換
    temp_np = ctx.temp.to_numpy()
    v_np = ctx.v.to_numpy()
    speed_np = np.linalg.norm(v_np, axis=-1)
    
    cell_id_np = ctx.cell_id.to_numpy()
    
    # 固体セルをマスク（NaNにすることで、Matplotlibが自動で透明・背景色にしてくれる）
    mask_np = (cell_id_np == SOLID) | (cell_id_np == SOLID_HEAT_SOURCE)
    temp_plot = np.where(mask_np, np.nan, temp_np)
    speed_plot = np.where(mask_np, np.nan, speed_np)
    
    # 描画対象の断面インデックス
    y_mid = cfg.ny // 2
    x_mid = cfg.nx // 2
    z_top = int(cfg.nz * 0.7)
    
    # 速度表示用の最大値（カラーバーの固定用）
    v_max = np.nanmax(speed_plot)
    if v_max < 1e-6: v_max = 1e-6  # ゼロ割回避

    # Figureの作成 (1200 x 900 ピクセル程度の画像になる)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')
    
    # ==========================================
    # 1. 左上: Front (XZ断面) - 温度場
    # ==========================================
    ax1 = axes[0, 0]
    # Matplotlibのimshowは (行, 列) のため転置(.T)が必要
    im1 = ax1.imshow(temp_plot[:, y_mid, :].T, origin='lower', cmap='coolwarm', vmin=0.0, vmax=1.0)
    ax1.set_title("Temperature (Front: XZ-plane)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ==========================================
    # 2. 左下: Front (XZ断面) - 速度場 ＋ 流線 (Streamline)
    # ==========================================
    ax2 = axes[1, 0]
    im2 = ax2.imshow(speed_plot[:, y_mid, :].T, origin='lower', cmap='viridis', vmin=0.0, vmax=v_max)
    ax2.set_title("Velocity & Streamlines (Front: XZ-plane)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 流線の描画
    u_2d = v_np[:, y_mid, :, 0].T
    w_2d = v_np[:, y_mid, :, 2].T
    # 速度がほぼゼロの領域は流線を引かないようマスク
    u_2d = np.where(np.isnan(speed_plot[:, y_mid, :].T), 0, u_2d)
    w_2d = np.where(np.isnan(speed_plot[:, y_mid, :].T), 0, w_2d)
    
    X, Z = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.nz))
    ax2.streamplot(X, Z, u_2d, w_2d, color='white', linewidth=0.8, density=1.2, arrowsize=1.5)

    # ==========================================
    # 3. 右上: Top (XY断面) - 温度場
    # ==========================================
    ax3 = axes[0, 1]
    im3 = ax3.imshow(temp_plot[:, :, z_top].T, origin='lower', cmap='coolwarm', vmin=0.0, vmax=1.0)
    ax3.set_title(f"Temperature (Top: XY-plane, Z={z_top})")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # ==========================================
    # 4. 右下: Side (YZ断面) - 温度場
    # ==========================================
    ax4 = axes[1, 1]
    im4 = ax4.imshow(temp_plot[x_mid, :, :].T, origin='lower', cmap='coolwarm', vmin=0.0, vmax=1.0)
    ax4.set_title("Temperature (Side: YZ-plane)")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 全体のタイトルに時間を表示
    fig.suptitle(f"Time: {current_time_p:.3f} s  |  Step: {step}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # --- Figure を RGB NumPy 配列に変換 ---
    buf = io.BytesIO()
    plt.savefig(buf, format='raw', dpi=100)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(
        int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1
    )
    
    # RGBAのA(アルファ値)を落としてRGBにする
    canvas = img_arr[:, :, :3]
    
    plt.close(fig)
    return canvas

def export_gif_frames(frames, filename, fps=12):
    imageio.mimsave(filename, frames, fps=fps)
    log.info("GIF saved: %s", filename)

def export_mp4_frames(frames, filename, fps=30):
    with imageio.get_writer(
        filename,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    log.info("MP4 saved: %s", filename)

def export_animation_frames(frames, filename, fps=12):
    lower_name = str(filename).lower()
    if lower_name.endswith(".mp4"):
        export_mp4_frames(frames, filename, fps=max(1, int(fps)))
    else:
        export_gif_frames(frames, filename, fps=max(1, int(fps)))