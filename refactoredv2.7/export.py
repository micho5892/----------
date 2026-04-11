# ==========================================================
# export.py — 可視化フレーム構築・GIF出力（タスク3）
# タスク8: GIF はレガシー/オプションとして残す。VTI は vtk_export.export_step を使用。
# ==========================================================
import numpy as np
import imageio.v2 as imageio
from context import SOLID, SOLID_HEAT_SOURCE


def build_vis_frame(ctx, cfg):
    """1ステップ分の可視化キャンバス（三面図＋パーティクル）を返す。"""
    temp_np = ctx.temp.to_numpy()
    cell_id_np = ctx.cell_id.to_numpy()
    mask_np = (cell_id_np == SOLID) | (cell_id_np == SOLID_HEAT_SOURCE)

    front = temp_np[:, cfg.ny//2, :]
    front_mask = mask_np[:, cfg.ny//2, :]
    side = temp_np[cfg.nx//2, :, :]
    side_mask = mask_np[cfg.nx//2, :, :]
    top_z = int(cfg.nz * 0.7)
    top = temp_np[:, :, top_z]
    top_mask = mask_np[:, :, top_z]

    def make_rgb(val, msk):
        rgb = np.zeros((*val.shape, 3), dtype=np.uint8)
        t_mask = (msk == False)
        p_mask = (msk == True)
        rgb[:, :, 0][t_mask] = (np.clip(val[t_mask], 0, 1) * 255).astype(np.uint8)
        rgb[:, :, 2][t_mask] = ((1 - np.clip(val[t_mask], 0, 1)) * 255).astype(np.uint8)
        rgb[p_mask] = [128, 128, 128]
        return rgb

    front_rgb = np.flipud(np.transpose(make_rgb(front, front_mask), (1, 0, 2)))
    side_rgb = np.flipud(np.transpose(make_rgb(side, side_mask), (1, 0, 2)))
    top_rgb = np.flipud(np.transpose(make_rgb(top, top_mask), (1, 0, 2)))

    h_top, w_top = top_rgb.shape[:2]
    h_front, w_front = front_rgb.shape[:2]
    h_side, w_side = side_rgb.shape[:2]
    total_h = h_top + h_front
    total_w = max(w_top, w_front + w_side)
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[0:h_top, 0:w_top] = top_rgb
    canvas[h_top:total_h, 0:w_front] = front_rgb
    canvas[h_top:total_h, w_front:w_front+w_side] = side_rgb

    p_pos = ctx.particle_pos.to_numpy()
    for p in p_pos:
        if not np.isfinite(p).all():
            continue
        x, y, z = int(p[0]), int(p[1]), int(p[2])
        z_draw = cfg.nz - 1 - z
        y_draw_top = cfg.ny - 1 - y
        if 0 <= x < cfg.nx and 0 <= y_draw_top < cfg.ny:
            canvas[y_draw_top, x] = [255, 255, 255]
        if 0 <= x < cfg.nx and 0 <= z_draw < cfg.nz:
            canvas[h_top + z_draw, x] = [255, 255, 255]
        if 0 <= y < cfg.ny and 0 <= z_draw < cfg.nz:
            canvas[h_top + z_draw, w_front + y] = [255, 255, 255]

    return canvas


def export_gif_frames(frames, filename, fps=12):
    """フレーム列をGIFファイルに保存する。"""
    imageio.mimsave(filename, frames, fps=fps)
