# ==========================================================
# data_exporter.py — AI学習用 DNSスナップショット エクスポーター
# ==========================================================
import taichi as ti
import numpy as np
import os
import json
from lbm_logger import get_logger

_log = get_logger(__name__)

class DataExporter:
    """
    AI学習用の高解像度（DNS相当）スナップショットを保存するクラス。
    指定ステップごとにVRAMからマクロ量を取得し、圧縮NumPyバイナリとして書き出します。
    """
    def __init__(self, context, cfg, output_dir="training_data/run_01"):
        self.ctx = context
        self.cfg = cfg
        self.output_dir = output_dir
        self.save_count = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            _log.info("[DataExporter] Created directory: %s", self.output_dir)

        # AI学習時に物理スケールを復元できるよう、メタデータも保存しておく
        self._save_metadata()

    def _save_metadata(self):
        meta_path = os.path.join(self.output_dir, "metadata.json")
        metadata = {
            "nx": self.cfg.nx, "ny": self.cfg.ny, "nz": self.cfg.nz,
            "dx": self.cfg.dx, "dt": self.cfg.dt,
            "Lx_p": self.cfg.Lx_p,
            "U_inlet_p": self.cfg.U_inlet_p
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

    def export_snapshot(self, step, current_time_p):
        """
        Taichiフィールド（GPU）からNumPy配列（CPU）へ変換し、圧縮保存する。
        """
        # 1. GPUからCPUへデータを一括転送
        # ※ ti.Vector.field の to_numpy() は (nx, ny, nz, 3) の配列になります
        v_np      = self.ctx.v.to_numpy()
        temp_np   = self.ctx.temp.to_numpy()
        # rho_np    = self.ctx.rho.to_numpy() # 密度変動が不要なら削って容量節約可能
        ctype_np  = self.ctx.cell_id.to_numpy()

        # 2. ファイルサイズの肥大化を防ぐため圧縮保存 (.npz)
        filename = os.path.join(self.output_dir, f"snapshot_{self.save_count:05d}.npz")
        
        # np.savez_compressed を使うことでファイルサイズを劇的に小さくできます
        np.savez_compressed(
            filename,
            v=v_np,              # 速度場: shape (nx, ny, nz, 3)
            temp=temp_np,        # 温度場: shape (nx, ny, nz)
            cell_id=ctype_np,    # 境界情報: shape (nx, ny, nz)
            step=step,
            time_p=current_time_p
        )
        
        _log.info(
            "[DataExporter] Saved snapshot %04d at t=%.3fs -> %s",
            self.save_count,
            current_time_p,
            filename,
        )
        self.save_count += 1