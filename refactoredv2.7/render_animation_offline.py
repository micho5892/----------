import argparse
import glob
import json
import os

import numpy as np

from config import SimConfig
from export import build_vis_frame_from_arrays, export_animation_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Offline snapshots から動画を再生成する")
    parser.add_argument("--snapshot-dir", required=True, help="vis_snapshots ディレクトリ")
    parser.add_argument("--output", default=None, help="出力ファイル(.mp4/.gif)。未指定なら render_meta.json を使用")
    parser.add_argument("--fps", type=int, default=None, help="FPS上書き")
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot_dir = os.path.abspath(args.snapshot_dir)
    meta_path = os.path.join(snapshot_dir, "render_meta.json")
    cell_id_path = os.path.join(snapshot_dir, "cell_id.npy")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"render_meta.json が見つかりません: {meta_path}")
    if not os.path.isfile(cell_id_path):
        raise FileNotFoundError(f"cell_id.npy が見つかりません: {cell_id_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    output_path = args.output or meta.get("filename")
    if not output_path:
        raise ValueError("出力先を決定できません。--output を指定してください。")
    fps = max(1, int(args.fps if args.fps is not None else meta.get("fps", 12)))

    cell_id_np = np.load(cell_id_path)
    cfg = SimConfig(
        nx=int(meta["nx"]),
        ny=int(meta["ny"]),
        nz=int(meta["nz"]),
        filename=output_path,
    )

    frame_files = sorted(glob.glob(os.path.join(snapshot_dir, "step_*.npz")))
    if not frame_files:
        raise FileNotFoundError(f"step_*.npz が見つかりません: {snapshot_dir}")

    frames = []
    for snap_path in frame_files:
        snap = np.load(snap_path)
        temp_np = snap["temp"]
        v_np = snap["v"]
        current_time_p = float(snap["time_p"])
        step = int(snap["step"])
        frame = build_vis_frame_from_arrays(temp_np, v_np, cell_id_np, cfg, current_time_p, step)
        frames.append(frame)

    export_animation_frames(frames, output_path, fps=fps)
    print(f"[SUCCESS] Animation rendered: {output_path} ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
