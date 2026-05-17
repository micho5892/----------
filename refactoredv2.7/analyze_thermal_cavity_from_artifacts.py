"""
保存済みの thermal_cavity 実行フォルダ内の *_meta.json と *_fields.npz から
run_benchmark_thermal.analyze_and_plot_thermal_cavity を再実行する。

例:
  conda run -n lbm-sim python refactoredv2.7/analyze_thermal_cavity_from_artifacts.py ^
    "results/validation_thermal/thermal_cavity_20260515_010755" --target-ra 1e5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from run_benchmark_thermal import analyze_and_plot_thermal_cavity  # noqa: E402


def _load_meta(meta_path: str) -> dict:
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _fluid_props(meta: dict) -> dict:
    dp = meta.get("domain_properties") or {}
    fluid = dp.get("0")
    if fluid is None:
        fluid = dp.get(0)
    if not isinstance(fluid, dict):
        raise KeyError(
            "meta.json に domain_properties['0']（流体）がありません。"
        )
    return fluid


def _resolve_paths(run_dir: str, stem: str | None) -> tuple[str, str]:
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"フォルダが見つかりません: {run_dir}")

    if stem:
        meta_path = os.path.join(run_dir, f"{stem}_meta.json")
        npz_path = os.path.join(run_dir, f"{stem}_fields.npz")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(meta_path)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(npz_path)
        return meta_path, npz_path

    metas = sorted(glob.glob(os.path.join(run_dir, "*_meta.json")))
    if not metas:
        raise FileNotFoundError(
            f"*_meta.json が見つかりません: {run_dir}\n"
            "（複数ある場合は --stem で接頭辞を指定してください）"
        )
    if len(metas) > 1:
        latest = max(metas, key=os.path.getmtime)
        print(
            f"[WARN] *_meta.json が {len(metas)} 件あります。"
            f"更新時刻が最新のものを使います:\n  {latest}"
        )
        meta_path = latest
    else:
        meta_path = metas[0]

    base = os.path.basename(meta_path)
    if not base.endswith("_meta.json"):
        raise ValueError(f"想定外のファイル名: {base}")
    prefix = base[: -len("_meta.json")]
    npz_path = os.path.join(run_dir, f"{prefix}_fields.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"対応する *_fields.npz が見つかりません: {npz_path}\n"
            "--stem で正しい接頭辞を指定できるか確認してください。"
        )
    return meta_path, npz_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "指定フォルダの meta.json / fields.npz から "
            "analyze_and_plot_thermal_cavity を実行する。"
        )
    )
    p.add_argument(
        "run_dir",
        help="thermal_cavity の出力フォルダ（*_meta.json と *_fields.npz を含む）",
    )
    p.add_argument(
        "--target-ra",
        type=float,
        required=True,
        help="ベンチマーク比較用の Ra（例: 1e4, 1e5）。de Vahl Davis 表は 1e4/1e5 のみ。",
    )
    p.add_argument(
        "--stem",
        default=None,
        help=(
            "ファイル接頭辞（例: thermal_cavity_20260515_010755）。"
            "省略時はフォルダ内の *_meta.json を自動選択。"
        ),
    )
    args = p.parse_args()

    meta_path, npz_path = _resolve_paths(args.run_dir, args.stem)
    meta = _load_meta(meta_path)
    nx = int(meta["nx"])
    nz = int(meta["nz"])
    dx = float(meta["dx"])
    dt = float(meta["dt"])

    fluid = _fluid_props(meta)
    k = float(fluid["k"])
    rho = float(fluid["rho"])
    Cp = float(fluid["Cp"])
    alpha_f = k / (rho * Cp)

    nx_f = nx - 2
    L_eff = dx * nx_f
    out_dir = os.path.abspath(args.run_dir)

    print(f"meta:    {meta_path}")
    print(f"npz:     {npz_path}")
    print(f"out_dir: {out_dir}")
    print(f"nx={nx}, nz={nz}, dx={dx}, dt={dt}, L_eff={L_eff}, alpha_f={alpha_f}")

    analyze_and_plot_thermal_cavity(
        npz_path,
        nx,
        nz,
        args.target_ra,
        L_eff,
        alpha_f,
        dx,
        dt,
        out_dir,
    )


if __name__ == "__main__":
    main()
