#!/usr/bin/env python3
"""
parallel_plates *_fields.npz: bulk temperature T_b(k) vs z-index, gradient |dT_b/dk|,
and a coarse check for thermal development near the outlet (low k; inlet is k=nz-1).

Usage (lbm-sim):
  python analyze_npz_thermal_development_z.py path/to/*_fields.npz
  python analyze_npz_thermal_development_z.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# parallel_plates（run_benchmark_channel と同じ）: y の上下固体帯
DEFAULT_WALL_THICKNESS = 10
# FLUID_A
FLUID = 0


def bulk_temperature_slice(
    temp: np.ndarray,
    v: np.ndarray,
    cell_id: np.ndarray,
    k: int,
    wall_thickness: int,
) -> tuple[float, float, int]:
    """
    固定 z=k の断面で、壁帯を除いた流体セル (cell_id==0) について
    質量流束重み付きバルク温度と、単純平均、セル数を返す。
    """
    ny = temp.shape[1]
    j0, j1 = wall_thickness, ny - wall_thickness
    T = temp[:, j0:j1, k]
    w = v[:, j0:j1, k, 2]
    cid = cell_id[:, j0:j1, k]
    mask = cid == FLUID
    if not np.any(mask):
        return float("nan"), float("nan"), 0
    Tm = T[mask]
    wm = np.abs(w[mask])
    wsum = float(np.sum(wm))
    if wsum < 1e-30:
        tb = float(np.mean(Tm))
    else:
        tb = float(np.sum(wm * Tm) / wsum)
    ta = float(np.mean(Tm))
    return tb, ta, int(np.sum(mask))


def analyze(
    npz_path: str,
    wall_thickness: int,
    skip_inlet_outlet_k: int,
    plateau_frac: float,
    grad_thresh_rel: float,
) -> None:
    data = np.load(npz_path)
    temp = np.asarray(data["temp"])
    v = np.asarray(data["v"])
    cell_id = np.asarray(data["cell_id"])
    nx, ny, nz = temp.shape

    tb = np.full(nz, np.nan, dtype=np.float64)
    ta = np.full(nz, np.nan, dtype=np.float64)
    ncells = np.zeros(nz, dtype=np.int32)

    for k in range(nz):
        tb[k], ta[k], ncells[k] = bulk_temperature_slice(
            temp, v, cell_id, k, wall_thickness
        )

    # 入口・出口付近は境界条件の影響が強いので解析から除外（オプション）
    k_lo = skip_inlet_outlet_k
    k_hi = nz - skip_inlet_outlet_k
    k_core = slice(k_lo, k_hi)
    z_index = np.arange(nz, dtype=np.float64)

    # 中心差分勾配 |dT_b/dk|（内部点）
    dtdk = np.full(nz, np.nan, dtype=np.float64)
    for k in range(1, nz - 1):
        if np.isfinite(tb[k + 1]) and np.isfinite(tb[k - 1]):
            dtdk[k] = 0.5 * (tb[k + 1] - tb[k - 1])
    abs_grad = np.abs(dtdk)

    # 下流プレートー: parallel_plates は入口 k=nz-1 → 出口 k=0 なので、
    # 「下流」は k が小さい側（出口付近）。plateau_frac 分を出口側から取る。
    n_tail = max(3, int(round((k_hi - k_lo) * plateau_frac)))
    tail_ks = np.arange(k_lo, min(k_lo + n_tail, k_hi), dtype=int)
    if tail_ks.size < 2:
        tail_ks = np.arange(k_lo, min(k_lo + 5, k_hi), dtype=int)

    tb_tail = tb[tail_ks]
    valid_tail = np.isfinite(tb_tail)
    tb_tail = tb_tail[valid_tail]
    tail_mean = float(np.mean(tb_tail)) if tb_tail.size else float("nan")
    tail_std = float(np.std(tb_tail)) if tb_tail.size > 1 else 0.0

    # 代表スケール（全体のバルク温度レンジ）
    tb_core = tb[k_core]
    finite = np.isfinite(tb_core)
    tmin = float(np.nanmin(tb_core))
    tmax = float(np.nanmax(tb_core))
    trange = max(tmax - tmin, 1e-12)

    grad_core = abs_grad[k_core]
    grad_max = float(np.nanmax(grad_core))
    grad_tail = abs_grad[tail_ks]
    grad_tail_mean = float(np.nanmean(grad_tail[np.isfinite(grad_tail)]))

    # 判定: 下流で勾配が小さく、かつプレートー内のばらつきが小さい → 発達に近い
    rel_fluct = tail_std / trange if trange > 0 else float("nan")
    rel_grad_tail = grad_tail_mean / trange if trange > 0 else float("nan")
    developed_like = (rel_grad_tail < grad_thresh_rel) and (
        rel_fluct < grad_thresh_rel * 2.0
    )

    out_dir = os.path.dirname(os.path.abspath(npz_path))
    base = os.path.splitext(os.path.basename(npz_path))[0]
    plot_path = os.path.join(out_dir, f"{base}_thermal_development_z.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        axes[0].plot(z_index, tb, "b-", lw=1.2, label=r"$T_b$ (|w|-weighted)")
        axes[0].plot(z_index, ta, "c--", lw=0.8, alpha=0.7, label=r"$\langle T \rangle$ (area)")
        axes[0].axvspan(0, k_lo, color="gray", alpha=0.15)
        axes[0].axvspan(k_hi, nz - 1, color="gray", alpha=0.15)
        if tail_ks.size:
            axes[0].axvspan(
                float(tail_ks[0]),
                float(tail_ks[-1]),
                color="green",
                alpha=0.12,
                label="plateau (outlet side)",
            )
        axes[0].set_ylabel("temperature")
        axes[0].set_title("Bulk temperature vs z index (fluid strip, cell_id==0)")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(z_index, np.clip(abs_grad, 1e-16, None), "r-", lw=1.0)
        axes[1].axvspan(0, k_lo, color="gray", alpha=0.15)
        axes[1].axvspan(k_hi, nz - 1, color="gray", alpha=0.15)
        axes[1].set_xlabel("z index k")
        axes[1].set_ylabel(r"$|dT_b/dk|$ (central diff.)")
        axes[1].set_title("Magnitude of axial gradient (small downstream ⇒ thermally developed-like)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[saved] {plot_path}")
    except Exception as e:
        print(f"[warn] could not save figure: {e}", file=sys.stderr)

    print("--- npz ---", npz_path)
    print(f"shape (nx,ny,nz)=({nx},{ny},{nz}), wall_thickness={wall_thickness}")
    print(
        f"flow: inlet high-k (k=nz-1) -> outlet low-k (k=0); "
        f"core k in [{k_lo}, {k_hi})"
    )
    print(f"T_b range over core: [{tmin:.6g}, {tmax:.6g}] (span {trange:.6g})")
    print(
        f"downstream (outlet-side) plateau k in [{tail_ks[0]}, {tail_ks[-1]}] "
        f"({tail_ks.size} indices, frac={plateau_frac:.2f})"
    )
    print(f"  mean(T_b) in plateau: {tail_mean:.6g}, std: {tail_std:.6g}")
    print(f"  std/span (plateau): {rel_fluct:.6g}")
    print(f"max |dT_b/dk| in core: {grad_max:.6g}")
    print(f"mean |dT_b/dk| in plateau window: {grad_tail_mean:.6g}")
    print(f"mean |dT_b/dk| / span: {rel_grad_tail:.6g}")
    print("--- summary ---")
    if developed_like:
        print(
            "Small gradient and low scatter near outlet -> "
            "bulk T likely saturated (thermally developed-like along z)."
        )
    else:
        print(
            "Gradient or plateau scatter still large -> longer domain, check BC/sponge, "
            "or not yet thermally developed."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="fields.npz から z 方向バルク温度の発達を見る"
    )
    p.add_argument(
        "npz",
        nargs="?",
        default=None,
        help="*_fields.npz のパス",
    )
    p.add_argument(
        "--wall-thickness",
        type=int,
        default=DEFAULT_WALL_THICKNESS,
        help="y 方向の上下壁厚（セル数）",
    )
    p.add_argument(
        "--skip-inlet-outlet-k",
        type=int,
        default=2,
        help="両端から除外する z セル数（入口出口境界の影響を弱める）",
    )
    p.add_argument(
        "--plateau-frac",
        type=float,
        default=0.15,
        help="下流「プレートー」判定に使う末尾の z の割合（core 内）",
    )
    p.add_argument(
        "--grad-thresh-rel",
        type=float,
        default=0.02,
        help="|dT_b/dk|/span がこの値より小さければ「小さい勾配」とみなす目安",
    )
    args = p.parse_args()
    path = args.npz
    if not path:
        print("エラー: npz ファイルを指定してください。", file=sys.stderr)
        p.print_help()
        sys.exit(1)
    if not os.path.isfile(path):
        print(f"エラー: ファイルがありません: {path}", file=sys.stderr)
        sys.exit(1)

    analyze(
        path,
        wall_thickness=args.wall_thickness,
        skip_inlet_outlet_k=args.skip_inlet_outlet_k,
        plateau_frac=args.plateau_frac,
        grad_thresh_rel=args.grad_thresh_rel,
    )


if __name__ == "__main__":
    main()
