import argparse
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Y(target) の分布を対数変換前後で比較して保存する"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="ai_dataset",
        help="学習データセットのルートフォルダ（*.npz を再帰探索）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="出力先の親フォルダ",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="対数変換前のスケーリング係数（log1p(scale*y+eps) の scale）",
    )
    parser.add_argument(
        "--log_eps",
        type=float,
        default=0.0,
        help="対数変換前のオフセット（log1p(scale*y+eps) の eps）",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="ヒストグラムのビン数",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="先頭から読む最大ファイル数（0 は全件）",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.9,
        help="可視化用に上側を切るパーセンタイル（重尾対策）",
    )
    parser.add_argument(
        "--fluid_only",
        action="store_true",
        help="X の4ch目マスクを使い、流体セルのみで分布を作る",
    )
    parser.add_argument(
        "--fluid_threshold",
        type=float,
        default=0.5,
        help="流体判定しきい値（X[3] < threshold を流体とみなす）",
    )
    return parser.parse_args()


def collect_targets(npz_files, fluid_only=False, fluid_threshold=0.5):
    chunks = []
    total_cells = 0
    selected_cells = 0
    for path in npz_files:
        data = np.load(path)
        y = data["Y"]
        if y.ndim == 3:
            y = np.expand_dims(y, axis=0)
        y = y.astype(np.float64, copy=False)

        if fluid_only:
            x = data["X"]
            if x.ndim != 4 or x.shape[0] < 4:
                raise ValueError(f"X shape が想定外です: {path}, shape={x.shape}")
            fluid_mask = x[3] < fluid_threshold
            selected = y[0][fluid_mask]
        else:
            selected = y.ravel()

        chunks.append(selected)
        total_cells += y.size
        selected_cells += selected.size

    if not chunks:
        return np.array([], dtype=np.float64), total_cells, selected_cells
    return np.concatenate(chunks), total_cells, selected_cells


def safe_log_transform(y_linear, scale, eps):
    transformed = np.log1p(np.maximum(scale * y_linear + eps, 0.0))
    return transformed


def describe(arr):
    if arr.size == 0:
        return {
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p99": np.nan,
            "p999": np.nan,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
    }


def save_stats_text(path, linear_stats, log_stats, args, num_files, total_cells, selected_cells):
    lines = []
    lines.append("# Target distribution statistics")
    lines.append("")
    lines.append(f"- num_files: {num_files}")
    lines.append(f"- mode: {'fluid_only' if args.fluid_only else 'all_cells'}")
    lines.append(f"- total_cells: {total_cells}")
    lines.append(f"- selected_cells: {selected_cells}")
    lines.append(
        f"- selected_ratio: {selected_cells / total_cells:.6f}" if total_cells > 0 else "- selected_ratio: nan"
    )
    lines.append(f"- scale: {args.scale}")
    lines.append(f"- log_eps: {args.log_eps}")
    lines.append(f"- transform: log1p(scale * y + eps)")
    if args.fluid_only:
        lines.append(f"- fluid_threshold: {args.fluid_threshold} (X[3] < threshold)")
    lines.append("")
    lines.append("## Linear target (y)")
    for k, v in linear_stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Log-transformed target (log1p(scale*y+eps))")
    for k, v in log_stats.items():
        lines.append(f"- {k}: {v}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.dataset_dir, "**", "*.npz"), recursive=True))
    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]

    if len(npz_files) == 0:
        print("エラー: npz ファイルが見つかりません。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"target_distribution_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(npz_files)} files.")
    mode_label = "fluid_only" if args.fluid_only else "all_cells"
    print(f"Collecting Y values ... mode={mode_label}")
    y_linear, total_cells, selected_cells = collect_targets(
        npz_files,
        fluid_only=args.fluid_only,
        fluid_threshold=args.fluid_threshold,
    )
    if y_linear.size == 0:
        print("エラー: Y が空です。")
        return

    y_log = safe_log_transform(y_linear, args.scale, args.log_eps)

    linear_stats = describe(y_linear)
    log_stats = describe(y_log)

    # 可視化の上限を揃える（重尾で全体が潰れないように）
    linear_hi = np.percentile(y_linear, args.clip_percentile)
    log_hi = np.percentile(y_log, args.clip_percentile)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(y_linear, bins=args.bins, color="steelblue", alpha=0.85)
    axes[0, 0].set_title("Linear target distribution (full range)")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(y_linear[y_linear <= linear_hi], bins=args.bins, color="royalblue", alpha=0.85)
    axes[0, 1].set_title(f"Linear target (<= p{args.clip_percentile})")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(y_log, bins=args.bins, color="darkorange", alpha=0.85)
    axes[1, 0].set_title("Log-transformed distribution (full range)")
    axes[1, 0].set_xlabel("log1p(scale*y+eps)")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(y_log[y_log <= log_hi], bins=args.bins, color="chocolate", alpha=0.85)
    axes[1, 1].set_title(f"Log-transformed (<= p{args.clip_percentile})")
    axes[1, 1].set_xlabel("log1p(scale*y+eps)")
    axes[1, 1].set_ylabel("count")

    fig.suptitle(
        "Target Distribution Comparison\n"
        f"files={len(npz_files)}, mode={mode_label}, selected={selected_cells}/{total_cells}, "
        f"scale={args.scale}, eps={args.log_eps}"
    )
    fig.tight_layout()

    png_path = os.path.join(out_dir, "distribution_comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    stats_path = os.path.join(out_dir, "stats.txt")
    save_stats_text(
        stats_path,
        linear_stats,
        log_stats,
        args,
        len(npz_files),
        total_cells,
        selected_cells,
    )

    # CDF も保存
    q = np.linspace(0, 100, 1001)
    linear_q = np.percentile(y_linear, q)
    log_q = np.percentile(y_log, q)
    cdf_path = os.path.join(out_dir, "quantiles.csv")
    with open(cdf_path, "w", encoding="utf-8") as f:
        f.write("percentile,linear,log_transformed\n")
        for p, lq, gq in zip(q, linear_q, log_q):
            f.write(f"{p:.1f},{lq:.12e},{gq:.12e}\n")

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"- {png_path}")
    print(f"- {stats_path}")
    print(f"- {cdf_path}")


if __name__ == "__main__":
    main()
