import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_fields(npz_path: str):
    data = np.load(npz_path)
    v = data["v"]            # (nx, ny, nz, 3)
    temp = data["temp"]      # (nx, ny, nz)
    # rho = data["rho"]      # 使いたくなったらコメントアウトを外す
    # cell_id = data["cell_id"]

    speed = np.linalg.norm(v, axis=-1)  # (nx, ny, nz)
    return temp, speed


def make_views(field: np.ndarray):
    """
    field: (nx, ny, nz)
    戻り値: (front, top, side)
      - front: x-z 平面 (y 中央スライス)
      - top  : x-y 平面 (z 70% スライス)
      - side : y-z 平面 (x 中央スライス)
    """
    nx, ny, nz = field.shape
    y_mid = ny // 2
    x_mid = nx // 2
    z_top = int(nz * 0.7)

    # front: (x, z)
    front = field[:, y_mid, :].T  # (nz, nx)

    # top: (x, y)
    top = field[:, :, z_top].T    # (ny, nx)

    # side: (y, z)
    side = field[x_mid, :, :].T   # (nz, ny)

    return front, top, side


def plot_views(views, titles, cmap, suptitle, save_path=None):
    front, top, side = views

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    for ax, img, title in zip(axes, [front, top, side], titles):
        im = ax.imshow(img, origin="lower", aspect="equal", cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle(suptitle)

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="温度・速度場を正面／平面／側面図で可視化するツール")
    parser.add_argument(
        "npz_path",
        help="`*_fields.npz` ファイルへのパス (例: results_.../heat_exchanger_..._fields.npz)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="プロットを表示せず、画像ファイル保存のみ行う",
    )
    args = parser.parse_args()

    npz_path = args.npz_path
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz ファイルが見つかりません: {npz_path}")

    temp, speed = load_fields(npz_path)

    # ビュー生成
    temp_views = make_views(temp)
    speed_views = make_views(speed)

    base, _ = os.path.splitext(npz_path)
    temp_png = base + "_temp_views.png"
    speed_png = base + "_speed_views.png"

    # 温度分布
    plot_views(
        temp_views,
        titles=["Front (x-z)", "Top (x-y)", "Side (y-z)"],
        cmap="turbo",
        suptitle="Temperature field views",
        save_path=temp_png,
    )

    # 速度絶対値分布
    plot_views(
        speed_views,
        titles=["Front (x-z)", "Top (x-y)", "Side (y-z)"],
        cmap="viridis",
        suptitle="Speed magnitude views",
        save_path=speed_png,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

