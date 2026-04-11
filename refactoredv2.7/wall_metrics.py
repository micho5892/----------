"""
wall_metrics.py
幾何依存の壁面評価パラメータを analytics から切り離すためのヘルパー。
"""
from lbm_logger import get_logger

log = get_logger(__name__)


def get_wall_neighbor_dirs_xy():
    """
    壁面熱流束評価で使う近傍方向インデックス。
    D3Q19 のうち XY 平面内の4近傍（±x, ±y）を使う。
    """
    return [1, 2, 3, 4]


def channel_fluid_height_p(nx, ny, lx_p, wall_thickness_cells=10):
    """
    平行平板 benchmark 用の流体高さ H [m] を返す。
    壁は y 方向に配置されるため、H は y 方向セル数から算出する。
    物理長は等方格子 (dx=dy) を仮定し、dx = Lx_p / nx を用いる。
    """
    dx = lx_p / nx
    fluid_cells_y = max(0, ny - 2 * wall_thickness_cells)
    return fluid_cells_y * dx


def channel_hydraulic_diameter_p(nx, ny, lx_p, wall_thickness_cells=10):
    """
    平行平板 benchmark 用の代表長さ D_h を返す（D_h = 2H）。
    """
    h_p = channel_fluid_height_p(nx, ny, lx_p, wall_thickness_cells=wall_thickness_cells)
    d_h = 2.0 * h_p
    log.debug("channel_hydraulic_diameter_p: D_h=%s (nx=%s ny=%s)", d_h, nx, ny)
    return d_h


def benchmark_cylinder_diameter_p(lx_p):
    """
    build_benchmark_cylinder の幾何定義に対応する代表直径 D [m]。
    geometry.py で radius = nx * 0.05 のため、直径比は 0.1。
    """
    return 0.1 * lx_p
