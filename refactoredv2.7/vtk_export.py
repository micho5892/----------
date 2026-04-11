# ==========================================================
# vtk_export.py — VTKExporter（タスク8: VTI 出力・ParaView 連携）
# ==========================================================
"""
指定ステップで ctx の rho, vel, temp, cell_id を VTI 形式で書き出す。
PyVista を使用（pip install pyvista）。
"""
import numpy as np
from lbm_logger import get_logger

log = get_logger(__name__)

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def export_step(ctx, step, filepath_template, dx=1.0):
    """
    ctx の rho, v, temp, cell_id を VTI で保存する。
    filepath_template: 例 "results/step_{:06d}.vti" → step=100 で results/step_000100.vti
    dx: 格子間隔（物理スケール、省略時は 1.0）
    """
    if not HAS_PYVISTA:
        raise RuntimeError("VTI export requires pyvista. Install with: pip install pyvista")

    path = filepath_template.format(step)
    nx, ny, nz = ctx.nx, ctx.ny, ctx.nz

    # Taichi field → numpy (i,j,k) の順（float16 の場合は PyVista 互換のため float32 に変換）
    rho_np = np.asarray(ctx.rho.to_numpy(), dtype=np.float32)
    temp_np = np.asarray(ctx.temp.to_numpy(), dtype=np.float32)
    cell_id_np = ctx.cell_id.to_numpy()
    v_np = np.asarray(ctx.v.to_numpy(), dtype=np.float32)  # (nx, ny, nz, 3)

    # PyVista ImageData: dimensions=(nx, ny, nz), データは Fortran 順で渡す
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dx, dx), origin=(0, 0, 0))
    grid.point_data["rho"] = rho_np.flatten(order="F")
    grid.point_data["temp"] = temp_np.flatten(order="F")
    grid.point_data["cell_id"] = cell_id_np.flatten(order="F").astype(np.int32)
    grid.point_data["velocity"] = v_np.reshape(-1, 3, order="F")

    grid.save(path)
    log.debug("VTI saved: %s", path)
    return path
