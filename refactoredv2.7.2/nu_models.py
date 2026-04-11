"""
nu_models.py
Nu 定義を benchmark ごとに差し替えるための軽量モデル定義。
"""

from dataclasses import dataclass

from wall_metrics import benchmark_cylinder_diameter_p, channel_hydraulic_diameter_p


@dataclass
class NuModelSpec:
    name: str
    l_ref_p: float
    # 0: bulk fluid k, 1: wall-side k
    k_ref_mode: int


def build_nu_model(cfg):
    """
    cfg から Nu 定義を構築する。
    - parallel_plates: Nu = h * D_h / k_bulk
    - cylinder(仮):    Nu = h * D / k_bulk （D は現状 fallback で D_h）
    """
    benchmark_name = getattr(cfg, "benchmark_name", "unknown")

    if benchmark_name == "parallel_plates":
        d_h = float(channel_hydraulic_diameter_p(cfg.nx, cfg.ny, cfg.Lx_p, wall_thickness_cells=10))
        return NuModelSpec(
            name="internal_flow_dh_bulk_k",
            l_ref_p=d_h,
            k_ref_mode=0,
        )

    if benchmark_name in ("benchmark_cylinder", "cylinder"):
        d_cyl = float(benchmark_cylinder_diameter_p(cfg.Lx_p))
        return NuModelSpec(
            name="cylinder_d_bulk_k",
            l_ref_p=d_cyl,
            k_ref_mode=0,
        )

    # fallback
    d_h = float(channel_hydraulic_diameter_p(cfg.nx, cfg.ny, cfg.Lx_p, wall_thickness_cells=10))
    return NuModelSpec(
        name="fallback_internal_flow",
        l_ref_p=d_h,
        k_ref_mode=0,
    )
