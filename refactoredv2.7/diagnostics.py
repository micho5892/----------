# ==========================================================
# diagnostics.py — 数値暴走の原因究明用ログ（rho / temp / v / 分布関数）
# ==========================================================
"""
暴走の切り分けのため、流体セル（FLUID_A, INLET, OUTLET）だけを対象に以下を記録する:

平行平板ベンチマーク用の輸送診断は `log_parallel_plates_transport_diagnostics` を参照。

- rho: min / max / mean、NaN/Inf 個数、rho<=0 のセル数、rho<1e-6 のセル数
- temp: min / max / mean、NaN/Inf 個数、異常個数（temp<0 または temp>2 のセル数）
- |v|: min / max / mean、NaN/Inf 個数
- (log_distributions=True のとき) f_old, g_old: min / max、負の値の個数

ログの取り場所:
- main の「step % vis_interval == 0」のブロック内（update_macro の後）。
- step<=2500 のあいだは 30 ステップごと、step 1200～1600 で 100 ステップごとに f_old/g_old も出力。

結果からわかること:
- rho_min が急に下がる / rho_neg や rho<1e-6 が増える → collide の Pi/rho や update_macro の new_rho が怪しい（rho≈0 での除算）。
- temp に NaN/Inf や temp_out 増加 → update_macro の new_temp か apply_heat_flux_bc の書き換えが怪しい。
- |v| が NaN/Inf → update_macro の v = new_v/rho で rho≈0 による除算が怪しい。
- f_old/g_old が負や巨大 → collide か stream_and_bc の出力が破綻。次の update_macro で rho/temp が壊れる。
"""
import math
import numpy as np
from context import FLUID_A, INLET, OUTLET, SOLID, SOLID_HEAT_SOURCE
from lbm_logger import get_logger

_log = get_logger(__name__)


def log_parallel_plates_transport_diagnostics(
    ctx,
    cfg,
    *,
    k_target,
    logger=None,
    step=None,
    wall_thickness_cells=10,
):
    """
    平行平板（ポアズイユ＋熱）検証向けに、(1)τ からの Pr、(2)熱助走長さの目安、(3)横流れ、(4)ρ とマッハ数をログする。

    - (1) SimConfig と同じ τ–ν 関係: nu_lbm = (tau_f-0.5)/3, alpha_lbm = (tau_g-0.5)/3, Pr = nu_lbm/alpha_lbm
    - (2) 物性は domain_properties[0] の nu、幾何は wall_metrics.channel_hydraulic_diameter_p。
          主流方向最大 |vz| を格子単位で取り、代表スケール U_inlet_p / u_lbm_inlet で m/s に換算。
          L_th ≈ 0.05 * Re_Dh * Pr * D_h（入口助走の経験則; 数量級の確認用）。
    - (3) 流体マスク上の max|vx|, max|vy|
    - (4) 流体かつ rho>0.1 の min/max rho、格子音速 cs=1/√3 に対する max|vz|/cs
    """
    log = logger if logger is not None else _log
    prefix = f"[parallel_plates transport] step={step} " if step is not None else "[parallel_plates transport] "

    cid = int(FLUID_A)
    tau_f = float(ctx.tau_f_table.to_numpy()[cid])
    tau_g = float(ctx.tau_g_table.to_numpy()[cid])
    nu_lbm = (tau_f - 0.5) / 3.0
    alpha_lbm = (tau_g - 0.5) / 3.0
    pr_lbm = nu_lbm / alpha_lbm if alpha_lbm > 1e-30 else float("nan")
    log.info(
        "%s(1) tau_f=%.8f tau_g=%.8f | nu*dt/dx^2=%.6e alpha*dt/dx^2=%.6e | Pr_LBM=%.6f",
        prefix,
        tau_f,
        tau_g,
        nu_lbm,
        alpha_lbm,
        pr_lbm,
    )

    from wall_metrics import channel_hydraulic_diameter_p

    v_np = ctx.v.to_numpy()
    rho_np = ctx.rho.to_numpy()
    cell_id = ctx.cell_id.to_numpy()
    fluid_mask = (cell_id == FLUID_A) | (cell_id == INLET) | (cell_id == OUTLET)

    vz_all = np.abs(v_np[:, :, :, 2])
    max_vz_lbm = float(np.max(vz_all[fluid_mask])) if np.any(fluid_mask) else 0.0

    props0 = cfg.domain_properties.get(0, {}) if isinstance(cfg.domain_properties, dict) else {}
    nu_p = float(props0.get("nu", 0.0)) if props0 else 0.0
    d_h_p = float(
        channel_hydraulic_diameter_p(
            cfg.nx, cfg.ny, cfg.Lx_p, wall_thickness_cells=wall_thickness_cells
        )
    )
    u_scale = float(cfg.U_inlet_p) / float(cfg.u_lbm_inlet) if cfg.u_lbm_inlet > 1e-30 else 0.0
    v_max_phys = max_vz_lbm * u_scale
    re_dh = v_max_phys * d_h_p / nu_p if nu_p > 1e-30 else float("nan")
    l_th_p = 0.05 * re_dh * pr_lbm * d_h_p if math.isfinite(re_dh) and math.isfinite(pr_lbm) else float("nan")
    dz_phys = float(cfg.dx)
    channel_len_p = float(cfg.nz) * dz_phys

    log.info(
        "%s(2) max|vz|_lbm=%.6e  U_scale=U_inlet_p/u_lbm_inlet=%.6e  max|vz|_phys≈%.6e m/s | "
        "D_h=%.6e m nu_phys=%.6e m2/s | Re_Dh≈%.4f Pr=%.6f | L_th≈0.05*Re*Pr*D_h=%.6e m | "
        "nz*dz≈%.6e m k_target=%s (main と同じ断面; dz=dx)",
        prefix,
        max_vz_lbm,
        u_scale,
        v_max_phys,
        d_h_p,
        nu_p,
        re_dh,
        pr_lbm,
        l_th_p,
        channel_len_p,
        int(k_target),
    )

    max_vx = float(np.max(np.abs(v_np[:, :, :, 0][fluid_mask]))) if np.any(fluid_mask) else 0.0
    max_vy = float(np.max(np.abs(v_np[:, :, :, 1][fluid_mask]))) if np.any(fluid_mask) else 0.0
    log.info(
        "%s(3) max|vx|_lbm=%.6e max|vy|_lbm=%.6e (横流れ・数値ノイズの目安)",
        prefix,
        max_vx,
        max_vy,
    )

    rho_ok = fluid_mask & (rho_np > 0.1)
    if np.any(rho_ok):
        r_sub = rho_np[rho_ok]
        min_rho = float(np.min(r_sub))
        max_rho = float(np.max(r_sub))
    else:
        min_rho = float("nan")
        max_rho = float("nan")
    cs = 1.0 / math.sqrt(3.0)
    mach_lbm = max_vz_lbm / cs if cs > 0 else float("nan")
    log.info(
        "%s(4) rho (fluid & rho>0.1): min=%.6f max=%.6f | Mach_lbm=max|vz|/cs=%.6f (cs=1/sqrt(3))",
        prefix,
        min_rho,
        max_rho,
        mach_lbm,
    )


def log_field_diagnostics(ctx, step, log_distributions=False):
    """
    ctx の流体セル（FLUID_A, INLET, OUTLET）のみをマスクし、
    rho, temp, |v| および必要なら f_old, g_old の統計を共通ロガーへ出力する。
    log_distributions=True のとき f_old, g_old の min/max も出す（重い）。
    """
    cell_id = ctx.cell_id.to_numpy()
    fluid_mask = (cell_id == FLUID_A) | (cell_id == INLET) | (cell_id == OUTLET)
    n_fluid = int(np.sum(fluid_mask))

    def stats(label, arr, mask):
        a = np.asarray(arr)
        if a.shape != mask.shape and a.ndim == 4:
            flat = a[mask].reshape(-1)
        else:
            flat = a[mask].ravel()
        nan_c = int(np.sum(np.isnan(flat)))
        inf_c = int(np.sum(np.isinf(flat)))
        ok = np.isfinite(flat)
        if np.sum(ok) == 0:
            return f"{label}: min=n/a max=n/a mean=n/a nan={nan_c} inf={inf_c}"
        valid = flat[ok]
        return (
            f"{label}: min={valid.min():.6f} max={valid.max():.6f} mean={valid.mean():.6f} "
            f"nan={nan_c} inf={inf_c}"
        )

    def count_anomaly(arr, mask, low=None, high=None):
        a = np.asarray(arr).ravel()
        m = mask.ravel()
        a = a[m]
        c = 0
        if low is not None:
            c += int(np.sum(a <= low))
        if high is not None:
            c += int(np.sum(a >= high))
        return c

    rho_np = ctx.rho.to_numpy()
    temp_np = ctx.temp.to_numpy()
    v_np = ctx.v.to_numpy()  # (nx,ny,nz,3)
    v_mag = np.sqrt(np.sum(v_np ** 2, axis=-1))

    rho_low = count_anomaly(rho_np, fluid_mask, low=1e-6)
    rho_neg = count_anomaly(rho_np, fluid_mask, low=0.0)  # <=0
    temp_out = count_anomaly(temp_np, fluid_mask, high=2.0) + count_anomaly(temp_np, fluid_mask, low=-0.01)

    line = (
        f"[DIAG] step={step} n_fluid={n_fluid} | "
        f"{stats('rho', rho_np, fluid_mask)} rho<=0:{rho_neg} rho<1e-6:{rho_low} | "
        f"{stats('temp', temp_np, fluid_mask)} temp_out:{temp_out} | "
        f"{stats('|v|', v_mag, fluid_mask)}"
    )
    _log.info("%s", line)

    if log_distributions:
        f_old = ctx.f_old.to_numpy()
        g_old = ctx.g_old.to_numpy()
        # 流体セル・全方向の値を1次元に
        f_flat = f_old[fluid_mask].reshape(-1)
        g_flat = g_old[fluid_mask].reshape(-1)
        f_ok = np.isfinite(f_flat)
        g_ok = np.isfinite(g_flat)
        f_min = f_flat[f_ok].min() if np.sum(f_ok) else float("nan")
        f_max = f_flat[f_ok].max() if np.sum(f_ok) else float("nan")
        g_min = g_flat[g_ok].min() if np.sum(g_ok) else float("nan")
        g_max = g_flat[g_ok].max() if np.sum(g_ok) else float("nan")
        f_neg = int(np.sum(f_flat < 0))
        g_neg = int(np.sum(g_flat < 0))
        _log.info(
            "[DIAG] step=%s f_old: min=%.6f max=%.6f neg_count=%s | g_old: min=%.6f max=%.6f neg_count=%s",
            step,
            f_min,
            f_max,
            f_neg,
            g_min,
            g_max,
            g_neg,
        )


def should_log_diagnostic(step, vis_interval, diag_start=0, diag_end=2500, diag_every=30):
    """
    暴走しうる区間（diag_start～diag_end）では diag_every ステップごとにログを取る。
    それ以外は vis_interval ごとでよい。
    """
    if diag_start <= step <= diag_end:
        return step % diag_every == 0
    return step % vis_interval == 0
