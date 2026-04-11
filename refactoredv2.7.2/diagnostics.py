# ==========================================================
# diagnostics.py — 数値暴走の原因究明用ログ（rho / temp / v / 分布関数）
# ==========================================================
"""
暴走の切り分けのため、流体セル（FLUID_A, INLET, OUTLET）だけを対象に以下を記録する:

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
import numpy as np
from context import FLUID_A, INLET, OUTLET, SOLID, SOLID_HEAT_SOURCE


def log_field_diagnostics(ctx, step, log_distributions=False):
    """
    ctx の流体セル（FLUID_A, INLET, OUTLET）のみをマスクし、
    rho, temp, |v| および必要なら f_old, g_old の統計を print する。
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
    print(line)

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
        print(f"[DIAG] step={step} f_old: min={f_min:.6f} max={f_max:.6f} neg_count={f_neg} | g_old: min={g_min:.6f} max={g_max:.6f} neg_count={g_neg}")


def should_log_diagnostic(step, vis_interval, diag_start=0, diag_end=2500, diag_every=30):
    """
    暴走しうる区間（diag_start～diag_end）では diag_every ステップごとにログを取る。
    それ以外は vis_interval ごとでよい。
    """
    if diag_start <= step <= diag_end:
        return step % diag_every == 0
    return step % vis_interval == 0
