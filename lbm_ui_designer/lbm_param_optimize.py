# ==========================================================
# lbm_param_optimize.py — 複数範囲を同時に満たすよう一次パラメータを最適化（ranges は SLSQP 制約で厳守）
# 前提: SymPy（lbm_solve_core が eqs_solver による整合チェックに使用）
# 使用例:
#   conda run -n lbm-sim python lbm_ui_designer/lbm_param_optimize.py -c lbm_ui_designer/lbm_param_optimize.example.yaml
# ==========================================================
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from lbm_solve_core import (
    ALL_VARIABLE_NAMES,
    COOLPROP_AVAILABLE,
    PRIMARY_PARAM_NAMES,
    SOLID_MATS,
    compute_all_state_from_primary,
    count_conflicts_from_state_dict,
    default_primary_from_var_info,
    get_fluid_properties_coolprop,
    get_solid_properties,
)

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML が必要です: pip install pyyaml") from e

KEY_ALIASES = {"u": "U"}

# min≈max を「一点制約」とみなす閾値
RANGE_POINT_TOL = 1e-12
# 報告・固定解チェック用（SLSQP の数値誤差を考慮）
RANGE_FEAS_TOL = 1e-7


def normalize_name(key: str) -> str:
    k = str(key).strip()
    return KEY_ALIASES.get(k, k)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_primary_from_coolprop(cfg: dict) -> dict:
    """CoolProp + 固体 + 幾何デフォルトから一次パラメータを構築。"""
    fluid = cfg.get("fluid", "Water")
    t_k = float(cfg.get("temperature_K", 300.0))
    p_pa = float(cfg.get("pressure_Pa", 101325.0))
    solid = cfg.get("solid", "Copper")

    if solid not in SOLID_MATS:
        raise ValueError(f"固体名が不正です: {solid}. 候補: {list(SOLID_MATS.keys())}")

    out = default_primary_from_var_info()

    if not COOLPROP_AVAILABLE:
        raise RuntimeError("CoolProp が利用できません。pip install CoolProp を実行してください。")

    fp = get_fluid_properties_coolprop(fluid, t_k, p_pa)
    if fp is None:
        raise RuntimeError(f"CoolProp で流体 '{fluid}' の物性取得に失敗しました。")

    sp = get_solid_properties(solid)
    out["nu"] = fp["nu"]
    out["k_f"] = fp["k_f"]
    out["rho_f"] = fp["rho_f"]
    out["Cp_f"] = fp["Cp_f"]
    out["k_s"] = sp["k_s"]
    out["rho_s"] = sp["rho_s"]
    out["Cp_s"] = sp["Cp_s"]
    return out


def parse_fixed_ranges(cfg: dict):
    fixed_raw = cfg.get("fixed") or {}
    ranges_raw = cfg.get("ranges") or {}

    # 値は float（明示）または True（CoolProp/固体ロード後の primary0 の値で固定）
    fixed: dict[str, float | bool] = {}
    for k, v in fixed_raw.items():
        nk = normalize_name(k)
        if nk not in PRIMARY_PARAM_NAMES:
            raise ValueError(
                f"fixed は一次パラメータのみ指定できます {list(PRIMARY_PARAM_NAMES)}。不明: {k}"
            )
        if v is True:
            fixed[nk] = True
        else:
            try:
                fixed[nk] = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"fixed.{k} は数値か true（ロード物性で固定）のみ指定できます。"
                ) from e

    ranges = {}
    for k, spec in ranges_raw.items():
        nk = normalize_name(k)
        if nk not in ALL_VARIABLE_NAMES:
            raise ValueError(f"不明な変数名 (ranges): {k}")
        if not isinstance(spec, dict) or "min" not in spec or "max" not in spec:
            raise ValueError(f"ranges.{k} は {{min, max}} 形式にしてください。")
        lo, hi = float(spec["min"]), float(spec["max"])
        if lo > hi:
            raise ValueError(f"ranges.{k}: min <= max にしてください。")
        ranges[nk] = (lo, hi)

    overlap = set(fixed.keys()) & set(ranges.keys())
    if overlap:
        raise ValueError(f"同一変数が fixed と ranges の両方にあります: {overlap}")

    return fixed, ranges


def resolve_fixed_to_values(fixed: dict[str, float | bool], primary0: dict[str, float]) -> dict[str, float]:
    """True の項目を primary0（物性ロード済み）の値に置き換えた固定値 dict。"""
    out = {}
    for k, spec in fixed.items():
        if spec is True:
            if k not in primary0:
                raise ValueError(f"内部エラー: primary0 に {k} がありません")
            out[k] = float(primary0[k])
        else:
            out[k] = float(spec)
    return out


def _value_in_range(v: float, lo: float, hi: float) -> bool:
    """ranges の区間／一点を満たすか（数値誤差付き）。"""
    if abs(lo - hi) < RANGE_POINT_TOL:
        return abs(v - lo) <= RANGE_FEAS_TOL * max(1.0, abs(lo))
    margin = RANGE_FEAS_TOL * max(1.0, abs(lo), abs(hi), abs(v))
    return (lo - margin) <= v <= (hi + margin)


def state_satisfies_ranges(st: dict, ranges: dict) -> bool:
    return all(_value_in_range(float(st[name]), lo, hi) for name, (lo, hi) in ranges.items())


def build_primary_vector(free_keys, theta, fixed: dict, template: dict) -> dict:
    p = dict(template)
    p.update(fixed)
    for k, val in zip(free_keys, theta):
        p[k] = float(val)
    return p


def run_optimize(cfg: dict) -> dict:
    primary0 = default_primary_from_coolprop(cfg)
    fixed_spec, ranges = parse_fixed_ranges(cfg)
    fixed = resolve_fixed_to_values(fixed_spec, primary0)

    for k, v in fixed.items():
        primary0[k] = v

    fix_cr = bool(cfg.get("fix_cr", False))
    cr_target = None
    if fix_cr:
        st0 = compute_all_state_from_primary(primary0)
        cr_target = float(st0["C_r"])

    reg = float(cfg.get("regularization", 1e-6))
    targets = cfg.get("targets") or {}
    target_reg = float(cfg.get("target_regularization", 0.0))
    weights = cfg.get("range_weights") or {}
    free_keys = [k for k in PRIMARY_PARAM_NAMES if k not in fixed]

    def ref_vector(keys):
        return np.array([primary0[k] for k in keys], dtype=float)

    x0 = ref_vector(free_keys)
    scales = np.maximum(np.abs(x0), 1e-30)

    # ranges は SLSQP の等式・不等式制約で厳守（区間外に出ない）
    # 目的関数は初期解への正則化のみ（range_weights は互換のため無視）
    def objective(theta):
        obj_val = 0.0
        # Existing regularization: keep free primary keys close to initial values
        if reg > 0 and len(free_keys) > 0:
            diff = (theta - x0) / scales
            obj_val += reg * float(np.dot(diff, diff))

        # New: Target-based regularization for output variables
        if target_reg > 0 and targets:
            current_state = _state_from_theta(theta)
            for name, target_val in targets.items():
                # Normalize target_val if it's a dict {value, weight}
                if isinstance(target_val, dict):
                    target_weight = float(target_val.get("weight", 1.0))
                    target_val = float(target_val["value"])
                else:
                    target_weight = 1.0
                    target_val = float(target_val)

                if name in current_state:
                    val = float(current_state[name])
                    # Penalize deviation from target. Scale by target_val if non-zero, otherwise by 1.
                    if target_val != 0:
                        deviation = (val - target_val) / target_val
                        obj_val += target_reg * target_weight * (deviation ** 2)
                    else: # Target is 0, penalize if value is not 0
                        obj_val += target_reg * target_weight * (val ** 2)
        return obj_val

    bounds = []
    for k, v in zip(free_keys, x0):
        if k == "nx":
            bounds.append((1.0, 1e7))
        elif k in ("L_domain", "L_ref"):
            bounds.append((1e-12, 1e3))
        elif k == "U":
            bounds.append((1e-12, 1e3))
        elif k == "u_lbm":
            bounds.append((1e-8, 0.5))
        elif k == "nu":
            bounds.append((1e-15, 1e-2))
        elif k in ("k_f", "k_s"):
            bounds.append((1e-6, 500.0))
        elif k in ("rho_f", "rho_s"):
            bounds.append((1.0, 20000.0))
        elif k in ("Cp_f", "Cp_s"):
            bounds.append((50.0, 20000.0))
        else:
            raise RuntimeError(f"内部エラー: 未対応の一次パラメータ {k}")

    constraints = []

    def _state_from_theta(theta):
        p = build_primary_vector(free_keys, theta, fixed, primary0)
        return compute_all_state_from_primary(p)

    def combined_eq(theta):
        """C_r 固定 + ranges の min==max 一点制約。"""
        st = _state_from_theta(theta)
        parts = []
        if fix_cr and cr_target is not None:
            parts.append(float(st["C_r"]) - float(cr_target))
        for name, (lo, hi) in ranges.items():
            if abs(lo - hi) < RANGE_POINT_TOL:
                parts.append(float(st[name]) - float(lo))
        return np.array(parts, dtype=float)

    def range_interval_ineq(theta):
        """区間 [lo,hi] について g>=0 形式: v-lo>=0, hi-v>=0。"""
        st = _state_from_theta(theta)
        parts = []
        for name, (lo, hi) in ranges.items():
            if abs(lo - hi) < RANGE_POINT_TOL:
                continue
            v = float(st[name])
            parts.append(v - lo)
            parts.append(hi - v)
        return np.array(parts, dtype=float)

    eq_parts = []
    if fix_cr and cr_target is not None:
        eq_parts.append("C_r")
    for name, (lo, hi) in ranges.items():
        if abs(lo - hi) < RANGE_POINT_TOL:
            eq_parts.append(name)

    ineq_parts = []
    for name, (lo, hi) in ranges.items():
        if abs(lo - hi) >= RANGE_POINT_TOL:
            ineq_parts.append(f"{name}>=min")
            ineq_parts.append(f"{name}<=max")

    if len(eq_parts) > 0:
        constraints.append({"type": "eq", "fun": combined_eq})

    if len(ineq_parts) > 0:
        constraints.append({"type": "ineq", "fun": range_interval_ineq})

    meta = {
        "fluid": cfg.get("fluid"),
        "temperature_K": cfg.get("temperature_K"),
        "pressure_Pa": cfg.get("pressure_Pa"),
        "solid": cfg.get("solid"),
        "fix_cr": fix_cr,
        "C_r_target": cr_target,
        "free_primary_keys": free_keys,
        "fixed_primary": dict(fixed),
        "fixed_spec": {k: ("load" if v is True else v) for k, v in fixed_spec.items()},
        "ranges_mode": "hard_constraints",
        "range_equality_vars": eq_parts,
        "range_inequality_labels": ineq_parts,
        "range_weights_ignored": bool(weights),
        "targets_active": bool(targets),
        "target_regularization": target_reg,
        "targets": targets,
    }

    if len(free_keys) == 0:
        st = compute_all_state_from_primary(build_primary_vector([], [], fixed, primary0))
        ranges_ok = state_satisfies_ranges(st, ranges) if ranges else True
        conflicts = count_conflicts_from_state_dict(st)
        range_report_fixed = {}
        for name, (lo, hi) in ranges.items():
            v = float(st[name])
            range_report_fixed[name] = {
                "value": v,
                "min": lo,
                "max": hi,
                "satisfied": _value_in_range(v, lo, hi),
            }
        return {
            "success": ranges_ok and conflicts == 0,
            "message": "最適化変数なし（一次パラメータはすべて固定）",
            "objective_value": 0.0,
            "final_penalty": 0.0,
            "state": st,
            "primary": build_primary_vector([], [], fixed, primary0),
            "scipy": None,
            "meta": meta,
            "range_report": range_report_fixed,
            "equation_conflicts": conflicts,
        }

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints if constraints else (),
        options={"maxiter": int(cfg.get("maxiter", 3000)), "ftol": 1e-12},
    )

    theta_f = res.x
    primary_f = build_primary_vector(free_keys, theta_f, fixed, primary0)
    st_f = compute_all_state_from_primary(primary_f)
    conflicts_f = count_conflicts_from_state_dict(st_f)

    range_report = {}
    for name, (lo, hi) in ranges.items():
        v = float(st_f[name])
        range_report[name] = {
            "value": v,
            "min": lo,
            "max": hi,
            "satisfied": _value_in_range(v, lo, hi),
        }

    ranges_feas = state_satisfies_ranges(st_f, ranges) if ranges else True

    return {
        "success": bool(res.success) and conflicts_f == 0 and ranges_feas,
        "message": str(res.message),
        "objective_value": float(res.fun),
        "final_penalty": float(res.fun),
        "state": st_f,
        "primary": primary_f,
        "scipy": {
            "nit": int(res.nit) if res.nit is not None else None,
            "status": int(res.status),
        },
        "meta": meta,
        "range_report": range_report,
        "equation_conflicts": conflicts_f,
    }


def main():
    ap = argparse.ArgumentParser(description="LBM パラメータ多目的範囲最適化 (CoolProp 物性ベース)")
    ap.add_argument("--config", "-c", required=True, help="YAML 設定ファイル")
    ap.add_argument("--output", "-o", default=None, help="結果 JSON の出力先 (省略時は標準出力のみ)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = run_optimize(cfg)
    text = json.dumps(out, ensure_ascii=False, indent=2, default=float)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
