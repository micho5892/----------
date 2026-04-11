# ==========================================================
# lbm_solve_core.py — LBM 方程式・物性・状態計算（SymPy 前提・app.py と共有）
# ==========================================================
import sympy as sp

try:
    import CoolProp.CoolProp as CP

    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

SOLID_MATS = {
    "Copper": {"k": 400.0, "rho": 8960.0, "Cp": 385.0},
    "Aluminum": {"k": 237.0, "rho": 2700.0, "Cp": 900.0},
    "Stainless Steel": {"k": 16.0, "rho": 8000.0, "Cp": 500.0},
}

# ==========================================
# SymPy シンボル（app.py と同一）
# ==========================================
nx = sp.Symbol("n_x", positive=True, real=True)
L_domain = sp.Symbol("L_{domain}", positive=True, real=True)
L_ref = sp.Symbol("L_{ref}", positive=True, real=True)

dx = sp.Symbol("dx", positive=True, real=True)
u = sp.Symbol("U", positive=True, real=True)
u_lbm = sp.Symbol("u_{lbm}", positive=True, real=True)
dt = sp.Symbol("dt", positive=True, real=True)

nu = sp.Symbol(r"\nu", positive=True, real=True)
k_f = sp.Symbol("k_f", positive=True, real=True)
rho_f = sp.Symbol(r"\rho_f", positive=True, real=True)
Cp_f = sp.Symbol("Cp_f", positive=True, real=True)
alpha_f = sp.Symbol(r"\alpha_f", positive=True, real=True)

k_s = sp.Symbol("k_s", positive=True, real=True)
rho_s = sp.Symbol(r"\rho_s", positive=True, real=True)
Cp_s = sp.Symbol("Cp_s", positive=True, real=True)
alpha_s = sp.Symbol(r"\alpha_s", positive=True, real=True)

Re = sp.Symbol("Re", positive=True, real=True)
Pr = sp.Symbol("Pr", positive=True, real=True)
C_r = sp.Symbol("C_r", positive=True, real=True)

Re_Delta = sp.Symbol(r"Re_{\Delta}", positive=True, real=True)
Pe_Delta = sp.Symbol(r"Pe_{\Delta,f}", positive=True, real=True)
Pe_Delta_s = sp.Symbol(r"Pe_{\Delta,s}", positive=True, real=True)

Fo_Delta_nu = sp.Symbol(r"Fo_{\Delta,\nu}", positive=True, real=True)
Fo_Delta_f = sp.Symbol(r"Fo_{\Delta,f}", positive=True, real=True)
Fo_Delta_s = sp.Symbol(r"Fo_{\Delta,s}", positive=True, real=True)

tau_f_margin = sp.Symbol(r"\tau_{f\_margin}", positive=True, real=True)
tau_gf_margin = sp.Symbol(r"\tau_{gf\_margin}", positive=True, real=True)
tau_gs_margin = sp.Symbol(r"\tau_{gs\_margin}", positive=True, real=True)

var_info = {
    nx: {"name": "nx", "desc": "空間解像度 (領域のセル数)", "default": 128.0},
    L_domain: {"name": "L_domain", "desc": "空間全体の長さ [m]", "default": 0.1},
    L_ref: {"name": "L_ref", "desc": "代表長さ (直径など) [m]", "default": 0.04},
    dx: {"name": "dx", "desc": "空間刻み [m]", "default": 0.1 / 128},
    u: {"name": "U", "desc": "物理流速 [m/s]", "default": 0.05},
    u_lbm: {"name": "u_lbm", "desc": "LBM流速", "default": 0.1},
    dt: {"name": "dt", "desc": "時間刻み [s]", "default": 0.1 / 128 * 0.1 / 0.05},
    nu: {"name": "nu", "desc": "動粘度[m^2/s]", "default": 1e-6},
    k_f: {"name": "k_f", "desc": "熱伝導率 (流体)", "default": 0.6},
    rho_f: {"name": "rho_f", "desc": "密度 (流体)", "default": 1000.0},
    Cp_f: {"name": "Cp_f", "desc": "比熱 (流体)", "default": 4180.0},
    alpha_f: {"name": "alpha_f", "desc": "熱拡散率 (流体)", "default": 0.6 / (1000.0 * 4180.0)},
    k_s: {"name": "k_s", "desc": "熱伝導率 (固体)", "default": 400.0},
    rho_s: {"name": "rho_s", "desc": "密度 (固体)", "default": 8960.0},
    Cp_s: {"name": "Cp_s", "desc": "比熱 (固体)", "default": 385.0},
    alpha_s: {"name": "alpha_s", "desc": "熱拡散率 (固体)", "default": 400.0 / (8960.0 * 385.0)},
    Re: {"name": "Re", "desc": "レイノルズ数", "default": 500.0},
    Pr: {"name": "Pr", "desc": "プラントル数", "default": 7.0},
    C_r: {"name": "C_r", "desc": "熱容量比 (固体/流体)", "default": (8960.0 * 385.0) / (1000.0 * 4180.0)},
    Re_Delta: {"name": "Re_Delta", "desc": "セル・レイノルズ数", "default": 5.0},
    Pe_Delta: {"name": "Pe_Delta_f", "desc": "セル・ペクレ数 (流体)", "default": 35.0},
    Pe_Delta_s: {"name": "Pe_Delta_s", "desc": "セル・ペクレ数 (固体)", "default": 5.0},
    Fo_Delta_nu: {"name": "Fo_Delta_nu", "desc": "セル拡散数 (粘性)", "default": 0.05},
    Fo_Delta_f: {"name": "Fo_Delta_f", "desc": "セル拡散数 (流体熱)", "default": 0.05},
    Fo_Delta_s: {"name": "Fo_Delta_s", "desc": "セル拡散数 (固体熱)", "default": 0.05},
    tau_f_margin: {"name": "tau_f マージン", "desc": "τ_f - 0.5", "default": 0.02},
    tau_gf_margin: {"name": "tau_gf マージン", "desc": "τ_gf - 0.5", "default": 0.02},
    tau_gs_margin: {"name": "tau_gs マージン", "desc": "τ_gs - 0.5", "default": 0.3},
}
all_vars = list(var_info.keys())

PRIMARY_PARAM_NAMES = ("nx", "L_domain", "L_ref", "U", "u_lbm", "nu", "k_f", "rho_f", "Cp_f", "k_s", "rho_s", "Cp_s")

ALL_VARIABLE_NAMES = {var_info[s]["name"] for s in all_vars}


def default_primary_from_var_info():
    """var_info の default から一次パラメータのみを返す。"""
    out = {}
    for sym in all_vars:
        name = var_info[sym]["name"]
        if name in PRIMARY_PARAM_NAMES:
            out[name] = float(var_info[sym]["default"])
    return out


def get_fluid_properties_coolprop(substance_name, temperature_kelvin, pressure_pa):
    """CoolProp から流体物性。失敗時は None。"""
    if not COOLPROP_AVAILABLE:
        return None
    try:
        density = CP.PropsSI("D", "T", temperature_kelvin, "P", pressure_pa, substance_name)
        dynamic_viscosity = CP.PropsSI("VISCOSITY", "T", temperature_kelvin, "P", pressure_pa, substance_name)
        kinematic_viscosity = dynamic_viscosity / density
        thermal_conductivity = CP.PropsSI("CONDUCTIVITY", "T", temperature_kelvin, "P", pressure_pa, substance_name)
        specific_heat_cp = CP.PropsSI("CPMASS", "T", temperature_kelvin, "P", pressure_pa, substance_name)
        return {
            "rho_f": float(density),
            "nu": float(kinematic_viscosity),
            "k_f": float(thermal_conductivity),
            "Cp_f": float(specific_heat_cp),
        }
    except Exception:
        return None


def get_solid_properties(solid_name):
    s = SOLID_MATS[solid_name]
    return {"k_s": float(s["k"]), "rho_s": float(s["rho"]), "Cp_s": float(s["Cp"])}


def compute_all_state_from_primary(primary):
    """一次パラメータ (PRIMARY_PARAM_NAMES) から全変数を計算。キーは var_info の name（流速は U）。"""
    nx_v = float(primary["nx"])
    L_dom = float(primary["L_domain"])
    L_ref_v = float(primary["L_ref"])
    U = float(primary["U"])
    u_lbm_v = float(primary["u_lbm"])
    nu_v = float(primary["nu"])
    k_f_v = float(primary["k_f"])
    rho_f_v = float(primary["rho_f"])
    Cp_f_v = float(primary["Cp_f"])
    k_s_v = float(primary["k_s"])
    rho_s_v = float(primary["rho_s"])
    Cp_s_v = float(primary["Cp_s"])

    dx_v = L_dom / nx_v
    dt_v = dx_v * (u_lbm_v / U)
    alpha_f_v = k_f_v / (rho_f_v * Cp_f_v)
    alpha_s_v = k_s_v / (rho_s_v * Cp_s_v)
    Re_v = U * L_ref_v / nu_v
    Pr_v = nu_v / alpha_f_v
    C_r_v = (rho_s_v * Cp_s_v) / (rho_f_v * Cp_f_v)
    Re_Delta_v = U * dx_v / nu_v
    Pe_Delta_f_v = U * dx_v / alpha_f_v
    Pe_Delta_s_v = U * dx_v / alpha_s_v
    Fo_nu = nu_v * dt_v / dx_v**2
    Fo_f = alpha_f_v * dt_v / dx_v**2
    Fo_s = alpha_s_v * dt_v / dx_v**2
    tau_f = 3.0 * Fo_nu
    tau_gf = 3.0 * Fo_f
    tau_gs = 3.0 * Fo_s

    return {
        "nx": int(nx_v),
        "L_domain": L_dom,
        "L_ref": L_ref_v,
        "U": U,
        "u_lbm": u_lbm_v,
        "nu": nu_v,
        "k_f": k_f_v,
        "rho_f": rho_f_v,
        "Cp_f": Cp_f_v,
        "k_s": k_s_v,
        "rho_s": rho_s_v,
        "Cp_s": Cp_s_v,
        "dx": dx_v,
        "dt": dt_v,
        "alpha_f": alpha_f_v,
        "alpha_s": alpha_s_v,
        "Re": Re_v,
        "Pr": Pr_v,
        "C_r": C_r_v,
        "Re_Delta": Re_Delta_v,
        "Pe_Delta_f": Pe_Delta_f_v,
        "Pe_Delta_s": Pe_Delta_s_v,
        "Fo_Delta_nu": Fo_nu,
        "Fo_Delta_f": Fo_f,
        "Fo_Delta_s": Fo_s,
        "tau_f マージン": tau_f,
        "tau_gf マージン": tau_gf,
        "tau_gs マージン": tau_gs,
    }


eqs_solver = [
    sp.Eq(dx, L_domain / nx),
    sp.Eq(dt, dx * (u_lbm / u)),
    sp.Eq(alpha_f, k_f / (rho_f * Cp_f)),
    sp.Eq(alpha_s, k_s / (rho_s * Cp_s)),
    sp.Eq(Re, u * L_ref / nu),
    sp.Eq(Pr, nu / alpha_f),
    sp.Eq(C_r, (rho_s * Cp_s) / (rho_f * Cp_f)),
    sp.Eq(Re_Delta, u * dx / nu),
    sp.Eq(Pe_Delta, u * dx / alpha_f),
    sp.Eq(Pe_Delta_s, u * dx / alpha_s),
    sp.Eq(Fo_Delta_nu, nu * dt / dx**2),
    sp.Eq(Fo_Delta_f, alpha_f * dt / dx**2),
    sp.Eq(Fo_Delta_s, alpha_s * dt / dx**2),
    sp.Eq(tau_f_margin, 3 * Fo_Delta_nu),
    sp.Eq(tau_gf_margin, 3 * Fo_Delta_f),
    sp.Eq(tau_gs_margin, 3 * Fo_Delta_s),
    sp.Eq(Fo_Delta_nu, u_lbm / Re_Delta),
    sp.Eq(Fo_Delta_f, u_lbm / Pe_Delta),
    sp.Eq(Fo_Delta_s, u_lbm / Pe_Delta_s),
    sp.Eq(tau_f_margin, 3 * (L_ref / dx) * u_lbm / Re),
    sp.Eq(tau_gf_margin, 3 * (L_ref / dx) * u_lbm / (Re * Pr)),
]

eqs_macro_display = [
    sp.Eq(alpha_f, k_f / (rho_f * Cp_f)),
    sp.Eq(alpha_s, k_s / (rho_s * Cp_s)),
    sp.Eq(C_r, (rho_s * Cp_s) / (rho_f * Cp_f)),
    sp.Eq(Re, u * L_ref / nu),
    sp.Eq(Pr, nu / alpha_f),
    sp.Eq(dt, dx * (u_lbm / u)),
    sp.Eq(tau_f_margin, 3 * (L_ref / dx) * u_lbm / Re),
    sp.Eq(tau_gf_margin, 3 * (L_ref / dx) * u_lbm / (Re * Pr)),
]

eqs_cell_display = [
    sp.Eq(Fo_Delta_nu, nu * dt / dx**2),
    sp.Eq(Fo_Delta_f, alpha_f * dt / dx**2),
    sp.Eq(Fo_Delta_s, alpha_s * dt / dx**2),
    sp.Eq(tau_f_margin, 3 * Fo_Delta_nu),
    sp.Eq(tau_gf_margin, 3 * Fo_Delta_f),
    sp.Eq(tau_gs_margin, 3 * Fo_Delta_s),
    sp.Eq(Re_Delta, u * dx / nu),
    sp.Eq(Pe_Delta, u * dx / alpha_f),
    sp.Eq(Pe_Delta_s, u * dx / alpha_s),
    sp.Eq(Fo_Delta_nu, u_lbm / Re_Delta),
    sp.Eq(Fo_Delta_f, u_lbm / Pe_Delta),
    sp.Eq(Fo_Delta_s, u_lbm / Pe_Delta_s),
]


def state_dict_to_sympy_known(state_dict):
    known = {}
    for sym in all_vars:
        name = var_info[sym]["name"]
        if name in state_dict:
            known[sym] = float(state_dict[name])
    return known


def solve_system(eqs, fixed_vals):
    known = fixed_vals.copy()
    new_found = True
    while new_found:
        new_found = False
        for eq in eqs:
            eq_sub = eq.subs(known)
            free_syms = list(eq_sub.free_symbols)
            if len(free_syms) == 1:
                sym = free_syms[0]
                try:
                    sol = sp.solve(eq_sub, sym)
                    if sol:
                        for s in sol:
                            if s.is_real and float(s) > 0:
                                known[sym] = float(s)
                                new_found = True
                                break
                        if not new_found and sol[0].is_real:
                            known[sym] = float(sol[0])
                            new_found = True
                except Exception:
                    pass
    return known


def count_conflicts(eqs, known):
    conflicts = 0
    for eq in eqs:
        try:
            lhs = float(eq.lhs.subs(known))
            rhs = float(eq.rhs.subs(known))
            if abs(lhs - rhs) > 1e-4:
                conflicts += 1
        except Exception:
            pass
    return conflicts


def count_conflicts_from_state_dict(state_dict):
    """状態 dict から eqs_solver との矛盾数（SymPy 代入）。"""
    return count_conflicts(eqs_solver, state_dict_to_sympy_known(state_dict))
