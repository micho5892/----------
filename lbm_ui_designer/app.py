# ==========================================================
# app.py — LBM Parameter Auto-Designer Pro (Representative Length Separated)
# cd C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション
# streamlit run lbm_ui_designer\app.py
# ==========================================================
import streamlit as st
import sympy as sp

from lbm_solve_core import (
    COOLPROP_AVAILABLE,
    SOLID_MATS,
    all_vars,
    var_info,
    eqs_solver,
    eqs_macro_display,
    eqs_cell_display,
    solve_system,
    count_conflicts,
    nx,
    L_domain,
    L_ref,
    dx,
    u,
    u_lbm,
    dt,
    nu,
    k_f,
    rho_f,
    Cp_f,
    alpha_f,
    k_s,
    rho_s,
    Cp_s,
    alpha_s,
    Re,
    Pr,
    C_r,
    Re_Delta,
    Pe_Delta,
    Pe_Delta_s,
    Fo_Delta_nu,
    Fo_Delta_f,
    Fo_Delta_s,
    tau_f_margin,
    tau_gf_margin,
    tau_gs_margin,
    get_fluid_properties_coolprop,
)

st.set_page_config(page_title="LBM Auto-Designer Pro", layout="wide")

# ==========================================
# 3. コールバック関数群 (Shadow State管理)
# ==========================================
def on_fix_change(v_name, def_val):
    fix_key = f"fix_{v_name}"
    val_key = f"val_{v_name}"
    shadow_key = f"shadow_{v_name}"
    if st.session_state[fix_key]:
        st.session_state[val_key] = float(st.session_state.get(shadow_key, def_val))

def load_real_properties():
    if not COOLPROP_AVAILABLE: return
    fluid = st.session_state.get("fluid_sel", "Water")
    T = st.session_state.get("temp_in", 300.0)
    P = st.session_state.get("press_in", 101325.0)
    solid = st.session_state.get("solid_sel", "Copper")
    
    props = get_fluid_properties_coolprop(fluid, T, P)
    if not props:
        st.error(f"CoolProp で流体 '{fluid}' の物性取得に失敗しました。")
        return

    rho_f_v = float(props["rho_f"])
    Cp_f_v = float(props["Cp_f"])
    st.session_state[f"shadow_{nu.name}"] = float(props["nu"])
    st.session_state[f"shadow_{k_f.name}"] = float(props["k_f"])
    st.session_state[f"shadow_{rho_f.name}"] = rho_f_v
    st.session_state[f"shadow_{Cp_f.name}"] = Cp_f_v

    s_props = SOLID_MATS[solid]
    rho_s_v = float(s_props["rho"])
    Cp_s_v  = float(s_props["Cp"])
    st.session_state[f"shadow_{k_s.name}"] = float(s_props["k"])
    st.session_state[f"shadow_{rho_s.name}"] = rho_s_v
    st.session_state[f"shadow_{Cp_s.name}"] = Cp_s_v
    
    calc_Cr = (rho_s_v * Cp_s_v) / (rho_f_v * Cp_f_v)
    st.session_state[f"shadow_{C_r.name}"] = calc_Cr
    st.session_state[f"val_{C_r.name}"] = calc_Cr
    st.session_state[f"fix_{C_r.name}"] = True

def apply_similarity():
    val_u = st.session_state.get(f"shadow_{u.name}", var_info[u]['default'])
    val_Lref = st.session_state.get(f"shadow_{L_ref.name}", var_info[L_ref]['default'])
    val_nu = st.session_state.get(f"shadow_{nu.name}", var_info[nu]['default'])
    val_kf = st.session_state.get(f"shadow_{k_f.name}", var_info[k_f]['default'])
    val_rhof = st.session_state.get(f"shadow_{rho_f.name}", var_info[rho_f]['default'])
    val_Cpf = st.session_state.get(f"shadow_{Cp_f.name}", var_info[Cp_f]['default'])
    
    if val_nu > 0 and val_rhof * val_Cpf > 0:
        calc_Re = (val_u * val_Lref) / val_nu
        calc_alpha = val_kf / (val_rhof * val_Cpf)
        calc_Pr = val_nu / calc_alpha
        
        st.session_state[f"val_{Re.name}"] = float(calc_Re)
        st.session_state[f"fix_{Re.name}"] = True
        st.session_state[f"shadow_{Re.name}"] = float(calc_Re)
        
        st.session_state[f"val_{Pr.name}"] = float(calc_Pr)
        st.session_state[f"fix_{Pr.name}"] = True
        st.session_state[f"shadow_{Pr.name}"] = float(calc_Pr)
        
    for v in[tau_f_margin, tau_gf_margin]:
        st.session_state[f"val_{v.name}"] = 0.05
        st.session_state[f"fix_{v.name}"] = True
        st.session_state[f"shadow_{v.name}"] = 0.05
        
    st.session_state[f"val_{tau_gs_margin.name}"] = 0.5
    st.session_state[f"fix_{tau_gs_margin.name}"] = True
    st.session_state[f"shadow_{tau_gs_margin.name}"] = 0.5
    
    for v in[Cp_f, Cp_s, dt, dx, u_lbm]:
        st.session_state[f"fix_{v.name}"] = False

# ==========================================
# 4. UI 描画
# ==========================================
st.title("🧩 LBM Parameter Auto-Designer Pro")

with st.expander("🌍 現実の物性データベース (CoolProp & 相似則適用)", expanded=True):
    if not COOLPROP_AVAILABLE:
        st.error("⚠️ CoolProp ライブラリが必要です。")
    else:
        st.markdown("現実の流体と固体を指定し、相似則と熱容量比($C_r$)を維持しながらLBMで計算可能な仮想物性へと安全に変換します。")
        c1, c2, c3, c4 = st.columns(4)
        c1.selectbox("流体名 (Fluid)",["Water", "Air", "R134a", "Nitrogen"], key="fluid_sel")
        c2.number_input("温度[K]", value=300.0, key="temp_in")
        c3.number_input("圧力 [Pa]", value=101325.0, key="press_in")
        c4.selectbox("固体名 (Solid)", list(SOLID_MATS.keys()), key="solid_sel")
        
        st.button("📥 現実の物性値をロード", on_click=load_real_properties, type="primary", use_container_width=True)

fixed_vals = {}
for var in all_vars:
    if st.session_state.get(f"fix_{var.name}", False):
        val_key = f"val_{var.name}"
        if val_key not in st.session_state:
            st.session_state[val_key] = st.session_state.get(f"shadow_{var.name}", var_info[var]['default'])
        fixed_vals[var] = float(st.session_state[val_key])

known_vars = solve_system(eqs_solver, fixed_vals)
for var, val in known_vars.items():
    st.session_state[f"shadow_{var.name}"] = float(val)

unresolved =[v for v in all_vars if v not in known_vars]
conflict_count = count_conflicts(eqs_solver, known_vars)

# --- 矛盾検知インジケータ ---
if len(unresolved) == 0:
    if conflict_count == 0:
        st.success("🎉 全ての変数が確定しました！（自由度: 0）矛盾なく計算可能です。下のコードをコピーして実行してください。")
    else:
        st.error(f"❌ 全ての変数が確定しましたが、**{conflict_count} 個の方程式で矛盾**が発生しています！固定値を見直してください。")
else:
    if conflict_count == 0:
        st.warning(f"⚠️ 未確定の変数がまだ **{len(unresolved)} 個** あります。以下のカードからさらに変数を固定してください。")
    else:
        st.error(f"⚠️ 未確定変数がありますが、すでに **{conflict_count} 個の方程式で矛盾** が発生しています！")

st.header("📊 変数のステータスと入力")
cols = st.columns(6)
for i, var in enumerate(all_vars):
    with cols[i % 6]:
        is_fixed = var in fixed_vals
        is_derived = var in known_vars and not is_fixed
        status_color = "🟢 **固定**" if is_fixed else "🔵 **自動導出**" if is_derived else "🔴 **自由**"
            
        with st.container(border=True):
            st.markdown(f"{status_color} <br> **{var_info[var]['name']}**", unsafe_allow_html=True)
            st.caption(var_info[var]['desc'])
            
            fix_key = f"fix_{var.name}"
            val_key = f"val_{var.name}"
            st.checkbox("この値を固定", key=fix_key, on_change=on_fix_change, args=(var.name, var_info[var]['default']))
            
            if st.session_state.get(fix_key, False):
                init_val = st.session_state.get(val_key, st.session_state.get(f"shadow_{var.name}", var_info[var]['default']))
                st.number_input("固定値", value=float(init_val), format="%.4e", key=val_key, label_visibility="collapsed")
            else:
                shadow_val = st.session_state.get(f"shadow_{var.name}", var_info[var]['default'])
                if is_derived:
                    st.success(f"{shadow_val:.4e}")
                else:
                    st.info(f"{shadow_val:.4e} (参考値)")

st.divider()

# ==========================================
# 5. 実行用コードの自動生成
# ==========================================
st.header("📋 実行コードの生成")
st.markdown("変数がすべて確定すると、そのまま `main.py` に貼り付けられるコードが生成されます。")

def get_val(sym): 
    return known_vars.get(sym, st.session_state.get(f"shadow_{sym.name}", 0.0))

fluid_name = st.session_state.get("fluid_sel", "Fluid")
solid_name = st.session_state.get("solid_sel", "Solid")

sim_code = f"""run_simulation(
    benchmark="heat_exchanger",
    fp_dtype="float32",
    steady_detection=True,
    nx={int(get_val(nx))}, ny={int(get_val(nx))}, nz={int(get_val(nx)) * 4},
    Lx_p={get_val(L_domain):.4e}, 
    
    U_inlet_p={get_val(u):.4e},
    max_time_p=30.0, ramp_time_p=2.0,
    vis_interval=100, vti_export_interval=500,
    
    # ▼ Auto-Designed Similarity Parameters
    # Target Re={get_val(Re):.1f} (based on L_ref={get_val(L_ref):.4f}), Pr_sim={get_val(Pr):.2f}, Cr={get_val(C_r):.4f}
    domain_properties={{
        0:  {{"nu": {get_val(nu):.2e}, "k": {get_val(k_f):.3f}, "rho": {get_val(rho_f):.1f}, "Cp": {get_val(Cp_f):.2f}}},
        22: {{"nu": {get_val(nu):.2e}, "k": {get_val(k_f):.3f}, "rho": {get_val(rho_f):.1f}, "Cp": {get_val(Cp_f):.2f}}},
        23: {{"nu": {get_val(nu):.2e}, "k": {get_val(k_f):.3f}, "rho": {get_val(rho_f):.1f}, "Cp": {get_val(Cp_f):.2f}}},
        
        2:  {{"nu": {get_val(nu):.2e}, "k": {get_val(k_f):.3f}, "rho": {get_val(rho_f):.1f}, "Cp": {get_val(Cp_f):.2f}}},
        24: {{"nu": {get_val(nu):.2e}, "k": {get_val(k_f):.3f}, "rho": {get_val(rho_f):.1f}, "Cp": {get_val(Cp_f):.2f}}},
        25: {{"nu": {get_val(nu):.2e}, "k": {get_val(k_f):.3f}, "rho": {get_val(rho_f):.1f}, "Cp": {get_val(Cp_f):.2f}}},
        
        10: {{"nu": 0.0, "k": {get_val(k_s):.3f}, "rho": {get_val(rho_s):.1f}, "Cp": {get_val(Cp_s):.2f}}}
    }},
    
    boundary_conditions={{
        22: {{"type": "inlet",  "velocity":[0.0, 0.0,  {get_val(u_lbm):.4f}], "temperature": 1.0}},
        24: {{"type": "inlet",  "velocity":[0.0, 0.0, -{get_val(u_lbm)/2:.4f}], "temperature": 0.0}},
        23: {{"type": "outlet"}},
        25: {{"type": "outlet"}},
        10: {{"type": "adiabatic_wall"}}, 
    }},
    flow_type="counter"
)"""

st.code(sim_code, language="python")

st.divider()

# ==========================================
# 6. 方程式の可視化
# ==========================================
st.header("📐 導出プロセスの可視化")
tab1, tab2 = st.tabs(["📝 マクロ無次元数モード (Re, Pr等)", "🧩 セル基準無次元数モード"])

def render_equations(eqs):
    for eq in eqs:
        st.markdown("#### 🔹 元の方程式")
        st.latex(sp.latex(eq))
        
        lhs_sub = eq.lhs.subs(known_vars)
        rhs_sub = eq.rhs.subs(known_vars)
        free_syms = list(lhs_sub.free_symbols | rhs_sub.free_symbols)
        
        st.markdown("**▼ 既知の変数を代入した結果:**")
        lhs_disp = lhs_sub.evalf(4) if hasattr(lhs_sub, 'evalf') else lhs_sub
        rhs_disp = rhs_sub.evalf(4) if hasattr(rhs_sub, 'evalf') else rhs_sub
        st.latex(sp.latex(sp.Eq(lhs_disp, rhs_disp)))
        
        if len(free_syms) == 0:
            try:
                if abs(float(lhs_sub) - float(rhs_sub)) < 1e-4:
                    st.caption("✅ 矛盾はありません。")
                else:
                    st.error(f"❌ 矛盾が発生しています！ (誤差: {abs(float(lhs_sub) - float(rhs_sub)):.2e})")
            except Exception:
                pass
        else:
            free_names =[var_info[sym]['name'] for sym in free_syms if sym in var_info]
            st.caption(f"⚠️ 未確定変数: **{', '.join(free_names)}**")
        st.markdown("---")

with tab1: render_equations(eqs_macro_display)
with tab2: render_equations(eqs_cell_display)