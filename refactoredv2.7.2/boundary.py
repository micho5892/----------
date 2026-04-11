# ==========================================================
# boundary.py — BoundaryConditionManager (Phase 2 Object-Oriented)
# 精度向上版：NEEMの本格実装、逆流防止の削除
# ==========================================================
import taichi as ti
import config
from context import FLUID_A

@ti.data_oriented
class VelocityInlet:
    def __init__(self, d3q19, cfg, target_id, velocity, temperature):
        self.d3q19 = d3q19
        self.cfg = cfg
        self.target_id = target_id
        self.velocity = ti.Vector(velocity)
        self.temperature = temperature

    @ti.kernel
    def apply_before_macro(self, ctx: ti.template(), ramp_factor: ti.f32):
        vel_vec = self.velocity * ramp_factor
        temp_val = self.temperature
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                # --- 速度場 NEEM ---
                f_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                rho_sum = 0.0
                count_f = 0
                for d in ti.static(range(1, 19)):
                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]
                    
                    if ti.static(self.cfg.periodic_x): ip = ip % ctx.nx
                    if ti.static(self.cfg.periodic_y): jp = jp % ctx.ny
                    is_inside = True
                    if ti.static(not self.cfg.periodic_x) and (ip < 0 or ip >= ctx.nx): is_inside = False
                    if ti.static(not self.cfg.periodic_y) and (jp < 0 or jp >= ctx.ny): is_inside = False
                    if kp < 0 or kp >= ctx.nz: is_inside = False

                    if is_inside and ctx.cell_id[ip, jp, kp] == FLUID_A:
                        rho_f = ctx.rho[ip, jp, kp]
                        v_f = ctx.v[ip, jp, kp]
                        rho_sum += rho_f
                        for d2 in ti.static(range(19)):
                            feq_f = self.d3q19.get_feq(rho_f, v_f, d2)
                            f_neq_sum[d2] += (ctx.f_old[ip, jp, kp, d2] - feq_f)
                        count_f += 1
                
                if count_f > 0:
                    rho_in = rho_sum / count_f
                    for d in ti.static(range(19)):
                        feq_b = self.d3q19.get_feq(rho_in, vel_vec, d)
                        ctx.f_new[i, j, k, d] = feq_b + f_neq_sum[d] / count_f
                else:
                    for d in ti.static(range(19)):
                        ctx.f_new[i, j, k, d] = self.d3q19.get_feq(1.0, vel_vec, d)

                # --- 温度場 NEEM ---
                g_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                count_g = 0
                for d in ti.static(range(1, 19)):
                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]
                    
                    if ti.static(self.cfg.periodic_x): ip = ip % ctx.nx
                    if ti.static(self.cfg.periodic_y): jp = jp % ctx.ny
                    is_inside = True
                    if ti.static(not self.cfg.periodic_x) and (ip < 0 or ip >= ctx.nx): is_inside = False
                    if ti.static(not self.cfg.periodic_y) and (jp < 0 or jp >= ctx.ny): is_inside = False
                    if kp < 0 or kp >= ctx.nz: is_inside = False

                    if is_inside and ctx.cell_id[ip, jp, kp] == FLUID_A:
                        v_f = ctx.v[ip, jp, kp]
                        t_f = ctx.temp[ip, jp, kp]
                        for d2 in ti.static(range(19)):
                            geq_f = self.d3q19.get_geq(t_f, v_f, d2)
                            g_neq_sum[d2] += (ctx.g_old[ip, jp, kp, d2] - geq_f)
                        count_g += 1
                
                if count_g > 0:
                    for d in ti.static(range(19)):
                        geq_b = self.d3q19.get_geq(temp_val, vel_vec, d)
                        ctx.g_new[i, j, k, d] = geq_b + g_neq_sum[d] / count_g
                else:
                    for d in ti.static(range(19)):
                        ctx.g_new[i, j, k, d] = self.d3q19.get_geq(temp_val, vel_vec, d)


@ti.data_oriented
class PressureOutlet:
    def __init__(self, d3q19, cfg, target_id):
        self.d3q19 = d3q19
        self.cfg = cfg
        self.target_id = target_id

    @ti.kernel
    def apply_before_macro(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                v_sum = ti.Vector.zero(config.TI_FLOAT, 3)
                t_sum = 0.0
                f_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                g_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                count = 0
                
                for d in ti.static(range(1, 19)):
                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]
                    
                    if ti.static(self.cfg.periodic_x): ip = ip % ctx.nx
                    if ti.static(self.cfg.periodic_y): jp = jp % ctx.ny
                    is_inside = True
                    if ti.static(not self.cfg.periodic_x) and (ip < 0 or ip >= ctx.nx): is_inside = False
                    if ti.static(not self.cfg.periodic_y) and (jp < 0 or jp >= ctx.ny): is_inside = False
                    if kp < 0 or kp >= ctx.nz: is_inside = False

                    if is_inside and ctx.cell_id[ip, jp, kp] == FLUID_A:
                        rho_f = ctx.rho[ip, jp, kp]
                        v_f = ctx.v[ip, jp, kp]
                        t_f = ctx.temp[ip, jp, kp]
                        v_sum += v_f
                        t_sum += t_f
                        
                        for d2 in ti.static(range(19)):
                            feq_f = self.d3q19.get_feq(rho_f, v_f, d2)
                            f_neq_sum[d2] += (ctx.f_old[ip, jp, kp, d2] - feq_f)
                            geq_f = self.d3q19.get_geq(t_f, v_f, d2)
                            g_neq_sum[d2] += (ctx.g_old[ip, jp, kp, d2] - geq_f)
                        count += 1
                
                if count > 0:
                    v_out = v_sum / count
                    t_out = t_sum / count
                    rho_out = 1.0
                    
                    # 改善案3: 逆流のゼロ固定を削除し、純粋なNEEM外挿のみとする
                    for d in ti.static(range(19)):
                        feq_b = self.d3q19.get_feq(rho_out, v_out, d)
                        ctx.f_new[i, j, k, d] = feq_b + f_neq_sum[d] / count
                        
                        geq_b = self.d3q19.get_geq(t_out, v_out, d)
                        ctx.g_new[i, j, k, d] = geq_b + g_neq_sum[d] / count
                else:
                    for d in ti.static(range(19)):
                        v_zero = ti.Vector([0.0, 0.0, 0.0])
                        ctx.f_new[i, j, k, d] = self.d3q19.get_feq(1.0, v_zero, d)
                        ctx.g_new[i, j, k, d] = self.d3q19.get_geq(0.0, v_zero, d)


@ti.data_oriented
class IsothermalWall:
    def __init__(self, d3q19, cfg, target_id, temperature):
        self.d3q19 = d3q19
        self.cfg = cfg
        self.target_id = target_id
        self.temperature = temperature

    @ti.kernel
    def apply_before_macro(self, ctx: ti.template(), ramp_factor: ti.f32):
        temp_val = self.temperature
        v_zero = ti.Vector([0.0, 0.0, 0.0])
        
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                # 等温壁の温度場(g)にNEEMを適用
                g_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                count = 0
                
                for d in ti.static(range(1, 19)):
                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]
                    
                    if ti.static(self.cfg.periodic_x): ip = ip % ctx.nx
                    if ti.static(self.cfg.periodic_y): jp = jp % ctx.ny
                    is_inside = True
                    if ti.static(not self.cfg.periodic_x) and (ip < 0 or ip >= ctx.nx): is_inside = False
                    if ti.static(not self.cfg.periodic_y) and (jp < 0 or jp >= ctx.ny): is_inside = False
                    if kp < 0 or kp >= ctx.nz: is_inside = False


                    if is_inside and ctx.cell_id[ip, jp, kp] == FLUID_A:
                        v_f = ctx.v[ip, jp, kp]
                        t_f = ctx.temp[ip, jp, kp]
                        for d2 in ti.static(range(19)):
                            geq_f = self.d3q19.get_geq(t_f, v_f, d2)
                            g_neq_sum[d2] += (ctx.g_old[ip, jp, kp, d2] - geq_f)
                        count += 1
                
                if count > 0:
                    for d in ti.static(range(19)):
                        geq_b = self.d3q19.get_geq(temp_val, v_zero, d)
                        ctx.g_new[i, j, k, d] = geq_b + g_neq_sum[d] / count
                else:
                    for d in ti.static(range(19)):
                        ctx.g_new[i, j, k, d] = self.d3q19.get_geq(temp_val, v_zero, d)

    @ti.kernel
    def apply_after_macro(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                ctx.temp[i, j, k] = self.temperature


@ti.data_oriented
class ConstantHeatFluxWall:
    def __init__(self, d3q19, target_id, q_val):
        self.d3q19 = d3q19
        self.target_id = target_id
        self.q_val = q_val

    @ti.kernel
    def apply_after_macro(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                ctx.temp[i, j, k] += self.q_val
                for d in ti.static(range(19)):
                    ctx.g_old[i, j, k, d] += self.d3q19.w[d] * self.q_val


class BoundaryManager:
    def __init__(self, d3q19, cfg):
        self.d3q19 = d3q19
        self.cfg = cfg
        self.before_macro_bcs =[]
        self.after_macro_bcs =[]

        for target_id, bc_info in cfg.boundary_conditions.items():
            bc_type = bc_info.get("type")
            
            if bc_type == "inlet":
                vel = bc_info.get("velocity",[0.0, 0.0, 0.0])
                temp = bc_info.get("temperature", 0.0)
                self.before_macro_bcs.append(VelocityInlet(self.d3q19, self.cfg, target_id, vel, temp))
            
            elif bc_type == "outlet":
                self.before_macro_bcs.append(PressureOutlet(self.d3q19, self.cfg, target_id))
            
            elif bc_type == "isothermal_wall":
                temp = bc_info.get("temperature", 1.0)
                # NEEM再構成のため before_macro に追加
                self.before_macro_bcs.append(IsothermalWall(self.d3q19, self.cfg, target_id, temp))
                # マクロ温度固定のため after_macro に追加
                self.after_macro_bcs.append(IsothermalWall(self.d3q19, self.cfg, target_id, temp))
            
            elif bc_type == "constant_heat_flux":
                q = bc_info.get("q", 0.0)
                self.after_macro_bcs.append(ConstantHeatFluxWall(self.d3q19, target_id, q))

    def apply_all_before_macro(self, ctx, ramp_factor):
        for bc in self.before_macro_bcs:
            if hasattr(bc, 'apply_before_macro'):
                if isinstance(bc, VelocityInlet) or isinstance(bc, IsothermalWall):
                    bc.apply_before_macro(ctx, ramp_factor)
                else:
                    bc.apply_before_macro(ctx)

    def apply_all_after_macro(self, ctx):
        for bc in self.after_macro_bcs:
            if hasattr(bc, 'apply_after_macro'):
                bc.apply_after_macro(ctx)