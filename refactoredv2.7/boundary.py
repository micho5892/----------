# ==========================================================
# boundary.py — BoundaryConditionManager（NEEM 共通化）
# ==========================================================
import taichi as ti
import config
from context import FLUID_A
from lbm_logger import get_logger

_log = get_logger(__name__)


@ti.data_oriented
class NeemOps:
    """非平衡補外法（NEEM）用: 近傍座標の折り返しと FLUID_A からの neq 集約を共通化する。"""

    def __init__(self, d3q19, periodic_x: bool, periodic_y: bool):
        self.d3q19 = d3q19
        self.periodic_x = bool(periodic_x)
        self.periodic_y = bool(periodic_y)

    @ti.func
    def neighbor_cell(
        self, i: ti.i32, j: ti.i32, k: ti.i32, d: ti.i32, nx: ti.i32, ny: ti.i32, nz: ti.i32
    ):
        """有効時は (1, ip, jp, kp)、無効時は (0,0,0,0) を返す。
        Taichi では非 static 分岐内の return が禁止のため、valid フラグでまとめて末尾 return する。
        """
        ip = i + self.d3q19.e[d][0]
        jp = j + self.d3q19.e[d][1]
        kp = k + self.d3q19.e[d][2]
        valid = 1

        if ti.static(self.periodic_x):
            ip = (ip % nx + nx) % nx
        else:
            if ip < 0 or ip >= nx:
                valid = 0

        if ti.static(self.periodic_y):
            jp = (jp % ny + ny) % ny
        else:
            if jp < 0 or jp >= ny:
                valid = 0

        if kp < 0 or kp >= nz:
            valid = 0

        return ti.Vector([valid, ip, jp, kp])


@ti.data_oriented
class VelocityInlet:
    """速度・温度ディリクレ入口（f, g とも NEEM）。"""

    def __init__(self, d3q19, neem: NeemOps, target_id, velocity, temperature):
        self.d3q19 = d3q19
        self.neem = neem
        self.target_id = target_id
        self.velocity = ti.Vector(velocity)
        self.temperature = temperature

    @ti.kernel
    def apply_before_macro(self, ctx: ti.template(), ramp_factor: ti.f32):
        vel_vec = self.velocity * ramp_factor
        temp_val = self.temperature
        nx, ny, nz = ctx.nx, ctx.ny, ctx.nz
        for i, j, k in ti.ndrange(nx, ny, nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                f_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                g_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                rho_sum = 0.0
                count = 0

                for d in ti.static(range(1, 19)):
                    nb = self.neem.neighbor_cell(i, j, k, d, nx, ny, nz)
                    if nb[0] == 1:
                        ip, jp, kp = nb[1], nb[2], nb[3]
                        if ctx.cell_id[ip, jp, kp] == FLUID_A:
                            rho_f = ctx.rho[ip, jp, kp]
                            v_f = ctx.v[ip, jp, kp]
                            t_f = ctx.temp[ip, jp, kp]
                            rho_sum += rho_f
                            for d2 in ti.static(range(19)):
                                feq_f = self.d3q19.get_feq(rho_f, v_f, d2)
                                geq_f = self.d3q19.get_geq(t_f, v_f, d2)
                                f_neq_sum[d2] += ctx.f_old[ip, jp, kp, d2] - feq_f
                                g_neq_sum[d2] += ctx.g_old[ip, jp, kp, d2] - geq_f
                            count += 1

                if count > 0:
                    inv_c = 1.0 / count
                    rho_in = rho_sum * inv_c
                    for d in ti.static(range(19)):
                        feq_b = self.d3q19.get_feq(rho_in, vel_vec, d)
                        geq_b = self.d3q19.get_geq(temp_val, vel_vec, d)
                        ctx.f_new[i, j, k, d] = feq_b + f_neq_sum[d] * inv_c
                        ctx.g_new[i, j, k, d] = geq_b + g_neq_sum[d] * inv_c
                else:
                    for d in ti.static(range(19)):
                        ctx.f_new[i, j, k, d] = self.d3q19.get_feq(1.0, vel_vec, d)
                        ctx.g_new[i, j, k, d] = self.d3q19.get_geq(temp_val, vel_vec, d)


@ti.data_oriented
class PressureOutlet:
    """圧力（密度）固定出口: 内挿した v, T から f,g を NEEM で再構成。逆流時は v をゼロに。"""

    def __init__(self, d3q19, neem: NeemOps, target_id):
        self.d3q19 = d3q19
        self.neem = neem
        self.target_id = target_id

    @ti.kernel
    def apply_before_macro(self, ctx: ti.template()):
        nx, ny, nz = ctx.nx, ctx.ny, ctx.nz
        for i, j, k in ti.ndrange(nx, ny, nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                f_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                g_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                v_sum = ti.Vector.zero(config.TI_FLOAT, 3)
                t_sum = 0.0
                n_in = ti.Vector.zero(config.TI_FLOAT, 3)
                count = 0

                for d in ti.static(range(1, 19)):
                    nb = self.neem.neighbor_cell(i, j, k, d, nx, ny, nz)
                    if nb[0] == 1:
                        ip, jp, kp = nb[1], nb[2], nb[3]
                        if ctx.cell_id[ip, jp, kp] == FLUID_A:
                            rho_f = ctx.rho[ip, jp, kp]
                            v_f = ctx.v[ip, jp, kp]
                            t_f = ctx.temp[ip, jp, kp]
                            v_sum += v_f
                            t_sum += t_f
                            n_in += ti.Vector(
                                [
                                    float(self.d3q19.e[d][0]),
                                    float(self.d3q19.e[d][1]),
                                    float(self.d3q19.e[d][2]),
                                ]
                            )
                            for d2 in ti.static(range(19)):
                                feq_f = self.d3q19.get_feq(rho_f, v_f, d2)
                                geq_f = self.d3q19.get_geq(t_f, v_f, d2)
                                f_neq_sum[d2] += ctx.f_old[ip, jp, kp, d2] - feq_f
                                g_neq_sum[d2] += ctx.g_old[ip, jp, kp, d2] - geq_f
                            count += 1

                if count > 0:
                    inv_c = 1.0 / count
                    v_out = v_sum * inv_c
                    t_out = t_sum * inv_c
                    rho_out = 1.0
                    if v_out.dot(n_in) > 0.0:
                        v_out = ti.Vector([0.0, 0.0, 0.0])
                    for d in ti.static(range(19)):
                        feq_b = self.d3q19.get_feq(rho_out, v_out, d)
                        geq_b = self.d3q19.get_geq(t_out, v_out, d)
                        ctx.f_new[i, j, k, d] = feq_b + f_neq_sum[d] * inv_c
                        ctx.g_new[i, j, k, d] = geq_b + g_neq_sum[d] * inv_c
                else:
                    v_zero = ti.Vector([0.0, 0.0, 0.0])
                    for d in ti.static(range(19)):
                        ctx.f_new[i, j, k, d] = self.d3q19.get_feq(1.0, v_zero, d)
                        ctx.g_new[i, j, k, d] = self.d3q19.get_geq(0.0, v_zero, d)

# boundary.py の IsothermalWall を修正
@ti.data_oriented
class IsothermalWall:
    def __init__(self, d3q19, neem: NeemOps, use_neem: bool, target_id, temperature):
        # use_neem や neem 引数は残しておいても良いですが、処理は使いません
        self.target_id = target_id
        self.temperature = temperature

    @ti.kernel
    def apply_after_macro(self, ctx: ti.template()):
        # 可視化や Analytics のため、固体セルに温度値だけ入れておく
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                ctx.temp[i, j, k] = self.temperature
                # ★ g_old をNEEMで上書きする処理は、ABBと干渉しシステムを破壊するため全て削除

@ti.data_oriented
class _IsothermalWall_old:
    """等温壁: マクロ温度固定。g は隣接流体から NEEM（オプション）。"""

    def __init__(self, d3q19, neem: NeemOps, use_neem: bool, target_id, temperature):
        self.d3q19 = d3q19
        self.neem = neem
        self.use_neem = bool(use_neem)
        self.target_id = target_id
        self.temperature = temperature

    @ti.kernel
    def apply_after_macro(self, ctx: ti.template()):
        v_zero = ti.Vector([0.0, 0.0, 0.0])
        nx, ny, nz = ctx.nx, ctx.ny, ctx.nz
        Tw = self.temperature
        for i, j, k in ti.ndrange(nx, ny, nz):
            if ctx.cell_id[i, j, k] == self.target_id:
                ctx.temp[i, j, k] = Tw

                if ti.static(self.use_neem):
                    g_neq_sum = ti.Vector.zero(config.TI_FLOAT, 19)
                    count = 0
                    for d in ti.static(range(1, 19)):
                        nb = self.neem.neighbor_cell(i, j, k, d, nx, ny, nz)
                        if nb[0] == 1:
                            ip, jp, kp = nb[1], nb[2], nb[3]
                            if ctx.cell_id[ip, jp, kp] == FLUID_A:
                                v_f = ctx.v[ip, jp, kp]
                                t_f = ctx.temp[ip, jp, kp]
                                for d2 in ti.static(range(19)):
                                    geq_f = self.d3q19.get_geq(t_f, v_f, d2)
                                    g_neq_sum[d2] += ctx.g_old[ip, jp, kp, d2] - geq_f
                                count += 1
                    if count > 0:
                        inv_c = 1.0 / count
                        for d in ti.static(range(19)):
                            geq_b = self.d3q19.get_geq(Tw, v_zero, d)
                            ctx.g_old[i, j, k, d] = geq_b + g_neq_sum[d] * inv_c
                    else:
                        for d in ti.static(range(19)):
                            ctx.g_old[i, j, k, d] = self.d3q19.get_geq(Tw, v_zero, d)
                else:
                    for d in ti.static(range(19)):
                        ctx.g_old[i, j, k, d] = self.d3q19.get_geq(Tw, v_zero, d)


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
        self.neem = NeemOps(d3q19, cfg.periodic_x, cfg.periodic_y)
        self.before_macro_bcs = []
        self.after_macro_bcs = []

        for target_id, bc_info in cfg.boundary_conditions.items():
            bc_type = bc_info.get("type")

            if bc_type == "inlet":
                vel = bc_info.get("velocity", [0.0, 0.0, 0.0])
                temp = bc_info.get("temperature", 0.0)
                self.before_macro_bcs.append(
                    VelocityInlet(self.d3q19, self.neem, target_id, vel, temp)
                )

            elif bc_type == "outlet":
                self.before_macro_bcs.append(PressureOutlet(self.d3q19, self.neem, target_id))

            elif bc_type == "isothermal_wall":
                temp = bc_info.get("temperature", 1.0)
                self.after_macro_bcs.append(
                    IsothermalWall(
                        self.d3q19,
                        self.neem,
                        cfg.neem_isothermal_wall,
                        target_id,
                        temp,
                    )
                )

            elif bc_type == "constant_heat_flux":
                q = bc_info.get("q", 0.0)
                self.after_macro_bcs.append(ConstantHeatFluxWall(self.d3q19, target_id, q))

            elif bc_type == "adiabatic_wall":
                pass

        _log.debug(
            "BoundaryManager: before_macro=%s, after_macro=%s",
            len(self.before_macro_bcs),
            len(self.after_macro_bcs),
        )

    def apply_all_before_macro(self, ctx, ramp_factor):
        for bc in self.before_macro_bcs:
            if isinstance(bc, VelocityInlet):
                bc.apply_before_macro(ctx, ramp_factor)
            else:
                bc.apply_before_macro(ctx)

    def apply_all_after_macro(self, ctx):
        for bc in self.after_macro_bcs:
            bc.apply_after_macro(ctx)
