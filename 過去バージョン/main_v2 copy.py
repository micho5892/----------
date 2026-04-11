import taichi as ti
import numpy as np
import imageio.v2 as imageio
from IPython.display import Image
import os
from datetime import datetime

ti.init(arch=ti.gpu)

# ==========================================================
# 1. パラメータ管理 (Configuration)
# ==========================================================
class SimConfig:
    def __init__(self, **kwargs):
        self.nx = kwargs.get('nx', 128)
        self.ny = kwargs.get('ny', 128)
        self.nz = kwargs.get('nz', 32)

        # 物理・物性パラメータ (SI単位)
        self.Lx_p = kwargs.get('Lx_p', 0.1)
        self.U_inlet_p = kwargs.get('U_inlet_p', 1.0)
        self.T_inlet_p = kwargs.get('T_inlet_p', 300.0)
        self.T_wall_p = kwargs.get('T_wall_p', 350.0)
        self.q_wall_p = kwargs.get('q_wall_p', 5000.0)

        self.bc_type = kwargs.get('bc_type', 'T') # 'T':定温, 'q':定熱流束
        self.bc_type_val = 0 if self.bc_type == 'T' else 1

        self.Pr = kwargs.get('Pr', 0.71)
        self.nu_p = kwargs.get('nu_p', 1.5e-5)
        self.k_p = kwargs.get('k_p', 0.026)
        self.rho_p = kwargs.get('rho_p', 1.2)
        self.Cp_p = kwargs.get('Cp_p', 1005.0)

        self.dx = self.Lx_p / self.nx

        use_legacy = ('tau_f' not in kwargs and 'nu_p' not in kwargs)

        if use_legacy or 'tau_f' in kwargs:
            self.tau_f = kwargs.get('tau_f', 1.0)
            self.tau_g = kwargs.get('tau_g', 1.0)
            self.v_inlet_val = kwargs.get('v_inlet',[0.0, 0.0, -0.1])
            self.v_inlet = ti.Vector(self.v_inlet_val)

            u_lbm = abs(self.v_inlet_val[2]) if self.v_inlet_val[2] != 0 else 0.1
            self.dt = self.dx * u_lbm / self.U_inlet_p
            self.nu_p = (self.tau_f - 0.5) / 3.0 * (self.dx**2) / self.dt
            alpha_p = (self.tau_g - 0.5) / 3.0 * (self.dx**2) / self.dt
            self.Pr = self.nu_p / alpha_p if alpha_p > 0 else 0.71
            self.k_p = self.rho_p * self.Cp_p * alpha_p
        else:
            self.u_lbm_inlet = 0.1
            self.dt = self.dx * self.u_lbm_inlet / self.U_inlet_p
            nu_lbm = self.nu_p * self.dt / (self.dx**2)
            self.tau_f = 3.0 * nu_lbm + 0.5

            alpha_p = self.k_p / (self.rho_p * self.Cp_p)
            alpha_lbm = alpha_p * self.dt / (self.dx**2)
            self.tau_g = 3.0 * alpha_lbm + 0.5
            self.v_inlet_val =[0.0, 0.0, -self.u_lbm_inlet]
            self.v_inlet = ti.Vector(self.v_inlet_val)

        if self.bc_type == 'T':
            self.delta_T_ref = self.T_wall_p - self.T_inlet_p
        else:
            self.delta_T_ref = self.q_wall_p * self.Lx_p / self.k_p

        self.n_particles = kwargs.get('n_particles', 10000)
        self.steps = kwargs.get('steps', 600)
        self.vis_interval = kwargs.get('vis_interval', 20)
        self.filename = kwargs.get('filename', 'cooling_modular_advanced.gif')

# ==========================================================
# 2. LBM 定数 (D3Q19 Constants & MRT Matrices)
# ==========================================================
@ti.data_oriented
class D3Q19:
    def __init__(self):
        self.e = ti.Vector.field(3, ti.i32, shape=19)
        e_np = np.array(
            [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0], [0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1], [-1,0,-1],[1,0,-1],[-1,0,1],
             [0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]], dtype=np.int32)
        self.e.from_numpy(e_np)

        self.inv_d = ti.field(ti.i32, shape=19)
        self.inv_d.from_numpy(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int32))

        self.w = ti.field(ti.f32, shape=19)
        self.w.from_numpy(np.array([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                                    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
                                    1/36, 1/36, 1/36, 1/36], dtype=np.float32))

        M_np = np.zeros((19, 19), dtype=np.float32)
        for i in range(19):
            x, y, z = float(e_np[i][0]), float(e_np[i][1]), float(e_np[i][2])
            n2 = x*x + y*y + z*z

            M_np[0, i]  = 1.0
            M_np[1, i]  = 19.0 * n2 - 30.0
            M_np[2, i]  = 21.0 * n2 * n2 - 53.0 * n2 + 24.0
            M_np[3, i]  = x
            M_np[4, i]  = (5.0 * n2 - 9.0) * x
            M_np[5, i]  = y
            M_np[6, i]  = (5.0 * n2 - 9.0) * y
            M_np[7, i]  = z
            M_np[8, i]  = (5.0 * n2 - 9.0) * z
            M_np[9, i]  = 3.0 * x*x - n2
            M_np[10, i] = (3.0 * n2 - 5.0) * (3.0 * x*x - n2)
            M_np[11, i] = y*y - z*z
            M_np[12, i] = (3.0 * n2 - 5.0) * (y*y - z*z)
            M_np[13, i] = x * y
            M_np[14, i] = y * z
            M_np[15, i] = z * x
            M_np[16, i] = (y*y - z*z) * x
            M_np[17, i] = (z*z - x*x) * y
            M_np[18, i] = (x*x - y*y) * z

        M_inv_np = np.linalg.inv(M_np).astype(np.float32)

        self.M = ti.field(ti.f32, shape=(19, 19))
        self.M_inv = ti.field(ti.f32, shape=(19, 19))
        self.M.from_numpy(M_np)
        self.M_inv.from_numpy(M_inv_np)

        self.S = ti.field(ti.f32, shape=19)
        self.S.from_numpy(np.array([
            0.0, 1.19, 1.4, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2,
            1.0, 1.4, 1.0, 1.4, 1.0, 1.0, 1.0, 1.98, 1.98, 1.98
        ], dtype=np.float32))

# ==========================================================
# 3. シミュレーター本体 (Solver)
# ==========================================================
@ti.data_oriented
class LBMSimulator:
    def __init__(self, config):
        self.cfg = config
        self.d3q19 = D3Q19()

        self.f_old = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz, 19))
        self.f_post= ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz, 19))
        self.f_new = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz, 19))

        self.g_old = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz, 19))
        self.g_post= ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz, 19))
        self.g_new = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz, 19))

        self.v = ti.Vector.field(3, ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz))
        self.rho = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz))
        self.temp = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz))

        self.mask = ti.field(ti.i32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz))
        self.sdf = ti.field(ti.f32, shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz))

        self.particle_pos = ti.Vector.field(3, ti.f32, shape=self.cfg.n_particles)
        self.img = ti.field(ti.f32, shape=(self.cfg.nx * 2, self.cfg.ny))

    @ti.func
    def get_feq(self, rho_val, v_val, d):
        eu = self.d3q19.e[d].dot(v_val)
        uv = v_val.dot(v_val)
        return self.d3q19.w[d] * rho_val * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*uv)

    @ti.func
    def get_geq(self, temp_val, v_val, d):
        return self.d3q19.w[d] * temp_val * (1.0 + 3.0 * self.d3q19.e[d].dot(v_val))

    @ti.kernel
    def init_fields(self):
        for i, j, k in self.rho:
            self.rho[i, j, k] = 1.0
            self.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.temp[i, j, k] = 1.0 if self.mask[i, j, k] == 1 else 0.0
            for d in range(19):
                self.f_old[i, j, k, d] = self.get_feq(1.0, self.v[i, j, k], d)
                self.g_old[i, j, k, d] = self.get_geq(self.temp[i, j, k], self.v[i, j, k], d)
        for n in range(self.cfg.n_particles):
            self.particle_pos[n] = ti.Vector([ti.random()*self.cfg.nx, ti.random()*self.cfg.ny, ti.random()*self.cfg.nz])

    @ti.kernel
    def collide(self):
        C_s = 0.16
        for i, j, k in ti.ndrange(self.cfg.nx, self.cfg.ny, self.cfg.nz):
            rho = self.rho[i, j, k]
            v = self.v[i, j, k]
            temp = self.temp[i, j, k]

            g_eq_cache = ti.Vector.zero(ti.f32, 19)
            for d in range(19):
                g_eq_cache[d] = self.get_geq(temp, v, d)
                self.g_post[i, j, k, d] = self.g_old[i, j, k, d] - (self.g_old[i, j, k, d] - g_eq_cache[d]) / self.cfg.tau_g

            if self.mask[i, j, k] == 0:
                Pi_xx, Pi_xy, Pi_xz = 0.0, 0.0, 0.0
                Pi_yy, Pi_yz, Pi_zz = 0.0, 0.0, 0.0

                f_eq_cache = ti.Vector.zero(ti.f32, 19)

                for d in range(19):
                    f_eq_cache[d] = self.get_feq(rho, v, d)
                    f_neq = self.f_old[i, j, k, d] - f_eq_cache[d]

                    ex = float(self.d3q19.e[d][0])
                    ey = float(self.d3q19.e[d][1])
                    ez = float(self.d3q19.e[d][2])

                    Pi_xx += ex * ex * f_neq
                    Pi_xy += ex * ey * f_neq
                    Pi_xz += ex * ez * f_neq
                    Pi_yy += ey * ey * f_neq
                    Pi_yz += ey * ez * f_neq
                    Pi_zz += ez * ez * f_neq

                Pi_mag = ti.math.sqrt(2.0 * (Pi_xx**2 + Pi_yy**2 + Pi_zz**2 + 2.0*(Pi_xy**2 + Pi_yz**2 + Pi_xz**2)))
                tau_0 = self.cfg.tau_f
                tau_total = 0.5 * (tau_0 + ti.math.sqrt(tau_0**2 + 18.0 * 1.41421356 * (C_s**2) * Pi_mag / rho))
                s_nu = 1.0 / tau_total

                m = ti.Vector.zero(ti.f32, 19)
                m_eq = ti.Vector.zero(ti.f32, 19)
                for d in range(19):
                    for d2 in range(19):
                        m[d] += self.d3q19.M[d, d2] * self.f_old[i, j, k, d2]
                        m_eq[d] += self.d3q19.M[d, d2] * f_eq_cache[d2]

                m_star = ti.Vector.zero(ti.f32, 19)
                for d in range(19):
                    s_val = self.d3q19.S[d]
                    if d == 9 or d == 11 or d == 13 or d == 14 or d == 15:
                        s_val = s_nu
                    m_star[d] = m[d] - s_val * (m[d] - m_eq[d])

                for d in range(19):
                    f_post_val = 0.0
                    for d2 in range(19):
                        f_post_val += self.d3q19.M_inv[d, d2] * m_star[d2]
                    self.f_post[i, j, k, d] = f_post_val

    @ti.kernel
    def stream_and_bc(self):
        for i, j, k in ti.ndrange(self.cfg.nx, self.cfg.ny, self.cfg.nz):
            if self.mask[i, j, k] == 0:
                for d in ti.static(range(19)):
                    inv_d = self.d3q19.inv_d[d]
                    ip = (i + self.d3q19.e[d][0]) % self.cfg.nx
                    jp = (j + self.d3q19.e[d][1]) % self.cfg.ny
                    kp = k + self.d3q19.e[d][2]

                    if 0 <= kp < self.cfg.nz:
                        if self.mask[ip, jp, kp] == 0:
                            self.g_new[ip, jp, kp, d] = self.g_post[i, j, k, d]
                        else:
                            T_w = 1.0
                            self.g_new[i, j, k, inv_d] = -self.g_post[i, j, k, d] + 2.0 * self.d3q19.w[d] * T_w

                        if self.mask[ip, jp, kp] == 0:
                            self.f_new[ip, jp, kp, d] = self.f_post[i, j, k, d]
                        else:
                            q = self.sdf[i, j, k] / (self.sdf[i, j, k] - self.sdf[ip, jp, kp])
                            q = ti.math.clamp(q, 0.001, 1.0)

                            i_back = (i - self.d3q19.e[d][0]) % self.cfg.nx
                            j_back = (j - self.d3q19.e[d][1]) % self.cfg.ny
                            k_back = k - self.d3q19.e[d][2]

                            f_curr = self.f_post[i, j, k, d]
                            f_back = f_curr
                            if 0 <= k_back < self.cfg.nz and self.mask[i_back, j_back, k_back] == 0:
                                f_back = self.f_post[i_back, j_back, k_back, d]
                            else:
                                q = 0.5

                            f_bb = 0.0
                            if q < 0.5:
                                f_bb = 2.0 * q * f_curr + (1.0 - 2.0 * q) * f_back
                            else:
                                f_inv = self.f_post[i, j, k, inv_d]
                                f_bb = (1.0 / (2.0 * q)) * f_curr + ((2.0 * q - 1.0) / (2.0 * q)) * f_inv

                            self.f_new[i, j, k, inv_d] = f_bb

    @ti.kernel
    def apply_bc(self):
        for i, j in ti.ndrange(self.cfg.nx, self.cfg.ny):
            for d in range(19):
                if self.d3q19.e[d][2] < 0: # Inlet
                    self.f_new[i, j, self.cfg.nz-1, d] = self.get_feq(1.0, self.cfg.v_inlet, d)
                    self.g_new[i, j, self.cfg.nz-1, d] = self.get_geq(0.0, self.cfg.v_inlet, d)
                if self.d3q19.e[d][2] > 0: # Outlet
                    self.f_new[i, j, 0, d] = self.f_new[i, j, 1, d]
                    self.g_new[i, j, 0, d] = self.g_new[i, j, 1, d]

    @ti.kernel
    def update_macro(self):
        for i, j, k in self.rho:
            if self.mask[i, j, k] == 0:
                new_rho, new_temp = 0.0, 0.0
                new_v = ti.Vector([0.0, 0.0, 0.0])
                for d in range(19):
                    f_val = self.f_new[i, j, k, d]
                    g_val = self.g_new[i, j, k, d]
                    new_rho += f_val
                    new_v += f_val * self.d3q19.e[d]
                    new_temp += g_val
                    self.f_old[i, j, k, d] = f_val
                    self.g_old[i, j, k, d] = g_val
                self.rho[i, j, k] = new_rho
                self.v[i, j, k] = new_v / new_rho
                self.temp[i, j, k] = new_temp
            else:
                self.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.temp[i, j, k] = 1.0

    @ti.kernel
    def apply_heat_flux_bc(self):
        sum_t, count = 0.0, 0
        for i, j, k in self.rho:
            if self.mask[i, j, k] == 1:
                for d in range(1, 7):
                    ip, jp, kp = (i + self.d3q19.e[d][0]) % self.cfg.nx, (j + self.d3q19.e[d][1]) % self.cfg.ny, k + self.d3q19.e[d][2]
                    if 0 <= kp < self.cfg.nz and self.mask[ip, jp, kp] == 0:
                        sum_t += self.temp[ip, jp, kp]
                        count += 1
        avg_tf = sum_t / count if count > 0 else 0.0
        new_t_solid = avg_tf + (1.0 / self.cfg.nx)
        for i, j, k in self.rho:
            if self.mask[i, j, k] == 1:
                self.temp[i, j, k] = new_t_solid

    @ti.kernel
    def get_avg_grad(self) -> ti.f32:
        sum_q, area = 0.0, 0.0
        for i, j, k in self.rho:
            if self.mask[i, j, k] == 1:
                for d in range(1, 7):
                    ip = (i + self.d3q19.e[d][0]) % self.cfg.nx
                    jp = (j + self.d3q19.e[d][1]) % self.cfg.ny
                    kp = k + self.d3q19.e[d][2]

                    if 0 <= kp < self.cfg.nz and self.mask[ip, jp, kp] == 0:
                        grad = (1.0 - self.temp[ip, jp, kp]) / 0.5
                        sum_q += grad
                        area += 1.0
        return sum_q / area if area > 0 else 0.0

    @ti.kernel
    def get_avg_t_out(self) -> ti.f32:
        sum_t, count = 0.0, 0
        for i, j, k in self.rho:
            if k == 1 and self.mask[i, j, k] == 0:
                sum_t += self.temp[i, j, k]
                count += 1
        return sum_t / count if count > 0 else 0.0

    @ti.kernel
    def get_solid_temp(self) -> ti.f32:
        sum_t, count = 0.0, 0
        for i, j, k in self.temp:
            if self.mask[i, j, k] == 1:
                sum_t += self.temp[i, j, k]
                count += 1
        return sum_t / count if count > 0 else 0.0

    @ti.kernel
    def move_particles(self):
        for n in range(self.cfg.n_particles):
            pos = self.particle_pos[n]
            i, j, k = int(pos[0]), int(pos[1]), int(pos[2])
            if 0 <= i < self.cfg.nx and 0 <= j < self.cfg.ny and 0 <= k < self.cfg.nz:
                self.particle_pos[n] += self.v[i, j, k] * 2.0

            new_pos = self.particle_pos[n]
            if new_pos[2] < 1.0 or new_pos[2] >= self.cfg.nz-1 or self.mask[int(new_pos[0]%self.cfg.nx), int(new_pos[1]%self.cfg.ny), int(new_pos[2])] == 1:
                self.particle_pos[n] = ti.Vector([ti.random()*self.cfg.nx, ti.random()*self.cfg.ny, self.cfg.nz-1.1])

# ==========================================================
# 4. 形状定義 & 実行管理
# ==========================================================
@ti.kernel
def build_pin_fin_sdf(mask: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
    for i, j, k in mask:
        sdf[i, j, k] = 100.0
        mask[i, j, k] = 0
        if 2 <= k < 18:
            x = float(i % 16) - 7.5
            y = float(j % 16) - 7.5
            radius = 4.0
            dist = ti.math.sqrt(x*x + y*y) - radius
            sdf[i, j, k] = dist

            if dist <= 0.0:
                mask[i, j, k] = 1

@ti.kernel
def build_validation_channel(mask: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
    for i, j, k in mask:
        mask[i, j, k] = 0
        sdf[i, j, k] = 100.0
        if i < 10 or i > (nx - 10):
            mask[i, j, k] = 1
            sdf[i, j, k] = 0.0

# ===== 追加：カルマン渦観測用の円柱形状の定義 =====
@ti.kernel
def build_karman_cylinder(mask: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
    for i, j, k in mask:
        mask[i, j, k] = 0
        sdf[i, j, k] = 100.0

        # 上流側 (Z軸方向が主流) に円柱を配置
        cx = nx * 0.5
        cz = nz * 0.8
        radius = nx * 0.1

        # X-Z平面の断面が円の円柱をY方向に伸ばす
        dist = ti.math.sqrt((float(i) - cx)**2 + (float(k) - cz)**2) - radius
        sdf[i, j, k] = dist

        if dist <= 0.0:
            mask[i, j, k] = 1

@ti.kernel
def get_local_Nu(sim: ti.template(), k_target: int) -> ti.f32:
    sum_uT, sum_u = 0.0, 0.0
    sum_q, area = 0.0, 0.0

    for i, j in ti.ndrange(sim.cfg.nx, sim.cfg.ny):
        if sim.mask[i, j, k_target] == 0:
            u_z = -sim.v[i, j, k_target][2]
            t = sim.temp[i, j, k_target]
            sum_uT += u_z * t
            sum_u += u_z

    T_bulk_nd = sum_uT / sum_u if sum_u > 0 else 0.0

    for i, j in ti.ndrange(sim.cfg.nx, sim.cfg.ny):
        if sim.mask[i, j, k_target] == 1:
            for d in ti.static([1, 2]):
                ip = (i + sim.d3q19.e[d][0]) % sim.cfg.nx
                if sim.mask[ip, j, k_target] == 0:
                    sum_q += (sim.temp[i, j, k_target] - sim.temp[ip, j, k_target]) * 2
                    area += 1.0

    avg_q_nd = sum_q / area if area > 0 else 0.0

    q_wall_p = sim.cfg.k_p * avg_q_nd * sim.cfg.delta_T_ref / sim.cfg.dx
    T_bulk_p = sim.cfg.T_inlet_p + T_bulk_nd * sim.cfg.delta_T_ref
    T_wall_p = sim.cfg.T_wall_p

    h_local = q_wall_p / (T_wall_p - T_bulk_p) if (T_wall_p - T_bulk_p) != 0 else 0.0

    H_p = (sim.cfg.nx - 20) / sim.cfg.nx * sim.cfg.Lx_p
    D_h = H_p * 2.0

    Nu_local = h_local * D_h / sim.cfg.k_p
    return Nu_local

@ti.kernel
def update_vis_image(sim: ti.template()):
    for i, j in ti.ndrange(sim.cfg.nx, sim.cfg.ny):
        sim.img[i, j] = sim.temp[i, j, 10]

    slice_y = sim.cfg.ny // 2 - 8

    for i, k in ti.ndrange(sim.cfg.nx, sim.cfg.nz):
        sim.img[sim.cfg.nx + i, k * 2] = sim.temp[i, slice_y, k]
        sim.img[sim.cfg.nx + i, k * 2 + 1] = sim.temp[i, slice_y, k]

    for n in range(sim.cfg.n_particles):
        pos = sim.particle_pos[n]
        if 9.5 <= pos[2] <= 10.5:
            sim.img[int(pos[0]), int(pos[1])] = 2.0
        if (slice_y - 1) <= pos[1] <= (slice_y + 1):
            sim.img[sim.cfg.nx + int(pos[0]), int(pos[2]*2)] = 2.0

def run_simulation(**kwargs):
    # ==== 追加：日時を用いたファイル名と保存先フォルダの設定 ====
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = os.path.join(out_dir, f"karman_vortex_{timestamp}.gif")

    # kwargsにfilenameが指定されていなければ、自動生成のものを適用
    if 'filename' not in kwargs:
        kwargs['filename'] = default_filename

    cfg = SimConfig(**kwargs)
    sim = LBMSimulator(cfg)

    
    build_validation_channel(sim.mask, sim.sdf, cfg.nx, cfg.ny, cfg.nz)
    sim.init_fields()

    frames =[]
    cylinder_diameter_p = cfg.Lx_p * (cfg.nx * 0.2) / cfg.nx
    Re = (cfg.U_inlet_p * cylinder_diameter_p) / cfg.nu_p
    print(f"Starting advanced simulation: {cfg.nx}x{cfg.ny}x{cfg.nz}, steps={cfg.steps}")
    print(f"[Properties] nu: {cfg.nu_p:.2e}, k: {cfg.k_p:.3f}, Pr: {cfg.Pr:.2f}, BC Type: {'Constant Temp' if cfg.bc_type=='T' else 'Constant Heat Flux'}")
    print(f"[Dimensionless Numbers] Reynolds Number (Re): {Re:.2f}, Prandtl Number (Pr): {cfg.Pr:.2f}")
    instance_attributes = [attr for attr in dir(cfg) if not attr.startswith('__') and not callable(getattr(cfg, attr))]
    for instance_attribute in instance_attributes:
        instance_value = getattr(cfg, instance_attribute)
        print(instance_attribute,": ",instance_value)

    k_target_nu = int(cfg.nz - 10) 



    for step in range(cfg.steps):
        sim.collide()
        sim.stream_and_bc()
        sim.apply_bc()
        sim.update_macro()

        if cfg.bc_type_val == 1:
            sim.apply_heat_flux_bc()

        sim.move_particles()

        if step % cfg.vis_interval == 0:
            avg_grad_nd = sim.get_avg_grad()
            avg_t_out_nd = sim.get_avg_t_out()

            q_wall_avg = cfg.k_p * avg_grad_nd * cfg.delta_T_ref / cfg.dx

            if cfg.bc_type == 'T':
                T_w_p = cfg.T_wall_p
            else:
                T_w_nd = sim.get_solid_temp()
                T_w_p = cfg.T_inlet_p + T_w_nd * cfg.delta_T_ref

            T_out_p = cfg.T_inlet_p + avg_t_out_nd * cfg.delta_T_ref
            h_coeff = q_wall_avg / (T_w_p - cfg.T_inlet_p) if (T_w_p - cfg.T_inlet_p) != 0 else 0.0
            effectiveness = (T_out_p - cfg.T_inlet_p) / (T_w_p - cfg.T_inlet_p) if (T_w_p - cfg.T_inlet_p) != 0 else 0.0

            if step % 100 == 0:
                H_g = cfg.nx - 20
                H_p = (H_g / cfg.nx) * cfg.Lx_p
                D_h = H_p * 2
                Hu_sim = h_coeff * D_h / cfg.k_p
                local_nu = get_local_Nu(sim, k_target_nu)
                P_Error = abs(local_nu - 7.54) / 7.54 * 100

                print(f"Step {step:4d} | T_wall: {T_w_p:.1f} K | h: {h_coeff:.2f} W/m2K | Eff(ε): {effectiveness:.4f} | T_out: {T_out_p:.2f} K | Nu_local({k_target_nu}): {local_nu:.2f} | Error: {P_Error:.2f} %")

            # ==== 変更：製図のような三面図配置・大解像度表示対応の可視化 ====
            temp_np = sim.temp.to_numpy()
            mask_np = sim.mask.to_numpy()

            # 各平面でのスライス取得
            front = temp_np[:, cfg.ny//2, :]      # 正面図 (XZ平面)
            front_mask = mask_np[:, cfg.ny//2, :]

            side = temp_np[cfg.nx//2, :, :]       # 側面図 (YZ平面)
            side_mask = mask_np[cfg.nx//2, :, :]

            top_z = int(cfg.nz * 0.7)             # 平面図 (XY平面)
            top = temp_np[:, :, top_z]
            top_mask = mask_np[:, :, top_z]

            # 画像の色付けと壁マスクの処理
            def make_rgb(val, msk):
                rgb = np.zeros((*val.shape, 3), dtype=np.uint8)
                t_mask = (msk == 0)
                p_mask = (msk == 1)
                rgb[:, :, 0][t_mask] = (np.clip(val[t_mask], 0, 1) * 255).astype(np.uint8)
                rgb[:, :, 2][t_mask] = ((1 - np.clip(val[t_mask], 0, 1)) * 255).astype(np.uint8)
                rgb[p_mask] =[128, 128, 128] # 壁をグレー表示
                return rgb

            # XYZ軸を画像座標に合わせるため転置、および主流が上から下へ流れるように上下反転
            front_rgb = np.flipud(np.transpose(make_rgb(front, front_mask), (1, 0, 2)))
            side_rgb  = np.flipud(np.transpose(make_rgb(side, side_mask), (1, 0, 2)))
            top_rgb   = np.flipud(np.transpose(make_rgb(top, top_mask), (1, 0, 2)))

            h_top, w_top = top_rgb.shape[:2]
            h_front, w_front = front_rgb.shape[:2]
            h_side, w_side = side_rgb.shape[:2]

            total_h = h_top + h_front
            total_w = max(w_top, w_front + w_side)

            # 三面図を格納するキャンバス
            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

            # 製図の配置風に組み合わせる
            canvas[0:h_top, 0:w_top] = top_rgb                             # 左上に平面図
            canvas[h_top:total_h, 0:w_front] = front_rgb                   # 左下に正面図
            canvas[h_top:total_h, w_front:w_front+w_side] = side_rgb       # 右下に側面図

            # パーティクルの投影
            p_pos = sim.particle_pos.to_numpy()
            for p in p_pos:
                x, y, z = int(p[0]), int(p[1]), int(p[2])
                z_draw = cfg.nz - 1 - z
                y_draw_top = cfg.ny - 1 - y

                if 0 <= x < cfg.nx and 0 <= y_draw_top < cfg.ny:
                    canvas[y_draw_top, x] =[255, 255, 255]

                if 0 <= x < cfg.nx and 0 <= z_draw < cfg.nz:
                    canvas[h_top + z_draw, x] =[255, 255, 255]

                if 0 <= y < cfg.ny and 0 <= z_draw < cfg.nz:
                    canvas[h_top + z_draw, w_front + y] = [255, 255, 255]

            frames.append(canvas)

    imageio.mimsave(cfg.filename, frames, fps=12)
    print(f"Finished. Result saved as {cfg.filename}")
    return cfg.filename

# ==========================================================
# 実行例
# ==========================================================
# レイノルズ数を上げてカルマン渦が発生しやすくするため、流速を上げ、動粘性係数を下げています
gif_file_val = run_simulation(
    nx=128,
    ny=128,
    nz=512,
    Lx_p=0.05,
    U_inlet_p=0.05,  # 代表流速を上昇
    nu_p=3.0e-6,     # 動粘性係数を低下 (Re数を約150付近に設定)
    k_p=0.026,
    steps=3000,      # カルマン渦の発達を見るためステップ数を確保
    vis_interval=30, # インターバル調整
    bc_type='T'
)

Image(open(gif_file_val, 'rb').read())