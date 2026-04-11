import taichi as ti
import numpy as np
import imageio.v2 as imageio
from IPython.display import Image

ti.init(arch=ti.cpu)

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

        # Local Grid Refinement (AMR) 用の緩和時間スケーリング
        self.tau_f_fine = 2.0 * (self.tau_f - 0.5) + 0.5
        self.tau_g_fine = 2.0 * (self.tau_g - 0.5) + 0.5

        if self.bc_type == 'T':
            self.delta_T_ref = self.T_wall_p - self.T_inlet_p
        else:
            self.delta_T_ref = self.q_wall_p * self.Lx_p / self.k_p

        self.n_particles = kwargs.get('n_particles', 10000)
        self.steps = kwargs.get('steps', 600)
        self.vis_interval = kwargs.get('vis_interval', 20)
        self.filename = kwargs.get('filename', 'cooling_modular_advanced_amr.gif')

# ==========================================================
# 2. LBM 定数 (D3Q19 Constants & MRT Matrices)
# ==========================================================
@ti.data_oriented
class D3Q19:
    def __init__(self):
        self.e = ti.Vector.field(3, ti.i32, shape=19)
        e_np = np.array(
            [[0,0,0],[1,0,0],[-1,0,0], [0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
             [1,1,0],[-1,-1,0],[1,-1,0], [-1,1,0],[1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],
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
# 補間用ヘルパー関数群 (Trilinear Interpolation)
# ==========================================================
@ti.func
def interp3d_scalar(field: ti.template(), x: ti.f32, y: ti.f32, z: ti.f32):
    i0 = int(ti.math.floor(x)); j0 = int(ti.math.floor(y)); k0 = int(ti.math.floor(z))
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    wx = x - float(i0); wy = y - float(j0); wz = z - float(k0)
    
    nx1 = field.shape[0] - 1; ny1 = field.shape[1] - 1; nz1 = field.shape[2] - 1
    i0 = ti.math.clamp(i0, 0, nx1); j0 = ti.math.clamp(j0, 0, ny1); k0 = ti.math.clamp(k0, 0, nz1)
    i1 = ti.math.clamp(i1, 0, nx1); j1 = ti.math.clamp(j1, 0, ny1); k1 = ti.math.clamp(k1, 0, nz1)

    c00 = field[i0, j0, k0] * (1.0 - wx) + field[i1, j0, k0] * wx
    c10 = field[i0, j1, k0] * (1.0 - wx) + field[i1, j1, k0] * wx
    c01 = field[i0, j0, k1] * (1.0 - wx) + field[i1, j0, k1] * wx
    c11 = field[i0, j1, k1] * (1.0 - wx) + field[i1, j1, k1] * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz

@ti.func
def interp3d_vec(field: ti.template(), x: ti.f32, y: ti.f32, z: ti.f32):
    i0 = int(ti.math.floor(x)); j0 = int(ti.math.floor(y)); k0 = int(ti.math.floor(z))
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    wx = x - float(i0); wy = y - float(j0); wz = z - float(k0)
    
    nx1 = field.shape[0] - 1; ny1 = field.shape[1] - 1; nz1 = field.shape[2] - 1
    i0 = ti.math.clamp(i0, 0, nx1); j0 = ti.math.clamp(j0, 0, ny1); k0 = ti.math.clamp(k0, 0, nz1)
    i1 = ti.math.clamp(i1, 0, nx1); j1 = ti.math.clamp(j1, 0, ny1); k1 = ti.math.clamp(k1, 0, nz1)

    c00 = field[i0, j0, k0] * (1.0 - wx) + field[i1, j0, k0] * wx
    c10 = field[i0, j1, k0] * (1.0 - wx) + field[i1, j1, k0] * wx
    c01 = field[i0, j0, k1] * (1.0 - wx) + field[i1, j0, k1] * wx
    c11 = field[i0, j1, k1] * (1.0 - wx) + field[i1, j1, k1] * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz

@ti.func
def interp3d_d(field: ti.template(), x: ti.f32, y: ti.f32, z: ti.f32, d: ti.i32):
    i0 = int(ti.math.floor(x)); j0 = int(ti.math.floor(y)); k0 = int(ti.math.floor(z))
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    wx = x - float(i0); wy = y - float(j0); wz = z - float(k0)
    
    nx1 = field.shape[0] - 1; ny1 = field.shape[1] - 1; nz1 = field.shape[2] - 1
    i0 = ti.math.clamp(i0, 0, nx1); j0 = ti.math.clamp(j0, 0, ny1); k0 = ti.math.clamp(k0, 0, nz1)
    i1 = ti.math.clamp(i1, 0, nx1); j1 = ti.math.clamp(j1, 0, ny1); k1 = ti.math.clamp(k1, 0, nz1)

    c00 = field[i0, j0, k0, d] * (1.0 - wx) + field[i1, j0, k0, d] * wx
    c10 = field[i0, j1, k0, d] * (1.0 - wx) + field[i1, j1, k0, d] * wx
    c01 = field[i0, j0, k1, d] * (1.0 - wx) + field[i1, j0, k1, d] * wx
    c11 = field[i0, j1, k1, d] * (1.0 - wx) + field[i1, j1, k1, d] * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz

# ==========================================================
# 3. シミュレーター本体 (Solver)
# ==========================================================
@ti.data_oriented
class LBMSimulator:
    def __init__(self, config):
        self.cfg = config
        self.d3q19 = D3Q19()

        # --- Coarse Grid Fields ---
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

        # --- Fine Grid Fields (AMR) ---
        self.f_old_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, 19))
        self.f_post_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, 19))
        self.f_new_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, 19))

        self.g_old_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, 19))
        self.g_post_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, 19))
        self.g_new_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, 19))

        self.v_f = ti.Vector.field(3, ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2))
        self.rho_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2))
        self.temp_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2))

        self.mask_f = ti.field(ti.i32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2))
        self.sdf_f = ti.field(ti.f32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2))
        self.refine_type = ti.field(ti.i32, shape=(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2)) # 0:Out, 1:Compute, 2:Buffer

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
        # Coarse init
        for i, j, k in self.rho:
            self.rho[i, j, k] = 1.0
            self.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.temp[i, j, k] = 1.0 if self.mask[i, j, k] == 1 else 0.0
            for d in range(19):
                self.f_old[i, j, k, d] = self.get_feq(1.0, self.v[i, j, k], d)
                self.g_old[i, j, k, d] = self.get_geq(self.temp[i, j, k], self.v[i, j, k], d)

        # Fine init
        for i, j, k in self.rho_f:
            if self.refine_type[i, j, k] > 0:
                self.rho_f[i, j, k] = 1.0
                self.v_f[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.temp_f[i, j, k] = 1.0 if self.mask_f[i, j, k] == 1 else 0.0
                for d in range(19):
                    self.f_old_f[i, j, k, d] = self.get_feq(1.0, self.v_f[i, j, k], d)
                    self.g_old_f[i, j, k, d] = self.get_geq(self.temp_f[i, j, k], self.v_f[i, j, k], d)

        for n in range(self.cfg.n_particles):
            self.particle_pos[n] = ti.Vector([ti.random()*self.cfg.nx, ti.random()*self.cfg.ny, ti.random()*self.cfg.nz])

    @ti.func
    def do_collide(self, i, j, k, rho_field: ti.template(), v_field: ti.template(), temp_field: ti.template(),
                   mask_field: ti.template(), f_old_field: ti.template(), g_old_field: ti.template(),
                   f_post_field: ti.template(), g_post_field: ti.template(), tau_f, tau_g):
        rho = rho_field[i, j, k]
        v = v_field[i, j, k]
        temp = temp_field[i, j, k]
        
        g_eq_cache = ti.Vector.zero(ti.f32, 19)
        for d in range(19):
            g_eq_cache[d] = self.get_geq(temp, v, d)
            g_post_field[i, j, k, d] = g_old_field[i, j, k, d] - (g_old_field[i, j, k, d] - g_eq_cache[d]) / tau_g

        if mask_field[i, j, k] == 0:
            Pi_xx, Pi_xy, Pi_xz, Pi_yy, Pi_yz, Pi_zz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            f_eq_cache = ti.Vector.zero(ti.f32, 19)

            for d in range(19):
                f_eq_cache[d] = self.get_feq(rho, v, d)
                f_neq = f_old_field[i, j, k, d] - f_eq_cache[d]
                ex, ey, ez = float(self.d3q19.e[d][0]), float(self.d3q19.e[d][1]), float(self.d3q19.e[d][2])
                Pi_xx += ex * ex * f_neq
                Pi_xy += ex * ey * f_neq
                Pi_xz += ex * ez * f_neq
                Pi_yy += ey * ey * f_neq
                Pi_yz += ey * ez * f_neq
                Pi_zz += ez * ez * f_neq

            Pi_mag = ti.math.sqrt(2.0 * (Pi_xx**2 + Pi_yy**2 + Pi_zz**2 + 2.0*(Pi_xy**2 + Pi_yz**2 + Pi_xz**2)))
            C_s = 0.16 
            tau_total = 0.5 * (tau_f + ti.math.sqrt(tau_f**2 + 18.0 * 1.41421356 * (C_s**2) * Pi_mag / rho))
            s_nu = 1.0 / tau_total 

            m = ti.Vector.zero(ti.f32, 19)
            m_eq = ti.Vector.zero(ti.f32, 19)
            for d in range(19):
                for d2 in range(19):
                    m[d] += self.d3q19.M[d, d2] * f_old_field[i, j, k, d2]
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
                f_post_field[i, j, k, d] = f_post_val

    @ti.kernel
    def collide(self):
        for i, j, k in ti.ndrange(self.cfg.nx, self.cfg.ny, self.cfg.nz):
            self.do_collide(i, j, k, self.rho, self.v, self.temp, self.mask, self.f_old, self.g_old, self.f_post, self.g_post, self.cfg.tau_f, self.cfg.tau_g)

    @ti.kernel
    def collide_fine(self):
        for i, j, k in ti.ndrange(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2):
            if self.refine_type[i, j, k] > 0:
                self.do_collide(i, j, k, self.rho_f, self.v_f, self.temp_f, self.mask_f, self.f_old_f, self.g_old_f, self.f_post_f, self.g_post_f, self.cfg.tau_f_fine, self.cfg.tau_g_fine)

    @ti.func
    def do_stream_and_bc(self, i, j, k, nx, ny, nz, mask_field: ti.template(), sdf_field: ti.template(),
                         f_post_field: ti.template(), g_post_field: ti.template(),
                         f_new_field: ti.template(), g_new_field: ti.template()):
        for d in ti.static(range(19)):
            ip, jp, kp = (i + self.d3q19.e[d][0]) % nx, (j + self.d3q19.e[d][1]) % ny, k + self.d3q19.e[d][2]
            if 0 <= kp < nz:
                g_new_field[ip, jp, kp, d] = g_post_field[i, j, k, d]

        if mask_field[i, j, k] == 0:
            for d in ti.static(range(19)):
                inv_d = self.d3q19.inv_d[d]
                ip, jp, kp = (i + self.d3q19.e[d][0]) % nx, (j + self.d3q19.e[d][1]) % ny, k + self.d3q19.e[d][2]
                if 0 <= kp < nz:
                    if mask_field[ip, jp, kp] == 0:
                        f_new_field[ip, jp, kp, d] = f_post_field[i, j, k, d]
                    else:
                        q = sdf_field[i, j, k] / (sdf_field[i, j, k] - sdf_field[ip, jp, kp])
                        q = ti.math.clamp(q, 0.001, 1.0)
                        i_back, j_back, k_back = (i - self.d3q19.e[d][0]) % nx, (j - self.d3q19.e[d][1]) % ny, k - self.d3q19.e[d][2]
                        
                        f_curr = f_post_field[i, j, k, d]
                        f_back = f_curr
                        f_bb = 0
                        if 0 <= k_back < nz and mask_field[i_back, j_back, k_back] == 0:
                            f_back = f_post_field[i_back, j_back, k_back, d]
                        else:
                            q = 0.5 

                        if q < 0.5:
                            f_bb = 2.0 * q * f_curr + (1.0 - 2.0 * q) * f_back
                        else:
                            f_inv = f_post_field[i, j, k, inv_d]
                            f_bb = (1.0 / (2.0 * q)) * f_curr + ((2.0 * q - 1.0) / (2.0 * q)) * f_inv

                        f_new_field[i, j, k, inv_d] = f_bb

    @ti.kernel
    def stream_and_bc(self):
        for i, j, k in ti.ndrange(self.cfg.nx, self.cfg.ny, self.cfg.nz):
            self.do_stream_and_bc(i, j, k, self.cfg.nx, self.cfg.ny, self.cfg.nz, self.mask, self.sdf, self.f_post, self.g_post, self.f_new, self.g_new)

    @ti.kernel
    def stream_and_bc_fine(self):
        for i, j, k in ti.ndrange(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2):
            if self.refine_type[i, j, k] > 0:
                self.do_stream_and_bc(i, j, k, self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2, self.mask_f, self.sdf_f, self.f_post_f, self.g_post_f, self.f_new_f, self.g_new_f)

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

    @ti.func
    def do_update_macro(self, i, j, k, mask_field: ti.template(), rho_field: ti.template(), v_field: ti.template(),
                        temp_field: ti.template(), f_new_field: ti.template(), g_new_field: ti.template(),
                        f_old_field: ti.template(), g_old_field: ti.template(), bc_type_val):
        if mask_field[i, j, k] == 0:
            new_rho, new_temp = 0.0, 0.0
            new_v = ti.Vector([0.0, 0.0, 0.0])
            for d in range(19):
                f_val = f_new_field[i, j, k, d]
                g_val = g_new_field[i, j, k, d]
                new_rho += f_val
                new_v += f_val * self.d3q19.e[d]
                new_temp += g_val
                f_old_field[i, j, k, d] = f_val
                g_old_field[i, j, k, d] = g_val
            rho_field[i, j, k] = new_rho
            v_field[i, j, k] = new_v / new_rho
            temp_field[i, j, k] = new_temp
        else:
            v_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            new_temp = 0.0
            for d in range(19):
                g_val = g_new_field[i, j, k, d]
                new_temp += g_val
                g_old_field[i, j, k, d] = g_val
            if bc_type_val == 0:
                temp_field[i, j, k] = 1.0
            else:
                temp_field[i, j, k] = new_temp

    @ti.kernel
    def update_macro(self):
        for i, j, k in ti.ndrange(self.cfg.nx, self.cfg.ny, self.cfg.nz):
            self.do_update_macro(i, j, k, self.mask, self.rho, self.v, self.temp, self.f_new, self.g_new, self.f_old, self.g_old, self.cfg.bc_type_val)

    @ti.kernel
    def update_macro_fine(self):
        for i, j, k in ti.ndrange(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2):
            if self.refine_type[i, j, k] > 0:
                self.do_update_macro(i, j, k, self.mask_f, self.rho_f, self.v_f, self.temp_f, self.f_new_f, self.g_new_f, self.f_old_f, self.g_old_f, self.cfg.bc_type_val)

    # --- AMR 補間と同期 ---
    @ti.kernel
    def interpolate_c2f(self):
        for i, j, k in ti.ndrange(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2):
            if self.refine_type[i, j, k] == 2: # Buffer zone
                x_c = float(i) / 2.0
                y_c = float(j) / 2.0
                z_c = float(k) / 2.0
                
                rho_c = interp3d_scalar(self.rho, x_c, y_c, z_c)
                v_c = interp3d_vec(self.v, x_c, y_c, z_c)
                temp_c = interp3d_scalar(self.temp, x_c, y_c, z_c)
                
                self.rho_f[i, j, k] = rho_c
                self.v_f[i, j, k] = v_c
                self.temp_f[i, j, k] = temp_c
                
                for d in ti.static(range(19)):
                    f_old_c_val = interp3d_d(self.f_old, x_c, y_c, z_c, d)
                    feq_c = self.get_feq(rho_c, v_c, d)
                    f_neq_c = f_old_c_val - feq_c
                    self.f_old_f[i, j, k, d] = feq_c + 2.0 * f_neq_c
                    
                    g_old_c_val = interp3d_d(self.g_old, x_c, y_c, z_c, d)
                    geq_c = self.get_geq(temp_c, v_c, d)
                    g_neq_c = g_old_c_val - geq_c
                    self.g_old_f[i, j, k, d] = geq_c + 2.0 * g_neq_c

    @ti.kernel
    def restrict_f2c(self):
        for i, j, k in ti.ndrange(self.cfg.nx, self.cfg.ny, self.cfg.nz):
            if self.refine_type[i*2, j*2, k*2] == 1:
                rho_avg = 0.0
                v_avg = ti.Vector([0.0, 0.0, 0.0])
                temp_avg = 0.0
                for dx in ti.static(range(2)):
                    for dy in ti.static(range(2)):
                        for dz in ti.static(range(2)):
                            fi, fj, fk = i*2 + dx, j*2 + dy, k*2 + dz
                            rho_avg += self.rho_f[fi, fj, fk]
                            v_avg += self.v_f[fi, fj, fk]
                            temp_avg += self.temp_f[fi, fj, fk]
                rho_avg *= 0.125
                v_avg *= 0.125
                temp_avg *= 0.125
                
                self.rho[i, j, k] = rho_avg
                self.v[i, j, k] = v_avg
                self.temp[i, j, k] = temp_avg
                
                for d in ti.static(range(19)):
                    f_avg = 0.0
                    g_avg = 0.0
                    for dx in ti.static(range(2)):
                        for dy in ti.static(range(2)):
                            for dz in ti.static(range(2)):
                                fi, fj, fk = i*2 + dx, j*2 + dy, k*2 + dz
                                f_avg += self.f_old_f[fi, fj, fk, d]
                                g_avg += self.g_old_f[fi, fj, fk, d]
                    f_avg *= 0.125
                    g_avg *= 0.125
                    
                    feq_f = self.get_feq(rho_avg, v_avg, d)
                    f_neq_f = f_avg - feq_f
                    self.f_old[i, j, k, d] = feq_f + 0.5 * f_neq_f
                    
                    geq_f = self.get_geq(temp_avg, v_avg, d)
                    g_neq_f = g_avg - geq_f
                    self.g_old[i, j, k, d] = geq_f + 0.5 * g_neq_f

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
    def apply_heat_flux_bc_fine(self):
        sum_t, count = 0.0, 0
        for i, j, k in ti.ndrange(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2):
            if self.refine_type[i, j, k] == 1 and self.mask_f[i, j, k] == 1:
                for d in range(1, 7):
                    ip = (i + self.d3q19.e[d][0]) % (self.cfg.nx*2)
                    jp = (j + self.d3q19.e[d][1]) % (self.cfg.ny*2)
                    kp = k + self.d3q19.e[d][2]
                    if 0 <= kp < self.cfg.nz*2 and self.mask_f[ip, jp, kp] == 0:
                        sum_t += self.temp_f[ip, jp, kp]
                        count += 1
        avg_tf = sum_t / count if count > 0 else 0.0
        new_t_solid = avg_tf + (1.0 / (self.cfg.nx * 2))
        for i, j, k in ti.ndrange(self.cfg.nx*2, self.cfg.ny*2, self.cfg.nz*2):
            if self.refine_type[i, j, k] == 1 and self.mask_f[i, j, k] == 1:
                self.temp_f[i, j, k] = new_t_solid

    @ti.kernel
    def get_avg_grad(self) -> ti.f32:
        sum_q, area = 0.0, 0.0
        for i, j, k in self.rho:
            if self.mask[i, j, k] == 1:
                for d in range(1, 7):
                    ip, jp, kp = (i + self.d3q19.e[d][0]) % self.cfg.nx, (j + self.d3q19.e[d][1]) % self.cfg.ny, k + self.d3q19.e[d][2]
                    if 0 <= kp < self.cfg.nz and self.mask[ip, jp, kp] == 0:
                        sum_q += (self.temp[i, j, k] - self.temp[ip, jp, kp])
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
def build_pin_fin_sdf(mask: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int,
                      mask_f: ti.template(), sdf_f: ti.template(), refine_type: ti.template()):
    # 1. Coarse definition
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

    # 2. Fine definition
    for i, j, k in mask_f:
        sdf_f[i, j, k] = 100.0
        mask_f[i, j, k] = 0
        refine_type[i, j, k] = 0
        
        z_phys = k * 0.5
        if 2.0 <= z_phys < 18.0:
            x_phys = float((i * 0.5) % 16.0) - 7.5
            y_phys = float((j * 0.5) % 16.0) - 7.5
            radius = 4.0
            dist = ti.math.sqrt(x_phys*x_phys + y_phys*y_phys) - radius
            sdf_f[i, j, k] = dist * 2.0 # Scale to fine grid size
            if dist <= 0.0:
                mask_f[i, j, k] = 1

    # 3. Computing Domain (壁面から距離2.5以内の領域)
    for i, j, k in mask_f:
        z_phys = k * 0.5
        if 1.0 <= z_phys < 19.0:
            x_phys = float((i * 0.5) % 16.0) - 7.5
            y_phys = float((j * 0.5) % 16.0) - 7.5
            radius = 4.0
            dist = ti.math.sqrt(x_phys*x_phys + y_phys*y_phys) - radius
            if dist < 2.5:
                refine_type[i, j, k] = 1

    # 4. Buffer Domain (Computing Domainの外側1層の補間層)
    for i, j, k in mask_f:
        if refine_type[i, j, k] == 0:
            is_neighbor_of_1 = False
            for dx, dy, dz in ti.static([(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]):
                ni, nj, nk = i + dx, j + dy, k + dz
                if 0 <= ni < nx*2 and 0 <= nj < ny*2 and 0 <= nk < nz*2:
                    if refine_type[ni, nj, nk] == 1:
                        is_neighbor_of_1 = True
            if is_neighbor_of_1:
                refine_type[i, j, k] = 2

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
    cfg = SimConfig(**kwargs)
    sim = LBMSimulator(cfg)

    build_pin_fin_sdf(sim.mask, sim.sdf, cfg.nx, cfg.ny, cfg.nz, sim.mask_f, sim.sdf_f, sim.refine_type)
    sim.init_fields()

    frames =[]
    print(f"Starting advanced simulation with AMR: {cfg.nx}x{cfg.ny}x{cfg.nz}, steps={cfg.steps}")
    print(f"[Properties] nu: {cfg.nu_p:.2e}, k: {cfg.k_p:.3f}, Pr: {cfg.Pr:.2f}, BC Type: {'Constant Temp' if cfg.bc_type=='T' else 'Constant Heat Flux'}")

    for step in range(cfg.steps):
        # 1. Coarse 衝突
        sim.collide()

        # 2. Fine Step 1
        sim.interpolate_c2f()
        sim.collide_fine()
        sim.stream_and_bc_fine()
        sim.update_macro_fine()

        # 3. Fine Step 2
        sim.interpolate_c2f()
        sim.collide_fine()
        sim.stream_and_bc_fine()
        sim.update_macro_fine()

        # 4. Coarse Stream & Macro
        sim.stream_and_bc()
        sim.apply_bc()
        sim.update_macro()

        # 5. Synchronize Fine to Coarse
        sim.restrict_f2c()

        if cfg.bc_type_val == 1:
            sim.apply_heat_flux_bc()
            sim.apply_heat_flux_bc_fine()

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

            if step % 20 == 0:
                print(f"Step {step:4d} | T_wall: {T_w_p:.1f} K | h: {h_coeff:.2f} W/m2K | Eff(ε): {effectiveness:.4f} | T_out: {T_out_p:.2f} K")

            update_vis_image(sim)
            f_np = sim.img.to_numpy().T
            frame_rgb = np.zeros((cfg.ny, cfg.nx*2, 3), dtype=np.uint8)
            t_mask = (f_np <= 1.0)
            p_mask = (f_np > 1.5)

            frame_rgb[:,:,0][t_mask] = (np.clip(f_np[t_mask], 0, 1) * 255).astype(np.uint8)
            frame_rgb[:,:,2][t_mask] = ((1 - np.clip(f_np[t_mask], 0, 1)) * 255).astype(np.uint8)
            frame_rgb[p_mask] =[255, 255, 255]

            frames.append(np.flipud(frame_rgb))

    imageio.mimsave(cfg.filename, frames, fps=12)
    print(f"Finished. Result saved as {cfg.filename}")
    return cfg.filename

# ==========================================================
# 実行例
# ==========================================================
gif_file_advanced = run_simulation()

Image(open(gif_file_advanced, 'rb').read())