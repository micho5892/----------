import taichi as ti
import numpy as np
import math
from lbm_logger import get_logger

_log = get_logger(__name__)


@ti.data_oriented
class IBManager:
    def __init__(self, fp_dtype, num_points, mass_lbm, gravity_lbm, dA_lbm):
        self.fp_dtype = fp_dtype
        self.num_points = num_points
        self.mass = mass_lbm
        self.gravity = ti.Vector(gravity_lbm)
        self.dA = dA_lbm # マーカー1個あたりの表面積 (LBM単位)

        # ラグランジュ点群
        self.pos = ti.Vector.field(3, dtype=fp_dtype, shape=num_points)
        self.vel = ti.Vector.field(3, dtype=fp_dtype, shape=num_points)
        self.force = ti.Vector.field(3, dtype=fp_dtype, shape=num_points)
        
        self.center = ti.Vector.field(3, dtype=fp_dtype, shape=())
        self.center_vel = ti.Vector.field(3, dtype=fp_dtype, shape=())
        _log.debug("IBManager: num_points=%s", num_points)

    @ti.func
    def peskin_delta(self, r):
        r_abs = ti.abs(r)
        val = 0.0
        if r_abs < 2.0:
            val = 0.25 * (1.0 + ti.math.cos(ti.math.pi * r_abs / 2.0))
        return val

    @ti.kernel
    def interp_velocity_and_calc_force(self, ctx: ti.template()):
        """1. 補間 と 2. Direct Forcing法による力の計算"""
        for p in range(self.num_points):
            p_pos = self.pos[p]
            u_interp = ti.Vector([0.0, 0.0, 0.0])
            
            base_i = int(ti.math.floor(p_pos[0]))
            base_j = int(ti.math.floor(p_pos[1]))
            base_k = int(ti.math.floor(p_pos[2]))
            
            for i in range(base_i - 1, base_i + 3):
                for j in range(base_j - 1, base_j + 3):
                    for k in range(base_k - 1, base_k + 3):
                        if 0 <= i < ctx.nx and 0 <= j < ctx.ny and 0 <= k < ctx.nz:
                            dist_x = p_pos[0] - float(i)
                            dist_y = p_pos[1] - float(j)
                            dist_z = p_pos[2] - float(k)
                            weight = self.peskin_delta(dist_x) * self.peskin_delta(dist_y) * self.peskin_delta(dist_z)
                            u_interp += ctx.v[i, j, k] * weight

            # ★ Direct Forcing: 速度差をそのまま必要な力密度とする (dt_lbm = 1.0)
            # ※係数2.0は、Guoフォーシングでの半ステップ遅れを補正するための安定化ゲイン
            self.force[p] = (self.vel[p] - u_interp) * 2.0

    @ti.kernel
    def spread_force(self, ctx: ti.template()):
        """3. 物体が流体を押す力を分配 (面積 dA を掛ける)"""
        for p in range(self.num_points):
            p_pos = self.pos[p]
            f_ib = self.force[p] * self.dA # 力密度 × 面積 = 実際の力
            
            base_i = int(ti.math.floor(p_pos[0]))
            base_j = int(ti.math.floor(p_pos[1]))
            base_k = int(ti.math.floor(p_pos[2]))
            
            for i in range(base_i - 1, base_i + 3):
                for j in range(base_j - 1, base_j + 3):
                    for k in range(base_k - 1, base_k + 3):
                        if 0 <= i < ctx.nx and 0 <= j < ctx.ny and 0 <= k < ctx.nz:
                            dist_x = p_pos[0] - float(i)
                            dist_y = p_pos[1] - float(j)
                            dist_z = p_pos[2] - float(k)
                            weight = self.peskin_delta(dist_x) * self.peskin_delta(dist_y) * self.peskin_delta(dist_z)
                            
                            ctx.F_int[i, j, k] += f_ib * weight

    @ti.kernel
    def update_rigid_body(self):
        """剛体の運動方程式 (dt_lbm = 1.0)"""
        total_hydro_force = ti.Vector([0.0, 0.0, 0.0])
        for p in range(self.num_points):
            total_hydro_force -= self.force[p] * self.dA # 作用・反作用
            
        acc = total_hydro_force / self.mass + self.gravity
        self.center_vel[None] += acc
        self.center[None] += self.center_vel[None]
        
        for p in range(self.num_points):
            self.pos[p] += self.center_vel[None]
            self.vel[p] = self.center_vel[None]

    def step(self, ctx):
        self.interp_velocity_and_calc_force(ctx)
        self.spread_force(ctx)
        self.update_rigid_body()

def create_sphere_markers(radius_lbm, center_lbm, num_samples):
    points =[]
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2 
        radius_at_y = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        points.append([center_lbm[0] + radius_lbm * x, center_lbm[1] + radius_lbm * y, center_lbm[2] + radius_lbm * z])
    return np.array(points, dtype=np.float32)