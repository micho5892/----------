import taichi as ti
import numpy as np
import math
from lbm_logger import get_logger

_log = get_logger(__name__)

# --- 形状生成ヘルパー関数 ---
def create_sphere_markers(radius_lbm, center_lbm, num_samples):
    points = []
    normals = []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2 
        radius_at_y = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        points.append([center_lbm[0] + radius_lbm * x, center_lbm[1] + radius_lbm * y, center_lbm[2] + radius_lbm * z])
        n_norm = math.sqrt(x * x + y * y + z * z) + 1e-12
        normals.append([x / n_norm, y / n_norm, z / n_norm])
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)

def create_cylinder_markers(radius_lbm, center_lbm, ny):
    """Y方向に貫通する円柱のマーカーを生成 (2D的なシミュレーション用)"""
    # 円周上の1セルにつき約1.2個のマーカーを配置
    num_circumference = int(2.0 * np.pi * radius_lbm * 2.0)
    theta = np.linspace(0, 2*np.pi, num_circumference, endpoint=False)
    
    points = []
    normals = []
    for j in range(ny):
        for t in theta:
            x = center_lbm[0] + radius_lbm * np.cos(t)
            z = center_lbm[2] + radius_lbm * np.sin(t)
            points.append([x, float(j) + 0.5, z])
            normals.append([np.cos(t), 0.0, np.sin(t)])
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)


def create_y_plate_markers(nz, wall_thickness_lbm, side, i_min, i_max_excl, ny):
    """
    法線が Y 方向の平板（チャネル上下壁）用マーカー。
    指定された壁の厚み (wall_thickness_lbm) の内部空間「すべて」にマーカーを敷き詰める（Volume IBM）。
    """
    points = []
    normals = []
    
    # 敷き詰めるマーカーの間隔 (1.0セル間隔だとLBM格子と完全に一致してしまうため、
    # わずかにずらすか、0.8セル間隔などで密に配置するのが一般的です)
    spacing = 0.8 
    
    # 壁のY方向の範囲を決定
    if side == "lower":
        y_start = 0.0
        y_end = float(wall_thickness_lbm)
    else: # "upper"
        y_start = float(ny - wall_thickness_lbm)
        y_end = float(ny)

    # Y方向（壁の厚み分）のマーカー座標リスト
    y_coords = np.arange(y_start + spacing/2.0, y_end, spacing)
    
    # X方向のマーカー座標リスト
    x_coords = np.arange(float(i_min) + spacing/2.0, float(i_max_excl), spacing)
    
    # Z方向のマーカー座標リスト
    z_coords = np.arange(0.0 + spacing/2.0, float(nz), spacing)

    for y in y_coords:
        for x in x_coords:
            for z in z_coords:
                points.append([x, y, z])
                if side == "lower":
                    normals.append([0.0, 1.0, 0.0])
                else:
                    normals.append([0.0, -1.0, 0.0])
                
    return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)

# --- IBM マネージャ ---
@ti.data_oriented
class IBManager:
    def __init__(
        self,
        fp_dtype,
        objects_data,
        total_points,
        dA_lbm,
        phase1_epsilon_lbm=1.5,
        phase1_num_iterations=3,
        phase2_num_iterations=3,
        phase2_heat_relax=1.0,
        phase2_enable_iterative_thermal=True,
        phase3_probe_distance_lbm=1.5,
        phase3_enable_fsi_rotation=True,
        phase3_gravity_lbm=(0.0, 0.0, -9.8e-4),
    ):
        """
        objects_data: 辞書のリスト。各物体のメタデータを含む
        """
        self.fp_dtype = fp_dtype
        self.num_points = total_points
        self.num_objects = len(objects_data)
        self.dA = dA_lbm

        # 全マーカーのデータ (平坦化された配列)
        self.pos = ti.Vector.field(3, dtype=fp_dtype, shape=total_points)
        self.vel = ti.Vector.field(3, dtype=fp_dtype, shape=total_points)
        self.force = ti.Vector.field(3, dtype=fp_dtype, shape=total_points)
        self.marker_obj_id = ti.field(dtype=ti.i32, shape=total_points) # 各マーカーが属する物体ID
        self.marker_normal = ti.Vector.field(3, dtype=fp_dtype, shape=total_points)

        # 物体ごとのメタデータ
        self.obj_type = ti.field(dtype=ti.i32, shape=self.num_objects) # 0: fixed, 1: free, 2: rotating
        self.obj_mass = ti.field(dtype=fp_dtype, shape=self.num_objects)
        self.obj_center = ti.Vector.field(3, dtype=fp_dtype, shape=self.num_objects)
        self.obj_vel = ti.Vector.field(3, dtype=fp_dtype, shape=self.num_objects)
        self.obj_omega = ti.Vector.field(3, dtype=fp_dtype, shape=self.num_objects) # 角速度ベクトル
        self.obj_shape = ti.field(dtype=ti.i32, shape=self.num_objects)  # 0:sphere, 1:cylinder, 2:y_plate
        self.obj_radius_lbm = ti.field(dtype=fp_dtype, shape=self.num_objects)
        self.obj_inertia = ti.field(dtype=fp_dtype, shape=self.num_objects)
        self.obj_torque = ti.Vector.field(3, dtype=fp_dtype, shape=self.num_objects)
        
        # 力計測用 (毎ステップの抗力・揚力を記録)
        self.obj_hydro_force = ti.Vector.field(3, dtype=fp_dtype, shape=self.num_objects)
        self.phase1_epsilon_lbm = float(phase1_epsilon_lbm)
        self.phase1_num_iterations = max(1, int(phase1_num_iterations))
        self.phase2_num_iterations = max(1, int(phase2_num_iterations))
        self.phase2_heat_relax = float(phase2_heat_relax)
        self.phase2_enable_iterative_thermal = 1 if phase2_enable_iterative_thermal else 0
        self.phase3_probe_distance_lbm = float(phase3_probe_distance_lbm)
        self.phase3_enable_fsi_rotation = 1 if phase3_enable_fsi_rotation else 0
        self.phase3_gravity_lbm = ti.Vector(
            [float(phase3_gravity_lbm[0]), float(phase3_gravity_lbm[1]), float(phase3_gravity_lbm[2])]
        )

        #  温度場 (Thermal IBM) 用のフィールド
        self.marker_temp = ti.field(dtype=fp_dtype, shape=total_points)     # 各マーカーの目標温度
        self.marker_q = ti.field(dtype=fp_dtype, shape=total_points)        # 各マーカーが流体に与える熱量(Q)
        self.obj_temp = ti.field(dtype=fp_dtype, shape=self.num_objects)    # 各物体の設定温度
        self.obj_is_thermal = ti.field(dtype=ti.i32, shape=self.num_objects)# 1なら熱源として機能

        # データの初期化 (Python側からTaichiフィールドへ転送)
        self._init_data(objects_data)
        _log.debug("IBManager initialized: %d objects, %d total markers", self.num_objects, self.num_points)

    def _init_data(self, objects_data):
        pos_np = np.zeros((self.num_points, 3), dtype=np.float32)
        vel_np = np.zeros((self.num_points, 3), dtype=np.float32)
        marker_id_np = np.zeros(self.num_points, dtype=np.int32)
        normal_np = np.zeros((self.num_points, 3), dtype=np.float32)
        
        type_np = np.zeros(self.num_objects, dtype=np.int32)
        mass_np = np.zeros(self.num_objects, dtype=np.float32)
        center_np = np.zeros((self.num_objects, 3), dtype=np.float32)
        vel_obj_np = np.zeros((self.num_objects, 3), dtype=np.float32)
        omega_np = np.zeros((self.num_objects, 3), dtype=np.float32)
        shape_np = np.zeros(self.num_objects, dtype=np.int32)
        radius_np = np.zeros(self.num_objects, dtype=np.float32)
        inertia_np = np.zeros(self.num_objects, dtype=np.float32)

        temp_obj_np = np.zeros(self.num_objects, dtype=np.float32)
        is_thermal_np = np.zeros(self.num_objects, dtype=np.int32)
        marker_temp_np = np.zeros(self.num_points, dtype=np.float32)
        
        current_idx = 0
        for obj_id, obj in enumerate(objects_data):
            markers = obj['markers']
            marker_normals = obj.get("marker_normals", None)
            n_markers = len(markers)

            # ▼ 追加: 温度設定の読み込み
            if "temperature" in obj:
                temp_obj_np[obj_id] = float(obj["temperature"])
                is_thermal_np[obj_id] = 1
                marker_temp_np[current_idx : current_idx + n_markers] = float(obj["temperature"])
            else:
                is_thermal_np[obj_id] = 0 # 熱源ではない(断熱)
            
            pos_np[current_idx : current_idx + n_markers] = markers
            if marker_normals is not None and len(marker_normals) == n_markers:
                normal_np[current_idx : current_idx + n_markers] = marker_normals
            else:
                normal_np[current_idx : current_idx + n_markers] = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            vel_np[current_idx : current_idx + n_markers] = obj.get('v0', [0,0,0])
            marker_id_np[current_idx : current_idx + n_markers] = obj_id
            
            # タイプ判定
            t_str = obj.get('type', 'fixed')
            if t_str == 'fixed': type_np[obj_id] = 0
            elif t_str == 'free': type_np[obj_id] = 1
            elif t_str == 'rotating': type_np[obj_id] = 2

            shape_str = str(obj.get("shape", "sphere")).lower()
            if shape_str == "cylinder":
                shape_np[obj_id] = 1
            elif shape_str == "y_plate":
                shape_np[obj_id] = 2
            else:
                shape_np[obj_id] = 0
            radius_np[obj_id] = float(obj.get("radius_lbm", 0.0))
            mass_val = float(obj.get('mass', 1e9))
            radius_val = float(obj.get("radius_lbm", 0.0))
            if shape_np[obj_id] == 1:
                inertia_np[obj_id] = max(1e-12, 0.5 * mass_val * radius_val * radius_val)
            elif shape_np[obj_id] == 0:
                inertia_np[obj_id] = max(1e-12, 0.4 * mass_val * radius_val * radius_val)
            else:
                inertia_np[obj_id] = max(1e-12, mass_val)
            
            mass_np[obj_id] = obj.get('mass', 1e9)
            center_np[obj_id] = obj.get('center', [0,0,0])
            vel_obj_np[obj_id] = obj.get('v0', [0,0,0])
            omega_np[obj_id] = obj.get('omega', [0,0,0])
            
            current_idx += n_markers
            
        self.pos.from_numpy(pos_np)
        self.vel.from_numpy(vel_np)
        self.marker_obj_id.from_numpy(marker_id_np)
        self.marker_normal.from_numpy(normal_np)
        self.obj_type.from_numpy(type_np)
        self.obj_mass.from_numpy(mass_np)
        self.obj_center.from_numpy(center_np)
        self.obj_vel.from_numpy(vel_obj_np)
        self.obj_omega.from_numpy(omega_np)
        self.obj_shape.from_numpy(shape_np)
        self.obj_radius_lbm.from_numpy(radius_np)
        self.obj_inertia.from_numpy(inertia_np)
        self.obj_temp.from_numpy(temp_obj_np)
        self.obj_is_thermal.from_numpy(is_thermal_np)
        self.marker_temp.from_numpy(marker_temp_np)

    @ti.func
    def peskin_delta(self, r):
        r_abs = ti.abs(r)
        val = 0.0
        if r_abs < 2.0:
            val = 0.25 * (1.0 + ti.math.cos(ti.math.pi * r_abs / 2.0))
        return val

    @ti.kernel
    def interp_velocity_and_calc_force(self, ctx: ti.template()):
        for p in range(self.num_points):
            p_pos = self.pos[p]
            u_interp = ti.Vector([0.0, 0.0, 0.0])
            t_interp = 0.0

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
                            t_interp += ctx.temp[i, j, k] * weight

            self.force[p] = (self.vel[p] - u_interp) * 2.0

            obj_id = self.marker_obj_id[p]
            if self.obj_is_thermal[obj_id] != 0:
                # Direct forcing: マーカー目標温度と補間流体温度の差を埋める熱源強度（spread で dA 重み付け）
                self.marker_q[p] = (self.marker_temp[p] - t_interp) * 2.0
            else:
                self.marker_q[p] = 0.0

    @ti.kernel
    def interp_temperature_and_calc_heat(self, ctx: ti.template(), clear_before_accumulate: ti.i32):
        for p in range(self.num_points):
            obj_id = self.marker_obj_id[p]
            if self.obj_is_thermal[obj_id] != 0:
                p_pos = self.pos[p]
                t_interp = 0.0

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
                                t_interp += ctx.temp[i, j, k] * weight

                q_delta = (self.marker_temp[p] - t_interp)
                if clear_before_accumulate != 0:
                    self.marker_q[p] = q_delta
                else:
                    self.marker_q[p] += q_delta
            else:
                self.marker_q[p] = 0.0

    @ti.kernel
    def spread_force(self, ctx: ti.template()):
        for p in range(self.num_points):
            p_pos = self.pos[p]
            f_ib = self.force[p] * self.dA
            q_ib = self.marker_q[p] * self.dA

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
                            cid = ctx.cell_id[i, j, k]
                            if ctx.is_fluid_table[cid] == 1:
                                ti.atomic_add(ctx.S_g[i, j, k], q_ib * weight)

    @ti.kernel
    def update_objects_and_markers(self):
        # 1. 各物体が受ける流体力・流体トルクを集計
        for obj_id in range(self.num_objects):
            self.obj_hydro_force[obj_id] = ti.Vector([0.0, 0.0, 0.0])
            self.obj_torque[obj_id] = ti.Vector([0.0, 0.0, 0.0])
            
        for p in range(self.num_points):
            obj_id = self.marker_obj_id[p]
            f_fluid = -self.force[p] * self.dA
            self.obj_hydro_force[obj_id] += f_fluid
            r = self.pos[p] - self.obj_center[obj_id]
            self.obj_torque[obj_id] += ti.Vector([
                r[1] * f_fluid[2] - r[2] * f_fluid[1],
                r[2] * f_fluid[0] - r[0] * f_fluid[2],
                r[0] * f_fluid[1] - r[1] * f_fluid[0],
            ])
            
        # 2. 自由運動(Free)オブジェクトの重心運動・回転運動を更新
        for obj_id in range(self.num_objects):
            if self.obj_type[obj_id] == 1: # free
                ext_force = self.phase3_gravity_lbm * self.obj_mass[obj_id]
                total_force = self.obj_hydro_force[obj_id] + ext_force
                acc = total_force / self.obj_mass[obj_id]
                self.obj_vel[obj_id] += acc
                self.obj_center[obj_id] += self.obj_vel[obj_id]
                if self.phase3_enable_fsi_rotation != 0:
                    self.obj_omega[obj_id] += self.obj_torque[obj_id] / self.obj_inertia[obj_id]

        # 3. 各マーカーの座標と速度の更新
        for p in range(self.num_points):
            obj_id = self.marker_obj_id[p]
            o_type = self.obj_type[obj_id]
            
            if o_type == 0:
                # 固定 (Fixed): 動かさない、速度ゼロ
                self.vel[p] = ti.Vector([0.0, 0.0, 0.0])
                
            elif o_type == 1:
                v_c = self.obj_vel[obj_id]
                omega = self.obj_omega[obj_id]
                center = self.obj_center[obj_id]
                r = self.pos[p] - center
                v_rot = ti.Vector([
                    omega[1]*r[2] - omega[2]*r[1],
                    omega[2]*r[0] - omega[0]*r[2],
                    omega[0]*r[1] - omega[1]*r[0]
                ])
                self.vel[p] = v_c + v_rot
                self.pos[p] += self.vel[p]
                n = self.marker_normal[p]
                dn = ti.Vector([
                    omega[1]*n[2] - omega[2]*n[1],
                    omega[2]*n[0] - omega[0]*n[2],
                    omega[0]*n[1] - omega[1]*n[0]
                ])
                new_n = n + dn
                nlen = ti.math.length(new_n)
                if nlen > 1e-12:
                    self.marker_normal[p] = new_n / nlen
                
            elif o_type == 2:
                # 回転 (Rotating): 指定された omega で強制回転
                omega = self.obj_omega[obj_id]
                center = self.obj_center[obj_id]
                r = self.pos[p] - center
                
                # v = omega x r (クロス積)
                v_rot = ti.Vector([
                    omega[1]*r[2] - omega[2]*r[1],
                    omega[2]*r[0] - omega[0]*r[2],
                    omega[0]*r[1] - omega[1]*r[0]
                ])
                self.pos[p] += v_rot
                self.vel[p] = v_rot

    @ti.kernel
    def compute_probe_heat_flux(self, ctx: ti.template(), k_fluid: ti.f32) -> ti.types.vector(2, ti.f32):
        sum_q = 0.0
        sum_area = 0.0
        probe_dist = self.phase3_probe_distance_lbm
        for p in range(self.num_points):
            obj_id = self.marker_obj_id[p]
            if self.obj_is_thermal[obj_id] != 0:
                pos = self.pos[p]
                normal = self.marker_normal[p]
                probe_pos = pos + normal * probe_dist
                t_probe = 0.0
                base_i = int(ti.math.floor(probe_pos[0]))
                base_j = int(ti.math.floor(probe_pos[1]))
                base_k = int(ti.math.floor(probe_pos[2]))
                for i in range(base_i - 1, base_i + 3):
                    for j in range(base_j - 1, base_j + 3):
                        for k in range(base_k - 1, base_k + 3):
                            if 0 <= i < ctx.nx and 0 <= j < ctx.ny and 0 <= k < ctx.nz:
                                dx = probe_pos[0] - float(i)
                                dy = probe_pos[1] - float(j)
                                dz = probe_pos[2] - float(k)
                                weight = self.peskin_delta(dx) * self.peskin_delta(dy) * self.peskin_delta(dz)
                                t_probe += ctx.temp[i, j, k] * weight
                grad_t = (t_probe - self.marker_temp[p]) / (probe_dist + 1e-12)
                q_local = -k_fluid * grad_t
                sum_q += q_local * self.dA
                sum_area += self.dA
        return ti.Vector([sum_q, sum_area])

    @ti.kernel
    def clear_forces_and_sources(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            ctx.F_int[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            ctx.S_g[i, j, k] = 0.0

    @ti.kernel
    def update_sdf_and_phi(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            ctx.sdf[i, j, k] = 1.0e6
            ctx.phi[i, j, k] = 0.0

        if self.num_objects > 0:
            center = self.obj_center[0]
            radius_lbm = self.obj_radius_lbm[0]
            shape = self.obj_shape[0]
            eps = self.phase1_epsilon_lbm

            if shape == 1 and radius_lbm > 0.0:
                for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
                    dx = float(i) - center[0]
                    dz = float(k) - center[2]
                    dist = ti.math.sqrt(dx * dx + dz * dz)
                    sdf_val = dist - radius_lbm
                    ctx.sdf[i, j, k] = sdf_val

                    if sdf_val < -eps:
                        ctx.phi[i, j, k] = 1.0
                    elif sdf_val > eps:
                        ctx.phi[i, j, k] = 0.0
                    else:
                        ctx.phi[i, j, k] = 0.5 * (1.0 - ti.math.sin(ti.math.pi * sdf_val / (2.0 * eps)))

    @ti.kernel
    def apply_delta_force_to_velocity(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            cid = ctx.cell_id[i, j, k]
            if ctx.is_fluid_table[cid] == 1 and ctx.rho[i, j, k] > 1.0e-12:
                ctx.v[i, j, k] += ctx.F_int[i, j, k] / ctx.rho[i, j, k]

    @ti.kernel
    def apply_delta_heat_to_temperature(self, ctx: ti.template(), heat_relax: ti.f32):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            cid = ctx.cell_id[i, j, k]
            if ctx.is_fluid_table[cid] == 1 and ctx.rho[i, j, k] > 1.0e-12:
                ctx.temp[i, j, k] += heat_relax * ctx.S_g[i, j, k] / ctx.rho[i, j, k]

    def step(self, ctx):
        self.update_sdf_and_phi(ctx)
        self.clear_forces_and_sources(ctx)
        num_iterations = self.phase1_num_iterations
        if self.phase2_enable_iterative_thermal != 0:
            num_iterations = max(self.phase1_num_iterations, self.phase2_num_iterations)

        for it in range(num_iterations):
            self.interp_velocity_and_calc_force(ctx)
            if self.phase2_enable_iterative_thermal != 0:
                clear_q = 1 if it == 0 else 0
                self.interp_temperature_and_calc_heat(ctx, clear_q)
            self.spread_force(ctx)
            if it < num_iterations - 1:
                self.apply_delta_force_to_velocity(ctx)
                if self.phase2_enable_iterative_thermal != 0:
                    self.apply_delta_heat_to_temperature(ctx, self.phase2_heat_relax)
        self.update_objects_and_markers()
        
    def get_hydro_forces(self):
        """抗力・揚力計測用に、Python側に力をnumpy配列で返す"""
        return self.obj_hydro_force.to_numpy()