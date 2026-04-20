# ==========================================================
# physics.py — 物理モデルプラグイン (多相流、浮力、相変化など)
# ==========================================================
import taichi as ti
import math
import os
import csv
import config
from context import FLUID_A, INLET, OUTLET, SOLID, SOLID_HEAT_SOURCE, ROTATING_WALL
from immersed_boundary import IBManager, create_sphere_markers
from lbm_logger import get_logger

_log = get_logger(__name__)

@ti.data_oriented
class PhysicsModel:
    def apply(self, ctx, current_time):
        pass

    @ti.func
    def is_fluid(self, ctx: ti.template(), cid):
        return ctx.is_fluid_table[cid] == 1

@ti.data_oriented
class ShanChenMultiphase(PhysicsModel):
    """Shan-Chen 擬ポテンシャル多相流モデル (引力による気液分離)"""
    def __init__(self, d3q19, G_target=-5.0, phase_start_time=4.0, phase_ramp_time=2.0, sponge_thickness=40.0):
        self.d3q19 = d3q19
        self.G_target = G_target
        self.phase_start_time = phase_start_time
        self.phase_ramp_time = phase_ramp_time
        self.sponge_thickness = sponge_thickness


    @ti.func
    def calc_psi(self, rho):
        rho_0 = 1.0
        return rho_0 * (1.0 - ti.math.exp(-rho / rho_0))

    @ti.kernel
    def _calc_interaction_force_kernel(self, ctx: ti.template(), G_current: ti.f32):  # G_current はスカラーなので f32 のまま
        # 1. psi の計算
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if self.is_fluid(ctx, ctx.cell_id[i, j, k]):
                ctx.psi[i, j, k] = self.calc_psi(ctx.rho[i, j, k])
            else:
                ctx.psi[i, j, k] = 0.0

        # 2. 引力の計算
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if self.is_fluid(ctx, ctx.cell_id[i, j, k]):
                G_local = G_current
                if float(k) < self.sponge_thickness:
                    factor = float(k) / self.sponge_thickness
                    fade = 0.5 * (1.0 - ti.math.cos(ti.math.pi * factor))
                    G_local = G_current * fade
                
                force = ti.Vector([0.0, 0.0, 0.0])
                psi_center = ctx.psi[i, j, k]
                
                for d in ti.static(range(1, 19)):
                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]
                    
                    if 0 <= ip < ctx.nx and 0 <= jp < ctx.ny and 0 <= kp < ctx.nz:
                        psi_neighbor = ctx.psi[ip, jp, kp]
                        w = self.d3q19.w[d]
                        e_vec = ti.Vector([float(self.d3q19.e[d][0]), float(self.d3q19.e[d][1]), float(self.d3q19.e[d][2])])
                        force += w * psi_neighbor * e_vec
                
                F = -G_local * psi_center * force
                # ★他の物理モデル(浮力など)と共存できるように += にする
                ctx.F_int[i, j, k] += F
                ctx.v[i, j, k] += F / (2.0 * ctx.rho[i, j, k])

    def apply(self, ctx, current_time):
        if current_time < self.phase_start_time:
            G_current = 0.0
        elif current_time < self.phase_start_time + self.phase_ramp_time:
            progress = (current_time - self.phase_start_time) / self.phase_ramp_time
            G_current = self.G_target * 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            G_current = self.G_target
        self._calc_interaction_force_kernel(ctx, G_current)


@ti.data_oriented
class BoussinesqBuoyancy(PhysicsModel):
    """自然対流用 ブシネスク近似（温度差による浮力）テスト実装"""
    def __init__(self, cfg, g_vec, beta, T_ref): 
        # 物理単位の重力加速度 g_vec [m/s^2] を LBM単位に自動変換
        g_lbm =[g * (cfg.dt ** 2) / cfg.dx for g in g_vec]
        
        self.g_vec = ti.Vector(g_lbm)
        self.beta = beta
        self.T_ref = T_ref

    @ti.kernel
    def _apply_kernel(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if self.is_fluid(ctx, ctx.cell_id[i, j, k]):
                temp_diff = ctx.temp[i, j, k] - self.T_ref
                F = -ctx.rho[i, j, k] * self.beta * temp_diff * self.g_vec
                ctx.F_int[i, j, k] += F
                ctx.v[i, j, k] += F / (2.0 * ctx.rho[i, j, k])

    def apply(self, ctx, current_time):
        self._apply_kernel(ctx)

@ti.data_oriented 
class PhysicsManager:
    """指定された複数の物理モデルを一括管理・適用するマネージャ"""
    def __init__(self, d3q19, cfg):
        self.models =[]
        for p_name, p_kwargs in cfg.physics_models.items():
            if p_name == "shan_chen":
                self.models.append(ShanChenMultiphase(d3q19, **p_kwargs))
            elif p_name == "boussinesq":
                self.models.append(BoussinesqBuoyancy(cfg, **p_kwargs))
            elif p_name == "immersed_boundary":
                self.models.append(ImmersedBoundaryModel(cfg.fp_dtype, cfg))
            # 将来的に "phase_change" (沸騰モデル) などをここに追加できます

    def apply_all(self, ctx, current_time):
        self._reset_forces_and_sources(ctx)
        for model in self.models:
            model.apply(ctx, current_time)

    @ti.kernel
    def _reset_forces_and_sources(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            ctx.F_int[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            ctx.S_g[i, j, k] = 0.0


@ti.data_oriented
class ImmersedBoundaryModel(PhysicsModel):
    def __init__(self, fp_dtype, cfg):
        from immersed_boundary import (
            IBManager,
            create_sphere_markers,
            create_cylinder_markers,
            create_y_plate_markers,
        )

        self.cfg = cfg
        ibm_cfg = cfg.physics_models.get("immersed_boundary", {})
        objects_cfg = ibm_cfg.get("objects", [])
        
        self.dx = cfg.dx
        self.dt = cfg.dt
        
        parsed_objects = []
        total_markers = 0
        self.dA_lbm = 1.0 
        
        for obj in objects_cfg:
            shape = obj.get("shape", "sphere")
            r_p = obj.get("radius_p", 0.01)
            center_p = obj.get("center_p", [0.05, 0.05, 0.08])
            
            r_lbm = r_p / self.dx
            center_lbm = [c / self.dx for c in center_p]
            
            if shape == "cylinder":
                # Y方向貫通円柱
                markers, marker_normals = create_cylinder_markers(r_lbm, center_lbm, cfg.ny)
                self.dA_lbm = (2.0 * math.pi * r_lbm * float(cfg.ny)) / len(markers)

            elif shape == "y_plate":
                wall_th = int(obj.get("wall_thickness_lbm", 10))
                side = str(obj.get("side", "lower")).lower()
                i_min = int(obj.get("i_min_lbm", wall_th))
                i_max_excl = int(obj.get("i_max_excl_lbm", cfg.nx - wall_th))
                
                # ▼ 変更: 平面のY座標ではなく、厚みと方向を渡してブロック状のマーカーを生成
                markers, marker_normals = create_y_plate_markers(cfg.nz, wall_th, side, i_min, i_max_excl, cfg.ny)
                
                if len(markers) == 0:
                    raise ValueError(
                        "immersed_boundary y_plate: マーカーが0個です（i_min_lbm / i_max_excl_lbm / nx を確認）"
                    )
                n_mark = len(markers)
                
                # ▼ 体積分のマーカーになったため、dA (面積) ではなく dV (体積) として扱う必要がありますが、
                # LBMの Direct Forcing においては、マーカー1個あたりの影響体積として計算します。
                vol_lbm = float(i_max_excl - i_min) * float(wall_th) * float(cfg.nz)
                self.dA_lbm = vol_lbm / float(n_mark) # 1マーカーあたりの体積(重み)
                
                # 中心座標の計算 (大体でOK)
                y_center = float(wall_th)/2.0 if side == "lower" else float(cfg.ny) - float(wall_th)/2.0
                center_lbm = [
                    0.5 * float(i_min + i_max_excl - 1),
                    y_center,
                    0.5 * float(cfg.nz - 1),
                ]
            else:
                surface_area_lbm = 4.0 * math.pi * (r_lbm ** 2)
                num_points = int(surface_area_lbm / (0.8 ** 2))
                markers, marker_normals = create_sphere_markers(r_lbm, center_lbm, num_points)
                self.dA_lbm = surface_area_lbm / num_points
            
            parsed_obj = {
                "markers": markers,
                "type": obj.get("type", "fixed"),
                "mass": 1e9, # 固定なら質量無限大でOK
                "center": center_lbm,
                "v0": obj.get("v0_lbm", [0,0,0]),
                "omega": obj.get("omega_lbm", [0,0,0]),
                "shape": shape,
                "radius_lbm": r_lbm,
                "marker_normals": marker_normals,
            }
            if "temperature" in obj:
                parsed_obj["temperature"] = float(obj["temperature"])
            parsed_objects.append(parsed_obj)
            total_markers += len(markers)
            
        phase1_epsilon_lbm = ibm_cfg.get("phase1_epsilon_lbm", 1.5)
        phase1_num_iterations = ibm_cfg.get("phase1_num_iterations", 3)
        phase2_num_iterations = ibm_cfg.get("phase2_num_iterations", 3)
        phase2_heat_relax = ibm_cfg.get("phase2_heat_relax", 1.0)
        phase2_enable_iterative_thermal = ibm_cfg.get("phase2_enable_iterative_thermal", True)
        phase3_probe_distance_lbm = ibm_cfg.get("phase3_probe_distance_lbm", 1.5)
        phase3_enable_fsi_rotation = ibm_cfg.get("phase3_enable_fsi_rotation", True)
        phase3_gravity_lbm = ibm_cfg.get("phase3_gravity_lbm", [0.0, 0.0, -9.8e-4])
        self.ibm = IBManager(
            fp_dtype,
            parsed_objects,
            total_markers,
            self.dA_lbm,
            phase1_epsilon_lbm=phase1_epsilon_lbm,
            phase1_num_iterations=phase1_num_iterations,
            phase2_num_iterations=phase2_num_iterations,
            phase2_heat_relax=phase2_heat_relax,
            phase2_enable_iterative_thermal=phase2_enable_iterative_thermal,
            phase3_probe_distance_lbm=phase3_probe_distance_lbm,
            phase3_enable_fsi_rotation=phase3_enable_fsi_rotation,
            phase3_gravity_lbm=phase3_gravity_lbm,
        )
        
        # --- 毎ステップの力を記録するCSVの準備 ---
        # cfgに out_dir がなければカレントディレクトリの results フォルダに保存
        self.out_dir = getattr(cfg, "out_dir", "results")
        os.makedirs(self.out_dir, exist_ok=True)
        self.force_log_path = os.path.join(self.out_dir, "ibm_forces.csv")
        
        with open(self.force_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ["step"]
            for i in range(len(parsed_objects)):
                header.extend([f"obj{i}_Fx", f"obj{i}_Fy", f"obj{i}_Fz"])
            writer.writerow(header)
            
        self.step_count = 0

    def apply(self, ctx, current_time):
        self.ibm.step(ctx)
        
        # パフォーマンスを落とさないよう、10ステップに1回だけCSVへ追記
        if self.step_count % 10 == 0:
            forces = self.ibm.get_hydro_forces()
            with open(self.force_log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [self.step_count]
                for i in range(forces.shape[0]):
                    row.extend([forces[i, 0], forces[i, 1], forces[i, 2]])
                writer.writerow(row)
                
        self.step_count += 1
