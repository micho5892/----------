# ==========================================================
# physics.py — 物理モデルプラグイン (多相流、浮力、相変化など)
# ==========================================================
import taichi as ti
import math
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
                F = ctx.rho[i, j, k] * self.beta * temp_diff * self.g_vec
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
        # cfg全体を受け取るように変更
        p_kwargs = cfg.physics_models["immersed_boundary"]
        
        dx = cfg.dx
        dt = cfg.dt
        
        # ユーザーが指定する【物理パラメータ】を取得
        r_p = p_kwargs.get("radius_p", 0.01)               # 物理半径 [m]
        center_p = p_kwargs.get("center_p",[0.05, 0.05, 0.08]) # 物理座標 [m]
        rho_s_p = p_kwargs.get("rho_s_p", 7800.0)          # 剛体の密度 [kg/m^3] (鉄など)
        g_p = p_kwargs.get("gravity_p",[0.0, 0.0, -9.81]) # 物理重力 [m/s^2]
        
        # 流体の物理密度を取得 (デフォルトは空気1.2)
        rho_f_p = 1.2
        if 0 in cfg.domain_properties:
            rho_f_p = cfg.domain_properties[0].get("rho", 1.2)

        # ==========================================================
        # ★ 全自動・無次元化計算 (LBM単位系への翻訳)
        # ==========================================================
        # 1. 長さの変換
        self.r_lbm = r_p / dx
        center_lbm = [c / dx for c in center_p]
        
        # 2. 質量の変換 (密度比を使うのがCFDの鉄則)
        # 物理質量 = 4/3 * pi * r_p^3 * rho_s_p
        # LBM質量 = LBM体積 * (固体密度 / 流体密度)
        density_ratio = rho_s_p / rho_f_p
        self.mass_lbm = (4.0 / 3.0) * math.pi * (self.r_lbm ** 3) * density_ratio
        
        # 3. 加速度(重力)の変換: g_lbm = g_p * (dt^2 / dx)
        self.g_lbm = [g * (dt ** 2) / dx for g in g_p]
        
        # 4. マーカー数の自動計算 (マーカー間隔が約0.8セルになるように配置)
        surface_area_lbm = 4.0 * math.pi * (self.r_lbm ** 2)
        self.num_points = int(surface_area_lbm / (0.8 ** 2))
        self.dA_lbm = surface_area_lbm / self.num_points # 1マーカーあたりの面積
        
        _log.info(
            "[IBM Auto-Scale] Radius(LBM): %.2f cells, Mass(LBM): %.2f, Points: %s",
            self.r_lbm,
            self.mass_lbm,
            self.num_points,
        )
        _log.info("[IBM Auto-Scale] Gravity(LBM): %s", self.g_lbm)

        # IBMマネージャーの初期化
        self.ibm = IBManager(fp_dtype, self.num_points, self.mass_lbm, self.g_lbm, self.dA_lbm)
        
        # マーカーを配置
        points_np = create_sphere_markers(self.r_lbm, center_lbm, self.num_points)
        self.ibm.pos.from_numpy(points_np)
        self.ibm.center[None] = center_lbm

    def apply(self, ctx, current_time):
        self.ibm.step(ctx)
