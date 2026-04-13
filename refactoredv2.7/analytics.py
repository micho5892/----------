# ==========================================================
# analytics.py — 解析ユーティリティ（タスク9: Nu・局所値・統計を Solver から分離）
# ctx を引数に取り、Solver/Boundary に依存しない。
# ==========================================================
import taichi as ti
import numpy as np
from wall_metrics import channel_hydraulic_diameter_p, get_wall_neighbor_dirs_xy
from nu_models import build_nu_model
from lbm_logger import get_logger

_log = get_logger(__name__)


@ti.data_oriented
class Analytics:
    """Nu・平均温度・局所 Nu など、ctx ベースの解析を提供する。"""

    def __init__(self, d3q19, cfg):
        self.d3q19 = d3q19
        self.cfg = cfg
        self.wall_neighbor_dirs_xy = tuple(get_wall_neighbor_dirs_xy())
        self.channel_d_h_p = float(channel_hydraulic_diameter_p(cfg.nx, cfg.ny, cfg.Lx_p, wall_thickness_cells=10))
        nu_model = build_nu_model(cfg)
        self.nu_model_name = nu_model.name
        self.nu_l_ref_p = float(nu_model.l_ref_p)
        self.nu_k_ref_mode = int(nu_model.k_ref_mode)
        max_cid = 255
        if cfg.domain_properties:
            max_cid = max(max_cid, max(int(cid) for cid in cfg.domain_properties.keys()))
        self.k_table = ti.field(dtype=ti.f32, shape=max_cid + 1)
        self.rho_table = ti.field(dtype=ti.f32, shape=max_cid + 1)
        self.cp_table = ti.field(dtype=ti.f32, shape=max_cid + 1)
        self.nu_table = ti.field(dtype=ti.f32, shape=max_cid + 1)
        self.is_fluid_table = ti.field(dtype=ti.i32, shape=max_cid + 1)

        k_np = np.full(max_cid + 1, 0.6, dtype=np.float32)
        rho_np = np.full(max_cid + 1, 1000.0, dtype=np.float32)
        cp_np = np.full(max_cid + 1, 4180.0, dtype=np.float32)
        nu_np = np.full(max_cid + 1, 1.0e-5, dtype=np.float32)
        is_fluid_np = np.zeros(max_cid + 1, dtype=np.int32)

        materials_dict = cfg.get_materials_dict()
        for cid, (_, _, is_fluid_flag) in materials_dict.items():
            is_fluid_np[int(cid)] = int(is_fluid_flag)

        for cid, props in cfg.domain_properties.items():
            k_np[int(cid)] = float(props.get("k", 0.6))
            rho_np[int(cid)] = float(props.get("rho", 1000.0))
            cp_np[int(cid)] = float(props.get("Cp", 4180.0))
            nu_np[int(cid)] = float(props.get("nu", 1.0e-5))

        self.k_table.from_numpy(k_np)
        self.rho_table.from_numpy(rho_np)
        self.cp_table.from_numpy(cp_np)
        self.nu_table.from_numpy(nu_np)
        self.is_fluid_table.from_numpy(is_fluid_np)

    @ti.func
    def _is_fluid(self, cid: ti.i32) -> ti.i32:
        return self.is_fluid_table[cid]

    @ti.func
    def _is_solid_like(self, cid: ti.i32) -> ti.i32:
        return 1 - self._is_fluid(cid)

    @ti.kernel
    def _get_avg_grad_kernel(self, ctx: ti.template()) -> ti.types.f32:
        sum_q, area = 0.0, 0.0
        for i, j, k in ctx.rho:
            cid = ctx.cell_id[i, j, k]
            if self._is_solid_like(cid) == 1:
                for d in range(1, 7):
                    ip = (i + self.d3q19.e[d][0]) % ctx.nx
                    jp = (j + self.d3q19.e[d][1]) % ctx.ny
                    kp = k + self.d3q19.e[d][2]
                    if 0 <= kp < ctx.nz:
                        cid_n = ctx.cell_id[ip, jp, kp]
                        if self._is_fluid(cid_n) == 1:
                            grad = (1.0 - ctx.temp[ip, jp, kp]) / 0.5
                            sum_q += grad
                            area += 1.0
        return sum_q / area if area > 0 else 0.0

    def get_avg_grad(self, ctx):
        return self._get_avg_grad_kernel(ctx)

    @ti.kernel
    def _get_avg_t_out_kernel(self, ctx: ti.template()) -> ti.types.f32:
        sum_t, count = 0.0, 0
        for i, j, k in ctx.rho:
            cid = ctx.cell_id[i, j, k]
            if self._is_fluid(cid) != 1:
                continue
            kp = k - 1
            if kp >= 0 and self._is_fluid(ctx.cell_id[i, j, kp]) == 0:
                sum_t += ctx.temp[i, j, k]
                count += 1
        return sum_t / count if count > 0 else 0.0

    def get_avg_t_out(self, ctx):
        return self._get_avg_t_out_kernel(ctx)

    @ti.kernel
    def _get_solid_temp_kernel(self, ctx: ti.template()) -> ti.types.f32:
        sum_t, count = 0.0, 0
        for i, j, k in ctx.temp:
            cid = ctx.cell_id[i, j, k]
            if self._is_solid_like(cid) == 1:
                sum_t += ctx.temp[i, j, k]
                count += 1
        return sum_t / count if count > 0 else 0.0

    def get_solid_temp(self, ctx):
        return self._get_solid_temp_kernel(ctx)

    @ti.kernel
    def _get_local_Nu_kernel(
        self,
        ctx: ti.template(),
        k_target: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        dx: ti.types.f32,
        Lx_p: ti.types.f32,
        delta_T_ref: ti.types.f32,
        T_inlet_p: ti.types.f32,
    ) -> ti.types.vector(6, ti.f32):
        # バルク温度: 質量流束重み Σ(ρ u_z T) / Σ(ρ u_z)
        sum_mdT, sum_md = 0.0, 0.0
        sum_q_wall_p, area = 0.0, 0.0
        sum_kf, count_kf = 0.0, 0.0
        sum_kw, count_kw = 0.0, 0.0
        for i, j in ti.ndrange(nx, ny):
            cid = ctx.cell_id[i, j, k_target]
            if self._is_fluid(cid) == 1:
                u_z = -ctx.v[i, j, k_target][2]
                rho = ctx.rho[i, j, k_target]
                t = ctx.temp[i, j, k_target]
                md = rho * u_z
                sum_mdT += md * t
                sum_md += md
                sum_kf += self.k_table[cid]
                count_kf += 1.0
        T_bulk_nd = sum_mdT / sum_md if sum_md > 0 else 0.0
        
        sum_Tw, area_Tw = 0.0, 0.0  
        
        for i, j in ti.ndrange(nx, ny):
            cid = ctx.cell_id[i, j, k_target]
            if self._is_solid_like(cid) == 1:
                sum_Tw += ctx.temp[i, j, k_target]
                area_Tw += 1.0
                sum_kw += self.k_table[cid]
                count_kw += 1.0
                
                for d in ti.static(self.wall_neighbor_dirs_xy):
                    ip = (i + self.d3q19.e[d][0]) % nx
                    jp = (j + self.d3q19.e[d][1]) % ny
                    
                    cid_n = ctx.cell_id[ip, jp, k_target]
                    if self._is_fluid(cid_n) == 1:
                        k_fluid = self.k_table[cid_n]
                        grad_nd = (1.0 - ctx.temp[ip, jp, k_target]) / 0.5
                        q_face_p = k_fluid * grad_nd * delta_T_ref / dx
                        
                        sum_q_wall_p += q_face_p
                        area += 1.0
                        
        q_wall_p = sum_q_wall_p / area if area > 0 else 0.0
        T_bulk_p = T_inlet_p + T_bulk_nd * delta_T_ref
        
        T_wall_nd = sum_Tw / area_Tw if area_Tw > 0 else 0.0
        T_wall_actual_p = T_inlet_p + T_wall_nd * delta_T_ref
        k_bulk_ref = sum_kf / count_kf if count_kf > 0 else 0.6
        k_wall_ref = sum_kw / count_kw if count_kw > 0 else k_bulk_ref
    
        denom = T_wall_actual_p - T_bulk_p
        h_local = q_wall_p / denom if denom != 0 else 0.0

        k_ref = k_bulk_ref
        if ti.static(self.nu_k_ref_mode == 1):
            k_ref = k_wall_ref

        Nu_local = h_local * self.nu_l_ref_p / (k_ref + 1e-12)

        cx = nx // 2
        cy_mid = ny // 2
        T_center = ctx.temp[cx, cy_mid, k_target]
        T_fluid_edge = ctx.temp[cx, cy_mid - 10, k_target]
        return ti.Vector(
            [
                Nu_local,
                T_center,
                T_fluid_edge,
                T_bulk_p,
                T_wall_actual_p,
                q_wall_p,
            ]
        )

    @ti.kernel
    def _get_local_Nu_ibm_channel_kernel(
        self,
        ctx: ti.template(),
        k_target: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        dx: ti.types.f32,
        delta_T_ref: ti.types.f32,
        T_inlet_p: ti.types.f32,
    ) -> ti.types.vector(6, ti.f32):
        """
        IBM での汎用的な Nu 数計算。
        固体セルを探すのではなく、Thermal IBM が `ctx.S_g` に書き込んだ熱源強度の総和
        を直接計測して壁からの熱流束を逆算する。
        """
        # バルク温度計算
        sum_mdT = 0.0
        sum_md = 0.0
        sum_kf = 0.0
        count_kf = 0.0
        
        # 断面における熱源(S_g)の総和を計算
        sum_Sg_slice = 0.0
        area = 0.0
        
        for i, j in ti.ndrange(nx, ny):
            cid = ctx.cell_id[i, j, k_target]
            # 流体セル（IBMが適用される空間）
            if self._is_fluid(cid) == 1:
                u_z = -ctx.v[i, j, k_target][2]
                rho = ctx.rho[i, j, k_target]
                t = ctx.temp[i, j, k_target]
                md = rho * u_z
                sum_mdT += md * t
                sum_md += md
                sum_kf += self.k_table[cid]
                count_kf += 1.0
                
                # IBMマーカーから与えられた熱源(S_g)を積算
                # ※ LBMの S_g は「単位時間・単位体積あたりの熱の湧き出し」
                s_g_val = ctx.S_g[i, j, k_target]
                if ti.abs(s_g_val) > 1e-12:
                    sum_Sg_slice += s_g_val
                    area += 1.0

        T_bulk_nd = sum_mdT / sum_md if sum_md > 0 else 0.0
        T_bulk_p = T_inlet_p + T_bulk_nd * delta_T_ref

        # IBM壁での平均熱流束計算
        # LBM方程式のソース項 S_g を物理的な熱流束 q に変換する。
        # q = \sum S_g * k_fluid * (delta_T_ref / dx) / (加熱面積)
        k_bulk_ref = sum_kf / count_kf if count_kf > 0 else 0.6
        
        # 周辺長(面積)が 0 なら 2*nx (平行平板の上下壁の長さ) で代用
        actual_perimeter = area if area > 0 else float(2 * nx)
        
        # 無次元熱流束 q_nd = sum(S_g) / perimeter
        q_nd = sum_Sg_slice / actual_perimeter
        
        # 物理的な熱流束へ変換
        q_wall_p = k_bulk_ref * q_nd * delta_T_ref / dx

        # IBMマーカーの目標温度 (現在は 1.0 固定と仮定)
        T_wall_nd = 1.0 
        T_wall_actual_p = T_inlet_p + T_wall_nd * delta_T_ref
        
        k_ref = k_bulk_ref # k_ref_mode=0
        
        denom = T_wall_actual_p - T_bulk_p
        h_local = q_wall_p / denom if denom != 0 else 0.0
        Nu_local = h_local * self.nu_l_ref_p / (k_ref + 1e-12)

        cx = nx // 2
        cy_mid = ny // 2
        
        # 任意形状対応のため、壁際温度はサンプリングしない(0を返す)
        return ti.Vector([
            Nu_local,
            ctx.temp[cx, cy_mid, k_target],
            0.0, 
            T_bulk_p,
            T_wall_actual_p,
            q_wall_p
        ])

    def get_local_Nu(self, ctx, k_target, log_thermal_slice=False):
        # ベンチマーク名で呼び出すカーネルを分岐
        if self.cfg.benchmark_name == "parallel_plates_ibm":
            pack = self._get_local_Nu_ibm_channel_kernel(
                ctx, k_target, self.cfg.nx, self.cfg.ny, self.cfg.dx,
                self.cfg.delta_T_ref, self.cfg.T_inlet_p
            )
        else:
            pack = self._get_local_Nu_kernel(
                ctx, k_target, self.cfg.nx, self.cfg.ny, self.cfg.dx, self.cfg.Lx_p,
                self.cfg.delta_T_ref, self.cfg.T_inlet_p
            )
            
        nu = float(pack[0])

        if log_thermal_slice:
            t_center = float(pack[1])
            t_fluid_edge = float(pack[2])
            t_bulk = float(pack[3])
            t_wall = float(pack[4])
            q_wall = float(pack[5])
            _log.info("T_center: %.6f, T_fluid_edge: %.6f", t_center, t_fluid_edge)
            _log.info("T_bulk: %.6f, T_wall: %.6f", t_bulk, t_wall)
            _log.info("q_wall_p: %.6e", q_wall)
        return nu

    @ti.kernel
    def compute_global_hash(self, ctx: ti.template()) -> ti.types.vector(2, ti.f32):
        """
        空間全体の速度と温度を「2つの数字（ハッシュ値）」に圧縮して返す超高速カーネル。
        戻り値: [全運動エネルギー, 全熱エネルギー]
        """
        total_ke = 0.0
        total_temp = 0.0
        
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            if self._is_fluid(ctx.cell_id[i, j, k]) == 1:
                v = ctx.v[i, j, k]
                total_ke += v.dot(v)  
                total_temp += ctx.temp[i, j, k]
                
        return ti.Vector([total_ke, total_temp])