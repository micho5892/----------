import taichi as ti
import config
from context import FLUID_A, SOLID, SOLID_HEAT_SOURCE, INLET, OUTLET, ROTATING_WALL
from lbm_logger import get_logger

_log = get_logger(__name__)


@ti.data_oriented
class LBMSimulator:
    """
    Cumulant / Regularized LBM ベースの衝突モデルを搭載したシミュレータ。
    
    従来の標準MRTやSmagorinskyモデル（SGS）が抱えていた「壁面近傍での過剰な渦粘性による過減衰（双子渦の定常化）」を克服するため、
    Cumulant LBM の思想に基づく高次モーメント・フィルタリング（Regularization）を採用しています。
    
    【特徴】
    - 2次の非平衡モーメント（応力テンソル）のみを物理的な動粘性係数(tau_f)で緩和。
    - 3次以上の高次ゴーストモーメントを衝突ごとに強制破棄(ω=1で平衡化)し、チェスボード不安定性を根本から除去。
    - 外部のSGSモデルなしで、高Re数の乱流遷移ベンチマーク（カルマン渦の崩壊など）を精緻に再現します。
    """
    def __init__(self, sim_config):
        self.cfg = sim_config
        from d3q19 import D3Q19
        self.d3q19 = D3Q19(sim_config.fp_dtype)

        self.G = -5.0  # 引力の強さ（結合係数）。マイナスで引力。後でソフトスタートさせます
        _log.debug("LBMSimulator initialized (D3Q19, fp_dtype=%s)", sim_config.fp_dtype)

    @ti.func
    def is_fluid(self, ctx: ti.template(), cid):
        return ctx.is_fluid_table[cid] == 1
    
    @ti.kernel
    def init_fields(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            cid = ctx.cell_id[i, j, k]
            
            ctx.rho[i, j, k] = 1.0
            noise_x = (ti.random(ti.f32) - 0.5) * 1e-4
            noise_y = (ti.random(ti.f32) - 0.5) * 1e-4
            noise_z = (ti.random(ti.f32) - 0.5) * 1e-4
            
            if self.is_fluid(ctx, cid):
                # 流体セルにはノイズを持った初期速度を与える
                ctx.v[i, j, k] = ti.Vector([noise_x, noise_y, noise_z])
            else:
                # 固体セルは完全に静止
                ctx.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            
            if cid == SOLID or cid == SOLID_HEAT_SOURCE:
                ctx.temp[i, j, k] = 1.0
            else:
                ctx.temp[i, j, k] = 0.0
                
            for d in ti.static(range(19)):
                ctx.f_old[i, j, k, d] = self.d3q19.get_feq(1.0, ctx.v[i, j, k], d)
                ctx.g_old[i, j, k, d] = self.d3q19.get_geq(ctx.temp[i, j, k], ctx.v[i, j, k], d)
                
        for n in range(ctx.particle_pos.shape[0]):
            ctx.particle_pos[n] = ti.Vector([ti.random() * ctx.nx, ti.random() * ctx.ny, ti.random() * ctx.nz])

    @ti.kernel
    def collide_and_stream(self, ctx: ti.template(), omega_scale: ti.f32, sponge_amp: ti.f32):
        sponge_thickness = float(self.cfg.sponge_thickness)
        cs2 = 1.0 / 3.0
        coeff = 4.5

        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            cid = ctx.cell_id[i, j, k]
            
            # ==========================================================
            # 1. 温度場 g の緩和と移流 (統合)
            # ==========================================================
            if self.is_fluid(ctx, cid):
                tau_g = ctx.tau_g_table[cid]
                omega_g = 1.0 / tau_g
                v_g = ctx.v[i, j, k] if self.is_fluid(ctx, cid) else ti.Vector([0.0, 0.0, 0.0])
                temp = ctx.temp[i, j, k]
                
                for d in ti.static(range(19)):
                    geq = self.d3q19.get_geq(temp, v_g, d)
                    
                    # 計算した直後の値をローカル変数に持つ（g_postの代わり）
                    g_curr = ctx.g_old[i, j, k, d] - omega_g * (ctx.g_old[i, j, k, d] - geq) + self.d3q19.w[d] * ctx.S_g[i, j, k]
                    
                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]

                    if ti.static(self.cfg.periodic_x): ip = ip % ctx.nx
                    if ti.static(self.cfg.periodic_y): jp = jp % ctx.ny
                    if ti.static(self.cfg.periodic_z): kp = kp % ctx.nz

                    is_inside = True
                    if ti.static(not self.cfg.periodic_x):
                        if ip < 0 or ip >= ctx.nx: is_inside = False
                    if ti.static(not self.cfg.periodic_y):
                        if jp < 0 or jp >= ctx.ny: is_inside = False
                    if ti.static(not self.cfg.periodic_z):
                        if kp < 0 or kp >= ctx.nz: is_inside = False

                    if is_inside:
                        neighbor_cid = ctx.cell_id[ip, jp, kp]
                        if self.is_fluid(ctx, neighbor_cid):
                            ctx.g_new[ip, jp, kp, d] = g_curr
                        else:
                            inv_d = self.d3q19.inv_d[d]
                            # 固体: boundary_conditions の isothermal_wall なら ABB で Tw を課す。それ以外は BB。
                            if ctx.g_wall_use_abb[neighbor_cid] != 0:
                                Tw = ctx.g_wall_tw[neighbor_cid]
                                ctx.g_new[i, j, k, inv_d] = -g_curr + 2.0 * self.d3q19.w[d] * Tw
                            else:
                                ctx.g_new[i, j, k, inv_d] = g_curr
                    else:
                        inv_d = self.d3q19.inv_d[d]
                        ctx.g_new[i, j, k, inv_d] = g_curr
                
            # ==========================================================
            # 2. 速度場 f の緩和と移流 (統合)
            # ==========================================================
            if self.is_fluid(ctx, cid):
                rho = ctx.rho[i, j, k]
                v_vec = ctx.v[i, j, k]
                tau_f = ctx.tau_f_table[cid]
                
                omega_f = 1.0 / tau_f
                dist_to_z0 = float(k)
                dist_to_zn = float(ctx.nz - 1 - k)
                min_dist = ti.math.min(dist_to_z0, dist_to_zn)
                
                if sponge_thickness > 0.0 and sponge_amp > 0.0:
                    dist_to_z0 = float(k)
                    dist_to_zn = float(ctx.nz - 1 - k)
                    min_dist = ti.math.min(dist_to_z0, dist_to_zn)

                    if min_dist < sponge_thickness:
                        factor = min_dist / (sponge_thickness + 1e-7)
                        fade = 0.5 * (1.0 - ti.math.cos(ti.math.pi * factor))
                        # 空間プロファイルは従来どおり (1-fade)。時間で sponge_amp を掛けて強度のみ弱める
                        tau_sponge = tau_f + (5.0 - tau_f) * (1.0 - fade) * sponge_amp
                        omega_f = 1.0 / tau_sponge
                
                Pi_xx = 0.0; Pi_yy = 0.0; Pi_zz = 0.0
                Pi_xy = 0.0; Pi_yz = 0.0; Pi_zx = 0.0
                
                f_eq_cache = ti.Vector.zero(config.TI_FLOAT, 19)
                f_post_cache = ti.Vector.zero(config.TI_FLOAT, 19) # f_postの代わりのローカル配列
                
                # Step A: 応力テンソルの計算
                for d in ti.static(range(19)):
                    feq = self.d3q19.get_feq(rho, v_vec, d)
                    f_eq_cache[d] = feq
                    f_neq = ctx.f_old[i, j, k, d] - feq
                    ex = float(self.d3q19.e[d][0]); ey = float(self.d3q19.e[d][1]); ez = float(self.d3q19.e[d][2])
                    Pi_xx += f_neq * ex * ex; Pi_yy += f_neq * ey * ey; Pi_zz += f_neq * ez * ez
                    Pi_xy += f_neq * ex * ey; Pi_yz += f_neq * ey * ez; Pi_zx += f_neq * ez * ex
                
                Pi_xx *= (1.0 - omega_f); Pi_yy *= (1.0 - omega_f); Pi_zz *= (1.0 - omega_f)
                Pi_xy *= (1.0 - omega_f); Pi_yz *= (1.0 - omega_f); Pi_zx *= (1.0 - omega_f)
                F_vec = ctx.F_int[i, j, k] 
                
                # Step B: 緩和後の分布関数の構築
                for d in ti.static(range(19)):
                    ex = float(self.d3q19.e[d][0]); ey = float(self.d3q19.e[d][1]); ez = float(self.d3q19.e[d][2])
                    w = float(self.d3q19.w[d])
                    Q_xx = ex * ex - cs2; Q_yy = ey * ey - cs2; Q_zz = ez * ez - cs2
                    Q_xy = ex * ey; Q_yz = ey * ez; Q_zx = ez * ex
                    
                    f_neq_reg = w * coeff * (
                        Pi_xx * Q_xx + Pi_yy * Q_yy + Pi_zz * Q_zz +
                        2.0 * (Pi_xy * Q_xy + Pi_yz * Q_yz + Pi_zx * Q_zx)
                    )
                    
                    e_vec = ti.Vector([ex, ey, ez])
                    e_minus_v = e_vec - v_vec
                    e_dot_v = e_vec.dot(v_vec)
                    term1 = 3.0 * e_minus_v
                    term2 = 9.0 * e_dot_v * e_vec
                    S_i = (1.0 - 0.5 * omega_f) * w * (term1 + term2).dot(F_vec)
                    
                    # 計算結果をローカルキャッシュに保存
                    f_post_cache[d] = f_eq_cache[d] + f_neq_reg + S_i

                # Step C: 構築した分布関数をそのまま移流 (Stream) させる
                for d in ti.static(range(19)):
                    inv_d = self.d3q19.inv_d[d]

                    ip = i + self.d3q19.e[d][0]
                    jp = j + self.d3q19.e[d][1]
                    kp = k + self.d3q19.e[d][2]

                    if ti.static(self.cfg.periodic_x): ip = ip % ctx.nx
                    if ti.static(self.cfg.periodic_y): jp = jp % ctx.ny
                    if ti.static(self.cfg.periodic_z): kp = kp % ctx.nz

                    is_inside = True
                    if ti.static(not self.cfg.periodic_x):
                        if ip < 0 or ip >= ctx.nx: is_inside = False
                    if ti.static(not self.cfg.periodic_y):
                        if jp < 0 or jp >= ctx.ny: is_inside = False
                    if ti.static(not self.cfg.periodic_z):
                        if kp < 0 or kp >= ctx.nz: is_inside = False

                    if is_inside:
                        neighbor_cid = ctx.cell_id[ip, jp, kp]
                        if self.is_fluid(ctx, neighbor_cid):
                            ctx.f_new[ip, jp, kp, d] = f_post_cache[d]
                        else:
                            f_bb = f_post_cache[d]

                            # ▼ 追加：回転壁 (ID: 30) だった場合、壁の勢いを空気に乗せて跳ね返す (Moving Bounce-Back)
                            if neighbor_cid == ROTATING_WALL:
                                cx = float(self.cfg.cylinder_center[0])
                                cy = float(self.cfg.cylinder_center[1])
                                cz = float(self.cfg.cylinder_center[2]) # Z座標も取得
                                omega = float(self.cfg.omega_cylinder) * omega_scale
                                
                                # 中心からの相対ベクトル
                                rx = float(ip) - cx
                                ry = float(jp) - cy
                                rz = float(kp) - cz
                                
                                uw_x = 0.0
                                uw_y = 0.0
                                uw_z = 0.0
                                
                                # ▼ 修正：回転軸に応じた表面速度 (クロス積) の計算
                                if self.cfg.rot_axis_id == 0:   # X軸回転
                                    uw_y = -omega * rz
                                    uw_z =  omega * ry
                                elif self.cfg.rot_axis_id == 1: # Y軸回転 ★今回のターゲット
                                    uw_x =  omega * rz
                                    uw_z = -omega * rx
                                else:                           # Z軸回転 (従来互換)
                                    uw_x = -omega * ry
                                    uw_y =  omega * rx
                                    
                                ex_d = float(self.d3q19.e[d][0])
                                ey_d = float(self.d3q19.e[d][1])
                                ez_d = float(self.d3q19.e[d][2])
                                w_d  = float(self.d3q19.w[d])
                                
                                # 内積 (e_i ・ u_wall) を計算して運動量を注入
                                eu = ex_d * uw_x + ey_d * uw_y + ez_d * uw_z
                                f_bb -= 6.0 * w_d * rho * eu
                                
                                # 熱の移流もオンにする場合（今後のため）
                                Tw = 1.0 
                                g_bb = ctx.g_old[ip, jp, kp, d] 
                                g_bb -= 6.0 * w_d * Tw * eu
                                ctx.g_new[i, j, k, inv_d] = g_bb

                                # ▼==========================================
                                # ▼ 追加：ID 20 を「横にスライドする壁」として処理
                                # ▼==========================================
                            elif neighbor_cid == 20: 
                                # フタの移動速度 (LBM単位系の u_lbm)
                                uw_x = float(self.cfg.u_lbm_inlet) 
                                uw_y = 0.0
                                uw_z = 0.0
                                
                                ex_d = float(self.d3q19.e[d][0])
                                ey_d = float(self.d3q19.e[d][1])
                                ez_d = float(self.d3q19.e[d][2])
                                w_d  = float(self.d3q19.w[d])
                                
                                # 壁の運動量を空気に乗せて跳ね返す (Moving Bounce-Back)
                                eu = ex_d * uw_x + ey_d * uw_y + ez_d * uw_z
                                f_bb -= 6.0 * w_d * rho * eu
                                
                                ctx.f_new[i, j, k, inv_d] = f_bb


                            ctx.f_new[i, j, k, inv_d] = f_bb
                    else:
                        # ▼ 【追加】 領域外 (外の世界) に出ようとしたら、見えない壁として反射(Bounce-back)
                        ctx.f_new[i, j, k, inv_d] = f_post_cache[d]

    @ti.kernel
    def update_macro(self, ctx: ti.template()):
        for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
            cid = ctx.cell_id[i, j, k]
            
            if self.is_fluid(ctx, cid):
                new_temp = 0.0
                new_rho = 0.0
                new_v = ti.Vector([0.0, 0.0, 0.0])
                
                for d in ti.static(range(19)):
                    # g と f の更新をまとめる
                    g_val = ctx.g_new[i, j, k, d]
                    new_temp += g_val
                    ctx.g_old[i, j, k, d] = g_val
                    
                    f_val = ctx.f_new[i, j, k, d]
                    new_rho += f_val
                    new_v += f_val * self.d3q19.e[d]
                    ctx.f_old[i, j, k, d] = f_val
                
                ctx.temp[i, j, k] = new_temp
                ctx.rho[i, j, k] = new_rho
                if new_rho > 1e-12:
                    ctx.v[i, j, k] = new_v / new_rho
                else:
                    ctx.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # 固体セルはゼロ(または壁温度)固定
                ctx.rho[i, j, k] = 1.0
                ctx.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                if ctx.g_wall_use_abb[cid] != 0:
                    ctx.temp[i, j, k] = ctx.g_wall_tw[cid]
                else:
                    ctx.temp[i, j, k] = 0.0

    @ti.kernel
    def move_particles(self, ctx: ti.template(), inject_per_step: ti.i32):
        # 入口は z = nz-1、出口は z = 0。出た粒子はリサイクルせず、毎ステップ一定数だけ入口で新規生成する。
        ctx.inject_count[None] = 0
        for n in range(ctx.particle_pos.shape[0]):
            pos = ctx.particle_pos[n]
            i, j, k = int(pos[0]), int(pos[1]), int(pos[2])

            if 0 <= i < ctx.nx and 0 <= j < ctx.ny and 0 <= k < ctx.nz:
                ctx.particle_pos[n] += ctx.v[i, j, k] * 2.0

            new_pos = ctx.particle_pos[n]

            is_out = False
            if new_pos[2] < 1.0 or new_pos[2] >= ctx.nz - 1.0:
                is_out = True
            else:
                ni = int(new_pos[0]) % ctx.nx
                nj = int(new_pos[1]) % ctx.ny
                nk = int(new_pos[2])
                if 0 <= nk < ctx.nz:
                    cid = ctx.cell_id[ni, nj, nk]
                    if cid == SOLID or cid == SOLID_HEAT_SOURCE:
                        is_out = True

            if is_out:
                # 毎ステップ一定数だけ入口に新規生成（空いたスロットを再利用）。それ以外の「出た」粒子は消滅（描画されない位置へ）
                idx = ti.atomic_add(ctx.inject_count[None], 1)
                if idx < inject_per_step:
                    ctx.particle_pos[n] = ti.Vector([
                        ti.random() * ctx.nx,
                        ti.random() * ctx.ny,
                        float(ctx.nz) - 0.5,
                    ])
                else:
                    # 描画範囲外に置いて消滅させる（export の 0<=x<nx 等で弾かれる）
                    ctx.particle_pos[n] = ti.Vector([-1.0, -1.0, -1.0])
