import taichi as ti
import math

@ti.data_oriented
class ShapeModifier:
    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.cfg = cfg
        self.FLUID_ID = 0
        self.SOLID_ID = 10 
        self.INLET_ID = 20
        self.OUTLET_ID = 21

    # =========================================================
    # 既存のgeometry.pyを触らないための「初期ブロック生成機能」
    # =========================================================
    def build_initial_block(self):
        """トップフロー冷却用の初期形状を作成"""
        self._build_initial_block_kernel()

    @ti.kernel
    def _build_initial_block_kernel(self):
        # ヒートシンクの配置範囲 (全体の 3/4 のサイズにする)
        # 例: 両端 1/8 ずつを空ける
        margin_x = self.ctx.nx // 8
        margin_y = self.ctx.ny // 8
        
        # ヒートシンクの高さ (上部から風が来るので、上部は空間を空けておく)
        height_z = int(self.ctx.nz * 0.6) 

        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            # --- 1. ヒートシンクブロックの判定 ---
            is_inside_block = False
            if (margin_x <= i < self.ctx.nx - margin_x) and \
               (margin_y <= j < self.ctx.ny - margin_y) and \
               (k <= height_z):
                is_inside_block = True

            if is_inside_block:
                self.ctx.cell_id[i, j, k] = self.SOLID_ID
                self.ctx.sdf[i, j, k] = -1.0
                self.ctx.phi[i, j, k] = 1.0
            else:
                self.ctx.cell_id[i, j, k] = self.FLUID_ID
                self.ctx.sdf[i, j, k] = 1.0
                self.ctx.phi[i, j, k] = 0.0

            # --- 2. 境界条件の上書き ---
            # 天井 (真上) を INLET に
            if k == self.ctx.nz - 1:
                self.ctx.cell_id[i, j, k] = self.INLET_ID
            
            # 側面四方を OUTLET に (天井のINLETとは被らないように k < nz-1 とする)
            elif (i == 0 or i == self.ctx.nx - 1 or j == 0 or j == self.ctx.ny - 1) and k < self.ctx.nz - 1:
                self.ctx.cell_id[i, j, k] = self.OUTLET_ID

    # =========================================================
    # 【案A用】 メタボール（球）による減算加工カーネル
    # =========================================================
    @ti.kernel
    def subtract_sphere(self, cx: ti.f32, cy: ti.f32, cz: ti.f32, radius: ti.f32):
        """指定した球の範囲のSDFをえぐり取り、流体に変更する"""
        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            current_sdf = self.ctx.sdf[i, j, k]
            
            # 例: Z方向の下から5セルはベースプレート（熱源）として保護し、絶対に削らない
            protected_z = 5.0  
            if float(k) > protected_z:
                dx = float(i) - cx
                dy = float(j) - cy
                dz = float(k) - cz
                dist_to_center = ti.math.sqrt(dx*dx + dy*dy + dz*dz)
                sphere_sdf = dist_to_center - radius
                
                # ブーリアン減算
                new_sdf = ti.math.max(current_sdf, -sphere_sdf)
                self.ctx.sdf[i, j, k] = new_sdf
                
                # SDFがプラス（流体領域）になったら物理プロパティを更新
                if new_sdf > 0.0:
                    self.ctx.cell_id[i, j, k] = self.FLUID_ID
                    self.ctx.phi[i, j, k] = 0.0
                    self.ctx.u_solid[i, j, k] = ti.Vector([0.0, 0.0, 0.0])