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
        """強化学習用の初期形状（ブロック状の固体と、入口・出口）を構築"""
        self._build_initial_block_kernel()

    @ti.kernel
    def _build_initial_block_kernel(self):
        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            # 1. 基本は全て固体とする
            self.ctx.cell_id[i, j, k] = self.SOLID_ID
            self.ctx.sdf[i, j, k] = -1.0 # マイナスは固体内を表す
            self.ctx.phi[i, j, k] = 1.0  # 固体率100%

            # 2. ただし、Zの端は風洞の入口と出口にするため流体領域にする
            if k == self.ctx.nz - 1:
                self.ctx.cell_id[i, j, k] = self.INLET_ID
                self.ctx.sdf[i, j, k] = 1.0
                self.ctx.phi[i, j, k] = 0.0
            elif k == 0:
                self.ctx.cell_id[i, j, k] = self.OUTLET_ID
                self.ctx.sdf[i, j, k] = 1.0
                self.ctx.phi[i, j, k] = 0.0

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