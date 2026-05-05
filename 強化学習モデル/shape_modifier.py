import taichi as ti
import math

@ti.data_oriented
class ShapeModifier:
    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.cfg = cfg
        self.FLUID_ID = 0
        self.SOLID_ID = 10 
        self.SOLID_HEAT_SOURCE_ID = 11  # ★追加: 絶対に削れない熱源
        self.INLET_ID = 20
        self.OUTLET_ID = 21

    def build_initial_block(self):
        """トップフロー冷却用の初期形状を作成"""
        self._build_initial_block_kernel()

    @ti.kernel
    def _build_initial_block_kernel(self):
        margin_x = self.ctx.nx // 8
        margin_y = self.ctx.ny // 8
        height_z = int(self.ctx.nz * 0.6) 
        base_z = 5  # ★追加: Z=0〜5 を熱源（ベースプレート）とする

        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            is_inside_block = False
            if (margin_x <= i < self.ctx.nx - margin_x) and \
               (margin_y <= j < self.ctx.ny - margin_y) and \
               (k <= height_z):
                is_inside_block = True

            if is_inside_block:
                # ★修正: Z座標でIDを切り分ける
                if k <= base_z:
                    self.ctx.cell_id[i, j, k] = self.SOLID_HEAT_SOURCE_ID
                else:
                    self.ctx.cell_id[i, j, k] = self.SOLID_ID
                    
                self.ctx.sdf[i, j, k] = -1.0
                self.ctx.phi[i, j, k] = 1.0
            else:
                self.ctx.cell_id[i, j, k] = self.FLUID_ID
                self.ctx.sdf[i, j, k] = 1.0
                self.ctx.phi[i, j, k] = 0.0

            # 境界条件の上書き
            if k == self.ctx.nz - 1:
                self.ctx.cell_id[i, j, k] = self.INLET_ID
            elif (i == 0 or i == self.ctx.nx - 1 or j == 0 or j == self.ctx.ny - 1) and k < self.ctx.nz - 1:
                self.ctx.cell_id[i, j, k] = self.OUTLET_ID

    @ti.kernel
    def subtract_sphere(self, cx: ti.f32, cy: ti.f32, cz: ti.f32, radius: ti.f32):
        """指定した球の範囲のSDFをえぐり取り、流体に変更する"""
        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            current_sdf = self.ctx.sdf[i, j, k]
            
            # 熱源の保護ライン (base_z と合わせる)
            protected_z = 5.0  
            if float(k) > protected_z:
                dx = float(i) - cx
                dy = float(j) - cy
                dz = float(k) - cz
                dist_to_center = ti.math.sqrt(dx*dx + dy*dy + dz*dz)
                sphere_sdf = dist_to_center - radius
                
                new_sdf = ti.math.max(current_sdf, -sphere_sdf)
                self.ctx.sdf[i, j, k] = new_sdf
                
                if new_sdf > 0.0:
                    # ★修正: 削れるのは通常の金属（ID:10）だけ
                    if self.ctx.cell_id[i, j, k] == self.SOLID_ID:
                        self.ctx.cell_id[i, j, k] = self.FLUID_ID
                        self.ctx.phi[i, j, k] = 0.0
                        self.ctx.u_solid[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


    def update_from_density_tensor(self, density_np):
        """
        AIが出力した 0.0(流体)〜1.0(固体) の密度テンソルを受け取り、phi と sdf を一括更新する。
        """
        # TaichiフィールドにNumPy配列を転送するための受け皿を作成
        if not hasattr(self, "density_field"):
            self.density_field = ti.field(dtype=self.cfg.fp_dtype, shape=(self.ctx.nx, self.ctx.ny, self.ctx.nz))
        
        self.density_field.from_numpy(density_np)
        self._update_from_density_kernel()

    @ti.kernel
    def _update_from_density_kernel(self):
        protected_z = 5.0  
        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            cid = self.ctx.cell_id[i, j, k]
            
            # 熱源ベースプレートや、入口・出口境界は絶対に改変しない
            if float(k) > protected_z and cid != self.INLET_ID and cid != self.OUTLET_ID and cid != self.SOLID_HEAT_SOURCE_ID:
                
                # ターゲットの固体率 (0.0: 流体, 1.0: 固体)
                d_val = self.density_field[i, j, k]
                
                # 密度法のように、中間状態（半透明のスポンジ状態）を許容する
                self.ctx.phi[i, j, k] = d_val
                
                # セルIDとSDFは、計算の安定のため閾値(0.5)で二値化して判定
                if d_val < 0.5:
                    self.ctx.cell_id[i, j, k] = self.FLUID_ID
                    self.ctx.sdf[i, j, k] = 1.0   # 簡易的に流体側(プラス)とする
                    self.ctx.u_solid[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    self.ctx.cell_id[i, j, k] = self.SOLID_ID
                    self.ctx.sdf[i, j, k] = -1.0  # 簡易的に固体側(マイナス)とする

