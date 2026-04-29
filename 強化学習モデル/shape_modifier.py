import taichi as ti
import math

@ti.data_oriented
class ShapeModifier:
    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.cfg = cfg
        # LBMの固体/流体判定に使うID
        self.FLUID_ID = 0
        self.SOLID_ID = 10 

    # =========================================================
    # 【案A用】 メタボール（球）による減算加工カーネル
    # =========================================================
    @ti.kernel
    def subtract_sphere(self, cx: ti.f32, cy: ti.f32, cz: ti.f32, radius: ti.f32):
        """指定した球の範囲のSDFをえぐり取り、流体に変更する"""
        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            # 元々のSDF値
            current_sdf = self.ctx.sdf[i, j, k]
            
            # 熱源保護（Z座標が低い場所は絶対に削らないマスク処理）
            protected_z = 5.0  # 例: 下から5セルはベースプレート
            if float(k) > protected_z:
                # 1. 球のSDFを計算
                dx = float(i) - cx
                dy = float(j) - cy
                dz = float(k) - cz
                dist_to_center = ti.math.sqrt(dx*dx + dy*dy + dz*dz)
                sphere_sdf = dist_to_center - radius
                
                # 2. ブーリアン減算演算: new_sdf = max(current_sdf, -sphere_sdf)
                new_sdf = ti.math.max(current_sdf, -sphere_sdf)
                self.ctx.sdf[i, j, k] = new_sdf
                
                # 3. 物理プロパティの更新 (削られた部分は流体になる)
                if new_sdf > 0.0:
                    self.ctx.cell_id[i, j, k] = self.FLUID_ID
                    self.ctx.phi[i, j, k] = 0.0  # 完全な流体
                    # 必要に応じて u_solid などをリセット
                    self.ctx.u_solid[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    # =========================================================
    # 【案A拡張用】 カプセル（線分）による減算加工カーネル
    # =========================================================
    @ti.kernel
    def subtract_capsule(self, x1: ti.f32, y1: ti.f32, z1: ti.f32, 
                               x2: ti.f32, y2: ti.f32, z2: ti.f32, radius: ti.f32):
        """始点から終点までの線分と半径に基づくカプセル形状でえぐり取る"""
        # (線分とボクセルの最短距離を計算し、そこから radius を引いたものを sdf とする処理。実装略)
        pass

    # =========================================================
    # 【将来の案B用】 3D-CNN出力（密度テンソル）による一括更新
    # =========================================================
    @ti.kernel
    def update_from_density_tensor(self, density_field: ti.template()):
        """
        AIが出力した 0.0~1.0 の密度テンソルを受け取り、phi と sdf を一括更新する。
        density_field は PyTorch からゼロコピーで渡される Taichi フィールドを想定。
        """
        for i, j, k in ti.ndrange(self.ctx.nx, self.ctx.ny, self.ctx.nz):
            d_val = density_field[i, j, k]
            # 密度が 0.5 未満なら流体、それ以上なら固体とする（例）
            self.ctx.phi[i, j, k] = d_val
            if d_val < 0.5:
                self.ctx.cell_id[i, j, k] = self.FLUID_ID
                self.ctx.sdf[i, j, k] = 1.0 # 簡易的なSDF（流体内）
            else:
                self.ctx.cell_id[i, j, k] = self.SOLID_ID
                self.ctx.sdf[i, j, k] = -1.0 # 簡易的なSDF（固体内）