# ==========================================================
# Geometry モジュール（タスク4: 形状・領域管理を一か所にまとめる）
# ctx.cell_id と ctx.sdf を更新するメソッドのみ。境界判定は位置に依存しない。
# ==========================================================
import taichi as ti
from context import FLUID_A, SOLID, INLET, OUTLET, ROTATING_WALL, SOLID_HEAT_SOURCE
from lbm_logger import get_logger

log = get_logger(__name__)


@ti.data_oriented
class GeometryBuilder:
    """形状構築では cell_id に ID を書き込み、入口・出口は set_inlet_outlet で上書きする。"""

    @staticmethod
    @ti.kernel
    def _build_pin_fin_sdf_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            sdf[i, j, k] = 100.0
            cell_id[i, j, k] = FLUID_A
            if 2 <= k < 18:
                x = float(i % 16) - 7.5
                y = float(j % 16) - 7.5
                radius = 4.0
                dist = ti.math.sqrt(x*x + y*y) - radius
                sdf[i, j, k] = dist
                if dist <= 0.0:
                    cell_id[i, j, k] = SOLID

    @staticmethod
    @ti.kernel
    def _build_validation_channel_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = FLUID_A
            sdf[i, j, k] = 100.0
            if i < 10 or i > (nx - 10):
                cell_id[i, j, k] = SOLID
                sdf[i, j, k] = 0.0
            # Z 境界: k=nz-1 を入口、k=0 を出口（側壁間の流体帯のみ。固体セルは上書きしない）
            elif k == nz - 1:
                cell_id[i, j, k] = INLET
                sdf[i, j, k] = 100.0
            elif k == 0:
                cell_id[i, j, k] = OUTLET
                sdf[i, j, k] = 100.0

    @staticmethod
    @ti.kernel
    def _build_karman_cylinder_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = FLUID_A
            sdf[i, j, k] = 100.0
            cx = nx * 0.5
            cz = nz * 0.8
            radius = nx * 0.1
            dist = ti.math.sqrt((float(i) - cx)**2 + (float(k) - cz)**2) - radius
            sdf[i, j, k] = dist
            if dist <= 0.0:
                cell_id[i, j, k] = SOLID

    @staticmethod
    @ti.kernel
    def _set_inlet_outlet_kernel(cell_id: ti.template(), nz: int):
        """入口・出口面の cell_id を INLET / OUTLET に上書き（座標範囲で指定）。"""
        for i, j in ti.ndrange(cell_id.shape[0], cell_id.shape[1]):
            cell_id[i, j, nz - 1] = INLET
            cell_id[i, j, 0] = OUTLET

    @staticmethod
    @ti.kernel
    def _build_parallel_plates_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = FLUID_A
            sdf[i, j, k] = 100.0
            # Y方向に上下壁を作成（壁の厚さは10セル）
            if j < 10 or j >= ny - 10:
                cell_id[i, j, k] = SOLID
                sdf[i, j, k] = 0.0

    @staticmethod
    @ti.kernel
    def _set_parallel_plates_inlet_outlet_kernel(cell_id: ti.template(), ny: int, nz: int):
        """平行平板の流体領域にのみ入口/出口IDを割り当てる。"""
        for i, j in ti.ndrange(cell_id.shape[0], cell_id.shape[1]):
            # if 10 <= j < ny - 10:
            #     cell_id[i, j, nz - 1] = INLET
            #     cell_id[i, j, 0] = OUTLET
            # ▼修正：元々が流体(0: FLUID_A)のセルだけを入口にする
            if cell_id[i, j, nz - 1] == 0: 
                cell_id[i, j, nz - 1] = 20 # INLET
                
            # ▼修正：元々が流体(0: FLUID_A)のセルだけを出口にする
            if cell_id[i, j, 0] == 0:      
                cell_id[i, j, 0] = 21      # OUTLET

    def build_parallel_plates(self, ctx):
        """[ベンチマーク1用] 平行平板チャネル (ポアズイユ流れ・熱伝達検証)"""
        self._build_parallel_plates_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)
        self._set_parallel_plates_inlet_outlet_kernel(ctx.cell_id, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_lid_driven_cavity_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = FLUID_A
            sdf[i, j, k] = 100.0
            
            is_wall = False
            # 左右の壁 (X方向)
            if i == 0 or i == nx - 1: is_wall = True
            # 底面の壁 (Z=0)
            if k == 0: is_wall = True
            
            if k == nz - 1:
                # 上面 (移動壁)。速度境界を設定するためカスタムID(20)を付与
                cell_id[i, j, k] = 20
                sdf[i, j, k] = 0.0
            elif is_wall:
                # 静止壁。カスタムID(21)を付与
                cell_id[i, j, k] = 21
                sdf[i, j, k] = 0.0

    def build_lid_driven_cavity(self, ctx):
        """[ベンチマーク2用] リッド・ドリブン・キャビティ (Ghia et al. 速度プロファイル検証)"""
        self._build_lid_driven_cavity_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_benchmark_cylinder_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = FLUID_A
            sdf[i, j, k] = 100.0
            
            # 円柱の配置 (Xの中央、Zの上流側3/4)
            # ※対称性を微小に崩す(+0.1)ことで、物理的に正しいカルマン渦の誘発を早める
            cx = float(nx) * 0.5 + 0.1
            cz = float(nz) * 0.75
            
            # ブロック率 10% (直径が幅の1/10。境界の干渉を防ぐための標準的な比率)
            radius = float(nx) * 0.05 
            
            dist = ti.math.sqrt((float(i) - cx)**2 + (float(k) - cz)**2) - radius
            sdf[i, j, k] = dist
            if dist <= 0.0:
                cell_id[i, j, k] = SOLID

    @staticmethod
    @ti.kernel
    def _build_tube_heat_exchanger_kernel(
        cell_id: ti.template(), sdf: ti.template(), 
        nx: int, ny: int, nz: int, 
        flow_type: int
    ):
        # カスタムIDの定義
        INLET_IN = 22   # 内部流体の入口
        OUTLET_IN = 23  # 内部流体の出口
        INLET_OUT = 24  # 外部流体の入口
        OUTLET_OUT = 25 # 外部流体の出口
        SOLID = 10
        FLUID_A = 0

        cx = float(nx) * 0.5
        cy = float(ny) * 0.5
        
        # 内部チューブの半径と厚み
        r_inner = float(nx) * 0.2
        thickness = 2.0
        r_outer = r_inner + thickness

        for i, j, k in cell_id:
            dx = float(i) - cx
            dy = float(j) - cy
            r = ti.math.sqrt(dx*dx + dy*dy)

            # デフォルトは流体
            cell_id[i, j, k] = FLUID_A
            sdf[i, j, k] = 100.0

            
            # 2. チューブ壁（固体）
            if r >= r_inner and r < r_outer:
                cell_id[i, j, k] = SOLID
                sdf[i, j, k] = 0.0
            
                """
            # 1. 外部ケーシング壁（外枠）
            elif i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2:
                cell_id[i, j, k] = SOLID
                sdf[i, j, k] = 0.0
                """
            
            # 3. 入口・出口の割り当て
            else:
                if r < r_inner:
                    # --- 内部流体 (ベースID: 0) ---
                    if flow_type == 1:  # 向流
                        if k == 0: cell_id[i, j, k] = 22
                        elif k == nz - 1: cell_id[i, j, k] = 23
                        else: cell_id[i, j, k] = 0
                    else:  # 並流
                        if k == nz - 1: cell_id[i, j, k] = 22
                        elif k == 0: cell_id[i, j, k] = 23
                        else: cell_id[i, j, k] = 0
                elif r >= r_outer:
                    # --- 外部流体 (ベースID: 2 ★ここを変更★) ---
                    if k == nz - 1: cell_id[i, j, k] = 24
                    elif k == 0: cell_id[i, j, k] = 25
                    else: cell_id[i, j, k] = 2  # <== ここが重要！

    @staticmethod
    @ti.kernel
    def _build_rotating_cylinder_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = FLUID_A # FLUID_A
            sdf[i, j, k] = 100.0
            
            # 円柱の配置 (Xの中央、Zの上流側3/4)
            cx = float(nx) * 0.5 + 0.1
            cz = float(nz) * 0.75
            radius = float(nx) * 0.05 
            
            dist = ti.math.sqrt((float(i) - cx)**2 + (float(k) - cz)**2) - radius
            sdf[i, j, k] = dist
            if dist <= 0.0:
                cell_id[i, j, k] = ROTATING_WALL # ROTATING_WALL


    @staticmethod
    @ti.kernel
    def _build_rotating_hollow_cylinder_kernel(cell_id: ti.template(), sdf: ti.template(), 
                                               nx: int, ny: int, nz: int):
        cx = float(nx) * 0.5 + 0.1
        cy = float(ny) * 0.5
        
        # 円筒の定義パラメータ（厚みを増やして安定化）
        radius = float(nx) * 0.3
        thickness = 5.0  # ★ 2.0 から 5.0 に変更
        r_in = radius - thickness
        r_out = radius
        
        # Z方向の範囲設定
        z_start = float(nz) * 0.2
        z_end = z_start + (float(nz) * 0.3)
        
        for i, j, k in cell_id:
            # 座標を浮動小数点に変換
            x, y, z = float(i), float(j), float(k)
            
            # --- 1. 中空円筒のSDF計算 ---
            dist_xy = ti.math.sqrt((x - cx)**2 + (y - cy)**2)
            
            d_r = ti.math.max(dist_xy - r_out, r_in - dist_xy)
            d_z = ti.math.max(z_start - z, z - z_end)
            
            # ▼ 変更：角の外側ではピタゴラスの定理を使い、内部では max を使う「完全なユークリッド距離」
            dist_cyl = ti.math.sqrt(ti.math.max(d_r, 0.0)**2 + ti.math.max(d_z, 0.0)**2) + \
                       ti.math.min(ti.math.max(d_r, d_z), 0.0)
            

            
            # --- 3. セルIDとSDFの割り当て ---
            # 円筒と外壁、近い方の距離をSDFとして保存
            sdf[i, j, k] = dist_cyl
            
            # 初期値は流体
            cell_id[i, j, k] = FLUID_A
            
            if dist_cyl <= 0.0:
                cell_id[i, j, k] = SOLID  # 30: 中空円筒（回転壁）
            else:

                # 空間の内部（流体）の場合のみ、入口と出口を設定
                if k == 0:
                    cell_id[i, j, k] = INLET
                elif k == nz - 1:
                    cell_id[i, j, k] = OUTLET

    @staticmethod
    @ti.kernel
    def _build_ai_training_rotating_hollow_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        cx = float(nx) * 0.5 + 0.1
        cy = float(ny) * 0.5
        radius = float(nx) * 0.3
        thickness = 5.0
        r_in = radius - thickness
        r_out = radius
        z_start = float(nz) * 0.7
        z_end = float(nz) * 0.8
        
        for i, j, k in cell_id:
            x, y, z = float(i), float(j), float(k)
            
            dist_xy = ti.math.sqrt((x - cx)**2 + (y - cy)**2)
            d_r = ti.math.max(dist_xy - r_out, r_in - dist_xy)
            d_z = ti.math.max(z_start - z, z - z_end)
            
            dist_cyl = ti.math.sqrt(ti.math.max(d_r, 0.0)**2 + ti.math.max(d_z, 0.0)**2) + \
                       ti.math.min(ti.math.max(d_r, d_z), 0.0)
            
            sdf[i, j, k] = dist_cyl
            
            if dist_cyl <= 0.0:
                cell_id[i, j, k] = 30 # ROTATING_WALL
            else:
                cell_id[i, j, k] = 0  # FLUID_A
                
            if dist_cyl > 0.0:
                # 1. まず、側面(X,Yの端) と 底面(Z=0) をすべてOUTLETの「殻」にする
                if k == 0 or i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    cell_id[i, j, k] = 21 # OUTLET
                    
                # 2. その「殻」のさらに内側にある天井(Z=nz-1)だけをINLETにする
                elif k == nz - 1:
                    cell_id[i, j, k] = 20 # INLET

    def build_ai_training_rotating_hollow(self, ctx):
        """[AI事前学習用] 開放系・Z軸回転の中空円筒 (カウル・ドローンローター周囲の学習)"""
        self._build_ai_training_rotating_hollow_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    def build_rotating_hollow_cylinder(self, ctx):
        """
        中空円筒体形状を構築するラッパーメソッド
        ctxには cell_id, sdf, nx, ny, nz が含まれていることを想定
        """
        self._build_rotating_hollow_cylinder_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    def build_rotating_cylinder(self, ctx):
        self._build_rotating_cylinder_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)
        self.set_inlet_outlet(ctx)


    def build_tube_heat_exchanger(self, ctx, flow_type_str="counter"):
        """二重管熱交換器形状の構築。flow_type_str に 'counter' か 'co-current' を指定。"""
        flow_type = 1 if flow_type_str == "counter" else 0
        self._build_tube_heat_exchanger_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz, flow_type)

    def build_benchmark_cylinder(self, ctx):
        """[ベンチマーク3用] 円柱周り流れ (カルマン渦 St数・剥離熱伝達検証)"""
        self._build_benchmark_cylinder_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    def build_pin_fin_sdf(self, ctx):
        self._build_pin_fin_sdf_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    def build_validation_channel(self, ctx):
        """検証用チャネル: 左右 X 壁（固体）と Z=nz-1 入口・Z=0 出口を cell_id / sdf に書き込む。"""
        log.debug("Geometry: validation_channel (%s x %s x %s)", ctx.nx, ctx.ny, ctx.nz)
        self._build_validation_channel_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    def build_karman_cylinder(self, ctx):
        """カルマン渦観測用の円柱形状を ctx の cell_id / sdf に書き込む。"""
        self._build_karman_cylinder_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    def set_inlet_outlet(self, ctx, inlet_box=None, outlet_box=None):
        """
        入口・出口の cell_id を上書きする。
        inlet_box / outlet_box 未指定時は従来どおり z=nz-1 を INLET、z=0 を OUTLET にする。
        """
        self._set_inlet_outlet_kernel(ctx.cell_id, ctx.nz)

    def set_heat_source(self, ctx, target_surface_box):
        """指定範囲の固体セルを熱源面(ID=SOLID_HEAT_SOURCE)に上書き。将来拡張用。"""
        pass

    def load_stl(self, ctx, filename: str):
        """STL からボクセル化し ID マップに書き込む。将来拡張用。"""
        pass

    @staticmethod
    @ti.kernel
    def _build_ai_training_cylinder_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        
        # 形状パラメータ
        wall_thickness = 5.0
        cx = float(nx) * 0.5 + 0.1
        cz = float(nz) * 0.8
        radius = 16.0

        for i, j, k in cell_id:
            # ==========================================================
            # 1. 厳密な SDF (Signed Distance Field) の計算
            # ==========================================================
            
            # 左壁 (X = 5 の境界) までの符号付き距離
            dist_wall_left = float(i) - wall_thickness
            
            # 右壁 (X = nx - 5 の境界) までの符号付き距離
            dist_wall_right = float(nx) - wall_thickness - float(i)
            
            # 両壁までの距離のうち、近い方を採用 (和集合)
            dist_walls = ti.math.min(dist_wall_left, dist_wall_right)
            
            # 円柱表面までの符号付き距離
            dist_center = ti.math.sqrt((float(i) - cx)**2 + (float(k) - cz)**2)
            dist_cyl = dist_center - radius
            
            # 空間全体での最終的なSDF (すべての物体の中で最も近い距離)
            final_sdf = ti.math.min(dist_walls, dist_cyl)
            sdf[i, j, k] = final_sdf
            
            # ==========================================================
            # 2. セルIDの割り当て (SDFに基づく完全な判定)
            # ==========================================================
            if final_sdf <= 0.0:
                cell_id[i, j, k] = SOLID # SOLID
            else:
                cell_id[i, j, k] = FLUID_A # FLUID_A
                
            # ==========================================================
            # 3. 入口・出口の上書き (SDF > 0 の流体領域のみ)
            # ==========================================================
            if final_sdf > 0.0:
                if k == nz - 1:
                    cell_id[i, j, k] = INLET # INLET
                elif k == 0:
                    cell_id[i, j, k] = OUTLET # OUTLET

    def build_ai_training_01(self, ctx):
        """[AI事前学習用] 壁面境界層＋円柱後流の最強ハイブリッドデータ生成領域"""
        self._build_ai_training_cylinder_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_ai_training_backstep_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        
        wall_thickness = 5
        step_z = float(nz) * 0.7  # Zの入口から30%進んだところで突然段差が下がる
        step_x = float(nx) * 0.5  # Xの半分を塞ぐ巨大な段差

        for i, j, k in cell_id:
            x = float(i)
            z = float(k)
            
            # 1. 右壁からの距離 (常に直線)
            d_right = float(nx) - wall_thickness - x
            
            # 2. 左壁（L字型の段差）からの正確なSDF距離計算
            d_left = 1000.0
            if z >= step_z:
                # 段差より上流: 狭い流路 (x = step_x が壁)
                d_left = x - step_x
            else:
                # 段差より下流: 広い流路
                if x >= step_x:
                    # 流路の右寄り: 最も近い左の障害物は「段差の角(corner)」または「左の壁」
                    d_corner = ti.math.sqrt((x - step_x)**2 + (step_z - z)**2)
                    d_wall = x - wall_thickness
                    d_left = ti.math.min(d_corner, d_wall)
                else:
                    # 段差の陰 (凹み部分): 最も近いのは「左の壁」か「天井(段差の裏)」
                    d_wall = x - wall_thickness
                    d_ceil = step_z - z
                    d_left = ti.math.min(d_wall, d_ceil)
            
            # 空間全体での最終的なSDF
            # final_sdf = ti.math.min(d_left, d_right)
            final_sdf = d_left
            sdf[i, j, k] = final_sdf
            
            # 3. セルIDの割り当て
            if final_sdf <= 0.0:
                cell_id[i, j, k] = SOLID # SOLID
            else:
                cell_id[i, j, k] = FLUID_A  # FLUID_A
                
            # 4. 入口・出口の上書き
            if final_sdf > 0.0:
                if k == nz - 1:
                    cell_id[i, j, k] = INLET # INLET
                elif k == 0:
                    cell_id[i, j, k] = OUTLET # OUTLET

    def build_ai_training_backstep(self, ctx):
        """[AI事前学習用] バックステップ流れ (剥離・再循環・再付着の学習)"""
        self._build_ai_training_backstep_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_ai_training_rotating_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        
        # 円柱の配置 (中心と半径)
        cx = float(nx) * 0.5 + 0.1
        cz = float(nz) * 0.7  # 上流側
        radius = 24.0         # 直径48セルの巨大な回転円柱

        for i, j, k in cell_id:
            x = float(i)
            z = float(k)
            
            # 円柱中心からの距離
            dist_center = ti.math.sqrt((x - cx)**2 + (z - cz)**2)
            dist_cyl = dist_center - radius
            
            sdf[i, j, k] = dist_cyl
            
            # 30: ROTATING_WALL を割り当てる
            if dist_cyl <= 0.0:
                cell_id[i, j, k] = ROTATING_WALL # ROTATING_WALL
            else:
                cell_id[i, j, k] = FLUID_A  # FLUID_A
                
            if dist_cyl > 0.0:
                if k == nz - 1:
                    cell_id[i, j, k] = INLET # INLET
                elif k == 0:
                    cell_id[i, j, k] = OUTLET # OUTLET

    def build_ai_training_rotating(self, ctx):
        """[AI事前学習用] 回転円柱 (マグヌス効果・せん断・遠心力の学習)"""
        self._build_ai_training_rotating_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_ai_training_thermal_baffles_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        
        wall_thickness = 5.0
        
        # バッフルの設定
        baffle_width = float(nx) * 0.5  # 流路の半分まで張り出す
        baffle_thickness = 10.0
        
        z_baffle_1 = float(nz) * 0.65  # 上流側の邪魔板（左から）
        z_baffle_2 = float(nz) * 0.35  # 下流側の邪魔板（右から）

        for i, j, k in cell_id:
            x = float(i)
            z = float(k)
            
            # 左壁 (ID: 11 / SOLID_HEAT_SOURCE とする)
            d_left_wall = x - wall_thickness
            
            # 右壁 (ID: 10 / SOLID とする)
            d_right_wall = float(nx) - wall_thickness - x
            
            # バッフル1 (左から張り出す)
            d_b1 = 1000.0
            if z >= z_baffle_1 - baffle_thickness and z <= z_baffle_1 + baffle_thickness:
                d_b1 = x - baffle_width
            
            # バッフル2 (右から張り出す)
            d_b2 = 1000.0
            if z >= z_baffle_2 - baffle_thickness and z <= z_baffle_2 + baffle_thickness:
                d_b2 = float(nx) - baffle_width - x
                
            # 全ての物体のうち最も近い距離
            final_sdf = ti.math.min(d_left_wall, d_right_wall)
            final_sdf = ti.math.min(final_sdf, d_b1)
            final_sdf = ti.math.min(final_sdf, d_b2)
            
            sdf[i, j, k] = final_sdf
            
            # --- IDの割り当て ---
            if final_sdf <= 0.0:
                # 左壁とバッフル1は熱源 (ID=11)
                if x < float(nx) * 0.5 and (x <= wall_thickness or d_b1 <= 0.0):
                    cell_id[i, j, k] = SOLID_HEAT_SOURCE # SOLID_HEAT_SOURCE
                # 右壁とバッフル2は冷壁 (ID=10)
                else:
                    cell_id[i, j, k] = SOLID # SOLID
            else:
                cell_id[i, j, k] = FLUID_A  # FLUID_A
                
            # 入口・出口の上書き
            if final_sdf > 0.0:
                if k == nz - 1:
                    cell_id[i, j, k] = INLET # INLET
                elif k == 0:
                    cell_id[i, j, k] = OUTLET # OUTLET

    def build_ai_training_thermal_baffles(self, ctx):
        """[AI事前学習用] 非対称加熱バッフル (激しい温度勾配と熱剥離の学習)"""
        self._build_ai_training_thermal_baffles_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_ai_training_inclined_plate_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        cx = float(nx) * 0.5
        cz = float(nz) * 0.8
        
        # 板の長さ60セル、厚み6セルの薄い板
        length = 60.0
        thickness = 6.0
        
        # 迎角 (Attack Angle) = 30度 (約0.523598ラジアン)
        theta = 0.523598
        cos_t = ti.math.cos(theta)
        sin_t = ti.math.sin(theta)

        for i, j, k in cell_id:
            # 局所座標系への回転
            x = float(i) - cx
            z = float(k) - cz
            rx = x * cos_t - z * sin_t
            rz = x * sin_t + z * cos_t
            
            # 矩形の厳密なSDF計算
            dx = ti.abs(rx) - length * 0.5
            dz = ti.abs(rz) - thickness * 0.5
            dist = ti.math.sqrt(ti.math.max(dx, 0.0)**2 + ti.math.max(dz, 0.0)**2) + \
                   ti.math.min(ti.math.max(dx, dz), 0.0)
            
            sdf[i, j, k] = dist
            if dist <= 0.0:
                cell_id[i, j, k] = 10 # SOLID
            else:
                cell_id[i, j, k] = 0  # FLUID_A
                
            if dist > 0.0:
                if k == nz - 1:
                    cell_id[i, j, k] = 20 # INLET
                elif k == 0:
                    cell_id[i, j, k] = 21 # OUTLET

    def build_ai_training_inclined_plate(self, ctx):
        """[AI事前学習用] 傾いた平板 (迎角剥離・翼端渦の学習)"""
        self._build_ai_training_inclined_plate_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_ai_training_mixed_convection_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        wall_thickness = 5.0
        for i, j, k in cell_id:
            x = float(i)
            y = float(j)
            
            # 左壁 (熱源: 11) と 右壁 (冷源: 10)
            d_left = x - wall_thickness
            d_right = float(nx) - wall_thickness - x
            
            # ★追加：手前の壁 と 奥の壁 (断熱壁: 10 とする)
            d_front = y - wall_thickness
            d_back = float(ny) - wall_thickness - y
            
            # X方向とY方向の壁のうち、最も近いものをSDFにする
            d_x = ti.math.min(d_left, d_right)
            d_y = ti.math.min(d_front, d_back)
            final_sdf = ti.math.min(d_x, d_y)
            
            sdf[i, j, k] = final_sdf
            
            if final_sdf <= 0.0:
                # 左壁だけを熱源 (11) にする
                if x <= wall_thickness + 0.1:
                    cell_id[i, j, k] = 11 # SOLID_HEAT_SOURCE
                else:
                    cell_id[i, j, k] = 10 # SOLID
            else:
                cell_id[i, j, k] = 0  # FLUID_A
                
            # 入口・出口の上書き（流体領域のみ）
            if final_sdf > 0.0:
                if k == nz - 1:
                    cell_id[i, j, k] = 20 # INLET
                elif k == 0:
                    cell_id[i, j, k] = 21 # OUTLET

    def build_ai_training_mixed_convection(self, ctx):
        """[AI事前学習用] 混合対流 (浮力とサーマルプルームの学習)"""
        self._build_ai_training_mixed_convection_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

    @staticmethod
    @ti.kernel
    def _build_thermal_cavity_kernel(cell_id: ti.template(), sdf: ti.template(), nx: int, ny: int, nz: int):
        for i, j, k in cell_id:
            cell_id[i, j, k] = 0  # FLUID_A
            sdf[i, j, k] = 100.0
            
            # 左壁 (ID: 11 / 熱源)
            if i == 0:
                cell_id[i, j, k] = 11
                sdf[i, j, k] = 0.0
            # 右壁 (ID: 10 / 冷源)
            elif i == nx - 1:
                cell_id[i, j, k] = 10
                sdf[i, j, k] = 0.0
            # 上下の壁 (ID: 21 / 断熱壁)
            elif k == 0 or k == nz - 1:
                cell_id[i, j, k] = 21
                sdf[i, j, k] = 0.0

    def build_thermal_cavity(self, ctx):
        """[ベンチマーク用] 差温キャビティ (自然対流と浮力の検証)"""
        self._build_thermal_cavity_kernel(ctx.cell_id, ctx.sdf, ctx.nx, ctx.ny, ctx.nz)

# geometry.py の GeometryBuilder クラス内のどこかに追加
    def build_empty_domain(self, ctx):
        """[AI・IBM検証用] 障害物のない空の空間（境界条件や周期境界用）"""
        self._set_inlet_outlet_kernel(ctx.cell_id, ctx.nz)