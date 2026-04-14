# ==========================================================
# セルID定数 & SimulationContext（タスク1・タスク2）
# タスク6: tau_f_table / tau_g_table で ID → 物性 を保持
# ==========================================================
import taichi as ti
from lbm_logger import get_logger

log = get_logger(__name__)

# --- セルID（タグ）マップ用定数 ---
FLUID_A = 0
SOLID = 10
SOLID_HEAT_SOURCE = 11
INLET = 20
OUTLET = 21
ROTATING_WALL = 30 

# 物性テーブルの最大 ID 数（cell_id でインデックスするため 32 で十分）
MAX_CELL_ID = 32


class SimulationContext:
    """
    タスク1: すべてのデータを一つの「箱」にまとめる共通インターフェース。
    タスク6: materials_dict を反映する tau_f_table / tau_g_table を保持。
    """

    def __init__(self, nx: int, ny: int, nz: int, n_particles: int, fp_dtype=None):
        if fp_dtype is None:
            import config
            fp_dtype = config.TI_FLOAT
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n_particles = n_particles


        self.cell_id = ti.field(dtype=ti.i32)
        self.sdf = ti.field(dtype=fp_dtype)
        self.phi = ti.field(dtype=fp_dtype)
        self.closest_obj_id = ti.field(dtype=ti.i32)
        self.rho = ti.field(dtype=fp_dtype)
        self.temp = ti.field(dtype=fp_dtype)
        self.psi = ti.field(dtype=fp_dtype)
        self.S_g = ti.field(dtype=fp_dtype)
        
        self.v = ti.Vector.field(3, fp_dtype)
        self.u_solid = ti.Vector.field(3, fp_dtype)
        self.F_int = ti.Vector.field(3, fp_dtype)
        
        # 突破策①：f_post と g_post を廃止し、old と new だけにする
        self.f_old = ti.field(fp_dtype)
        self.f_new = ti.field(fp_dtype)
        self.g_old = ti.field(fp_dtype)
        self.g_new = ti.field(fp_dtype)
        
        # ID → 物性テーブル (これだけは常にDense)
        self.tau_f_table = ti.field(fp_dtype, shape=MAX_CELL_ID)
        self.tau_g_table = ti.field(fp_dtype, shape=MAX_CELL_ID)

        # ▼ 追加：流体か固体かを判定するフラグテーブル (0:固体, 1:流体)
        self.is_fluid_table = ti.field(ti.i32, shape=MAX_CELL_ID)

        # 温度 g の移流: 固体かつ等温壁のみ ABB（Tw は boundary_conditions 由来）
        # g_wall_use_abb[cid]==1 のとき g_wall_tw[cid] を壁温度として使用。0 のときは BB（断熱相当）。
        self.g_wall_use_abb = ti.field(ti.i32, shape=MAX_CELL_ID)
        self.g_wall_tw = ti.field(dtype=fp_dtype, shape=MAX_CELL_ID)
        
        self.particle_pos = ti.Vector.field(3, fp_dtype, shape=n_particles)
        self.img = ti.field(fp_dtype, shape=(nx * 2, ny))
        self.inject_count = ti.field(ti.i32, shape=())

        # ==========================================================
        # 修正：Sparse(pointer) をやめ、安全な Dense(密) レイアウトに戻す。
        # ただし、変数をまとめて配置することでメモリアクセス速度を向上させます。
        # ==========================================================
        grid = ti.root.dense(ti.ijk, (nx, ny, nz))
        
        # マクロ変数をまとめて配置
        grid.place(self.cell_id, self.sdf, self.phi, self.closest_obj_id, self.rho, self.temp, self.psi, self.S_g)
        grid.place(self.v, self.u_solid, self.F_int)
        
        # 分布関数は方向(19)の次元を追加して配置
        dirs = grid.dense(ti.l, 19)
        dirs.place(self.f_old, self.f_new)
        dirs.place(self.g_old, self.g_new)

        self._init_g_wall_tables_zero()

        log.debug("SimulationContext: grid %s x %s x %s", nx, ny, nz)

    def _init_g_wall_tables_zero(self):
        import numpy as np
        z = np.zeros(MAX_CELL_ID, dtype=np.int32)
        self.g_wall_use_abb.from_numpy(z)
        zf = np.zeros(MAX_CELL_ID, dtype=np.float32)
        self.g_wall_tw.from_numpy(zf)

    def set_materials(self, materials_dict):
        """タスク7: ID → (tau_f, tau_g) のマッピングをテーブルに反映する。"""
        for cid, (tau_f, tau_g, is_fluid_flag) in materials_dict.items():
            if 0 <= cid < MAX_CELL_ID:
                self.tau_f_table[cid] = tau_f
                self.tau_g_table[cid] = tau_g
                self.is_fluid_table[cid] = is_fluid_flag # ★追加

    def set_g_thermal_wall_tables_from_config(self, sim_config):
        """boundary_conditions の isothermal_wall に合わせて g 移流用 ABB テーブルを設定する。"""
        import numpy as np

        use_abb = np.zeros(MAX_CELL_ID, dtype=np.int32)
        tw = np.zeros(MAX_CELL_ID, dtype=np.float32)

        bcs = getattr(sim_config, "boundary_conditions", None) or {}
        for cid, bc_info in bcs.items():
            ic = int(cid)
            if ic < 0 or ic >= MAX_CELL_ID:
                continue
            if bc_info.get("type") == "isothermal_wall":
                use_abb[ic] = 1
                tw[ic] = float(bc_info.get("temperature", 1.0))

        self.g_wall_use_abb.from_numpy(use_abb)
        if sim_config.fp_dtype == ti.f16:
            self.g_wall_tw.from_numpy(tw.astype(np.float16))
        else:
            self.g_wall_tw.from_numpy(tw.astype(np.float32))