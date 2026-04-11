# ==========================================================
# d3q19.py — D3Q19・LBM定数（タスク3・タスク10: マジックナンバー削減）
# ==========================================================
import taichi as ti
import numpy as np

# タスク10: LBM で使う定数をここにまとめる
C_s2 = 1.0 / 3.0
C_s = 0.5773502691896257  # sqrt(1/3) （0.16から修正！）
SQRT2 = 1.4142135623730951


@ti.data_oriented
class D3Q19:
    def __init__(self, fp_dtype=None):
        if fp_dtype is None:
            import config
            fp_dtype = config.TI_FLOAT
        self._fp = fp_dtype
        self.C_s = C_s
        self.SQRT2 = SQRT2
        self.e = ti.Vector.field(3, ti.i32, shape=19)
        e_np = np.array(
            [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],
             [0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]], dtype=np.int32)
        self.e.from_numpy(e_np)

        self.inv_d = ti.field(ti.i32, shape=19)
        self.inv_d.from_numpy(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int32))

        np_fp = np.float32 if fp_dtype == ti.f32 else np.float16
        self.w = ti.field(fp_dtype, shape=19)
        self.w.from_numpy(np.array([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                                    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
                                    1/36, 1/36, 1/36, 1/36], dtype=np_fp))

        M_np = np.zeros((19, 19), dtype=np_fp)
        for i in range(19):
            x, y, z = float(e_np[i][0]), float(e_np[i][1]), float(e_np[i][2])
            n2 = x*x + y*y + z*z
            M_np[0, i]  = 1.0
            M_np[1, i]  = 19.0 * n2 - 30.0
            M_np[2, i]  = 21.0 * n2 * n2 - 53.0 * n2 + 24.0
            M_np[3, i]  = x
            M_np[4, i]  = (5.0 * n2 - 9.0) * x
            M_np[5, i]  = y
            M_np[6, i]  = (5.0 * n2 - 9.0) * y
            M_np[7, i]  = z
            M_np[8, i]  = (5.0 * n2 - 9.0) * z
            M_np[9, i]  = 3.0 * x*x - n2
            M_np[10, i] = (3.0 * n2 - 5.0) * (3.0 * x*x - n2)
            M_np[11, i] = y*y - z*z
            M_np[12, i] = (3.0 * n2 - 5.0) * (y*y - z*z)
            M_np[13, i] = x * y
            M_np[14, i] = y * z
            M_np[15, i] = z * x
            M_np[16, i] = (y*y - z*z) * x
            M_np[17, i] = (z*z - x*x) * y
            M_np[18, i] = (x*x - y*y) * z

        M_inv_np = np.linalg.inv(M_np.astype(np.float64)).astype(np_fp)
        self.M = ti.field(fp_dtype, shape=(19, 19))
        self.M_inv = ti.field(fp_dtype, shape=(19, 19))
        self.M.from_numpy(M_np)
        self.M_inv.from_numpy(M_inv_np)

        self.S = ti.field(fp_dtype, shape=19)
        self.S.from_numpy(np.array([
            0.0, 1.19, 1.4, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2,
            1.0, 1.4, 1.0, 1.4, 1.0, 1.0, 1.0, 1.98, 1.98, 1.98
        ], dtype=np_fp))

    @ti.func
    def get_feq(self, rho_val, v_val, d):
        eu = self.e[d].dot(v_val)
        uv = v_val.dot(v_val)
        return self.w[d] * rho_val * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*uv)

    @ti.func
    def get_geq(self, temp_val, v_val, d):
        return self.w[d] * temp_val * (1.0 + 3.0 * self.e[d].dot(v_val))
