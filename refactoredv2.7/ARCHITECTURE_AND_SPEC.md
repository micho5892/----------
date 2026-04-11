# refactoredv2.2 全体機能・モジュール仕様書

LBM（格子ボルツマン法）による熱流体シミュレーションのリファクタ版「refactoredv2.2」の全体機能と、各モジュールの仕様を具体的なコードとともに説明する文書です。

---

## 1. 全体の機能概要

### 1.1 目的

- **LBM（D3Q19）** による 3 次元の**非等温流体**シミュレーション
- **データ主導のモジュール化**：全データを `SimulationContext (ctx)` に集約し、形状・境界・ソルバ・解析・出力を分離
- **ID ベースの境界管理**：セル種別（流体・固体・入口・出口・熱源）を `cell_id` で判定し、位置に依存しない境界処理
- **ベンチマーク対応**：平行平板・リッドドリブンキャビティ・円柱周り流れの 3 種類を同一インターフェースで実行

### 1.2 処理の大まかな流れ

```
1. run_simulation(**kwargs) で benchmark と時間・定常パラメータを取得
2. SimConfig(**kwargs) で設定構築（格子・物性・境界・n_particles 等）
3. SimulationContext(nx, ny, nz, n_particles) で Taichi field を確保
4. ctx.set_materials(cfg.get_materials_dict()) で物性テーブルを反映
5. GeometryBuilder で形状・入口出口を cell_id / sdf に設定
6. LBMSimulator(cfg), BoundaryConditionManager(d3q19, cfg), Analytics(d3q19, cfg) を生成
7. sim.init_fields(ctx) で rho/v/temp/f_old/g_old を初期化
8. SteadyStateDetector を生成（定常判定用）
9. メインループ（物理時間ベース）:
   while current_time_p < max_time_p:
     collide → stream_and_bc → 速度 BC（ramp_factor 適用）→ 出口 BC → update_macro → 熱流束 BC → move_particles
     vis_interval ごとにモニタ値・ETA・GIF フレームを記録
     定常検知後 extra 時間経過で break
10. 終了後: メタデータ(JSON)・フィールド(.npz)・最終 VTI を保存
11. export_gif_frames で GIF 出力
```

**重要**：形状設定 → 物性反映 → `init_fields` の順序を変えないこと。`init_fields` は `ctx.cell_id` を見て固体/流体の初期温度を切り分けている。

---

## 2. ファイル構成と依存関係

| ファイル | 役割 | 主な依存 |
|----------|------|----------|
| **context.py** | セルID定数・SimulationContext（全 field の確保） | taichi |
| **config.py** | SimConfig・物性・境界条件の ID ベース管理 | context |
| **d3q19.py** | D3Q19 格子・LBM 定数（get_feq / get_geq） | なし |
| **geometry.py** | GeometryBuilder（形状・入口出口の cell_id / sdf 更新） | context |
| **boundary.py** | BoundaryConditionManager（速度・出口・熱流束の境界） | context |
| **solver.py** | LBMSimulator（collide / stream / update_macro / move_particles） | context, d3q19 |
| **analytics.py** | Analytics（Nu・平均温度・局所値など ctx ベースの解析） | context |
| **export.py** | GIF 用フレーム構築・保存 | context |
| **vtk_export.py** | VTI 出力（ParaView 連携） | なし（ctx を引数で受け取る） |
| **diagnostics.py** | 数値暴走切り分け用ログ（rho/temp/v/f_old/g_old）。main からはオプションで import | context |
| **main.py** | run_simulation・物理時間ループ・定常検知(SteadyStateDetector)・ETA・最終保存(JSON/npz/VTI)・GIF | 上記すべて |

---

## 3. 各モジュールの仕様とコード

### 3.1 context.py — セルID定数と SimulationContext

**役割**：シミュレーションで使う「セル種別」の定数と、すべての Taichi field を保持する `SimulationContext` を提供する。

#### セルID定数

```python
# context.py より
FLUID_A = 0
SOLID = 10
SOLID_HEAT_SOURCE = 11
INLET = 20
OUTLET = 21
MAX_CELL_ID = 32  # tau_f_table / tau_g_table のサイズ
```

境界判定は **cell_id の値のみ**で行い、座標（例：k）には依存しない。

#### SimulationContext のフィールド

```python
class SimulationContext:
    def __init__(self, nx: int, ny: int, nz: int, n_particles: int):
        # 形状・境界用
        self.cell_id = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.sdf = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        # ID → 物性（collide で tau_f, tau_g を参照）
        self.tau_f_table = ti.field(ti.f32, shape=MAX_CELL_ID)
        self.tau_g_table = ti.field(ti.f32, shape=MAX_CELL_ID)
        # LBM 分布関数・マクロ量
        self.f_old, self.f_post, self.f_new = ...  # shape (nx, ny, nz, 19)
        self.g_old, self.g_post, self.g_new = ...
        self.v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))
        self.rho = ti.field(ti.f32, shape=(nx, ny, nz))
        self.temp = ti.field(ti.f32, shape=(nx, ny, nz))
        # 可視化・パーティクル
        self.particle_pos = ti.Vector.field(3, ti.f32, shape=n_particles)
        self.img = ti.field(ti.f32, shape=(nx * 2, ny))
        self.inject_count = ti.field(ti.i32, shape=())  # パーティクル注入カウント
```

#### 物性テーブルの反映

```python
def set_materials(self, materials_dict):
    """ID → (tau_f, tau_g) のマッピングをテーブルに反映する。"""
    for cid, (tau_f, tau_g) in materials_dict.items():
        if 0 <= cid < MAX_CELL_ID:
            self.tau_f_table[cid] = tau_f
            self.tau_g_table[cid] = tau_g
```

`materials_dict` は `config.get_materials_dict()` で取得し、**形状設定の後・init_fields の前**に一度だけ `ctx.set_materials(...)` を呼ぶ。

---

### 3.2 config.py — SimConfig と境界の ID 管理

**役割**：格子サイズ・物理パラメータ・時間刻み・境界条件（速度・出口・熱流束）を一元管理する。境界は **セルID をキー**にした辞書で指定する。

#### 主要な設定項目（抜粋）

```python
class SimConfig:
    def __init__(self, **kwargs):
        self.nx = kwargs.get('nx', 128)
        self.ny, self.nz = ...
        self.Lx_p = kwargs.get('Lx_p', 0.1)
        self.U_inlet_p = kwargs.get('U_inlet_p', 1.0)
        self.T_inlet_p, self.T_wall_p, self.q_wall_p = ...
        self.bc_type = kwargs.get('bc_type', 'T')
        self.Pr, self.nu_p, self.k_p, self.rho_p, self.Cp_p = ...
        self.dx = self.Lx_p / self.nx
        # tau_f, tau_g は nu_p / k_p 等から計算、または kwargs で直接指定
        self.n_particles = kwargs.get('n_particles', 10000)
        self.steps = kwargs.get('steps', 600)       # 参考値（メインループは物理時間で制御）
        self.vis_interval, self.filename = ...
        self.particles_inject_per_step = ...
        self.bc_velocity_by_id = { ... }            # キーは ti.Vector に変換して保持
        self.bc_heat_flux_by_id = { ... }
        self.bc_outlet_ids = kwargs.get('bc_outlet_ids', [OUTLET])
        self.vti_export_interval, self.vti_path_template = ...
```

**run_simulation 専用の引数**（main 側で pop され、SimConfig には渡らない）:  
`max_time_p`, `ramp_time_p`, `steady_window_p`, `steady_tolerance`, `steady_extra_p`, `out_dir`。これらは物理時間ベースのループと定常検知に使う。

#### 物性辞書の取得

```python
def get_materials_dict(self):
    """ID → (tau_f, tau_g) のマッピングを返す。"""
    return {
        FLUID_A: (self.tau_f, self.tau_g),
        INLET: (self.tau_f, self.tau_g),
        OUTLET: (self.tau_f, self.tau_g),
        20: (self.tau_f, self.tau_g),  # カスタムID用
        21: (self.tau_f, self.tau_g),
    }
```

流体系のセル（FLUID_A, INLET, OUTLET 等）はすべてこの辞書で `tau_f`, `tau_g` が定義されている必要がある。

---

### 3.3 d3q19.py — D3Q19 格子と LBM 定数

**役割**：19 方向の格子ベクトル `e`、重み `w`、逆方向 `inv_d`、MRT 用の `M`, `M_inv`, `S`、およびカーネル内で使う **平衡分布** `get_feq` / `get_geq` を提供する。他モジュールに依存しない。

#### 定数と格子ベクトル

```python
C_s = 0.16   # 音速比（MRT の Pi スケール用）
SQRT2 = 1.4142135623730951

@ti.data_oriented
class D3Q19:
    def __init__(self):
        self.e = ti.Vector.field(3, ti.i32, shape=19)   # 方向ベクトル
        self.inv_d = ti.field(ti.i32, shape=19)         # 逆方向インデックス
        self.w = ti.field(ti.f32, shape=19)             # 重み
        self.M, self.M_inv = ...                       # MRT 変換（未使用でも保持）
        self.S = ...
```

#### 平衡分布（カーネル内で使用）

```python
@ti.func
def get_feq(self, rho_val, v_val, d):
    eu = self.e[d].dot(v_val)
    uv = v_val.dot(v_val)
    return self.w[d] * rho_val * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*uv)

@ti.func
def get_geq(self, temp_val, v_val, d):
    return self.w[d] * temp_val * (1.0 + 3.0 * self.e[d].dot(v_val))
```

差し替える場合は、**シグネチャと戻り値の意味**を変えないこと。solver / boundary / analytics は同じ `D3Q19` インスタンスを共有する必要がある。

---

### 3.4 geometry.py — GeometryBuilder（形状・入口出口）

**役割**：`ctx.cell_id` と `ctx.sdf` **のみ**を更新する。境界の「意味」は ID で決まり、geometry は座標範囲でどのセルを INLET/OUTLET にするかを決める。

#### 使用例（main.py からの呼び出し）

```python
geo = GeometryBuilder()
if benchmark == "parallel_plates":
    geo.build_parallel_plates(ctx)
    geo.set_inlet_outlet(ctx)
elif benchmark == "cavity":
    geo.build_lid_driven_cavity(ctx)
    # キャビティは set_inlet_outlet を呼ばない
elif benchmark == "cylinder":
    geo.build_benchmark_cylinder(ctx)
    geo.set_inlet_outlet(ctx)
```

#### 入口・出口の上書き（固定仕様）

```python
@staticmethod
@ti.kernel
def _set_inlet_outlet_kernel(cell_id: ti.template(), nz: int):
    for i, j in ti.ndrange(cell_id.shape[0], cell_id.shape[1]):
        cell_id[i, j, nz - 1] = INLET
        cell_id[i, j, 0] = OUTLET
```

- 入口: `k = nz - 1` の面を INLET
- 出口: `k = 0` の面を OUTLET

#### 検証用チャネル（左右壁）

```python
def _build_validation_channel_kernel(...):
    for i, j, k in cell_id:
        cell_id[i, j, k] = FLUID_A
        sdf[i, j, k] = 100.0
        if i < 10 or i > (nx - 10):
            cell_id[i, j, k] = SOLID
            sdf[i, j, k] = 0.0
```

#### 円柱（ベンチマーク用）

```python
def _build_benchmark_cylinder_kernel(...):
    cx = float(nx) * 0.5 + 0.1
    cz = float(nz) * 0.75
    radius = float(nx) * 0.05
    dist = ti.math.sqrt((float(i) - cx)**2 + (float(k) - cz)**2) - radius
    sdf[i, j, k] = dist
    if dist <= 0.0:
        cell_id[i, j, k] = SOLID
```

リッドドリブンキャビティでは上面に ID=20（速度境界）、他壁に ID=21 を割り当て、`config.bc_velocity_by_id` で 20 にのみ速度を指定する。

---

### 3.5 boundary.py — BoundaryConditionManager

**役割**：**ID ベース**で速度境界・出口境界・熱流束境界を適用する。NEEM（非平衡補外法）により、入口は速度固定・密度は内部から外挿、出口は密度固定・速度は内部から外挿する。

#### 速度境界（入口など）

`cell_id == target_id` のセルで、隣接流体の非平衡成分を平均しつつ、指定速度・温度の平衡分布に載せて `f_new`, `g_new` を上書きする。

```python
@ti.kernel
def apply_velocity_bc(self, ctx: ti.template(), target_id: ti.i32, vel_vec: ti.math.vec3, rho_val: ti.f32, temp_val: ti.f32):
    for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
        if ctx.cell_id[i, j, k] == target_id:
            # 隣接 FLUID_A セルから f_neq, g_neq をサンプリング
            # ...
            rho_in = rho_sum / count  # 密度は内部に合わせる（圧力浮き）
            for d in ti.static(range(19)):
                feq_b = self.d3q19.get_feq(rho_in, vel_vec, d)
                geq_b = self.d3q19.get_geq(temp_val, vel_vec, d)
                ctx.f_new[i, j, k, d] = feq_b + (f_neq_sum[d] / count)
                ctx.g_new[i, j, k, d] = geq_b + (g_neq_sum[d] / count)
```

**呼び出しタイミング**：`stream_and_bc` の直後、**update_macro の前**（f_new/g_new を上書きするため）。

#### 出口境界

```python
@ti.kernel
def apply_outlet_bc(self, ctx: ti.template(), target_id: ti.i32):
    # 隣接流体から v_out, t_out, f_neq, g_neq を取得
    # 逆流時は v_out = 0 にフォールバック
    rho_out = 1.0
    ctx.f_new[i, j, k, d] = feq_b + (f_neq_sum[d] / count)
    ctx.g_new[i, j, k, d] = geq_b + (g_neq_sum[d] / count)
```

#### 熱流束境界

固体セル（target_id）の温度を「隣接流体の平均温度 + flux_val * dx / k_p」で更新する。**update_macro の後**に呼ぶ（temp を直接書き換えるため）。

```python
@ti.kernel
def apply_heat_flux_bc(self, ctx: ti.template(), target_id: ti.i32, flux_val: ti.f32):
    if ctx.cell_id[i, j, k] == target_id:
        # 隣接 FLUID_A の temp を平均
        ctx.temp[i, j, k] = avg_tf + flux_val * self.cfg.dx / self.cfg.k_p
```

---

### 3.6 solver.py — LBMSimulator

**役割**：Cumulant / Regularized LBM に基づく衝突・移流・マクロ更新・パーティクル移動を行う。流体セル（FLUID_A, INLET, OUTLET）のみを扱い、固体は stream 時の bounce-back で処理する。

#### 初期化

```python
@ti.kernel
def init_fields(self, ctx: ti.template()):
    for i, j, k in ti.ndrange(ctx.nx, ctx.ny, ctx.nz):
        cid = ctx.cell_id[i, j, k]
        ctx.rho[i, j, k] = 1.0
        ctx.v[i, j, k] = ...  # 流体は v_inlet、固体は 0
        if cid == SOLID or cid == SOLID_HEAT_SOURCE:
            ctx.temp[i, j, k] = 1.0
        else:
            ctx.temp[i, j, k] = 0.0
        for d in ti.static(range(19)):
            ctx.f_old[i, j, k, d] = self.d3q19.get_feq(1.0, ctx.v[i, j, k], d)
            ctx.g_old[i, j, k, d] = self.d3q19.get_geq(ctx.temp[i, j, k], ctx.v[i, j, k], d)
    for n in range(ctx.particle_pos.shape[0]):
        ctx.particle_pos[n] = ti.Vector([ti.random()*ctx.nx, ...])
```

#### 衝突（collide）

流体セルのみ。`ctx.tau_f_table[cid]`, `ctx.tau_g_table[cid]` を参照する。

- **g**：BGK で `g_post = g_old - omega_g * (g_old - g_eq)`
- **f**：2 次モーメント（応力テンソル）のみを tau_f で緩和し、Regularization で f_post を再構築（高次モーメントは捨てる）。

```python
if cid == FLUID_A or cid == INLET or cid == OUTLET:
    tau_f = ctx.tau_f_table[cid]
    tau_g = ctx.tau_g_table[cid]
    # ...
    for d in ti.static(range(19)):
        f_neq = ctx.f_old[i, j, k, d] - feq
        Pi_xx += f_neq * ex * ex
        # ...
    Pi_xx *= (1.0 - omega_f)
    # ...
    ctx.f_post[i, j, k, d] = f_eq_cache[d] + f_neq_reg
```

#### 移流と固体境界（stream_and_bc）

Push 方式で、流体セルから出た分布を隣に配る。隣が固体の場合は SDF を使った Bouzidi 型補間 bounce-back を行い、`f_new[i,j,k,inv_d]` と `g_new`（壁温 T_w=1.0 の反射）を設定する。

```python
if neighbor_cid == FLUID_A or ...:
    ctx.f_new[ip, jp, kp, d] = ctx.f_post[i, j, k, d]
    ctx.g_new[ip, jp, kp, d] = ctx.g_post[i, j, k, d]
else:
    q = ctx.sdf[i, j, k] / (ctx.sdf[i, j, k] - ctx.sdf[ip, jp, kp])
    # f_bb を計算し
    ctx.f_new[i, j, k, inv_d] = f_bb
    ctx.g_new[i, j, k, inv_d] = -ctx.g_post[i, j, k, d] + 2.0 * self.d3q19.w[d] * T_w
```

#### マクロ更新（update_macro）

f_new, g_new から rho, v, temp を計算し、**同じセルで** f_old, g_old にコピーする。rho ≤ 1e-12 の場合は速度を 0 にする。

```python
if cid == FLUID_A or cid == INLET or cid == OUTLET:
    new_rho = sum f_new, new_v = sum f_new * e, new_temp = sum g_new
    ctx.f_old[i, j, k, d] = ctx.f_new[i, j, k, d]
    ctx.g_old[i, j, k, d] = ctx.g_new[i, j, k, d]
    ctx.rho[i, j, k] = new_rho
    ctx.v[i, j, k] = new_v / new_rho if new_rho > 1e-12 else 0
    ctx.temp[i, j, k] = new_temp
```

#### パーティクル移動（move_particles）

`ctx.v` で位置を更新し、領域外や固体に入った粒子は「毎ステップ一定数だけ入口で再生成」し、それ以外は描画外（-1,-1,-1）に置いて消滅させる。

```python
ctx.particle_pos[n] += ctx.v[i, j, k] * 2.0
# 出口 or 固体衝突で is_out
if is_out:
    idx = ti.atomic_add(ctx.inject_count[None], 1)
    if idx < inject_per_step:
        ctx.particle_pos[n] = ti.Vector([ti.random()*ctx.nx, ti.random()*ctx.ny, float(ctx.nz)-0.5])
    else:
        ctx.particle_pos[n] = ti.Vector([-1.0, -1.0, -1.0])
```

---

### 3.7 analytics.py — Analytics

**役割**：ctx を**読み取り専用**で参照し、ヌッセルト数・平均温度・局所 Nu などを計算する。Solver / Boundary に依存しない。

#### 主な API

- `get_avg_grad(ctx)`：固体面の平均温度勾配（無次元）
- `get_avg_t_out(ctx)`：出口直前の流体の平均温度
- `get_solid_temp(ctx)`：固体セルの平均温度
- `get_local_Nu(ctx, k_target)`：指定 k 断面の局所ヌッセルト数

#### 局所 Nu の利用例（main.py）

```python
if benchmark == "parallel_plates":
    k_target = int(cfg.nz * 0.1)
    local_nu = analytics.get_local_Nu(ctx, k_target)
    target_nu = 7.54 if cfg.bc_type == 'T' else 8.23
    p_error = abs(local_nu - target_nu) / target_nu * 100
    print(f"Step {step:5d} | Nu_local: {local_nu:.3f} | Target Nu: {target_nu} | Error: {p_error:.2f} %")
```

---

### 3.8 export.py — 可視化フレームと GIF

**役割**：1 ステップ分の温度＋固体マスク＋パーティクルを 2D キャンバスに描画し、GIF として保存する。

#### build_vis_frame

```python
def build_vis_frame(ctx, cfg):
    temp_np = ctx.temp.to_numpy()
    cell_id_np = ctx.cell_id.to_numpy()
    mask_np = (cell_id_np == SOLID) | (cell_id_np == SOLID_HEAT_SOURCE)
    # 正面・側面・上面の切り出しと RGB 化（温度は R/B、固体はグレー）
    # パーティクルを白でプロット
    return canvas  # shape (total_h, total_w, 3), dtype uint8
```

#### export_gif_frames

```python
def export_gif_frames(frames, filename, fps=12):
    imageio.mimsave(filename, frames, fps=fps)
```

---

### 3.9 vtk_export.py — VTI 出力

**役割**：指定ステップで `ctx` の rho, v, temp, cell_id を VTI 形式で書き出し、ParaView で可視化できるようにする。PyVista 必須。

```python
def export_step(ctx, step, filepath_template, dx=1.0):
    """
    ctx の rho, v, temp, cell_id を VTI で保存する。
    filepath_template: 例 "results/step_{:06d}.vti" → step=100 で results/step_000100.vti
    最終フレーム用に直接パスを渡す場合（例 "results/final.vti"）も可。その場合 .format(step) でもプレースホルダが無ければそのまま。
    dx: 格子間隔（物理スケール、省略時は 1.0）
    """
    path = filepath_template.format(step)
    # ...
    grid.save(path)
    return path
```

main では `vti_export_interval > 0` のとき `step % cfg.vti_export_interval == 0` で `export_step(ctx, step, cfg.vti_path_template, dx=cfg.dx)` を呼ぶ。終了後は `export_step(ctx, step, final_vti_path, dx=cfg.dx)` で最終状態を別名 VTI として保存する。

---

### 3.10 diagnostics.py — 数値診断ログ

**役割**：暴走切り分けのため、流体セル（FLUID_A, INLET, OUTLET）のみを対象に rho / temp / |v| の min・max・mean・NaN/Inf 個数・異常個数、およびオプションで f_old, g_old の統計を出力する。

```python
def log_field_diagnostics(ctx, step, log_distributions=False):
    fluid_mask = (cell_id == FLUID_A) | (cell_id == INLET) | (cell_id == OUTLET)
    # rho, temp, |v| の stats と rho<=0, rho<1e-6, temp 異常個数
    # log_distributions=True のとき f_old, g_old の min/max/neg_count
```

main では `step <= 2000` かつ `step % 100 == 0` などで呼び、必要に応じて `log_distributions=True` を指定する。

---

### 3.11 main.py — run_simulation・物理時間ループ・定常検知・最終保存

**役割**：上記モジュールを組み合わせ、ベンチマーク種別に応じて形状・境界を切り替え、**物理時間**でループし、定常検知・ETA 表示・終了後のメタデータ/フィールド/VTI 保存・GIF 出力を行う。

#### SteadyStateDetector（定常検知）

流れの定常化（または周期定常化）を検知し、検知後 `extra_time_p` だけ回してから終了する。

```python
class SteadyStateDetector:
    def __init__(self, window_time_p, tolerance, extra_time_p, U_ref):
        # 直近 2 ウィンドウ分の (物理時間, モニタ値) を保持
    def update(self, current_time_p, val):  # 定常判定、初回検知時に True を返す
    def should_stop(self, current_time_p):  # 定常検知から extra_time_p 経過で True
```

モニタ値はベンチマークごとに異なる（平行平板: 局所 Nu、キャビティ: 中心 |v|、円柱: 後流の vx）。

#### 1 ステップの順序（厳守）

```python
while current_time_p < max_time_p:
    sim.collide(ctx)
    sim.stream_and_bc(ctx)
    # ソフトスタート: ramp_time_p まで ramp_factor で速度を徐々に上げる
    ramp_factor = 0.5 * (1 - cos(π * current_time_p / ramp_time_p)) if current_time_p < ramp_time_p else 1.0
    for target_id, vel in cfg.bc_velocity_by_id.items():
        bc.apply_velocity_bc(ctx, target_id, vel * ramp_factor, rho_val=1.0, temp_val=0.0)
    for target_id in cfg.bc_outlet_ids:
        bc.apply_outlet_bc(ctx, target_id)
    sim.update_macro(ctx)
    for target_id, flux in cfg.bc_heat_flux_by_id.items():
        bc.apply_heat_flux_bc(ctx, target_id, flux)
    sim.move_particles(ctx, cfg.particles_inject_per_step)
    # vis_interval ごと: モニタ値・ETA・GIF フレーム、定常検知時は should_stop で break
    current_time_p += cfg.dt
    step += 1
```

#### 終了後の保存

1. **メタデータ**（物性・格子・最終時間・ステップ数）を JSON で `{out_dir}/{benchmark}_{timestamp}_meta.json`
2. **3D フィールド**（v, temp, rho, cell_id）を NumPy 圧縮で `{out_dir}/{benchmark}_{timestamp}_fields.npz`
3. **最終状態の VTI** を `export_step(ctx, step, final_vti_path, dx=cfg.dx)` で `{out_dir}/{benchmark}_{timestamp}_final.vti`
4. **GIF** を `export_gif_frames(frames, cfg.filename, fps=12)` で保存

#### 実行例（平行平板・物理時間・定常検知）

```python
run_simulation(
    benchmark="parallel_plates",
    nx=64, ny=64, nz=512,
    Lx_p=0.05, U_inlet_p=0.05, nu_p=2.5e-5, k_p=0.026,
    max_time_p=30.0,
    ramp_time_p=1.0,
    steady_window_p=1.0,
    steady_tolerance=0.01,
    steady_extra_p=1.0,
    vis_interval=100, bc_type="q",
    bc_velocity_by_id={INLET: [0.0, 0.0, -0.05]},
    bc_outlet_ids=[OUTLET],
    vti_export_interval=0, particles_inject_per_step=0,
)
```

円柱ベンチマークでは `steady_window_p=1.5`, `steady_tolerance=0.02`, `steady_extra_p=4.0` などでカルマン渦の周期定常を検知してから余分に回す。

---

## 4. 契約と差し替え時の注意

- **ctx の shape**：`(ctx.nx, ctx.ny, ctx.nz)` で統一。Context 生成後に nx, ny, nz を変えない。
- **セルID**：FLUID_A, SOLID, INLET, OUTLET 等は context の定数と一致させる。境界は ID だけで判定し、位置に依存しない。
- **物性**：流体系の ID は `get_materials_dict()` に含め、`set_materials()` を init 前に一度呼ぶ。
- **呼び出し順序**：  
  - 初期化: 形状設定 → set_materials → init_fields  
  - 1 ステップ: collide → stream_and_bc → 速度/出口 BC → update_macro → 熱流束 BC → move_particles  
  - 終了後: メタデータ(JSON)・フィールド(.npz)・最終 VTI を保存してから GIF 出力  

D3Q19 の `get_feq` / `get_geq` のシグネチャと `e`, `w`, `inv_d` の意味・shape を変えると、solver / boundary / analytics の整合性が崩れるため、差し替え時は維持すること。

---

## 5. 実行方法

```bash
cd refactoredv2.2
python main.py
```

パラメータは `main.py` の `run_simulation(...)` のキーワード引数で変更する。

- **時間・定常制御**: `max_time_p`, `ramp_time_p`, `steady_window_p`, `steady_tolerance`, `steady_extra_p`
- **出力**: `out_dir`（既定 "results"）、`filename`（GIF）、`vti_export_interval`, `vti_path_template`
- VTI を使う場合は `vti_export_interval` を正にし、`vti_path_template` を指定し、`pip install pyvista` で PyVista を入れておく。
- 終了後、同一 `out_dir` にメタデータ（JSON）・フィールド（.npz）・最終 VTI が自動保存される。

---

*この文書は、refactoredv2.2 の全体機能と各モジュールの仕様を、具体的なコードを交えてまとめたものです。*
