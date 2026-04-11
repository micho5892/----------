# refactoredv2 — 流体シミュレーション（データ主導モジュール化）

LBM（格子ボルツマン法）による熱流体シミュレーションのリファクタ版。  
設計書「データ主導のモジュール化設計」および `docs/タスク一覧_優先順位順.md` に沿った構成です。

---

## 1. ファイル構成と役割

| ファイル | 役割 |
|----------|------|
| **context.py** | セルID定数・`SimulationContext`（全 Taichi field の確保） |
| **config.py** | `SimConfig`・物性・境界条件の ID ベース管理 |
| **d3q19.py** | D3Q19 格子・LBM 定数（`get_feq` / `get_geq`） |
| **geometry.py** | `GeometryBuilder`（形状・入口出口の `cell_id` / `sdf` 更新） |
| **boundary.py** | `BoundaryConditionManager`（速度・出口・熱流束の境界） |
| **solver.py** | `LBMSimulator`（collide / stream / update_macro / move_particles） |
| **analytics.py** | `Analytics`（Nu・平均温度・局所値など ctx ベースの解析） |
| **export.py** | GIF 用フレーム構築・保存 |
| **vtk_export.py** | VTI 出力（ParaView 連携・オプション） |
| **main.py** | `run_simulation`・メインループ（コンポーネントの組み合わせのみ） |

---

## 2. 処理の流れ

### 2.1 初期化（main.py の run_simulation 冒頭）

```
1. SimConfig(**kwargs) で設定を構築
2. SimulationContext(nx, ny, nz, n_particles) で field を確保
3. cfg.get_materials_dict() → ctx.set_materials(materials) で物性テーブルを反映
4. GeometryBuilder().build_*() / set_inlet_outlet(ctx) で形状・入口出口を設定
5. LBMSimulator(cfg), BoundaryConditionManager(d3q19, cfg), Analytics(d3q19, cfg) を生成
6. sim.init_fields(ctx) で rho/v/temp/f_old/g_old を初期化
```

- **重要**: 形状設定（Geometry）→ 物性反映（set_materials）→ init_fields の順を変えないこと。  
  init は `ctx.cell_id` を参照して固体/流体の初期温度を切り分けている。

### 2.2 1 ステップのループ（設計書の疑似コードに沿った順序）

```
for step in range(cfg.steps):
    sim.collide(ctx)                    # 衝突（f_post, g_post を更新）
    sim.stream_and_bc(ctx)              # 移流＋固体壁面の bounce-back 等（f_new, g_new を更新）
    # 境界条件（ID ベース）
    for target_id, vel in cfg.bc_velocity_by_id.items():
        bc.apply_velocity_bc(ctx, target_id, vel, ...)
    for target_id in cfg.bc_outlet_ids:
        bc.apply_outlet_bc(ctx, target_id)
    sim.update_macro(ctx)               # rho, v, temp と f_old, g_old を更新
    for target_id, flux in cfg.bc_heat_flux_by_id.items():
        bc.apply_heat_flux_bc(ctx, target_id, flux)
    sim.move_particles(ctx)             # パーティクル位置更新
    # 可視化・VTI・ログ（vis_interval 等で間引く）
```

- **重要**: `apply_velocity_bc` / `apply_outlet_bc` は **f_new, g_new を上書き**するため、`stream_and_bc` の直後かつ `update_macro` の前に呼ぶこと。  
  `update_macro` は `f_new, g_new` を集計して `rho, v, temp` と `f_old, g_old` を更新する。  
  `apply_heat_flux_bc` は `temp` を直接書き換えるため、`update_macro` の後に呼ぶ。

---

## 3. 依存関係

### 3.1 モジュールの import 依存（矢印は「依存する」）

```
main.py
  → context, config, geometry, boundary, solver, export, analytics, vtk_export(optional)

config.py     → context（FLUID_A, INLET, OUTLET, SOLID, SOLID_HEAT_SOURCE）
geometry.py   → context（FLUID_A, SOLID, INLET, OUTLET）
boundary.py   → context（INLET, OUTLET, SOLID, SOLID_HEAT_SOURCE, FLUID_A）
solver.py     → context, d3q19
analytics.py  → context
export.py     → context（SOLID, SOLID_HEAT_SOURCE）
vtk_export.py → （context は引数 ctx で受け取るのみ。import なし）
d3q19.py      → （他モジュールに依存しない）
context.py    → taichi のみ
```

- **context** は「データの箱」であり、セルID定数と `SimulationContext` を提供する。他モジュールは context の定数または ctx インスタンスに依存する。
- **d3q19** は LBM 定数のみ。solver / boundary / analytics が `D3Q19` インスタンスを利用する。
- **main** だけが geometry / boundary / solver / analytics / export / vtk_export を組み合わせて呼ぶ。

### 3.2 インスタンスの受け渡し（main 内）

- `sim = LBMSimulator(cfg)` → `sim.d3q19` を **BoundaryConditionManager** と **Analytics** に渡す。
- **BoundaryConditionManager(d3q19, cfg)** と **Analytics(d3q19, cfg)** は同じ `d3q19` を共有する必要がある（格子ベクトル `e` 等の整合性のため）。

---

## 4. 各モジュールの API（引数・戻り値）

### 4.1 context.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| FLUID_A, SOLID, SOLID_HEAT_SOURCE, INLET, OUTLET | 定数 | — | int。セルID。境界判定にのみ使用。 |
| MAX_CELL_ID | 定数 | — | 32。tau_f_table / tau_g_table のサイズ。 |
| SimulationContext | クラス | — | — |
| `__init__(nx, ny, nz, n_particles)` | コンストラクタ | 格子サイズ・パーティクル数 | インスタンス。全 field を確保。 |
| `set_materials(materials_dict)` | メソッド | `dict[int, tuple[float,float]]`（ID → (tau_f, tau_g)） | None。tau_f_table / tau_g_table を更新。 |

**ctx が持つ field（契約）**:  
`cell_id`, `sdf`, `tau_f_table`, `tau_g_table`, `f_old`, `f_post`, `f_new`, `g_old`, `g_post`, `g_new`, `v`, `rho`, `temp`, `particle_pos`, `img`。  
形状は `(nx, ny, nz)` または `(nx, ny, nz, 19)` 等。`tau_*_table` は `shape=(MAX_CELL_ID,)`。

---

### 4.2 config.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| SimConfig | クラス | — | — |
| `__init__(**kwargs)` | コンストラクタ | nx, ny, nz, Lx_p, U_inlet_p, nu_p, k_p, bc_type, steps, vis_interval, vti_export_interval, vti_path_template 等 | インスタンス。bc_velocity_by_id, bc_heat_flux_by_id, bc_outlet_ids も設定。 |
| `get_materials_dict()` | メソッド | なし | `dict[int, tuple[float,float]]`。FLUID_A, INLET, OUTLET → (tau_f, tau_g)。 |

---

### 4.3 d3q19.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| C_s, SQRT2 | 定数 | — | float。MRT 等で使用。 |
| D3Q19 | クラス | — | — |
| `__init__()` | コンストラクタ | なし | インスタンス。e, inv_d, w, M, M_inv, S, C_s, SQRT2 を保持。 |
| `get_feq(rho_val, v_val, d)` | @ti.func | 密度・速度・方向 | 平衡分布 f_eq。 |
| `get_geq(temp_val, v_val, d)` | @ti.func | 温度・速度・方向 | 平衡分布 g_eq。 |

**差し替え時**: Taichi の `ti.func` として kernel 内から呼ばれるため、シグネチャ（引数と戻り値の型）を変えないこと。`e`, `w`, `M`, `M_inv`, `S`, `inv_d` の意味と shape を維持すること。

---

### 4.4 geometry.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| GeometryBuilder | クラス | — | — |
| `build_validation_channel(ctx)` | メソッド | ctx | None。検証用チャネル（左右壁）を cell_id / sdf に書き込む。 |
| `build_karman_cylinder(ctx)` | メソッド | ctx | None。円柱形状を cell_id / sdf に書き込む。 |
| `build_pin_fin_sdf(ctx)` | メソッド | ctx | None。ピンフィン用 SDF。 |
| `set_inlet_outlet(ctx, inlet_box=None, outlet_box=None)` | メソッド | ctx, 省略可 | None。入口 z=nz-1 → INLET、出口 z=0 → OUTLET。 |
| `set_heat_source(ctx, target_surface_box)` | メソッド | ctx, 範囲 | None。将来用。未実装可。 |
| `load_stl(ctx, filename)` | メソッド | ctx, パス | None。将来用。未実装可。 |

**契約**: いずれも **ctx の cell_id と sdf のみを更新**する。ctx.nx, ctx.ny, ctx.nz と一致する shape であること。使用する ID は context の定数（FLUID_A, SOLID, INLET, OUTLET）と一致させること。

---

### 4.5 boundary.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| BoundaryConditionManager | クラス | — | — |
| `__init__(d3q19, cfg)` | コンストラクタ | D3Q19 インスタンス, SimConfig | インスタンス。 |
| `apply_velocity_bc(ctx, target_id, vel_vec, rho_val=1.0, temp_val=0.0)` | メソッド | ctx, セルID, 速度ベクトル(ti.Vector), 密度, 温度 | None。cell_id==target_id のセルで f_new, g_new を feq/geq で上書き。 |
| `apply_outlet_bc(ctx, target_id)` | メソッド | ctx, セルID | None。cell_id==target_id で e[d][2]>0 の方向のみ隣 (i,j,k+1) から f_new, g_new をコピー。 |
| `apply_heat_flux_bc(ctx, target_id, flux_val)` | メソッド | ctx, セルID, 熱流束（スカラー） | None。固体 target_id の温度を「隣接流体の平均温度 + flux_val」に更新。 |

**契約**: 境界は **cell_id の値だけで判定**する。位置（k など）はハードコードしない。  
差し替え時は「同じ target_id に対して同じ物理意味の境界を適用する」こと。  
`apply_velocity_bc` の `vel_vec` は cfg の `v_inlet` など **ti.Vector** が渡る想定。

---

### 4.6 solver.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| LBMSimulator | クラス | — | — |
| `__init__(config)` | コンストラクタ | SimConfig | インスタンス。内部で D3Q19() を生成。 |
| `init_fields(ctx)` | メソッド | ctx | None。rho=1, v=0, temp（固体1/流体0）, f_old/g_old を平衡で初期化。パーティクル位置を乱数で設定。 |
| `collide(ctx)` | メソッド | ctx | None。FLUID_A/INLET/OUTLET のみ処理。ctx.tau_f_table[cid], ctx.tau_g_table[cid] を参照。f_post, g_post を更新。 |
| `stream_and_bc(ctx)` | メソッド | ctx | None。f_post/g_post を移流し f_new/g_new を更新。固体セルでは bounce-back 等。 |
| `update_macro(ctx)` | メソッド | ctx | None。f_new/g_new から rho, v, temp を計算し、f_old/g_old にコピー。rho≤1e-12 のセルは平衡リセット。 |
| `move_particles(ctx)` | メソッド | ctx | None。ctx.v でパーティクルを移動。固体衝突時は再配置。 |

**契約**:  
- **collide**: ctx の tau_f_table / tau_g_table を「流体系のセルID」で参照する。get_materials_dict で渡す ID と一致させること。  
- **stream_and_bc**: 隣接セルの cell_id で流体/固体を判定する。D3Q19 の e, inv_d, w に依存する。  
- **update_macro**: 流体セルでは f_new, g_new をそのまま f_old, g_old にコピーする。この順序（stream → BC → update_macro）を変えると境界が反映されない。

---

### 4.7 analytics.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| Analytics | クラス | — | — |
| `__init__(d3q19, cfg)` | コンストラクタ | D3Q19, SimConfig | インスタンス。 |
| `get_avg_grad(ctx)` | メソッド | ctx | float。固体面の平均温度勾配（無次元）。 |
| `get_avg_t_out(ctx)` | メソッド | ctx | float。出口直前の流体の平均温度（無次元）。 |
| `get_solid_temp(ctx)` | メソッド | ctx | float。固体セルの平均温度（無次元）。 |
| `get_local_Nu(ctx, k_target)` | メソッド | ctx, k インデックス | float。指定 k 断面の局所ヌッセルト数。 |

**契約**: いずれも **ctx を読み取り専用**として参照する。ctx の cell_id, temp, v 等の意味が Solver/BC と一致していること。  
差し替え時は「同じ物理量を返す」こと（例: get_avg_grad は壁面熱流束に関係する勾配）。

---

### 4.8 export.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| build_vis_frame(ctx, cfg) | 関数 | ctx, SimConfig | np.ndarray。RGB キャンバス (H, W, 3)。temp とパーティクルを可視化。 |
| export_gif_frames(frames, filename, fps=12) | 関数 | フレームリスト, パス, fps | None。GIF として保存。 |

**契約**: ctx の temp, cell_id, particle_pos と cfg の nx, ny, nz を参照する。shape は cfg と ctx が一致していること。

---

### 4.9 vtk_export.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| export_step(ctx, step, filepath_template, dx=1.0) | 関数 | ctx, ステップ番号, パステンプレート（`"dir/step_{:06d}.vti"` 等）, 格子間隔 | 保存したファイルパス（str）。PyVista 必須。 |

**契約**: ctx の rho, v, temp, cell_id を numpy に変換して VTI に書き出す。ctx.nx, ctx.ny, ctx.nz と一致した shape であること。

---

### 4.10 main.py

| 名前 | 種類 | 引数 | 戻り値 / 説明 |
|------|------|------|----------------|
| run_simulation(**kwargs) | 関数 | SimConfig のキーワード引数一式 | 保存した GIF のファイルパス（str）。 |

**契約**: 上記「処理の流れ」の順序でコンポーネントを呼ぶ。kwargs は Config にそのまま渡す。

---

## 5. 関数を差し替える際に守ること

### 5.1 共通契約（すべての ctx を扱う関数）

- **ctx の shape**: `(ctx.nx, ctx.ny, ctx.nz)` で統一。Context 生成後に nx, ny, nz を変えない。
- **セルID**: FLUID_A, SOLID, SOLID_HEAT_SOURCE, INLET, OUTLET のいずれか（または将来拡張用の 0～MAX_CELL_ID-1）。境界処理は「ID だけ」で判定し、位置（k など）に依存しない。
- **物性**: 流体系のセル（FLUID_A, INLET, OUTLET）では `ctx.tau_f_table[cid]`, `ctx.tau_g_table[cid]` が正しく設定されていること。`set_materials(get_materials_dict())` を init 後に一度呼ぶこと。

### 5.2 差し替え時の注意（モジュール別）

| 差し替え対象 | 守ること |
|--------------|----------|
| **GeometryBuilder の build_* / set_inlet_outlet** | cell_id と sdf だけを更新する。使用する ID は context の定数と一致させる。入口・出口は set_inlet_outlet で上書きする前提（build で INLET/OUTLET を立ててもよいが、座標で上書きするなら set_inlet_outlet の仕様に合わせる）。 |
| **BoundaryConditionManager の apply_*_bc** | 引数は (ctx, target_id, ...) の形を維持する。apply_velocity_bc は f_new, g_new を上書き。apply_outlet_bc は e[d][2]>0 の方向のみ隣からコピーする仕様を維持すると、元コードと同等の出口条件になる。apply_heat_flux_bc は固体セルの temp を書き換える。 |
| **LBMSimulator の collide** | ctx.tau_f_table[cid], ctx.tau_g_table[cid] を参照すること。流体セル（FLUID_A, INLET, OUTLET）のみ処理し、f_post, g_post を更新する。 |
| **LBMSimulator の stream_and_bc** | f_post, g_post を移流して f_new, g_new を更新。固体との境界では bounce-back 等、D3Q19 の e, inv_d, w を使う。 |
| **LBMSimulator の update_macro** | f_new, g_new から rho, v, temp を計算し、**同じセルで** f_old, g_old にコピーすること。順序は「境界適用の後」であること。 |
| **Analytics の各メソッド** | 戻り値の物理意味を変えない（get_avg_grad → 壁面勾配、get_local_Nu → 局所 Nu など）。ctx は読み取り専用。 |
| **D3Q19** | get_feq / get_geq のシグネチャと戻り値の意味を維持。e, w, M, M_inv, S, inv_d の shape と意味を維持。 |
| **SimConfig.get_materials_dict** | 少なくとも FLUID_A, INLET, OUTLET をキーにした dict を返すこと。値は (tau_f, tau_g) のタプル。 |
| **build_vis_frame / export_step** | ctx と cfg の格子サイズが一致していること。export_step は rho, v, temp, cell_id を書き出す前提で差し替え可能。 |

### 5.3 呼び出し順序を変えてはいけない部分

1. **init_fields** は **set_materials** および **Geometry による形状設定**の後で実行する。
2. **1 ステップ内**: collide → stream_and_bc → apply_velocity_bc / apply_outlet_bc → update_macro → apply_heat_flux_bc → move_particles。  
   （velocity/outlet BC は f_new, g_new を上書きするため update_macro の前。heat_flux_bc は temp を直接触るため update_macro の後。）
3. **可視化・VTI・ログ**は update_macro と move_particles の後でよい。

---

## 6. 実行方法・オプション

```bash
cd refactoredv2
python main.py
```

- パラメータは `main.py` の `run_simulation(...)` のキーワード引数で変更可能。
- VTI 出力: `vti_export_interval` を正の整数にし、`vti_path_template` でパスを指定。PyVista が必要（`pip install pyvista`）。
- GIF のみ使う場合は vti_export_interval=0（既定）のままでよい。

---

*この README は、処理の流れ・依存関係・各関数の引数と戻り値・関数を差し替える際の契約をまとめたものです。*
