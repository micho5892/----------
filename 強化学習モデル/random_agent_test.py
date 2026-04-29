import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import taichi as ti

# 既存のシミュレータのインポートを想定
from config import SimConfig
from context import SimulationContext
from geometry import GeometryBuilder
from solver import LBMSimulator
from analytics import Analytics
# 先ほど作成したモジュール
from shape_modifier import ShapeModifier

class LBMHeatSinkEnv(gym.Env):
    """
    共役熱伝達LBMシミュレータを用いたヒートシンク最適化環境
    """
    def __init__(self, mode="plan_a"):
        super().__init__()
        self.mode = mode  # "plan_a" (球減算) or "plan_b" (3D-CNN密度)
        
        # 1. LBMシミュレータの初期化準備
        self.cfg = SimConfig(nx=64, ny=64, nz=128, benchmark_name="rl_heatsink")
        self.ctx = SimulationContext(self.cfg.nx, self.cfg.ny, self.cfg.nz, self.cfg.n_particles, self.cfg.fp_dtype)
        self.modifier = ShapeModifier(self.ctx, self.cfg)
        self.sim = LBMSimulator(self.cfg)
        self.analytics = Analytics(self.sim.d3q19, self.cfg)
        
        # 2. 行動空間 (Action Space) の定義
        if self.mode == "plan_a":
            # 案A:[X, Y, Z, R] の連続値。値域は [-1, 1] に正規化しておくのがRLの定石
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        elif self.mode == "plan_b":
            # 案B: 空間全体のボクセルごとの削る確率。今回は将来用。
            self.action_space = spaces.Box(low=0.0, high=1.0, 
                                           shape=(self.cfg.nx, self.cfg.ny, self.cfg.nz), 
                                           dtype=np.float32)

        # 3. 観測空間 (Observation Space) の定義
        # [Channels, X, Y, Z] 形式 (例: Channel 0=Vx, 1=Vy, 2=Vz, 3=Temp, 4=SDF)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(5, self.cfg.nx, self.cfg.ny, self.cfg.nz), 
                                            dtype=np.float32)

        self.current_step = 0
        self.max_steps = 100  # 1エピソードあたり最大100回削れる

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. 形状を初期状態の「ソリッドブロック」にリセット
        geo = GeometryBuilder()
        geo.build_rl_initial_block(self.ctx) # ※GeometryBuilder側に初期ブロック生成関数を作る想定
        
        # 2. 流場・温度場の初期化
        self.sim.init_fields(self.ctx)
        
        # 3. 定常状態になるまで初期ウォームアップシミュレーションを回す
        # （例：最初は流れが安定するまで1000ステップ無条件で回す）
        self._run_lbm_steps(1000)
        
        # 初期状態のNu数や圧力を記録（報酬の基準値にする）
        self.base_nu = self._calculate_global_nu()
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. アクションに従って形状を変更 (SDFを更新)
        if self.mode == "plan_a":
            # 正規化されたaction [-1, 1] をシミュレーション座標 [0, nx] 等にスケール変換
            cx = (action[0] + 1.0) * 0.5 * self.cfg.nx
            cy = (action[1] + 1.0) * 0.5 * self.cfg.ny
            cz = (action[2] + 1.0) * 0.5 * self.cfg.nz
            r  = ((action[3] + 1.0) * 0.5 * 10.0) + 2.0  # 半径2〜12セル
            
            # カーネルを呼び出して削る
            self.modifier.subtract_sphere(cx, cy, cz, r)
            
        elif self.mode == "plan_b":
            # 将来用: actionテンソルをTaichiフィールドに転送して update_from_density_tensor を呼ぶ
            pass

        # 2. 形状が変わった状態で、少しだけシミュレーションを進める (ウォームスタート)
        # 流場が新しい形状に馴染むのに必要なステップ数（例: 200ステップ）
        self._run_lbm_steps(200)

        # 3. 報酬の計算
        current_nu = self._calculate_global_nu()
        pressure_drop = self._calculate_pressure_drop()
        
        # 報酬設計: Nu数の向上分 - 圧力損失ペナルティ
        reward = (current_nu - self.base_nu) * 10.0 - (pressure_drop * 0.1)
        
        # 4. 終了判定
        terminated = False
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        # ※例えばブロックの体積が初期の30%以下になったら terminated = True にするなど
            
        info = {"nu_number": current_nu, "pressure_drop": pressure_drop}
        return self._get_obs(), reward, terminated, truncated, info

    def _run_lbm_steps(self, num_steps):
        """指定回数だけLBMのCollide & Streamを実行するヘルパー関数"""
        for _ in range(num_steps):
            # 既存の physics_manager 等の呼び出しも必要に応じて入れる
            self.sim.collide_and_stream(self.ctx, ramp_factor=1.0, sponge_amp=1.0)
            self.sim.update_macro(self.ctx)

    def _get_obs(self):
        """Taichiフィールドから観測テンソルを構築してPyTorchテンソルで返す（ゼロコピーの理想形）"""
        # 注意: 現実には self.ctx.v.to_torch() で[nx, ny, nz, 3] を取得し、
        # permute(3,0,1,2) などでチャネルファーストに変形する処理が入る。
        
        # ここでは擬似コードとしてNumPy経由の書き方を記載（最初はこれで十分動く）
        v_np = self.ctx.v.to_numpy()       # (nx, ny, nz, 3)
        temp_np = self.ctx.temp.to_numpy() # (nx, ny, nz)
        sdf_np = self.ctx.sdf.to_numpy()   # (nx, ny, nz)
        
        obs = np.zeros((5, self.cfg.nx, self.cfg.ny, self.cfg.nz), dtype=np.float32)
        obs[0:3, :, :, :] = np.transpose(v_np, (3, 0, 1, 2))
        obs[3, :, :, :] = temp_np
        obs[4, :, :, :] = sdf_np
        
        return obs

    def _calculate_global_nu(self):
        """現在の全体のNu数を計算する"""
        # Analytics クラスを利用して計算
        # return self.analytics.get_local_Nu(...) などを全体に拡張したものを想定
        return 0.0 # ダミー

    def _calculate_pressure_drop(self):
        """入口と出口の圧力（密度 rho）差を計算する"""
        # LBMでは p = rho * cs^2
        # return 入口のrho平均 - 出口のrho平均
        return 0.0 # ダミー