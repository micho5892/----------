import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.ndimage

# 新しく作ったRunnerをインポート
from .simulation_runner import HeatSinkSimulationRunner

class LBMHeatSinkEnv(gym.Env):
    """
    共役熱伝達LBMシミュレータを用いたヒートシンク最適化環境
    """
    def __init__(
        self,
        mode="plan_a",
        nx: int = 64,
        ny: int = 64,
        nz: int = 128,
        sim_config_overrides: dict | None = None,
        warmup_time_sum_scale: float = 3.0,
        plan_b_downscale: int = 4,
    ):
        super().__init__()
        self.mode = mode
        self.warmup_time_sum_scale = float(warmup_time_sum_scale)
        self.plan_b_downscale = max(1, int(plan_b_downscale))
        
        # シミュレーション実行クラスのインスタンス化
        self.runner = HeatSinkSimulationRunner(
            nx=nx,
            ny=ny,
            nz=nz,
            mode=self.mode,
            sim_config_overrides=sim_config_overrides,
        )
        
        # 行動空間 (Action Space)
        if self.mode == "plan_a":
            # [X, Y, Z, R]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        elif self.mode == "plan_b":
            # ====================================================
            # ★修正: 指定ダウンスケール率で案Bの行動空間を定義
            # ====================================================
            self.action_shape = (
                max(1, self.runner.cfg.nx // self.plan_b_downscale),
                max(1, self.runner.cfg.ny // self.plan_b_downscale),
                max(1, self.runner.cfg.nz // self.plan_b_downscale),
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=self.action_shape,
                dtype=np.float32
            )

        # 観測空間 (Observation Space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(5, self.runner.cfg.nx, self.runner.cfg.ny, self.runner.cfg.nz), 
            dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if not hasattr(self, "_initial_state_backed_up"):
            print("\n[LBMHeatSinkEnv] Performing initial warmup and state backup...")
            self.runner.reset_simulation(
                # warmup_steps=1000, # ※本番時はお好みで変更してください
                warmup_time_sum_scale=self.warmup_time_sum_scale,
            )
            self.runner.backup_state()
            
            self.base_nu, self.base_pressure_drop = self.runner.get_metrics()
            self._base_nu_backup = self.base_nu
            self._base_pressure_drop_backup = self.base_pressure_drop
            self._initial_state_backed_up = True
        else:
            self.runner.restore_state()
            self.base_nu = self._base_nu_backup
            self.base_pressure_drop = self._base_pressure_drop_backup
            
        # ====================================================
        # ★追加: 「前回のステップのスコア」を記憶する変数を追加
        # ====================================================
        self.prev_nu = self.base_nu
        self.prev_pressure_drop = self.base_pressure_drop
            
        info = {}
        if self.runner.last_reset_warmup_meta is not None:
            info["reset_warmup"] = dict(self.runner.last_reset_warmup_meta)
        return self.runner.get_observation_arrays(), info

    def step(self, action):
        self.current_step += 1
        
        if self.mode == "plan_a":
            cx = (action[0] + 1.0) * 0.5 * self.runner.cfg.nx
            cy = (action[1] + 1.0) * 0.5 * self.runner.cfg.ny
            cz = (action[2] + 1.0) * 0.5 * self.runner.cfg.nz
            r  = ((action[3] + 1.0) * 0.5 * 10.0) + 2.0  
            
            ix = int(np.clip(cx, 0, self.runner.cfg.nx - 1))
            iy = int(np.clip(cy, 0, self.runner.cfg.ny - 1))
            iz = int(np.clip(cz, 0, self.runner.cfg.nz - 1))
            current_sdf = self.runner.ctx.sdf.to_numpy()[ix, iy, iz]
            
            if current_sdf > r - 1.0:
                reward = -2.0  # 完全な空振りペナルティ
                terminated = False
                truncated = (self.current_step >= self.max_steps)
                info = {"nu_number": 0.0, "pressure_drop": 0.0, "missed": True}
                return self.runner.get_observation_arrays(), reward, terminated, truncated, info

            self.runner.modify_shape_sphere(cx, cy, cz, r)

        elif self.mode == "plan_b":
            # ====================================================
            # ★修正: ダウンスケール率に応じたアップサンプリング
            # ====================================================
            action_3d = action.reshape(self.action_shape)
            target_phi_low_res = 0.5 * (1.0 - action_3d)

            if self.plan_b_downscale == 1:
                target_phi_high_res = target_phi_low_res
            else:
                scale_factors = (
                    self.runner.cfg.nx / self.action_shape[0],
                    self.runner.cfg.ny / self.action_shape[1],
                    self.runner.cfg.nz / self.action_shape[2],
                )
                target_phi_high_res = scipy.ndimage.zoom(
                    target_phi_low_res, scale_factors, order=3
                )
                target_phi_high_res = np.clip(target_phi_high_res, 0.0, 1.0)

                req_shape = (self.runner.cfg.nx, self.runner.cfg.ny, self.runner.cfg.nz)
                if target_phi_high_res.shape != req_shape:
                    temp = np.zeros(req_shape, dtype=np.float32)
                    cx = min(req_shape[0], target_phi_high_res.shape[0])
                    cy = min(req_shape[1], target_phi_high_res.shape[1])
                    cz = min(req_shape[2], target_phi_high_res.shape[2])
                    temp[:cx, :cy, :cz] = target_phi_high_res[:cx, :cy, :cz]
                    target_phi_high_res = temp

            target_phi_high_res = np.asarray(
                np.clip(target_phi_high_res, 0.0, 1.0), dtype=np.float32
            )
            self.runner.modify_shape_density(target_phi_high_res)

        self.runner.run_steps(num_steps=200)

        current_nu, current_pressure_drop = self.runner.get_metrics()
        
        # ====================================================
        # ★修正: base ではなく prev (前ステップ) と比較して差分を報酬にする
        # ====================================================
        nu_reward = (current_nu - self.prev_nu) * 1.0  
        
        # 圧力の改善係数を少し強くして「風を通す」ことのモチベーションを上げる
        pressure_improvement = (self.prev_pressure_drop - current_pressure_drop) * 500.0 
        
        raw_reward = nu_reward + pressure_improvement
        reward = float(raw_reward / 100.0)
        
        # ★追加: サボり防止のタイムペナルティ (毎ステップ少しずつ体力を奪う)
        # これにより、無意味な場所を削って reward=0 になってもマイナスになるため、AIは必死に改善点を探す
        reward -= 0.05 
        
        # 状態の更新
        self.prev_nu = current_nu
        self.prev_pressure_drop = current_pressure_drop
        
        terminated = False
        truncated = (self.current_step >= self.max_steps)
            
        info = {"nu_number": current_nu, "pressure_drop": current_pressure_drop, "missed": False}
        return self.runner.get_observation_arrays(), reward, terminated, truncated, info