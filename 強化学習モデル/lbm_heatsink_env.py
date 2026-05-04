import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
    ):
        super().__init__()
        self.mode = mode
        self.warmup_time_sum_scale = float(warmup_time_sum_scale)
        
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
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

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
        
        # Runnerにシミュレーションのリセットとウォームアップを依頼
        self.runner.reset_simulation(
            warmup_steps=None,
            warmup_time_sum_scale=self.warmup_time_sum_scale,
        )
        
        # 初期状態のメトリクスを基準値として記録
        self.base_nu, _ = self.runner.get_metrics()
        info = {}
        if self.runner.last_reset_warmup_meta is not None:
            info["reset_warmup"] = dict(self.runner.last_reset_warmup_meta)
        return self.runner.get_observation_arrays(), info

    def step(self, action):
        self.current_step += 1
        
        # 1. 行動をシミュレーション座標に変換し、Runnerに形状変更を指示
        if self.mode == "plan_a":
            cx = (action[0] + 1.0) * 0.5 * self.runner.cfg.nx
            cy = (action[1] + 1.0) * 0.5 * self.runner.cfg.ny
            cz = (action[2] + 1.0) * 0.5 * self.runner.cfg.nz
            r  = ((action[3] + 1.0) * 0.5 * 10.0) + 2.0  # 半径2〜12セル
            
            self.runner.modify_shape_sphere(cx, cy, cz, r)

        # 2. シミュレーションを進める
        self.runner.run_steps(num_steps=200)

        # 3. 評価指標を取得し、報酬を計算
        current_nu, pressure_drop = self.runner.get_metrics()
        
        # 報酬設計 (※重みは学習を見ながら調整します)
        reward = (current_nu - self.base_nu) * 10.0 - (pressure_drop * 0.1)
        
        # 4. 終了判定
        terminated = False
        truncated = (self.current_step >= self.max_steps)
            
        info = {"nu_number": current_nu, "pressure_drop": pressure_drop}
        return self.runner.get_observation_arrays(), reward, terminated, truncated, info