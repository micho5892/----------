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
        
        # ====================================================
        # ★修正: 最初の1回だけウォームアップしてバックアップを取る
        # 2回目以降は、バックアップをリストアするだけで一瞬で初期状態に戻す
        # ====================================================
        if not hasattr(self, "_initial_state_backed_up"):
            print("\n[LBMHeatSinkEnv] Performing initial warmup and state backup...")
            self.runner.reset_simulation(
                warmup_steps=1000, 
                warmup_time_sum_scale=self.warmup_time_sum_scale,
            )
            # ウォームアップ済みの状態を保存
            self.runner.backup_state()
            
            self.base_nu, _ = self.runner.get_metrics()
            self._base_nu_backup = self.base_nu
            self._initial_state_backed_up = True
        else:
            # 2回目以降は、LBMシミュレータを一瞬で初期状態に巻き戻す
            self.runner.restore_state()
            self.base_nu = self._base_nu_backup
            
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
        
        # --- 報酬のチューニング ---
        # 1. Nu数の変化 (削ると基本的に減るので、ペナルティを少しマイルドにする)
        nu_reward = (current_nu - self.base_nu) * 1.0  
        
        # 2. 圧力損失の改善 (初期状態の詰まったブロックからどれだけ風通しを良くしたか)
        # 初期状態の圧力損失を base_pressure_drop として __init__ または reset で取っておき、
        # (self.base_pressure_drop - pressure_drop) * 1000.0 のようにすると
        # 「風を通したこと」への強いプラス報酬を与えられます。
        
        # 今回は簡易的に「圧力損失そのものが小さいほど良い」とするため、係数を大きくします
        pressure_penalty = pressure_drop * 100.0 
        
        # 生の報酬
        raw_reward = nu_reward - pressure_penalty
        
        # ★重要: NNが学習しやすいように報酬全体を小さくスケーリング（100分の1など）
        reward = raw_reward / 100.0
        
        # 4. 終了判定
        terminated = False
        truncated = (self.current_step >= self.max_steps)
            
        info = {"nu_number": current_nu, "pressure_drop": pressure_drop}
        return self.runner.get_observation_arrays(), reward, terminated, truncated, info