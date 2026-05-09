# ==========================================================
# config.py — SimConfig（Pydantic）、materials_dict・境界条件の ID ベース管理
# ==========================================================
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import taichi as ti
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from lbm_logger import get_logger

log = get_logger(__name__)

# 浮動小数点型（run_simulation 内で cfg.fp_dtype に合わせて上書きされる）
TI_FLOAT = ti.f32


class RenderConfig(BaseModel):
    """可視化・出力に関する設定"""

    model_config = ConfigDict(extra="ignore")

    vis_interval: int = Field(20, ge=1, description="可視化のステップ間隔")
    """可視化のステップ間隔。動画用。"""
    filename: str = Field("output.gif", description="出力ファイル名")
    """出力ファイル名"""
    output_format: Literal["gif", "mp4"] = Field("gif", description="出力フォーマット")
    """出力フォーマット。gif または mp4 """
    vti_export_interval: int = Field(0, ge=0, description="VTI出力間隔(0で無効)")
    """VTI出力間隔(0で無効)"""
    vti_path_template: str = Field("results/step_{:06d}.vti")
    """VTI出力パステンプレート"""

    @field_validator("output_format", mode="before")
    @classmethod
    def _norm_output_format(cls, v: Any) -> str:
        s = str(v).lower().strip()
        return s if s in ("gif", "mp4") else "gif"


class ParticleConfig(BaseModel):
    """パーティクル（流線確認用）に関する設定"""

    model_config = ConfigDict(extra="ignore")

    n_particles: int = Field(10000, ge=0)
    particles_inject_per_step: Optional[int] = Field(
        None,
        ge=0,
        description="未指定時は max(1, n_particles//50)",
    )


class SimConfig(BaseModel):
    """
    シミュレーションの全設定を管理する Pydantic モデル
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    fp_dtype_str: Literal["float32", "float16"] = Field(
        "float32",
        validation_alias=AliasChoices("fp_dtype_str", "fp_dtype"),
        description="浮動小数点の精度（Taichi は文字列で受けて @property で変換）",
    )

    nx: int = Field(128, gt=0)
    ny: int = Field(128, gt=0)
    nz: int = Field(32, gt=0)
    benchmark_name: str = Field("unknown", description="ベンチマーク名")
    Lx_p: float = Field(0.1, gt=0.0, description="X方向の物理長[m]")

    render: RenderConfig = Field(default_factory=RenderConfig)
    """可視化・出力に関する設定"""
    particles: ParticleConfig = Field(default_factory=ParticleConfig)
    """パーティクル（流線確認用）に関する設定"""

    sponge_thickness: float = Field(0.0, ge=0.0)
    """スポンジ厚み（物理長）"""
    sponge_strength_decay_start_p: float = Field(0.0)
    """スポンジ強度減衰開始時間（秒）"""
    sponge_strength_decay_duration_p: float = Field(0.0)
    """スポンジ強度減衰期間（秒）"""

    steps: int = Field(600, ge=1)
    """ステップ数。max_time_p で物理時間。"""


    # キーはセル ID で int のことが多い（従来コード・ベンチスクリプト互換）。str も許容。
    boundary_conditions: Dict[Union[str, int], Any] = Field(default_factory=dict)
    """境界条件"""
    physics_models: Dict[Union[str, int], Any] = Field(default_factory=dict)
    """物理モデル"""
    domain_properties: Dict[Union[str, int], Any] = Field(default_factory=dict)
    """物性"""

    U_inlet_p: float = Field(1.0, description="代表物理流速")
    u_lbm_inlet: float = Field(0.1, description="LBM単位での代表流速（inlet から自動上書き可）")

    periodic_x: bool = Field(False)
    """X方向の周期境界条件"""
    periodic_y: bool = Field(False)
    """Y方向の周期境界条件"""
    periodic_z: bool = Field(False)
    """Z方向の周期境界条件"""

    neem_isothermal_wall: bool = Field(True)

    T_inlet_p: float = Field(0.0)
    T_wall_p: float = Field(1.0)
    delta_T_ref: Optional[float] = Field(
        None,
        description="未指定時は T_wall_p - T_inlet_p（ほぼ0のときは1.0）",
    )

    omega_cylinder_phys: Optional[float] = Field(None)
    omega_cylinder: float = Field(0.0)
    cylinder_center: Optional[List[float]] = Field(None)
    rotation_axis: str = Field("z")
    rot_axis_id: int = Field(2, exclude=True)

    out_dir: Optional[str] = Field(
        None,
        description="未実行時は「明示出力ディレクトリ」または None（自動）。run_simulation 開始後は確定パスの絶対パスへ更新される",
    )
    custom_physics_models: Optional[List[Any]] = Field(
        None,
        description="オプション: run_mpemba 等から注入する追加物理モデル",
    )

    arch: Literal["gpu", "cpu"] = Field("gpu", description="Taichi の arch（ti.init）")

    max_time_p: float = Field(10.0, gt=0, description="シミュレーションの最大実行時間（秒）")
    ramp_time_p: float = Field(1.0, ge=0, description="立ち上げ時間（秒）。この間外力が徐々に立ち上がる")

    steady_detection: bool = Field(True, description="定常状態を自動検知して早期終了するか")
    """定常状態を自動検知して早期終了するか"""
    steady_window_p: float = Field(1.0, gt=0, description="定常検知の判定ウィンドウ（秒）")
    """定常検知の判定ウィンドウ（秒）"""
    steady_tolerance: float = Field(0.001, gt=0, description="定常検知の許容誤差割合")
    """定常検知の許容誤差割合"""
    steady_extra_p: float = Field(2.0, ge=0, description="定常検知後の追加実行時間（秒）")
    """定常検知後の追加実行時間（秒）"""

    target_video_fps: Optional[float] = Field(None, description="出力動画の目標FPS（指定時は vis_interval を自動計算）")
    """出力動画の目標FPS（指定時は vis_interval を自動計算）"""

    visualization_mode: Literal["realtime", "offline", "none"] = Field(
        "realtime",
        description="可視化モード",
    )
    """可視化モード。計算速度を極限まで上げたい場合やデータ出力だけ欲しい場合は「none」にします。"""
    visualization_queue_size: int = Field(8, ge=1, description="非同期可視化のキューサイズ")
    """非同期可視化のキューサイズ。"""
    visualization_drop_policy: Literal["drop_oldest", "drop_newest"] = Field("drop_oldest")
    """非同期可視化のドロップポリシー"""

    data_export_interval: int = Field(0, ge=0, description="機械学習用データの保存間隔ステップ（0で無効）")
    data_export_start_p: float = Field(5.0, ge=0, description="データ保存を開始する物理時間（秒）")

    state: Dict[str, Any] = Field(default_factory=dict, description="状態維持用の辞書")
    artifact_parent: Optional[str] = Field(None, description="結果を出力する親ディレクトリ（run_subdir 配下を作成）")
    paths_out: Optional[Dict[str, Any]] = Field(None, description="パス結果を呼び出し元に返すミュータブル辞書")
    vti_dir: Optional[str] = Field(None, description="明示的な VTI 出力ディレクトリ（未指定時は <out>/vti）")

    flow_type: str = Field("counter", description="熱交換器などでの向き（例: counter / co-current）。geometry が参照する場合用")

    @field_validator("target_video_fps", mode="before")
    @classmethod
    def _norm_target_fps(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        return float(v)

    @field_validator("visualization_mode", mode="before")
    @classmethod
    def _norm_visualization_mode(cls, v: Any) -> str:
        s = str(v).lower().strip()
        return s if s in ("realtime", "offline", "none") else "realtime"

    @field_validator("visualization_drop_policy", mode="before")
    @classmethod
    def _norm_drop_policy(cls, v: Any) -> str:
        s = str(v).lower().strip()
        return s if s in ("drop_oldest", "drop_newest") else "drop_oldest"

    @property
    def fp_dtype(self):
        return ti.f32 if self.fp_dtype_str == "float32" else ti.f16

    @property
    def dx(self) -> float:
        return self.Lx_p / self.nx

    @property
    def dt(self) -> float:
        return self.dx * self.u_lbm_inlet / self.U_inlet_p

    # --- 旧コード互換: フラットアクセス → サブモデルへ委譲 ---

    @property
    def vis_interval(self) -> int:
        return self.render.vis_interval

    @vis_interval.setter
    def vis_interval(self, value: int) -> None:
        self.render.vis_interval = value

    @property
    def filename(self) -> str:
        return self.render.filename

    @filename.setter
    def filename(self, value: str) -> None:
        self.render.filename = value

    @property
    def output_format(self) -> Literal["gif", "mp4"]:
        return self.render.output_format

    @output_format.setter
    def output_format(self, value: Literal["gif", "mp4"]) -> None:
        self.render.output_format = value

    @property
    def vti_export_interval(self) -> int:
        return self.render.vti_export_interval

    @vti_export_interval.setter
    def vti_export_interval(self, value: int) -> None:
        self.render.vti_export_interval = value

    @property
    def vti_path_template(self) -> str:
        return self.render.vti_path_template

    @vti_path_template.setter
    def vti_path_template(self, value: str) -> None:
        self.render.vti_path_template = value

    @property
    def n_particles(self) -> int:
        return self.particles.n_particles

    @n_particles.setter
    def n_particles(self, value: int) -> None:
        self.particles.n_particles = value

    @property
    def particles_inject_per_step(self) -> int:
        v = self.particles.particles_inject_per_step
        if v is None:
            return max(1, self.particles.n_particles // 50)
        return int(v)

    @particles_inject_per_step.setter
    def particles_inject_per_step(self, value: int) -> None:
        self.particles.particles_inject_per_step = int(value)

    @model_validator(mode="before")
    @classmethod
    def _nest_subconfigs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)

        if "benchmark" in data:
            if "benchmark_name" not in data:
                data["benchmark_name"] = data["benchmark"]
            del data["benchmark"]

        render_d: Dict[str, Any] = {}
        if isinstance(data.get("render"), dict):
            render_d.update(data["render"])
        for key in (
            "vis_interval",
            "filename",
            "output_format",
            "vti_export_interval",
            "vti_path_template",
        ):
            if key in data and key not in render_d:
                render_d[key] = data[key]
                del data[key]
        if render_d:
            data["render"] = render_d

        part_d: Dict[str, Any] = {}
        if isinstance(data.get("particles"), dict):
            part_d.update(data["particles"])
        for key in ("n_particles", "particles_inject_per_step"):
            if key in data and key not in part_d:
                part_d[key] = data[key]
                del data[key]
        if part_d:
            data["particles"] = part_d

        return data

    @field_validator("rotation_axis", mode="before")
    @classmethod
    def _norm_rotation_axis(cls, v: Any) -> str:
        return str(v).lower() if v is not None else "z"

    @model_validator(mode="after")
    def _post_init(self) -> SimConfig:
        # 粒子注入: 未指定なら従来どおり max(1, n_particles // 50)
        if self.particles.particles_inject_per_step is None:
            self.particles.particles_inject_per_step = max(1, self.particles.n_particles // 50)

        # inlet から代表 LBM 流速
        max_u_lbm = 0.0
        for _bc_id, bc_info in self.boundary_conditions.items():
            if bc_info.get("type") == "inlet":
                v = bc_info.get("velocity", [0.0, 0.0, 0.0])
                v_mag = max(abs(v[0]), abs(v[1]), abs(v[2]))
                if v_mag > max_u_lbm:
                    max_u_lbm = float(v_mag)
        if max_u_lbm > 0.0:
            self.u_lbm_inlet = max_u_lbm

        if self.delta_T_ref is None:
            default_delta_t = self.T_wall_p - self.T_inlet_p
            if abs(default_delta_t) < 1e-12:
                default_delta_t = 1.0
            self.delta_T_ref = default_delta_t

        if self.cylinder_center is None:
            self.cylinder_center = [
                self.nx * 0.5,
                self.ny * 0.5,
                self.nz * 0.75,
            ]

        axis_str = self.rotation_axis.lower()
        if axis_str == "x":
            self.rot_axis_id = 0
        elif axis_str == "y":
            self.rot_axis_id = 1
        else:
            self.rot_axis_id = 2

        if self.omega_cylinder_phys is not None:
            self.omega_cylinder = float(self.omega_cylinder_phys) * float(self.dt)

        log.debug(
            "SimConfig: benchmark=%s grid=%sx%sx%s",
            self.benchmark_name,
            self.nx,
            self.ny,
            self.nz,
        )
        return self

    def sponge_strength_amp(self, time_p: float) -> float:
        """collide_and_stream 内スポンジ粘性ブーストの全体倍率 [0, 1]。"""
        dur = self.sponge_strength_decay_duration_p
        if dur <= 0.0:
            return 1.0
        t = float(time_p) - self.sponge_strength_decay_start_p
        if t <= 0.0:
            return 1.0
        if t >= dur:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * t / dur))

    def get_materials_dict(self):
        """設定された各IDの物性から、独立して tau_f, tau_g を計算してテーブルに登録する"""
        mat_dict: Dict[Any, Tuple[Any, Any, Any]] = {}

        for cid, props in self.domain_properties.items():
            nu = props.get("nu", 1.0e-5)
            k_val = props.get("k", 0.6)
            rho = props.get("rho", 1000.0)
            Cp = props.get("Cp", 4180.0)

            nu_lbm = nu * self.dt / (self.dx**2)
            tau_f = 3.0 * nu_lbm + 0.5

            alpha = k_val / (rho * Cp)
            alpha_lbm = alpha * self.dt / (self.dx**2)
            tau_g = 3.0 * alpha_lbm + 0.5

            is_fluid_flag = 0 if nu <= 1e-12 else 1

            if cid in self.boundary_conditions:
                bc_type = self.boundary_conditions[cid].get("type", "")
                if "wall" in bc_type:
                    is_fluid_flag = 0

            mat_dict[cid] = (tau_f, tau_g, is_fluid_flag)

        return mat_dict
