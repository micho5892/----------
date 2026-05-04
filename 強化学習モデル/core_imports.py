"""refactoredv2.7 配下モジュールの読み込みアダプタ。"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


_MODULE_FILES = {
    "context": "context.py",
    "config": "config.py",
    "geometry": "geometry.py",
    "solver": "solver.py",
    "analytics": "analytics.py",
    "boundary": "boundary.py",
    "physics": "physics.py",
}

_REF_DIR = Path(__file__).resolve().parents[1] / "refactoredv2.7"


def _load_module(module_name: str, file_name: str) -> ModuleType:
    module_path = _REF_DIR / file_name
    if not module_path.exists():
        raise ModuleNotFoundError(
            f"必要なモジュール '{module_name}' が見つかりません: {module_path}"
        )

    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"モジュール仕様を作成できません: {module_path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_modules() -> dict[str, ModuleType]:
    if not _REF_DIR.exists():
        raise ModuleNotFoundError(
            f"依存ディレクトリ 'refactoredv2.7' が見つかりません: {_REF_DIR}"
        )
    ref_dir_str = str(_REF_DIR)
    if ref_dir_str not in sys.path:
        sys.path.insert(0, ref_dir_str)

    loaded: dict[str, ModuleType] = {}
    for module_name, file_name in _MODULE_FILES.items():
        if module_name in sys.modules:
            loaded[module_name] = sys.modules[module_name]
            continue
        loaded[module_name] = _load_module(module_name, file_name)
    return loaded


try:
    _loaded_modules = _bootstrap_modules()
except Exception as exc:
    raise ImportError(
        "refactoredv2.7 の読み込みに失敗しました。"
        " 依存ファイルの存在と import エラーを確認してください。"
    ) from exc

SimConfig = _loaded_modules["config"].SimConfig
SimulationContext = _loaded_modules["context"].SimulationContext
GeometryBuilder = _loaded_modules["geometry"].GeometryBuilder
LBMSimulator = _loaded_modules["solver"].LBMSimulator
Analytics = _loaded_modules["analytics"].Analytics
BoundaryManager = _loaded_modules["boundary"].BoundaryManager
PhysicsManager = _loaded_modules["physics"].PhysicsManager

__all__ = [
    "SimConfig",
    "SimulationContext",
    "GeometryBuilder",
    "LBMSimulator",
    "Analytics",
    "BoundaryManager",
    "PhysicsManager",
]
