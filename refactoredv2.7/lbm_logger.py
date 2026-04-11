# ==========================================================
# lbm_logger.py — シミュレーション系モジュール共通のロガー
# ==========================================================
"""
使用例::

    from lbm_logger import configure_logging, get_logger

    # エントリポイントで1回（ファイル出力を有効にする）
    configure_logging("/path/to/run.log")

    # 各モジュール
    log = get_logger(__name__)
    log.info("message")
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

ROOT_NAME = "lbm"


def configure_logging(
    log_file_path: str,
    *,
    console: bool = True,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    ルートロガー `lbm` に FileHandler（UTF-8）と任意で StreamHandler を付与する。
    同一プロセスで再度呼ぶと既存ハンドラを置き換える。
    """
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    `lbm` 配下のロガーを返す。`name` は通常 `__name__` を渡す。
    `configure_logging` 前でも呼べるが、その場合は親に届かず警告のみの可能性がある。
    """
    base = logging.getLogger(ROOT_NAME)
    if not name or name == ROOT_NAME:
        return base
    if name.startswith(f"{ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")
