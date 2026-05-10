#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旧 main.run_simulation(**kwargs) で kwargs から除去していたキーが、SimConfig で表現できるか検証する。

実行: conda run -n lbm-sim python refactoredv2.7/verify_simconfig_main_parity.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    # 旧 main が SimConfig に渡す前に pop していたキー（kwargs 側で消費していたもの）
    MAIN_POPPED_KEYS = frozenset(
        [
            "arch",
            "benchmark",
            "state",
            "data_export_interval",
            "data_export_start_p",
            "ramp_time_p",
            "steady_detection",
            "steady_window_p",
            "steady_tolerance",
            "steady_extra_p",
            "max_time_p",
            "target_video_fps",
            "visualization_mode",
            "visualization_queue_size",
            "visualization_drop_policy",
            "artifact_parent",
            "paths_out",
            "out_dir",
            "vti_dir",
            "output_format",
        ]
    )

    # render / particles に昇格させるフラットキー（SimConfig に同名フィールドなしでも有効）
    HOIST_RENDER = frozenset(
        [
            "vis_interval",
            "filename",
            "output_format",
            "vti_export_interval",
            "vti_path_template",
            "target_video_fps",
        ]
    )
    HOIST_PARTICLES = frozenset(["n_particles", "particles_inject_per_step"])

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from config import SimConfig, RenderConfig, ParticleConfig  # noqa: E402

    top_level = set(SimConfig.model_fields.keys())
    render_keys = set(RenderConfig.model_fields.keys())
    particle_keys = set(ParticleConfig.model_fields.keys())

    missing_direct: list[str] = []

    specials = {"benchmark"}  # → benchmark_name + before バリデータ

    HOIST_RENDER_FOR_POPPED = MAIN_POPPED_KEYS & HOIST_RENDER
    MAIN_POPPED_RELEVANT = MAIN_POPPED_KEYS - HOIST_RENDER_FOR_POPPED - specials - {"output_format"}

    for k in sorted(MAIN_POPPED_RELEVANT):
        if k not in top_level:
            missing_direct.append(k)

    if HOIST_RENDER_FOR_POPPED:
        bad = HOIST_RENDER_FOR_POPPED - render_keys - {"output_format", "filename"}  # 全部 render にあるべき
        if bad:
            print("ERROR: 昇格 render キーが RenderConfig と一致しない:", bad)

    # output_format は render 側
    if "output_format" in MAIN_POPPED_KEYS and "output_format" not in render_keys:
        print("ERROR: output_format が RenderConfig に無い")

    print("SimConfig top-level keys count:", len(top_level))

    # benchmark → benchmark_name
    if "benchmark_name" not in top_level:
        print("ERROR: benchmark_name が無い")
        return 1

    # arch
    if "arch" not in top_level:
        print("ERROR: arch が無い")
        return 1

    if missing_direct:
        print("ERROR: SimConfig が旧 pop 済みパラメータをトップレベルで網羅していない:")
        print(" ", missing_direct)
        return 1

    print("OK: kwargs.pop で消費していたキーは SimConfig / benchmark 変換 / render 昇格で網羅されています")

    # 典型的なフラットキーが SimConfig 化できることを実際に読む（境界辞書など簡単な例）
    try:
        c = SimConfig(
            benchmark_name="test",
            arch="cpu",
            max_time_p=11.0,
            steady_detection=False,
            out_dir="/tmp/lbm_explicit_test_not_run",
            output_format="mp4",
            vis_interval=40,
            domain_properties={
                0: {"nu": 1e-5, "k": 0.6, "rho": 1000.0, "Cp": 4180.0},
                22: {"nu": 1e-5, "k": 0.6, "rho": 1000.0, "Cp": 4180.0},
            },
            boundary_conditions={
                22: {"type": "inlet", "velocity": [0.1, 0.0, 0.0], "temperature": 1.0}
            },
        )
    except Exception as e:
        print("ERROR: 合成 SimConfig が構築できない:", e)
        return 1

    assert c.max_time_p == 11.0 and c.arch == "cpu"
    assert c.output_format == "mp4" and c.vis_interval == 40
    print("OK: 合成 SimConfig のスモーク読み込み")

    print("フラット昇格チェック対象粒子キーが SimConfig で動く:")
    cp = SimConfig(benchmark_name="t", particles={"n_particles": 500}).n_particles
    assert cp == 500
    print("OK: particles=dict")

    dangling = (HOIST_RENDER - render_keys) | (HOIST_PARTICLES - particle_keys)
    if dangling:
        print("WARNING: sub-model key mismatch:", dangling)
        return 1

    print("parity check complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
