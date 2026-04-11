# Multi-material Analytics Migration Plan

このドキュメントは、`analytics.py` を単一物性前提から多物性・多形状対応へ段階的に移行する実装計画。

## Goal

- `k_p` の単一定数依存を除去し、局所物性ベースで Nu を計算する。
- 既存ベンチマーク（特に `parallel_plates`）を壊さず、段階的に拡張する。
- 将来的に形状別の Nu 定義を差し替え可能にする。

## Phase 1 (実装対象)

- `Analytics` 初期化時に `domain_properties` から `cell_id -> k` テーブルを構築。
- `get_local_Nu()` で `k_p` を使わず、壁面フェイスごとに `k_face`（調和平均）で熱流束を評価。
- Nu の分母 `k_ref` は断面流体の平均 `k_bulk_ref` を使用。
- 既存 API 名は維持し、`main.py` の呼び出し変更を最小化。
- `config.py` に `delta_T_ref`, `T_inlet_p`, `T_wall_p` の互換属性を追加。

## Phase 2

- `is_fluid` 判定を `materials_dict` と統一（ID固定判定を段階的に廃止）。
- `k`, `rho`, `Cp`, `nu` の lookup を analytics 用に統一インターフェース化。
- ログに `k_ref` と Nu 定義を明記。

## Phase 3

- 壁面フェイス抽出を `wall_metrics.py` へ分離し、analytics から幾何詳細を切り離す。
- `parallel_plates`, `cylinder`, 複雑内部流路で同じ集計関数を使えるようにする。

## Phase 4

- Nu 定義をプラガブル化（差し替え可能にする）。
  - internal flow: `Nu = h * D_h / k_ref`
  - cylinder: `Nu_theta = h(theta) * D / k_ref`
- benchmark ごとに定義セットを選択する。

## Validation Checklist

- `parallel_plates`: 既存結果と同オーダーで推移する。
- 壁材 `k` を変えると Nu が妥当に変化する。
- 単一物性ケースで旧実装との差分が過度に大きくない。
- 既存ログ・可視化が継続して出力される。
