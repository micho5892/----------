# エージェント向けメモ

## Python の実行環境

このリポジトリでは **conda 環境 `lbm-sim`** を使う。`(base)` や別環境の `python` でコマンドを実行しない（依存が欠けて失敗する）。

推奨:

```powershell
conda run -n lbm-sim python lbm_ui_designer/lbm_param_optimize.py -c lbm_ui_designer/lbm_param_optimize.example.yaml
```

または先に `conda activate lbm-sim` してから `python` / `streamlit`。

環境定義のたたき台はルートの `environment.yml`。詳細は `.cursor/rules/lbm-sim-environment.mdc` を参照。
