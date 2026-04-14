# 新PCでのMiniforge環境再構築手順

この手順は、本リポジトリを新しいPCへ移し、`lbm-sim` 環境で実行可能にするためのものです。

## 0. 事前準備

- 新PCに Miniforge をインストールする
- PowerShell を再起動する
- `conda --version` が通ることを確認する

```powershell
conda --version
```

## 1. リポジトリを新PCへ用意する

### A) Git管理している場合（推奨）

```powershell
git clone <リポジトリURL>
cd <リポジトリ名>
```

### B) フォルダコピーで移行する場合

- 旧PCのプロジェクトフォルダを zip / OneDrive / 外部ストレージで新PCへコピー
- 新PCで展開して、そのフォルダを PowerShell で開く

```powershell
cd "<展開したプロジェクトフォルダ>"
```

## 2. conda環境を作成する

プロジェクトルートにある `environment.yml` から環境を作成します。

```powershell
conda env create -f environment.yml
```

すでに `lbm-sim` が存在する場合は更新に切り替えます。

```powershell
conda env update -n lbm-sim -f environment.yml --prune
```

## 3. 環境を有効化する

```powershell
conda activate lbm-sim
```

## 4. 最低限の動作確認

```powershell
python -c "import taichi; print('taichi:', taichi.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
```

必要なら、プロジェクトの簡易実行で確認します。

```powershell
python refactoredv2.7/run_benchmark_ibm_y_wall_channel.py
```

## 5. このプロジェクトでの実行ルール

- Python実行は必ず `lbm-sim` 環境で行う
- `base` 環境やシステムPythonでの実行は避ける
- 迷ったら次の形式で実行する

```powershell
conda run -n lbm-sim python refactoredv2.7/main.py
```

## 6. よくあるトラブル

### 6-1. `ModuleNotFoundError: No module named 'taichi'`

- 原因: 環境が有効化されていない
- 対処:

```powershell
conda activate lbm-sim
python -c "import taichi"
```

### 6-2. `PackagesNotFoundError` や解決失敗

- ネットワーク不安定時に起きることがあります
- 再実行、または `conda clean --index-cache` 後に再試行

```powershell
conda clean --index-cache
conda env create -f environment.yml
```

### 6-3. GPUまわりが不安定

- まずCPUで動作確認して切り分ける
- NVIDIAドライバを更新する

## 7. 補足（再現性をさらに高めたい場合）

旧PCで lock 相当ファイルを作って持ち運ぶと、環境差分を減らせます。

```powershell
conda activate lbm-sim
conda env export --no-builds > environment.lock.yml
```

新PCで:

```powershell
conda env create -f environment.lock.yml
```

