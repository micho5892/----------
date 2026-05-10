# Git ワークツリー：作成・マージ・削除（初心者向け）

この文書では、**同じリポジトリを「別フォルダ」にもう一つ用意して、別ブランチで並行作業する**ための **Git ワークツリー（worktree）** の流れを、コマンドとともに説明します。

コマンド例は **PowerShell** を想定しています。パスはご自身の環境に読み替えてください。

---

## 0. まず押さえる用語

| 用語 | ざっくり意味 |
|------|----------------|
| **リポジトリ（repo）** | プロジェクトの履歴（コミット）が入っている `.git` を含むフォルダ一式 |
| **ブランチ** | 作業ライン。たとえば `main` はいつも安定させたいライン、実験用は別名にすることが多い |
| **ワークツリー** | **ディスク上のフォルダ**。通常はリポジトリを開いたときの「そのフォルダ」が 1 つ |
| **Git ワークツリー追加** | **同じ `.git` を共有しつつ、別フォルダで別ブランチをチェックアウトする**機能 |

**なぜ使うか**

- `main` を開いたフォルダでは軽い修正だけする  
- **別フォルダ**では長い実験ブランチ（例：`feat/rl-impl`）だけを進める  

といった **物理的な分離**ができます。フォルダを間違えにくくなるのがメリットです。

---

## 1. 作成前の確認（おすすめ）

いつものリポジトリのルート（例：`流体シミュレーション`）で次を実行します。

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション"
git status
git branch
```

- **`git status`**: 未コミットの変更がないか確認  
  - 変更が大量にあると切り替え・追加時に迷いやすいので、できればコミットするか退避してから進めると安全です  
- **`git branch`**: いまどのブランチにいるか（`*` が付いている行）

リモートの最新を取りたい場合:

```powershell
git fetch
```

---

## 2. ワークツリーを作成する

### 2-1. すでにあるブランチを別フォルダに出す

**パターン A**：ブランチ `feat/example` が既に存在するとき。

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション"

git worktree add "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション-wt-feat-example" feat/example
```

- **第 1 引数**: 新しく作るフォルダのパス（**空のフォルダ**になる。既にファイルだらけだと失敗しやすい）  
- **第 2 引数**: そのフォルダでチェックアウトしたい **ブランチ名**

### 2-2. 新しいブランチを作りながら追加する

**パターン B**：ブランチ自体もこれから作るとき。

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション"

git worktree add -b feat/example "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション-wt-feat-example"
```

- **`-b feat/example`**: この名前のブランチを新規作成し、そのブランチを新フォルダに関連付ける  

### 2-3. 作成できたか確認

```powershell
git worktree list
```

- メインのフォルダと、追加したフォルダの両方が表示され、それぞれブランチが付いていれば成功です  

---

## 3. 追加フォルダ側での作業（いつもの Git）

追加したフォルダに移動して、普段どおり編集・コミットします。

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション-wt-feat-example"

git status
git add .
git commit -m "feat: 〇〇を実装"
```

リモート（GitHub など）へ載せる場合:

```powershell
git push -u origin feat/example
```

**注意**

- メインフォルダと追加フォルダは **同じリポジトリ**なので、**同じブランチを二つのフォルダで同時に触る**と混乱しやすいです。基本は **フォルダごとにブランチを分ける**イメージで使います。

---

## 4. 作業完了後に「main」へ取り込む（マージ）

「マージ」とは、**別ブランチで積んだ変更を、取り込み先ブランチ（多くは `main`）の履歴に合体させる**ことです。やり方は大きく二つあります。

### 4-1. GitHub などでプルリクエスト（PR）してマージ（チーム運用でよく使う）

1. `feat/example` を `push` 済みにする  
2. リモートで **Pull Request** を作る（`feat/example` → `main`）  
3. レビュー後、**Merge** ボタンでマージ  

その後、手元の `main` を最新にします。

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション"
git checkout main
git pull
```

### 4-2. ローカルだけでマージする（シンプルな例）

**ワークツリー追加フォルダではなく**、普段使いのリポジトリルートで:

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション"

git checkout main
git pull
git merge feat/example
```

コンフリクトが出た場合は、Git が印を付けたファイルを直してから:

```powershell
git add （直したファイル）
git commit -m "merge: feat/example を main に取り込み"
```

マージが終わり、リモートに `main` を反映するなら:

```powershell
git push
```

---

## 5. ワークツリーを削除する

作業ブランチの内容が `main` に取り込まれ、**追加フォルダがもう不要**になったら削除します。

### 5-1. 正しい手順（推奨）

**追加していたフォルダを Git に任せて削除**します（手でフォルダだけ消すより安全です）。

```powershell
cd "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション"

git worktree remove "C:\Users\hainy\OneDrive\デスクトップ\短大\流体シミュレーション-wt-feat-example"
```

- そのフォルダでエディタやターミナルが開いていると **削除に失敗**することがあります。**Cursor でそのフォルダを閉じる**、ターミナルのカレントを別パスにする、などしてから再試行してください。

削除できたか確認:

```powershell
git worktree list
```

### 5-2. 参照の整理（必要なときだけ）

通常は不要ですが、怪しいときは:

```powershell
git worktree prune
```

### 5-3. ブランチ自体も消すかどうか

ワークツリーを消しただけでは **ブランチ `feat/example` はまだ残ります**。マージ済みで不要なら、メインリポジトリで:

```powershell
git branch -d feat/example
```

まだマージされていない変更だけが残っているブランチを消すと Git が止めてくれるので、そのときは内容を確認してから `-D`（強制削除）を検討します（**データが消えるので注意**）。

---

## 6. よくあるつまずき（Windows / OneDrive / Cursor）

| 現象 | 考えられる原因 | 対処のヒント |
|------|----------------|--------------|
| `Permission denied` で `worktree remove` できない | `.git` 配下やフォルダが **ロック**されている | Cursor で該当フォルダを閉じる、プロセスを終了、OneDrive の同期完了を待つ |
| ソース管理に変更が出ない（エディタ側） | GUI と実際の `git status` のずれ | 統合ターミナルで `git status` を信頼して `add` / `commit` |
| 間違えて別リポジトリを開いている | パスが `lbm-rl` など別フォルダ | **リポジトリのルート**を `git rev-parse --show-toplevel` で確認 |

---

## 7. 最短チェックリスト

1. **作成**: `git worktree add ...` または `git worktree add -b 新ブランチ ...`  
2. **作業**: 追加フォルダでコミット（必要なら `push`）  
3. **マージ**: PR でマージ、または `main` で `git merge ブランチ名`  
4. **削除**: メインルートで `git worktree remove 追加フォルダのパス`  
5. **掃除**: ブランチ不要なら `git branch -d`  

---

## 参考コマンド一覧

```powershell
git worktree list          # 一覧
git worktree add PATH BRANCH
git worktree add -b NEW_BRANCH PATH
git worktree remove PATH
git worktree prune
```

関連: 日々のブランチ操作は `docs/git-使い方.md` も参照してください。
