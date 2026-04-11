# Git の使い方（場面別・コマンド例付き）

コマンド例は PowerShell / Git Bash どちらでも同じように動くことが多いです（パスに日本語があってもそのまま使えます）。

---

## 1. いまの状態を確認したい

**いつ使う**: 何が変わったか、どのブランチにいるか分からなくなったとき。

```powershell
git status
git branch
git log --oneline -10
```

- `status`: 変更・ステージの有無
- `branch`: ブランチ一覧（先頭に `*` がいまのブランチ）
- `log`: 直近のコミット履歴（短く表示）

---

## 2. 変更を「履歴」として残す（日常のコミット）

**いつ使う**: 試行が一段落した、ここまで戻したい地点ができたとき。

```powershell
git add refactoredv2.7/boundary.py
git commit -m "試: 壁境界の〇〇を変更"
```

複数ファイルまとめて:

```powershell
git add refactoredv2.7/
git commit -m "refactoredv2.7: 境界まわりを更新"
```

**注意**: `git add .` はカレント以下全部が対象。巨大ファイルが `.gitignore` に入っていないと誤ってコミットしやすいです。

---

## 3. 直前のコミットメッセージだけ直したい

**いつ使う**: コミットしたあとで「説明が雑だった」と気づいたとき（まだ `push` していない前提が安全）。

```powershell
git commit --amend -m "試: 壁境界の〇〇を変更（説明を修正）"
```

**既にリモートに `push` 済み**なら、履歴を書き換えるので原則は避け、代わりに「打ち消しコミット」（後述の `revert`）や新しいコミットで説明を足す方が無難です。

---

## 4. 大きな改修・実験を分けたい（ブランチ）

**いつ使う**: `main` を動く状態に保ちつつ、長めの作業や壊れやすい変更をしたいとき。

```powershell
git checkout main
git pull
git checkout -b exp/boundary-refactor
```

作業してコミットを積む。終わって `main` に取り込む:

```powershell
git checkout main
git merge exp/boundary-refactor
git branch -d exp/boundary-refactor
```

**コンフリクト**が出たら、Git が「衝突」と印を付けたファイルをエディタで直し:

```powershell
git add 直したファイル
git commit -m "merge: exp/boundary-refactor を取り込み"
```

---

## 5. 「あの時点のコード」を見たいだけ（戻さず確認）

**いつ使う**: 過去の挙動と比べたい、当時のファイルを読みたいとき。

```powershell
git log --oneline
git checkout abc1234
```

この状態は **detached HEAD**（ブランチの先端ではない）になりがちです。**ここから新しい作業を続ける**なら、その場でブランチを切ると安全です。

```powershell
git checkout -b try/from-abc1234
```

**元の `main` に戻る**:

```powershell
git checkout main
```

---

## 6. 悪い変更を「なかったこと」にしたい（よくある3パターン）

### 6-A. まだコミットしていない（ファイルだけ戻す）

作業ツリーで特定ファイルを最後のコミットの内容に戻す:

```powershell
git restore refactoredv2.7/boundary.py
```

全部の未コミット変更を捨てる（**取り返しがつきにくい**）:

```powershell
git restore .
```

### 6-B. 直近のコミットを打ち消すが、履歴は残したい（共有リポジトリ向き）

**いつ使う**: もう `push` したあとで、安全に巻き戻したいとき。

```powershell
git revert HEAD
```

複数コミット分まとめて戻す場合は、対象の範囲を指定するやり方もあります（慣れてからでよいです）。

### 6-C. ローカルだけで「ブランチの先を過去に戻す」（1人開発でよく使う）

**いつ使う**: まだ共有していない `main` で、「このコミット以降は全部いらない」と決めたとき。

```powershell
git reset --hard abc1234
```

**危険**: その後のコミットはブランチから見えなくなることがあります。迷うなら先に別ブランチやタグで退避してください。

```powershell
git branch backup/before-reset
git reset --hard abc1234
```

---

## 7. 「この版」を名前で固定したい（タグ）

**いつ使う**: 論文用・レポート用、危ない実験の直前の「安全地点」。

```powershell
git tag paper-2026-04-11-fig3
```

過去のコミットに付ける:

```powershell
git tag baseline-old abc1234
```

一覧:

```powershell
git tag -l
```

タグ付きの状態をチェックアウト:

```powershell
git checkout paper-2026-04-11-fig3
```

---

## 8. リモート（GitHub 等）とつなぐ

**初回（ローカルに既にリポジトリがある場合）**:

```powershell
git remote add origin https://github.com/ユーザー名/リポジトリ名.git
git push -u origin main
```

**いつもの送受信**:

```powershell
git pull
git push
```

**別 PC で続ける**:

```powershell
git clone https://github.com/ユーザー名/リポジトリ名.git
cd リポジトリ名
```

---

## 9. 一時的に手を止めたいがコミットしたくない（stash）

**いつ使う**: 急に別ブランチで直したいが、いまの変更をコミットするほどでもないとき。

```powershell
git stash push -m "WIP: 境界の試行"
git checkout main
```

戻す:

```powershell
git stash list
git stash pop
```

---

## 10. うっかり消した・戻しすぎた（reflog）

**いつ使う**: `reset` 後に「やっぱりあのコミットが欲しい」となったとき。

```powershell
git reflog
```

表示されたハッシュに戻る例:

```powershell
git checkout def5678
```

必要ならそこからブランチを切る。

---

## 11. このプロジェクト向けの注意（補足）

- **OneDrive 上**のリポジトリは、同期と Git が同時に触れることがあるので、巨大成果物はルートの `.gitignore` に寄せる運用が向いています。
- **複数バリアント**（`refactoredv2.7` と `refactoredv2.7.2`）は「同じコミット内で並べる」と、**同じ履歴のまま A/B のツリー比較**がしやすいです。Git のブランチは「長く分岐する線」向き、という整理で併用すると扱いやすいです。

---

## クイック対応表

| やりたいこと     | 使うイメージ                          |
|------------------|---------------------------------------|
| 変更の一覧       | `git status`                          |
| 履歴に残す       | `git add` → `git commit`              |
| 大きい作業を分ける | `git checkout -b` → あとで `merge`   |
| 過去をのぞく     | `git checkout コミット`             |
| 版に名前を付ける | `git tag`                             |
| 未コミットを捨てる | `git restore ファイル`              |
| push 後に安全に戻す | `git revert`                        |
| ローカルで強く戻す | `git reset --hard`（要注意）        |

---

## Cursor の GUI について

同じ操作は **Source Control（ソース管理）** パネルからも行えます（変更のステージ、コミット、ブランチ切り替え、履歴の表示など）。GUI だけの手順が必要なら、このファイルに追記してください。
