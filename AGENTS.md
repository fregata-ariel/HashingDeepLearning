# Repository agent instructions

## 適用範囲とユーザーの方針

- 対象はGitHubの`fregata-ariel/HashingDeepLearning`と、そのlocal checkout/worktree。
- ユーザーは2026-10-23に、依頼した作業の**コミット・push・PR作成までをエージェントの判断で進める**方針を明示した。毎回の最終確認を理由に未コミットで止めない。
- 完了した変更は必要な検証を行い、task branchへコミットし、`origin`へ通常pushし、適切なbaseへPRを作成する。既存PRがあれば更新する。独立レビューや未完了の受入が残る場合はDraft PRとし、理由・残件・検証範囲を本文に記す。
- 現在の統合先は`research/lsh-lineage-vendor`。PR作成前にremote/default branchと実際の統合方針を確認する。
- このファイルは作業方針と許可範囲の記録。ツールのsandbox、network、承認prefixそのものを変更しない。実行環境が追加承認を要求する場合は所定のtool approvalを使い、制限を迂回しない。

## 許可済み動作の一覧

依頼された作業の達成に必要な範囲で適用する。各IDは追記・更新の参照用に維持する。

| ID | 動作 | 範囲・条件 |
| --- | --- | --- |
| A01 | 調査・閲覧 | local source、Git履歴・差分、当該repoのIssue/PR/CI/logの確認。外部一次資料・公式依存情報の取得 |
| A02 | 実装・修復・文書化 | 依頼に関係するsource、test、CI、provenance、実行証拠、引き継ぎの作成・更新 |
| A03 | ローカル検証 | 小さく上限を設けたCPU/GPU build、数値oracle、sanitizer、lint、既存の適用test。GPU使用はGPU作業が依頼された範囲で実施 |
| A04 | Gitの作業整理 | task branchの作成、変更を確認したファイルのstage、意味のまとまりごとのlocal commit。既存のユーザー変更を勝手に取り込まない |
| A05 | 通常push | 当該repoの`origin`にtask branchをpush・追記。force push、remote削除、統合先への直接pushは含めない |
| A06 | PR作成・更新 | 当該taskのPR作成、Draft設定、title/bodyの更新。関連Issueは参照し、未完了Issueを自動closeする文言を付けない |
| A07 | CI確認と修復 | PRに適用される既存CIの状態・失敗logを確認し、依頼範囲の修復・再検証・追加commit/pushを行う |
| A08 | 依存・診断ツールの取得 | 作業に必要な公式/出典固定のpackage・sourceを、管理可能なvenv/prefix/一時領域へ取得。版・URL・hash・licenseを必要に応じ記録し、system全体を変更しない |
| A09 | 不要な一時物の削除 | 自分が今回作成・取得した不要なarchive、展開物、venv、build/cache等を対象を確認して削除。証拠と再取得情報は残し、他作業の共用物を消さない |
| A10 | 不要worktreeの整理 | 一覧、状態、未統合commit、lock等を確認。不要と確認したcleanなlinked worktreeの通常removeと、実体のない登録のprune。使用中・未コミット変更・未保全の作業は維持 |
| A11 | 引き継ぎ・進捗の保存 | branch、commit/PR/CIリンク、実行証拠、未検証項目、次の具体的手順、download cleanup状態を文書へ残す |

## コマンドパターンの一覧

`<...>`は対象を確認して置き換える引数。ここに載ることは、任意の引数・任意のrepo・任意のshell scriptの包括許可を意味しない。既存コマンドの引数（build先・SM・suite等）を、同じ作業目的の範囲で変更してよい。

| ID | コマンド/操作のパターン | 制約・用途 |
| --- | --- | --- |
| C01 | `rg ...`、`rg --files ...`、`git status/diff/log/show/branch/rev-parse/remote ...` | source・履歴・状態の調査。検索にはまずrgを使う |
| C02 | `git switch -c <task-branch>`、`git add -- <確認済みpaths...>`、`git commit -m <message>` | A04に従う。stage後にcached diffとstatを確認する |
| C03 | `git push -u origin <task-branch>`、`git push origin <task-branch>` | A05に従う。force系optionやremote branch削除を使わない |
| C04 | GitHub connectorのPR作成/更新、`gh pr create/view/diff/checks/edit ...` | A06/A07。connectorで足りる場合は優先。複数行bodyは構造化引数か`--body-file`を使う |
| C05 | GitHub connectorのIssue/CI閲覧、`gh issue list/view ...`、`gh run list/view ...` | 読み取り。PR以外のIssue/comment投稿は別途ユーザーの指示がある場合に行う |
| C06 | `git worktree list --porcelain`、`git worktree prune --dry-run --verbose`、`git worktree remove <確認済みpath>`、`git worktree prune --verbose` | A10。`--force`で未保全変更を捨てない |
| C07 | `nvidia-smi ...`、`nvcc --version`、`lscpu`、`uname -a`、`cmake --version`、`g++-12 --version` | GPU/CPU/compiler/runtimeの実機診断 |
| C08 | `cmake -S <suite> -B <fresh-build> ...`、`cmake --build <build> --parallel <bounded-jobs>`、`ctest --test-dir <build> ...` | 対象を明示したbuild/test。CPU数・時間・memoryを考慮し、必要な場合はtimeoutと`--no-tests=error`を指定 |
| C09 | `python3 tools/check_nvidia_gpu.py --build-dir <fresh-dir> ...` | 既存の必須CUDA/4診断driver。`--cuda-arch`、host compiler、sanitizer pathを実環境へ合わせる。CPU fallbackでGPU成功にしない |
| C10 | `python3 tools/check_gslide_cpu.py ...`、`python3 tools/check_gslide_source_regressions.py ...`、`python3 tools/check_gslide_train_adapter.py ...` | CPU本文oracle、原典負例、mockを使うhost callerを区別。`--sanitize`を含む必要な検証 |
| C11 | `python3 tools/validate_traceability.py ...`、`python3 tools/audit_traceability_docs.py`、`python3 tests/traceability/<対象test>.py` | 台帳・文書・対象回帰の検証。台帳のvalidを実GPU実行と混同しない |
| C12 | `bash tools/bootstrap_nvidia_sanitizer.sh <new-prefix>`、`compute-sanitizer --tool <memcheck/racecheck/initcheck/synccheck> ... <対象binary>` | 固定出典/版/hash、local prefix、必要な診断と終了時cleanupを維持 |
| C13 | `curl --fail --silent --show-error ... <確認済みURL> -o <管理対象file>`、`sha256sum ...`、`tar ...` | A08。取得先と展開先を限定し、hash確認前に取得物を実行しない。今回のprefix承認は無関係な取得/送信の許可へ拡大しない |
| C14 | project用venvへの必要package導入、taskで採用した`uv run ...`等の実行 | A03/A08。依存profileを固定・分離し、既存の共用環境やsystem Pythonを勝手に変更しない |
| C15 | `git diff --check`、`git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check`、`bash -n <対象script>` | CRLFを保持する既存CUDAファイルにはcr-at-eolを考慮。無関係な改行変換を避ける |
| C16 | `rm -- <確認済み不要file>`、`rm -rf -- <確認済みtask用dir>`、同等の限定的なunlink/rmtree | A09のみ。広域glob、未確認path、repo全体、他者の環境を対象にしない |

ツール側へ再利用可能なapproval prefixを提案する場合は、たとえば`["python3", "tools/check_nvidia_gpu.py"]`や`["python3", "tools/check_gslide_cpu.py"]`のように対象scriptまで限定する。`python3`単独、任意の`bash -c`、破壊的削除の包括prefix等は提案しない。shellのpipe・redirection・環境変数・substitutionがあると既存prefixに一致しない場合があるため、承認回避のために書き換えない。

## 個別の指示が必要な操作

コミット・PRまでの委任には、以下の操作を含めない。後にユーザーから明示された場合は、その範囲で一覧を更新する。

| ID | 操作 |
| --- | --- |
| U01 | PR merge、auto-merge有効化、保護設定変更、repo/runner/secrets/permission等の管理変更 |
| U02 | force push、公開履歴のrewrite、ユーザー変更を失うreset/clean、未保全worktreeの強制削除 |
| U03 | 有料GPU/cloudの起動、購入、予算上限変更、大規模model/dataset取得、定期GPU jobの開始 |
| U04 | 当該repo外への変更、system package/driverの置換、他作業の共用cache/venvの削除 |
| U05 | Slack/email等への送信、依頼に含まれないIssue/comment投稿、未完了Issueのclose |

## 実装・検証の基準

- `third_party/`は固定原典。修復と保守は`ports/`へ、testは`tests/`へ、出典・paper/code/test対応は`papers/`へ置く。
- CPU実行、CPU本文adapter、mock、SIMDe、native ISA、原典CUDA、maintained CUDA、共有SYCL CPU/GPU、model全体を別の証拠として管理する。
- 独立した数値・状態oracle、実行profile、source/compiler/device/binaryのidentityを記録する。skip、未対応、未実行、FAILをPASSへ置き換えない。
- GPUの存在だけでdtype/library/SM対応を仮定しない。性能比較では同条件を揃え、許容低下率を測定前に決める。
- 変更に必要な検証を行い、成功後の無目的な全面再実行を避ける。対象CIを確認し、local成功とGitHub上のCI成功を分けてPRへ書く。
- downloadした不要物は削除する。復元に必要なURL/版/hash/licenseと実行証拠は保存し、cleanup結果を報告する。
- 詳細な統合規約は[RESEARCH_INTEGRATION_PLAYBOOK.md](docs/RESEARCH_INTEGRATION_PLAYBOOK.md)、CIは[CI_ARCHITECTURE.md](docs/CI_ARCHITECTURE.md)を参照する。

## 一覧の更新方法

1. 新しい許可・制約が明示されたら、対応するA/C/U項目を更新する。新規項目には未使用IDを付ける。
2. 対象repo、動作、引数/対象path、適用条件、例外を明記する。一度の具体的なコマンド実行を無制限の許可へ広げない。
3. 環境のtool approvalと、ユーザーが委任した作業範囲を混同しない。既存承認・ユーザーの最新指示を確認し、同じ判断を毎回聞き直さない。
4. 更新したAGENTS.mdも通常の変更としてコミット・PRへ含める。

## 方針の記録

| 日付 | 根拠・変更 |
| --- | --- |
| 2026-10-22 | GPU実行テストとCI定義を開始。G-SLIDEを優先する回答を受領。task branch作成、ホスト側GPU/CPU診断、公式診断tool取得を実施 |
| 2026-10-22 | ユーザーから、取得物は不要になったら削除するよう明示あり |
| 2026-10-23 | 不要なGit worktreeの整理を依頼。使用中のworktree 1個のみで削除対象なし |
| 2026-10-23 | 当該repoでコミット・PRまではエージェント判断で進める方針を明示。今回のコマンドを汎化した一覧をAGENTS.mdへ初期化する依頼 |
| 2026-10-23 | 後の許可追加に備え、許可動作・コマンド・個別指示が必要な操作をID付き一覧で管理する依頼 |
