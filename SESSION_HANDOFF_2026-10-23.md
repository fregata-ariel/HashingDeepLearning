# 次セッションへの引き継ぎ — 全系統のGPUテスト

## SYCLライブラリ化に集中（2026-10-26追記）

ユーザーから、意味論や安定APIの追加判断は後に回し、判断が影響しない部分のライブラリ化を進める指示。
今回の対象はpublic entry headers、standalone CMake install/export、relocatable find_package、外部consumer検証。
[package guide](ports/g-slide/library/README.md)を参照。既存の数値処理・共有controllerの本文は変更しない。
外部consumerはソースを証拠dirへコピーし、インストール先を移動した後、checkout includeなしでビルドする。
SDK不要hostと実SYCL CPU/NVIDIAで、複数translation unit・部品呼出・1更新＋評価の独立FP64 oracleを確認する。
CPU/NVIDIA各15/15既存スイート＋2/2外部consumer、SDK不要host consumerが成功。
[保存済み証拠](docs/evidence/gslide-package-ci-2026-10-26.json)を参照。意味論の追加判断は保留。
CIのartifactがcheckout内にある場合にも、consumerを外部一時領域でbuild/install/移動/利用し、snapshot保存後にscratchを削除する修正を含む。
最新公開commit/CIはPR #127を参照。

## 名前付きAPIと意味論の確定（2026-10-25追記）

推奨案の採用を受け、ModelSpec/IndexConfig/WorkspaceLimits/ExecutionOptionsとInitialStateを分離。
SDK不要のSpan/MatrixView/SegmentedIdsView、容量計算、checked component API、ModelState/IndexState/TrainingWorkspaceを実装。
Networkはqueueを保持し、旧constructorとpointer評価は互換adapterとして維持。
意味論は[論文・固定原典の照合](docs/GSLIDE_SEMANTICS.md)を優先し、今回のAPI整理で数値を変えない。
bias込みのSYCL評価、重複ラベルのmembership/listed-count、自然対数hash packing、順序未保証のbounded ring、更新後rebuildを明記。
論文はsave時のlayout変換に言及するが、固定原典の公開APIにはsave/loadがない点も区別した。
新規host configuration/viewsを含め、15スイートをCPU/NVIDIAで必須とする。公開commit/最新CIと実行証拠はPR #127を参照。

## G-SLIDEの設計整理（2026-10-25追記）

全機能追加の前に部品化するユーザー方針へ更新。[ライブラリ設計](docs/GSLIDE_LIBRARY_DESIGN.md)を優先する。
既存演算をhost contracts・USM/parameter ownership・候補選択・Softmax/backward・Adamへ分離し、`Network`を合成役にした。
新規機能の移植ではなく、既存数値挙動の分解。CPU/NVIDIA各13テストとSDK不要host 2テストが成功し、[証拠](docs/evidence/gslide-library-2026-10-25.json)を保存。
再検証は `bash tools/run_gpu_container.sh /tmp/gslide-library-NEW sycl run gslide-library`。
この節は最初の部品化checkpoint。named configuration/viewと意味論は上記追記を優先する。多層化・checkpoint追加・最適化は別作業とする。

## 最新の再開結果（2026-10-25）

以下の2026-10-23本文は履歴。現状は[全系統GPU checkpoint](docs/GPU_LINEAGE_CHECKPOINT.md)と[保存済みJSON](docs/evidence/gpu-lineages-2026-10-25.json)を優先する。

- PR #127のincoming head `f5812af`はCI run `39813806042`の3ジョブすべて成功（実checkoutはmerge candidate `680c7ff`）。
- ユーザーのDockerfile化依頼を実装。再実行は `bash tools/run_gpu_container.sh /tmp/hashing-gpu-NEW sycl`。GPU1台、非root、read-only、networkなし、resource上限付き。Docker/tool approval自体を不要にするものではない。
- 全列挙profileで実行成功。G-SLIDE native CUDA、MONGOOSE原典CUDA/実wrapper/Torch GPU fallback/CUDA RNG replay、Scatterbrain maintained CUDA/非causal hybrid/2更新/save-reload、MagicPIG小部品、共有SYCL NVIDIAのG-SLIDE/MONGOOSE/Scatterbrainを確認。
- G-SLIDEとScatterbrainのmemcheck/racecheck/initcheck/synccheckはすべて検出0。G-SLIDEのみfull leak checkも必須。Scatterbrain診断はsource/fixture/binary hashを照合し、plain実行と同じbinaryを使用。
- 原典Scatterbrain CUDAのpadding勾配とkey-gradient indexingに数値不一致を発見。負例を保持し、MIT/provenance付き `ports/scatterbrain/cuda/` で修復。非default streamとstrided autograd cotangentも確認。原典とmaintained成功を混ぜない。
- MagicPIGのnative BF16 server/model、DDP/NCCL、Scatterbrain causal hybrid、G-SLIDEの残るstress/serialization等、同条件CUDA/SYCL比較・性能受入は未完。aggregate `PARTIAL`は全研究受入の成功ではない。
- branchは `feat/nvidia-gpu-validation`、PR #127/base `research/lsh-lineage-vendor`を継続。公開headと更新後CIはPRのchecks/本文を参照。独立レビュー未了のためDraftを維持し、merge/Issue closeは行わない。
- 不要になった今回のhost venv/cache/source/sanitizer downloadsは削除済み。Docker imageは再利用するテスト環境として保持。生証拠は `/tmp/hashing-docker-final-20261025`、消えない証拠は上記JSON。

次は独立レビューとPR CI、MagicPIGに適したhardware/API profile、CUDA/SYCL同一入力の比較設計を優先する。性能低下率の閾値は測定前に決める。

## 依頼と最初に読むもの

ユーザーの次セッションの目的は、今回G-SLIDE優先のため実行しなかった**全系統のGPUを含むテスト**を行うこと。本ファイル作成時点では次のテストを開始していない。

- 今回の詳細・再現手順: [NVIDIA_GPU_CHECKPOINT.md](docs/NVIDIA_GPU_CHECKPOINT.md)
- 消えない実行証拠: [nvidia-2026-10-22.json](docs/evidence/nvidia-2026-10-22.json)
- 共通CPU/SYCLの既存契約: [SYCL_CPU_BACKEND.md](docs/SYCL_CPU_BACKEND.md)
- 既存の[2026-10-06引き継ぎ](SESSION_HANDOFF_2026-10-06.md)は履歴。現在のGPU進捗は本ファイルと上記checkpointを優先する。

## コミット・PR方針の追記（2026-10-23）

ユーザーは、当該repoでコミット・push・PR作成までをエージェント判断で進める方針を明示した。[AGENTS.md](AGENTS.md)のID付き許可一覧を参照する。以下の未コミット/未PRという記述は、この引き継ぎを最初に記録した時点のsnapshot。再開時はGitとGitHubの実際のbranch/PR状態を確認する。

## 公開checkpoint（2026-10-23追記）

- Draft PR: [#127 — Initialize repository permissions and validate native G-SLIDE CUDA](https://github.com/fregata-ariel/HashingDeepLearning/pull/127)
- head branch: `feat/nvidia-gpu-validation`、base: `research/lsh-lineage-vendor`。
- 方針のcommit: `13fdae0`、GPU実装・検証・CIのcommit: `a7d950e8295cbe51f5e5f798be8bb8331cb200c5`。この追記は後続の文書commitへ収録する。
- 最初のPR CI: [Paper-code traceability run 39813677470](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/39813677470)。確認時は実行中で、対象headは`a7d950e`。新しいhead/CI状態は再開時に確認する。
- コミット・push・PR作成まで実施済み。独立レビュー、merge、Issue更新・close、GitHub上の手動GPU job実行は未実施。

## 作業ツリーと証拠を保全する

作業場所: `/home/fischeri/Projects/DNN/HashingDeepLearning`

- ブランチ: `feat/nvidia-gpu-validation`
- HEAD: `c64e5526eca284640a8f04c2e8478677a0314f90`
- HEADのtree: `0414fd032da3469be40f51bb7cc37ee55f46ea29`
- **今回の実装・CI・文書は未コミット。新規ファイルも多数untracked。** `git status --short --branch`を読み、既存差分を引き継ぐこと。HEADの内容だけでは今回の修復を再現できない。
- push、PR作成、独立レビュー、統合、Issueへの書き込み・closeは行っていない。GitHub側にはまだローカル成功の追記がない。
- `third_party/`は原典のimmutable archive。修復は`ports/`、fixtureは`tests/`へ置く。
- 生ログ・ビルドは現ホストの`/tmp/hashing-nvidia-final/`にあるが、一時領域なので存続を前提にしない。リポジトリ内JSONに最終数値・診断ログとsource/binary hashesを保存済み。
- JSON内の`report`は実測時の記録。実測後にGPU workflowのrunner環境変数の初期化位置を直したため、そのファイルだけ`post_run_metadata_changes`に最終hashと理由がある。記録を都合よく書き換えず、新しい実行は別artifactにする。

今回追加した入口は`tools/check_nvidia_gpu.py`、`tools/bootstrap_nvidia_sanitizer.sh`、`tools/generate_gslide_cuda_original.py`、`tests/gslide_cuda/`、`.github/workflows/nvidia-gpu.yml`。対応する台帳は`papers/traceability/gslide-cuda.json`、失敗報告の回帰テストは`tests/traceability/test_nvidia_gpu_reporting.py`。

## 実機・環境上の注意

前セッションの実測値。再開時に再確認する。

| 項目 | 確認済み |
| --- | --- |
| OS | Ubuntu 24.04、x86-64 |
| GPU | RTX 2080 Ti、SM75、VRAM 11264 MiB、1台 |
| NVIDIA driver | 595.99.02 |
| CUDA toolkit / runtime | nvcc 12.0.140 / runtime 12.0。nvidia-smiの13.2表示はdriverの対応版 |
| CUDA host compiler | `/usr/bin/g++-12`、12.4.0。既定g++は13.3でnvcc 12.0のhost上限を超えるため、明示指定した |
| CMake / Python | 3.28.3 / system Python 3.12 |
| CPU | Ryzen 7 5700X、8 core / 16 thread、1 NUMA node。記録されたflagsにAVX-512なし |
| Python GPU依存 | 今回はTorch/CuPy/FlashInferを導入・実行していない。開始時のsystem PythonにはTorch/CuPy/NumPyがなかった。次回は既存venvの有無から再確認する |

サンドボックス内のnvidia-smiはdriver接続に失敗し、ホスト側の承認付き実行では成功した。GPU不在と即断せず、同じ制限なら適切な権限で再実行する。CPUのLeakSanitizerもサンドボックス内ではプロセス検査制限で失敗し、ホスト側では通過した。

SM75であることを依存選定の条件にする。CUDA Torch、CuPy/NVRTC、FlashInfer、BF16各経路について実際のwheel・compiler・device対応を確認する。GPUが見えることと各演算の実対応を混同しない。対応外の場合は理由付きの未実行として残す。

## 今回完了したG-SLIDEの範囲

- #124: dense IDsとbatch+1 offsetsを独立extentに修復。空/1ノード/端数、capacity、guard、両production ReLU callerをCUDAで確認。
- #125: alternate hashのnode/lane/shared-rowとlaunch coverageを修復。nodes 1/31/33/64/65、lanes 1/2/4、threads 32/64/128、正値/負値/同値、partial table tile、2回rebuild、実`LSH::build`とqueryを確認。
- #43: 既にCPU作業で接続済みだったfull-capacity resetを実`Network::train`の2更新で確認。FP64の独立期待値、ReLU gate変化、予備capacityのpoison/reset、別fixtureでcompressed IDs変更とstale deltaを確認。
- `LSH::~LSH`でrandom-node所有buffer 2個をfreeする修復を追加。
- 評価の旧Thrust `max_element`経路でinitcheckが2件報告したため、paddingのないvalue/index reductionに変更。cuBLAS logits、独立argmax、同値時の先頭選択を確認。元のdense evalがoutput biasを加算しない挙動は維持・明記している。
- 原典のdense/hashの失敗は、archive-verified CUDA本文をsymbolだけ改名してGPU負例に保持。CPU側の原典回帰も`third_party/g-slide`から抽出するよう更新済み。原典の失敗値を正しいGoldenにしていない。

最終結果: **9226 checks、max absolute error 3.8112e-8、tolerance 8e-6**。memcheck（leak含む）、racecheck、initcheck、synccheckすべて検出0。CPUの既存5 suiteもplainとASan/UBSan/LSanで成功。台帳85 entry、native文書監査61 entryを確認。これはsuite数やGPU test数と同じ意味ではない。

未受入: broad lock/pool/overflow stress、全adaptive/fallback branch、nondefault stream、一般のK/bin packing、model checkpoint保存復元、共有SYCL NVIDIA比較、論文規模の品質・性能。接続学習fixtureは全出力候補を使い、compressed IDs変更は別backward fixture。#34全体が完了した意味ではない。

## 全系統の再開順序と受入の分割

2026-10-23のIssue再取得では以下9件がopen。Issue題名の「延期」は過去の状態を含むため、ユーザーの今回・次回の再開指示と、実行済み証拠を照合する。タイトルだけでテストを停止しない。

| 系統 / Issue | 次回の最小受入と注意点 |
| --- | --- |
| 統括 [#113](https://github.com/fregata-ariel/HashingDeepLearning/issues/113) | CPU成功、CUDA原典、maintained修復、共有SYCL GPUを別profileとして記録する |
| G-SLIDE [#34](https://github.com/fregata-ariel/HashingDeepLearning/issues/34) / [#43](https://github.com/fregata-ariel/HashingDeepLearning/issues/43) / [#124](https://github.com/fregata-ariel/HashingDeepLearning/issues/124) / [#125](https://github.com/fregata-ariel/HashingDeepLearning/issues/125) | 今回の成功を保全。未検証branchや疎候補の接続、stream/ownershipを追加。既存fixtureを全系統完了として扱わない |
| MONGOOSE [#121](https://github.com/fregata-ariel/HashingDeepLearning/issues/121) | original SimHashのCUDA compile/launchとCuPy/PyNVRTC wrapperを別々に確認。小さいpacking→CPU oracle照合→CUDA RNG replay。DDP/NCCLは独立profile |
| Scatterbrain [#101](https://github.com/fregata-ariel/HashingDeepLearning/issues/101) | native local product 4操作のforward/backward→非causal hybrid→短いconnected training/save/reload。CUDA/causal sourceのpin/build/noticeから必要 |
| MagicPIG [#62](https://github.com/fregata-ariel/HashingDeepLearning/issues/62) | tiny projection/BF16、FlashInfer plan/paged append/decode/LSE merge、D2H/H2Dとstream/event/所有権。モデル取得や生成はcomponent成功の次の段階 |
| 校正 [#111](https://github.com/fregata-ariel/HashingDeepLearning/issues/111) | 小fixtureのcold/warm時間とmemoryを実測。今回の約20秒buildと短いGPU診断を、他系統やfull-modelの費用へ流用しない |

先に全系統の環境・最小component・未対応条件を棚卸しし、その後接続fixtureを広げると未着手系統が残りにくい。大規模学習・有料GPU・自動切替までが今回承認されたという意味ではない。

### MONGOOSEの具体的な入口

- `ports/mongoose/mongoose_slide/slide_lib/simHash.py`のCUDA文字列と`fingerprint`、同ディレクトリの`cupy_kernel.py`が対象。wrapperはlazy importで、CuPyのdevice/module APIと`pynvrtc.compiler.Program`を使用する。
- CUDA sourceの`long`値shuffle、32-lane mask/width、`1 <= k <= 32`、bit31のint32解釈、strictly-positiveと両符号zeroを独立期待値へ照合する。元CUDAのcompile成功をまだ確認していない。必要な移植修正は原典と区別する。
- `self.fp`が存在することだけではnative成功でない。compile/load、named function、raw pointer/scalar ABI、Torch current stream、module/buffer寿命を実際に確認する。Torch fallbackをGPU上で実行してもoriginal CUDAの受入にはならない。
- `ports/mongoose/sycl/tests/packing.cpp`と`tests/fallback.py`、[SYCL_CONTRACT.md](ports/mongoose/sycl/SYCL_CONTRACT.md)にliteral code/shape oracleがある。`tests/validate.py`の原典probeはCPU互換compile-onlyで、CUDA native実行証拠ではない。
- `ports/mongoose/lsh_lib/matrix_simhash.py`はlegacy int64 API。maintained int32契約と混ぜず、現production importとの関係を確認して採用範囲を記録する。
- reversible/dropoutのCPU RNG replayとCUDA device RNG replayを分ける。GPUは1台なので複数device DDPを成功扱いにしない。

### Scatterbrainの具体的な入口

- `ports/scatterbrain/src/models/modules/attention/sblocal_attention.py`と、後発の`ports/scatterbrain-later/scatterbrain_later/attention.py`を別lineageとして扱う。後者のCPU限定契約を単に`.cuda()`へ変えて対応済みとしない。
- 共有実装とoracleは`ports/scatterbrain/sycl/`。`include/scatterbrain/local_product.hpp`、`tests/validate.py`、`sblocal_sycl.py`、[SYCL_CONTRACT.md](ports/scatterbrain/sycl/SYCL_CONTRACT.md)を読む。
- 収録済み`third_party/scatterbrain-fast-transformers`はlocal CPU source。GPU local/causal sourceは未取込・未build。#101に記録されたIdiap commit `f22c13716fc748bb21a7b226ada7f7b5f87f867f`、local CUDA blob `4d10eee240a53ae0bda4499e92b5d64115578310`、causal blob `af60163dc631be5e008aa7d6f0af935bdb5f719b`を再照合してから取込判断する。
- tiny oracleと長系列性能測定を分ける。default maskやreturned dense weightsがN² memoryを使うため、最初から大きな系列へ進めない。
- 後発のSMYRF由来コンポーネントのGPLと、Apache/MIT noticeの区別を維持する。非causal成功をcausal成功にしない。

### MagicPIGの具体的な入口

- 原典server/modelは`third_party/magicpig/models/attnserver.py`と`models/llama.py`。maintained CPU制御は`ports/magicpig/magicpig_cpu/`、native CPU拡張は`ports/magicpig/library/`。今回maintained GPU serverを作ったわけではない。
- source pinは`papers/magicpig-source-pin.json`。CUDA/FlashInferの版・API・SM75対応を先に固定し、重みなしtiny componentで検証する。11 GiB GPUで大きなモデルや長contextが可能と仮定しない。
- host CPUにはAVX-512がない。元serverのCPU extension importが成立するか、部品切出しまたは既存SIMDe経路との接続が必要かを確認する。SIMDe実行をnative AVX-512成功として記録しない。
- `tools/check_magicpig_extension_cpu.py`はCPU wheelを要求する既存CPU gate。GPU Torch環境で通すためにCPU guardを外さず、venv/profileを分ける。
- [EXTENSION_CPU_CONTRACT.md](ports/magicpig/EXTENSION_CPU_CONTRACT.md)、[SERVER_CPU_CONTRACT.md](ports/magicpig/SERVER_CPU_CONTRACT.md)と`tests/traceability/test_magicpig_*.py`をoracle資料にする。AST/mock/CPU composition成功からFlashInfer、転送、実生成を推論しない。

## 共有SYCL NVIDIA backendは別の実装作業

- `tools/bootstrap_sycl_cpu.sh`はDebian 13専用で、CUDA/ROCm/OpenCL/Level Zero backendをOFFにしている。今回のUbuntuホスト用GPU installerではない。既存CPU toolchainを上書きしない。
- AdaptiveCpp pin `31db2e83903c1db9f50b03b620eb0a09224cd0eb`のNVIDIA backendを別prefixでbuild/校正する必要がある。compiler/runtime/driver/CUDAの組合せとSM75を記録する。
- `ports/sycl-common/include/hashing/sycl_backend.hpp`はGPU queueを作成できるが、現fixtureやbridgeはCPU指定。`tests/gslide_sycl/test_kernels.cpp`、`test_network.cpp`、`ports/mongoose/sycl/tests/packing.cpp`、`ports/scatterbrain/sycl/adapter.cpp`に明示CPU選択がある。
- 既存の必須CPU gateを維持してGPU profileを追加する。CPU選択を削除しただけ、default selector、環境変数変更だけ、CPU fallbackでGPU成功を主張しない。実device/vendor、queue、USM、subgroup、local memory、async error完了を記録する。
- 同一GPU・入力・精度・候補密度でCUDA対SYCLを比較する。許容性能低下率は計測前に決める。初期化、JIT/cold、warm kernel、転送、同期、接続全体を分ける。今回まだ閾値合意もGPU性能比較もない。

## CIと再実行の入口

既存の[手動GPU workflow](.github/workflows/nvidia-gpu.yml)は**G-SLIDE専用**。全系統を自動発見するrunnerではない。runner labelsは`self-hosted, linux, x64, nvidia-gpu`。登録・GitHub上での実行は未実施。次回は各profileを明示登録し、未対応を成功へ丸めないよう拡張する。CPU側は`.github/workflows/traceability.yml`の既存必須検証を維持する。

G-SLIDEの再実行コマンドは[checkpointのReproduce](docs/NVIDIA_GPU_CHECKPOINT.md#reproduce-locally-or-on-an-explicitly-provisioned-runner)を使う。`--build-dir`は毎回新しい場所を渡す。driverは既存dirの再利用を拒否する。

軽い既存検証の例（GPU実行証拠とは別）:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/traceability/test_nvidia_gpu_reporting.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/validate_traceability.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_traceability_docs.py
```

既存CUDAファイルはCRLFのため、追加行のCRを実質的な末尾空白と誤認しないdiff検査は次で行った。無関係な全ファイルの改行変換をしない。

```bash
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

## ダウンロードの後片付けと次回の完了記録

ユーザーの明示的な希望: **今回・次回にダウンロードしたものは、不要になったら削除する。** 再取得URL・版・SHA-256と必要なlicense notice、source provenance、実測ログは残す。

今回のNVIDIA manifest、tar.xz、展開したCompute Sanitizerは全て削除済み。`/tmp/hashing-cuda-redistrib.json`、`/tmp/hashing-cuda-sanitizer-12.9.79.tar.xz`、`/tmp/hashing-nvidia-tools`は存在しない。再実行時はbootstrapから再取得する。systemの古いCompute Sanitizer 2022.4.1は起動/注入に失敗したため、成功した公式component 12.9.79 / tool 2025.2.1.0のpinを利用する。

次回導入するTorch/CuPy等もisolatedなvenv/cache/build先と削除責任を記録する。共用cacheや他セッションの環境を不用意に消さない。CIにはdownloaded sanitizerを`always()`で削除するstepがある。

次の引き継ぎには、系統ごとにsource pin、fixture、実device、dependency versions、PASS/FAIL/未対応/未実行、残件Issue、実行artifact、cleanup状態を残す。共有SYCLとoriginal CUDA、native CPU/SIMDe、単一GPUとDDP、componentとmodel生成、正しさと性能を区別する。Issue更新・closeはレビューと該当CI・統合の状態を確認してから行う。
