# Specification

本仕様は、BioReason-Pro を用いた疾患関連ヒトタンパク質の GO 機能予測について、**最終的に採用する運用ルールだけ**を定める。  
検討過程、比較版、候補フィルタ、件数判断の学びは [learning-log](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/domain/learning-log) に分離し、本書には残さない。

前提:

- 現在採用する benchmark は **small-data 版**
- 現行版は `213 -> 221 -> 225 -> 228`
- train / validation / test は **protein-disjoint** かつ **temporal order** を守る
- local filesystem 上の生成物は scratch とみなし、**正本は W&B Artifact ref**
- 文中の `dev` は、コード上の split 名 `validation` と同義

## 1. 問題設定

本仕様で扱う問題は、**疾患関連ヒトタンパク質に対する protein-level GO term prediction と disease-aware reasoning** である。

固定する問い:

- 疾患関連ヒトタンパク質に対して、BioReason-Pro の公開モデルに追加学習を行う価値があるか
- 時系列的に独立した benchmark 上で、独自 tuning 前の比較モデルより良い予測と reasoning が出るか

本仕様で扱わないもの:

- variant pathogenicity classification
- ClinVar 単独 supervision
- cross-species disease transfer

## 2. 使用モデル・比較対象

### 2.1 使用モデル

**BioReason-Pro**（bowang-lab, bioRxiv 2026-03）

| 要素 | 内容 |
|---|---|
| アーキテクチャ | ESM3 + GO Graph Encoder + Qwen3-4B |
| タスク | タンパク質配列 -> GO term 予測 + 推論トレース生成 |
| 学習方式 | SFT -> RL |
| ベース学習データのカットオフ | UniProt GOA 2022-11 リリース |

### 2.2 比較対象

比較対象は次の 3 系統に固定する。

| 手法 | 位置づけ |
|---|---|
| `bioreason-pro-rl-paper` | 独自 tuning 前の比較モデル |
| `train-sft-output` | custom `train_sft` run が生成する output artifact |
| `train-rl-output` | custom `train_rl` run が生成する output artifact |

初期状態で先に必要な model artifact ref は `bioreason-pro-rl-paper` だけでよい。  
`train-sft-output` と `train-rl-output` は、それぞれの学習 run 実行後に成果物として現れる。

## 3. ベンチマーク

### 3.1 最終採用 benchmark

採用する shortlist query は次に固定する。

```text
reviewed:true AND organism_id:9606 AND cc_disease:* AND
(xref:mim-* OR xref:orphanet-*) AND
(go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR
 go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)
```

採用する benchmark version は次に固定する。

| Version | Train proteins | Train unique labels | Validation proteins | Test proteins |
|---|---:|---:|---:|---:|
| `213 -> 221 -> 225 -> 228` | 1,245 | 2,773 | 590 | 875 |

### 3.2 split の厳格ルール

固定ルール:

1. `train -> validation -> test` は必ず時系列順に並ぶ
2. 同一 protein を複数 split に入れてはならない
3. 各 protein は、**最初に新規 label が現れた split に 1 回だけ割り当てる**
4. 新規 label は `(DB_ID, GO_ID, Aspect)` 単位で定義する
5. evidence code の差だけでは別 label と数えない

split 定義:

- train: `213 -> 221`
- validation: `221 -> 225`
- test: `225 -> 228`

split validation の必須条件:

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`

protein overlap が 1 件でも見つかった run は不正とみなし、採用しない。

### 3.3 独立性判定

独立性判定は **GOA archive の release 差分** で行う。  
次は採用しない。

- `annotation_date` による独立性判定
- UniProt REST の `date_modified` による独立性判定
- `ClinVar` cross-reference を main filter の必須条件にする設計

### 3.4 正本 artifact

現行版の正本 artifact は次に固定する。

- temporal split artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production`
- reasoning dataset artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production`
- comparison model artifact: `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production`

local の build / download 結果は scratch であり、source-of-truth ではない。

## 4. データの仕様

### 4.1 使用する証拠コード

使用する GO evidence code は次に固定する。

- `EXP`
- `IDA`
- `IPI`
- `IGI`
- `IMP`
- `IEP`
- `IC`
- `TAS`

GAF では `DB == UniProtKB` かつ `DB_Type == protein` の行だけを対象にする。

### 4.2 reasoning dataset

primary dataset は reasoning dataset のみとする。  
config 名は `disease_temporal_hc_reasoning_v1` に固定する。

必要列:

| 列 | 必須 | 用途 |
|---|---|---|
| `protein_id` | Yes | 主キー |
| `sequence` | Yes | モデル入力 |
| `organism` | Yes | prompt 生成 |
| `go_bp` | Yes | BP 正解ラベル |
| `go_mf` | Yes | MF 正解ラベル |
| `go_cc` | Yes | CC 正解ラベル |
| `reasoning` | Yes | SFT 用 reasoning |
| `final_answer` | Yes | SFT 用 answer |
| `protein_function` | Yes | UniProt summary 文脈 |
| `go_pred` | Yes | GO-GPT 予測の事前計算列 |
| `interpro_formatted` | Optional | InterPro 文脈 |
| `ppi_formatted` | Optional | PPI 文脈 |

split 名は `train` / `validation` / `test` に固定する。  
任意列が存在しない場合も列自体は保持し、空文字列で埋める。

### 4.3 SFT / RL の split 利用ルール

固定ルール:

- SFT / RL / eval は **同じ benchmark split** を使う
- SFT は reasoning dataset の `train` を学習に使う
- SFT の checkpoint selection には `validation` を使う
- RL の rollout / reward 最適化には `train` を使う
- RL の checkpoint selection と offline sanity-check には `validation` を使う
- `test` は最終評価専用であり、学習信号には使わない
- RL 用の派生 dataset を作る場合も、元データは `train` split のみから作る

## 5. データの準備

### 5.1 orchestration

データ準備の高位 entry point は `scripts/run_temporal_split_artifact_pipeline.py` に固定する。  
この orchestration は少なくとも次を行う。

- temporal split artifact build
- sanity check
- reasoning dataset build
- W&B Artifact upload
- pipeline status 記録

低位 entry point:

- temporal split artifact build: `scripts/build_disease_temporal_split_artifact.py`
- reasoning dataset build: `scripts/build_disease_benchmark_datasets.py`

### 5.2 必須成果物

temporal split artifact に必須な成果物は次とする。

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `pipeline_status.json`

補助解析用の追加ファイルを生成してもよいが、上記以外を必須成果物にはしない。

### 5.3 保存方針

固定ルール:

- local 出力は scratch とみなし、恒久保存を前提にしない
- W&B Artifact への upload 成功後は、local scratch を cleanup してよい
- downstream の eval / SFT / RL は local directory を正本にせず、W&B Artifact ref から解決する
- 必要のない dataset や補助ファイルは生成しない

## 6. データ・モデルの upload for W&B

### 6.1 共通原則

W&B を experiment tracking と artifact lineage の標準にする。  
各 phase で最低限次を run config に残す。

- `job_type`
- `benchmark_version`
- `temporal_split_artifact`
- `dataset_config`
- `reasoning_dataset_config`
- `dataset_artifact`
- `shortlist_query`
- `shortlist_mode`
- `train_start_release`
- `train_end_release`
- `dev_end_release`
- `test_end_release`
- `base_checkpoint`
- `model_artifact`
- `seed`
- `learning_rate`
- `batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs`
- `job_time_limit`

### 6.2 dataset upload

固定ルール:

- dataset は W&B Artifact として登録する
- dataset artifact の version / alias を run config に残す
- downstream の eval / SFT / RL は W&B Artifact ref から dataset を解決する

### 6.3 model upload

固定ルール:

- model checkpoint は W&B Artifact として登録する
- model artifact の version / alias を run config に残す
- 比較モデルは `bioreason-pro-rl-paper` に固定する
- `bioreason-pro-rl-paper` は public Hugging Face source `wanglab/bioreason-pro-rl` から一度 materialize し、W&B model artifact に固定して使う
- `train_sft` と `train_rl` の出力は、それぞれの run の output artifact として登録する

### 6.4 artifact ref manifest

CoreWeave 上の実行では、任意の local directory を毎回手で渡すのではなく、repo 内の artifact ref manifest を entry point にする。

固定ファイル:

- data-bundle manifest: `configs/disease_benchmark/data_registry.json`
- evaluation-target manifest: `configs/disease_benchmark/eval_target_registry.json`
- asset publish manifest: `configs/disease_benchmark/artifact_publish_registry.json`
- artifact ref env template: `configs/disease_benchmark/wandb_registry_paths.env.example`
- source env template: `configs/disease_benchmark/wandb_asset_sources.env.example`

固定ルール:

- manifest に渡す ref は `entity/project/artifact_name:alias` 形式を使う
- 初期状態で人が明示的に用意する model ref は `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH` だけでよい
- `train-sft-output` と `train-rl-output` の ref は、各 training run 完了後に成果物として決まる

## 7. 評価

### 7.1 eval phase

評価の論理フェーズ名は `eval` に固定する。  
高位 entry point は `scripts/run_registered_eval.py`、低位 entry point は `eval.py` と `scripts/sh_eval.sh` を使う。  
W&B run は `wandb.init(..., job_type="eval")` で開始する。

split の使い分け:

- 開発中の比較、ablation、checkpoint 比較は `validation`
- 最終報告値は `test`

### 7.2 定量評価

主指標は **F_max** とする。必須 namespace は次。

- MF
- BP
- CC

評価対象:

- `bioreason-pro-rl-paper`
- `train-sft-output`
- `train-rl-output` がある場合はそれも含める

target family:

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-sft-output`, `train-rl-output`
- `spec-comparison`: 上記すべて

各 metric は `wandb.log()` で保存し、少なくとも `fmax_mf`, `fmax_bp`, `fmax_cc` を残す。

### 7.3 保存物と追跡

固定ルール:

- 評価 summary を **1 evaluated target = 1 row** の W&B Table として保存する
- sample-level 結果を **1 sample = 1 row** の W&B Table として保存する
- reasoning task の場合は `reasoning_full`, `final_answer`, `intermediate_trace` を保持する
- JSON 結果、summary export、sample export は W&B Artifact として version 管理する
- Weave Evaluation logger でも同一 eval run を追跡する
- local eval 出力は scratch とみなし、W&B 保存成功後は既定で cleanup してよい

## 8. 学習 for SFT

### 8.1 train_sft phase

SFT の論理フェーズ名は `train_sft` に固定する。  
entry point は `train_protein_llm.py` と `scripts/sh_train_protein_qwen_staged.sh` を使う。  
W&B run は `wandb.init(..., job_type="train_sft")` で開始する。

### 8.2 入力と厳格ルール

入力:

- `disease_temporal_hc_reasoning_v1`
- `bioreason-pro-rl-paper` checkpoint artifact

固定ルール:

- 学習には reasoning dataset の `train` split を使う
- checkpoint selection には `validation` split を使う
- `test` split は SFT 学習に使わない
- tuning 前の比較モデルは W&B Artifact ref から materialize して使う
- 前半フェーズと後半フェーズの両方で、比較モデルに含まれる projector / GO module 重みを warm-start に使ってよい
- train / validation metric を `wandb.log()` で保存する
- sample table を W&B Table として保存する
- output checkpoint を W&B Artifact として登録する

### 8.3 実行条件

固定ルール:

- 学習は GPU 前提
- `dataset_type=cafa5` 前提で dataset config を供給する
- 学習ジョブの wall time は最大 12 時間
- submission 時の time limit は `12:00:00`
- checkpoint / resume 前提で運用する

## 9. 学習 for RL

### 9.1 train_rl phase

RL の論理フェーズ名は `train_rl` に固定する。  
W&B run は `wandb.init(..., job_type="train_rl")` で開始する。

### 9.2 入力と厳格ルール

入力:

- canonical には `train-sft-output` checkpoint
- reward 設定
- 同じ benchmark 定義

固定ルール:

- rollout / reward 最適化には benchmark の `train` split を使う
- checkpoint selection と offline sanity-check には `validation` split を使う
- `test` split は RL 学習に使わない
- RL 用 dataset を派生生成する場合も、元データは `train` split のみから作る
- `bioreason-pro-rl-paper` から直接 RL を始める経路は、必要な場合に限って ablation として扱う
- reward 系 metric、KL 系 metric、学習安定性指標を `wandb.log()` で保存する
- rollout trace は Weave で保存する
- RL 出力 checkpoint を W&B Artifact として登録する

### 9.3 実行条件

固定ルール:

- RL でも同じ benchmark version を使う
- 学習ジョブの wall time は最大 12 時間
- submission 時の time limit は `12:00:00`
- checkpoint / resume 前提で運用する
