# Specification

本仕様は、BioReason-Pro を用いた疾患関連ヒトタンパク質の GO 機能予測について、2026-04-06 時点で確認できている実測結果と運用ルールを整理したものである。

最初に明記しておくと、現在採用する benchmark は **small-data 版** である。現行版は `213 -> 221 -> 225 -> 228` を用い、train は **1,245 proteins / 2,773 unique labels**、dev は **590 proteins**、test は **875 proteins** である。この点を前提に、データ設計・学習・評価・主張の仕方を定める。

Status:
- temporal split artifact の実測結果は `213` 版と `214` 版がある
- 現行版 benchmark は `213 -> 221 -> 225 -> 228`
- train / dev / test は protein-disjoint かつ temporal order を守って分割する
- 文中の `dev` は dataset split 名 `validation` と同義である
- local filesystem 上の生成物は作業用 scratch であり、正本は W&B Artifact ref とする

## 1. 問題設定

上位目的は `domain/foundation/philosophy.md` に従う。  
本仕様で扱う問題は、**疾患関連ヒトタンパク質に対する protein-level GO term prediction と disease-aware reasoning** である。

本仕様で固定する問いは次の 2 点である。

- 疾患関連ヒトタンパク質に対して、BioReason-Pro の既存 checkpoint へ追加学習を行う価値があるか
- 時系列的に独立した benchmark 上で、base より良い予測と reasoning が出るか

本仕様で扱わないものは次である。

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
| 学習方式 | SFT（合成推論トレース） -> RL |
| 公開物 | モデル weight、学習データ、推論コード |
| ベース学習データのカットオフ | UniProt GOA 2022-11 リリース |

### 2.2 比較対象

比較対象は次の 3 系統に固定する。

| 手法 | 位置づけ |
|---|---|
| bioreason-pro-rl-paper | 独自 tuning 前の比較モデル |
| train_sft output | reasoning dataset で追加学習したモデル |
| train_rl output | SFT 後に RL で最適化したモデル |

## 3. ベンチマーク

### 3.1 現在採用する benchmark 版

現在採用する benchmark は、次の high-confidence disease shortlist を使う。

```text
reviewed:true AND organism_id:9606 AND cc_disease:* AND
(xref:mim-* OR xref:orphanet-*) AND
(go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR
 go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)
```

この shortlist の live count は **5,088 proteins** である。

この値は **現在の UniProt shortlist size** を示す参考値であり、そのまま train / dev / test の件数になるわけではない。  
実際の benchmark 規模は、release 差分、evidence filter、protein-disjoint assignment を通した temporal split artifact の出力で決まる。

現在採用する split は次である。

| Version | Train proteins | Train unique labels | Dev proteins | Test proteins |
|---|---:|---:|---:|---:|
| `213 -> 221 -> 225 -> 228` | 1,245 | 2,773 | 590 | 875 |

以後、この版を `current implementation version` と呼ぶ。

### 3.2 comparison benchmark 版

comparison variant として、次も保持する。

| Version | Train proteins | Train unique labels | Dev proteins | Test proteins |
|---|---:|---:|---:|---:|
| `214 -> 221 -> 225 -> 228` | 932 | 1,898 | 662 | 969 |

`214` 版は、件数感度と archive choice の影響を確認するための comparison variant として使う。

### 3.3 broader alternative filter

broader alternative として、`cc_disease:*` を軸にした shortlist も存在する。

```text
reviewed:true AND organism_id:9606 AND cc_disease:* AND
(go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR
 go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)
```

この broader query の live count は **5,093 proteins** である。  
現時点の benchmark は `MIM/Orphanet` を含む high-confidence 版で進めるが、broader filter は追加 EDA や将来の拡張候補として保持する。

### 3.4 split の基本ルール

この benchmark で守る条件は 2 つである。

1. train / dev / test は **時系列順** に並ぶこと
2. train / dev / test は **同一 protein を共有しない**こと

current implementation version の split は次で固定する。

- train: `213 -> 221`
- dev: `221 -> 225`
- test: `225 -> 228`

時間の流れは必ず `train -> dev -> test` とする。  
train より後の時点で初めて現れた label を dev に入れ、dev より後の時点で初めて現れた label を test に入れる。逆流は許さない。

なお、benchmark の説明では中間 split を `dev` と呼ぶが、dataset config とコード上の split 名は `validation` で統一する。

### 3.5 新規 label の定義

新規 label は `(DB_ID, GO_ID, Aspect)` 単位で定義する。  
evidence code の違いだけでは別 label と数えない。

つまり、

- old release に無い
- new release にはある
- `(protein, GO, aspect)` が新しい

ときだけ、その label を temporal delta に含める。

### 3.6 protein-disjoint ルール

同じ protein を train / dev / test の複数 split に入れてはならない。  
各 protein は、**最初に新規 label が現れた split に 1 回だけ割り当てる**。

例:

- ある protein が `213 -> 221` で初めて新規 GO を持ったら train 専属
- その protein が `221 -> 225` で追加 GO を得ても dev には入れない
- その protein が `225 -> 228` でさらに追加 GO を得ても test には入れない

要するに、train / dev / test は **時系列でも分かれ、protein でも分かれる**。

### 3.7 split validation

temporal split artifact は、`summary.json` に次が出力されていることを必須条件とする。

- `split_validation.time_order_valid == true`
- `split_validation.protein_disjoint_valid == true`

protein overlap が 1 件でも見つかった run は不正とみなし、採用しない。

### 3.8 現在使う artifact

current implementation version の正本 artifact は次に固定する。

- temporal split artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production`
- reasoning dataset artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production`

comparison variant の artifact は次である。

- temporal split artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:214.221.225.228`
- reasoning dataset artifact: `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:214.221.225.228`

local の `data/artifacts/...` は build / download の作業領域として使ってよいが、  
spec 上の source-of-truth は W&B Artifact ref とする。

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

### 4.2 データ規模の解釈

現在の benchmark は small-data 版である。  
ただし、この規模で実装する正当性はある。

理由は次のとおり。

- 追加学習の対象は zero-from-scratch ではなく、既存の BioReason-Pro checkpoint である
- 時系列的に独立した benchmark がある
- `bioreason-pro-rl-paper` との直接比較ができる
- 小規模でも、疾患文脈の改善が出るかどうかは検証できる

`3,000 unique labels` は **full-scale study を安心して主張しやすい目安** であり、current implementation version を否定する hard gate ではない。

### 4.3 main label と propagated label

学習用の正解ラベルは `*_assigned_labels.tsv` を基準に作る。  
`*_assigned_propagated.tsv` は次に使う。

- coverage 確認
- benchmark 記録
- F_max 評価時の補助集計

### 4.4 除外する判定軸

以下は main benchmark の定義や独立性判定には使わない。

- `ClinVar` cross-reference を main filter の必須条件にする設計
- `annotation_date` を使って初回付与日を引く設計
- UniProt REST の `date_modified` だけで独立性を判定する設計

### 4.5 dataset config

現行の `train_protein_llm.py` と `eval.py` は `load_cafa5_dataset()` を通す実装になっているため、current implementation split は **CAFA5 互換の Hugging Face dataset config** として提供する。

config 名:

- primary reasoning dataset: `disease_temporal_hc_reasoning_v1`

役割:

- reasoning dataset は SFT の教師データ、および current evaluation flow の入力に使う

必要列:

| 列 | 必須 | 用途 |
|---|---|---|
| `protein_id` | Yes | 主キー |
| `sequence` | Yes | モデル入力 |
| `organism` | Yes | prompt 生成 |
| `go_bp` | Yes | BP 正解ラベル |
| `go_mf` | Yes | MF 正解ラベル |
| `go_cc` | Yes | CC 正解ラベル |
| `reasoning` | SFT 用 | assistant reasoning |
| `final_answer` | SFT 用 | assistant answer |
| `protein_function` | Yes | UniProt summary 文脈 |
| `go_pred` | Yes | GO-GPT 予測の事前計算列 |
| `interpro_formatted` | 任意 | InterPro 文脈 |
| `ppi_formatted` | 任意 | PPI 文脈 |

split 名は `train` / `validation` / `test` に固定する。  
任意列が存在しない場合は列自体を省略せず、空文字列で埋める。

RL 用に派生 dataset を作る場合は、少なくとも次の列を持つ形を推奨する。

| 列 | 必須 | 用途 |
|---|---|---|
| `protein_id` | Yes | 元データとの対応付け |
| `split` | Yes | `train` 固定の確認 |
| `prompt` | Yes | rollout 入力 |
| `response` | Yes | policy 出力または候補出力 |
| `reward` | Yes | scalar reward |
| `reward_components` | 任意 | reward 内訳 |
| `notes` | 任意 | 例外・監査メモ |

### 4.6 SFT と RL のデータの分け方

SFT 用と RL 用に benchmark split を別々に作るのではなく、**split 定義は 1 つに固定し、その上で各 phase が使う列と最適化目的を分ける**。

固定ルール:

- `train` / `validation` / `test` の protein membership は SFT / RL / eval で共通にする
- SFT は reasoning dataset の `train` split を使い、`reasoning` と `final_answer` を教師信号として使う
- RL は同じ benchmark version の `train` split を使い、reward に基づいて最適化する
- `validation` は SFT の early stopping と RL の offline sanity-check / checkpoint selection に使う
- `test` は最終評価専用であり、SFT / RL の学習信号には使わない
- RL 用の preference / reward dataset を派生生成する場合も、元データは `train` split のみから作る

要するに、SFT と RL は **別の benchmark を持つのではなく、同じ benchmark の train split を別目的で使う**。

### 4.7 NK / LK の仕様

NK / LK は benchmark の補助的な解析軸として保持する。  
current implementation version では、NK / LK を主要判定軸にはしないが、**どれぐらい含まれているかを EDA できる状態**にしておく。

固定ルール:

- NK / LK の bucket は split ごとに保存する
- EDA 用の集計表を保存する
- summary に split ごとの NK / LK 件数を残す
- CAFA5 access がある場合は実カウント、無い場合は status を明記する

保存物:

- `*_assigned_nk_lk.tsv`
- `*_assigned_nk_lk_propagated.tsv`
- `nk_lk_eda.tsv`
- `summary.json` の `nk_lk_status`, `nk_lk_error`, split 別 NK/LK 件数
- `report.md` の split summary table

## 5. データの準備

### 5.1 orchestration の原則

データ準備は、個別の生成スクリプトを手で順に叩く運用ではなく、  
高位 entry point である `scripts/run_temporal_split_artifact_pipeline.py` から実行する。

この orchestration では variant ごとに少なくとも次を行う。

- temporal split artifact build
- sanity check
- reasoning dataset build
- W&B Artifact upload
- pipeline status 記録

個別 build の低位 entry point としては次を使う。

- temporal split artifact build: `scripts/build_disease_temporal_split_artifact.py`
- reasoning dataset build: `scripts/build_disease_benchmark_datasets.py`

ただし、business rule として要求するのは **成果物と整合性** であり、  
local path や中間コマンド列そのものではない。

### 5.2 temporal split artifact の出力

temporal split artifact build では少なくとも次を出力する。

- `summary.json`
- `report.md`
- `train_assigned_labels.tsv`
- `dev_assigned_labels.tsv`
- `test_assigned_labels.tsv`
- `*_assigned_propagated.tsv`
- `*_assigned_nk_lk.tsv`
- `*_assigned_nk_lk_propagated.tsv`
- `nk_lk_eda.tsv`
- `pipeline_status.json`

これらは local に一時生成されてよいが、採用判定後は W&B Artifact に登録し、  
以後の参照は artifact version / alias で行う。

### 5.3 dataset 化

temporal split artifact を基に、reasoning dataset を作る。

必要に応じて、RL 用の preference / reward dataset を **train split からのみ**派生生成してよい。  
ただしこの派生 dataset も、元の benchmark version と `protein_id` 対応を保持する。

生成後は `train` / `validation` / `test` に分割し、学習と評価の両方で同じ dataset version を参照する。  
reasoning dataset は `protein_id` / `split` を保持し、temporal split artifact と一対一で対応していることを必須とする。

生成物は少なくとも次の単位で version 管理する。

- temporal split artifact
- reasoning dataset artifact
- RL preference / reward dataset artifact を作った場合はその artifact

current flow では、reasoning dataset だけを primary dataset として生成・登録する。  
補助用途の dataset を追加で作る必要が無い場合は、不要な dataset を生成しない。

## 6. データ・モデルのupload for W&B

本仕様では、W&B を experiment tracking と artifact lineage の標準にする。

### 6.1 共通の run config

各 phase で最低限 W&B に記録する config は次とする。

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
- `output_dir`
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
- stable な dataset version は artifact alias で安定参照してよい
- downstream の eval / SFT / RL は local dataset directory を正本にせず、W&B Artifact ref から解決する

### 6.3 model upload

固定ルール:

- model checkpoint は W&B Artifact として登録する
- model artifact の version / alias を run config に残す
- stable な model version は artifact alias で安定参照してよい
- public checkpoint は downstream 実行の直前に Hugging Face から毎回読むのではなく、一度 materialize して W&B model artifact に固定してから使う
- tuning 前の比較モデルは `bioreason-pro-rl-paper` に固定する
- `bioreason-pro-rl-paper` は public Hugging Face source `wanglab/bioreason-pro-rl` から W&B model artifact に固定してよい
- `train_sft` と `train_rl` の出力 checkpoint は、それぞれの学習 run の output artifact として登録する

### 6.4 artifact ref manifest

ここで扱うのは W&B Artifact ref である。  
CoreWeave 上の実行では、任意の local directory を毎回手で渡すのではなく、repo 内に置く **W&B Artifact ref manifest** を entry point にする。

固定ルール:

- data-bundle manifest は `configs/disease_benchmark/data_registry.json` を使う
- evaluation-target manifest は `configs/disease_benchmark/eval_target_registry.json` を使う
- asset publish manifest は `configs/disease_benchmark/artifact_publish_registry.json` を使う
- W&B Artifact ref の貼り付け用 env template は `configs/disease_benchmark/wandb_registry_paths.env.example` を使う
- source 用 env template は `configs/disease_benchmark/wandb_asset_sources.env.example` を使う
- manifest に渡す artifact ref は browser URL ではなく `entity/project/artifact_name:alias` 形式を使う
- data lineage は W&B Artifact ref を manifest に保持する
- public checkpoint の source 情報は manifest に保持し、一度 W&B artifact 化した後は artifact ref を source-of-truth にする
- private / internal checkpoint は W&B model artifact ref を manifest に保持する
- asset publish 後は `wandb_registry_paths.env` を更新し、eval / SFT / RL はその env file を既定入力として読む
- README と specification の両方で current ref が読める状態を維持する


## 7. 評価

### 7.1 eval phase

本仕様では、評価の論理フェーズ名を **`eval`** に固定する。  
repo 内の高位 entry point は `scripts/run_registered_eval.py`、低位 entry point は `eval.py` と `scripts/sh_eval.sh` を使う。  
W&B run は `wandb.init(..., job_type="eval")` で開始する。  
ProteinLLM 系の reasoning 評価では、同じ評価 run について `weave.Evaluation` の eval logger でも trace と score を追跡する。  
CoreWeave では login node から Slurm job を submit して評価を回し、GPU node へ直接入って手実行する前提にはしない。

評価 split の使い分けは次とする。

- 開発中の比較、ablation、checkpoint 比較は `validation`
- 最終報告値は lock 済みの `test`

### 7.2 定量評価

定量評価の主指標は **F_max** とし、次を必須とする。

- MF
- BP
- CC

評価対象:

- `bioreason-pro-rl-paper`
- `train_sft` の出力
- `train_rl` の出力がある場合はそれも含める

運用上の target family は次の 3 つに分ける。

- `comparison-family`: `bioreason-pro-rl-paper`
- `tuned-family`: `train-sft-output`, `train-rl-output`
- `spec-comparison`: 上記すべて

各 metric は `wandb.log()` で記録する。  
少なくとも split ごとに `fmax_mf`, `fmax_bp`, `fmax_cc` を保存する。

### 7.3 定性評価

定性評価では、少なくとも次の 3 点を確認する。

- 疾患名・経路名の具体的言及
- 変異と機能障害の因果記述
- 相互作用パートナーと疾患との関連記述

### 7.4 eval の保存物

必須事項:

- metric を `wandb.log()` する
- 評価 summary を **1 evaluated target = 1 row** の W&B Table として `wandb.log()` する
- sample-level 結果を **1 sample = 1 row** の W&B Table として `wandb.log()` する
- reasoning task の場合は sample-level table に reasoning 途中過程も含める
- JSON 結果、summary export、sample export は Artifact として version 管理する
- eval の local 出力は scratch とみなし、W&B artifact への保存が成功したら既定で cleanup してよい
- local の JSON / TSV を恒久保存の前提にしない

summary table の最低列:

- `model_name`
- `dataset_config`
- `split`
- `benchmark_version`
- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `macro_note`

sample-level table の最低列:

- `protein_id`
- `split`
- `model_name`
- `prompt`
- `prediction`
- `expected_output`
- `accuracy_or_match_note`
- `reasoning_excerpt`

reasoning task の場合は少なくとも次も保持する。

- `reasoning_full`
- `final_answer`
- `intermediate_trace`

## 8. 学習 for SFT

### 8.1 train_sft phase

本仕様では、SFT の論理フェーズ名を **`train_sft`** に固定する。  
repo 内の実装 entry point は `train_protein_llm.py` と `scripts/sh_train_protein_qwen_staged.sh` を使う。  
W&B run は `wandb.init(..., job_type="train_sft")` で開始する。

### 8.2 train_sft の入力

入力:

- `disease_temporal_hc_reasoning_v1`
- `bioreason-pro-rl-paper` checkpoint artifact

使い方:

- 学習には reasoning dataset の `train` split を使う
- checkpoint selection には reasoning dataset の `validation` split を使う
- `test` split は SFT 学習には使わず、最終評価まで保持する
- tuning 前の比較モデルは W&B Artifact ref から local に materialize して使う
- 前半フェーズと後半フェーズの両方で、`bioreason-pro-rl-paper` に含まれる projector / GO module 重みを warm-start に使ってよい

### 8.3 train_sft の必須事項

固定ルール:

- dataset を W&B Artifact として登録する
- 学習 run は W&B で追跡する
- train / validation の主要 metric を `wandb.log()` で保存する
- train sample / validation sample の代表例を W&B Table として `wandb.log()` する
- output checkpoint を W&B Artifact として登録する
- artifact alias を使う場合は、この phase の model artifact を安定参照先として昇格させてよい
- tuning 前の比較モデル checkpoint の取得元は local path を手入力せず、W&B Artifact ref を既定入力にする

W&B Table の最低列:

- `protein_id`
- `split`
- `input_summary`
- `reasoning`
- `final_answer`
- `expected_go_bp`
- `expected_go_mf`
- `expected_go_cc`

### 8.4 実行条件

固定ルール:

- 学習は GPU 前提
- `dataset_type=cafa5` 前提で dataset config を供給する
- 学習ジョブの wall time は **最大 12 時間**
- submission 時の time limit は `12:00:00` を使う
- checkpoint / resume 前提で運用する

## 9. 学習 for RL

### 9.1 train_rl phase

本仕様では、RL の論理フェーズ名を **`train_rl`** に固定する。  
phase 名は仕様上維持し、実装時には現行の学習スタックへ対応づけて運用する。  
W&B run は `wandb.init(..., job_type="train_rl")` で開始する。

### 9.2 train_rl の入力

入力:

- canonical には `train_sft` output checkpoint
- reward 設定
- 同じ benchmark 定義

使い方:

- rollout / reward 最適化には benchmark の `train` split を使う
- checkpoint selection と offline sanity-check には `validation` split を使う
- `test` split は RL 学習には使わず、最終評価まで保持する
- RL 用 dataset を派生生成する場合も `protein_id` と `benchmark_version` を保持し、SFT と同じ split 境界を崩さない
- `bioreason-pro-rl-paper` から直接 RL を始める経路は、必要な場合に限って ablation として扱う

### 9.3 train_rl の必須事項

固定ルール:

- RL run の config を W&B に保存する
- reward 系 metric、KL 系 metric、学習安定性指標を `wandb.log()` で保存する
- 途中のrolloutをweaveでtraceする
- RL input / reasoning 途中経過 / output sample を W&B Table として `wandb.log()` する
- RL 出力 checkpoint を W&B Artifact として登録する


W&B Table の最低列:

- `protein_id`
- `split`
- `prompt`
- `response`
- `reward`
- `parsed_go_terms`
- `notes`

### 9.4 実行条件

固定ルール:

- RL でも同じ benchmark version を使う
- 学習ジョブの wall time は **最大 12 時間**
- submission 時の time limit は `12:00:00` を使う
- checkpoint / resume 前提で運用する
