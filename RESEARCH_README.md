# Disease Benchmark Research README

この README は、`domain/specification/busiless-rules/specification.md` に沿って、疾患ベンチマークの研究実装をどう進めるかを実行順に整理したものである。  
流れは **データの準備 -> Slurm job の送信 -> ベースモデルの評価 -> SFT -> RL** とする。

前提は次の 3 点である。

- ローカル Mac では **データの準備だけ** を行う
- 学習と大規模評価は **CoreWeave GPU クラスター** に移ってから行う
- Python 環境は **uv** を前提にする

## 0. 命名と保存先

### 0.1 この README 内での呼び方

この README では次の呼び方を使う。

- `213-start main variant`: `213 -> 221 -> 225 -> 228`
- `214-start comparison variant`: `214 -> 221 -> 225 -> 228`
- `temporal split artifact`: release 差分、protein-disjoint split、label assignment を固定した基準 artifact

以後、人が読む文書では **`temporal split artifact`** に統一する。  
artifact family は `disease-temporal-split`、run config key は `temporal_split_artifact` に統一する。

`214-start comparison variant` という呼び方は **この README 内だけの便宜上の名称** である。  
W&B Artifact 側では、両者は同じ artifact family の別 version / alias として同等に扱う。

### 0.2 ローカル保存先

ローカル生成物は `domain/specification/.../artifacts` には置かず、**`data/artifacts`** 配下に置く。

使うディレクトリは次で固定する。

```text
data/artifacts/
├── benchmarks/
│   ├── 213_221_225_228/
│   │   └── temporal_split/
│   └── 214_221_225_228/
│       └── temporal_split/
├── datasets/
│   └── disease_temporal_hc_reasoning_v1/
│       ├── 213_221_225_228/
│       └── 214_221_225_228/
└── eval/
    └── ...   # eval の一時 scratch。W&B upload 成功後は既定で cleanup する
```

### 0.3 W&B Artifact の family, alias, URL

Artifact family は variant 名を family 名に埋め込まず、同じ family の中で alias で区別する。

| 用途 | Artifact family | 主 alias | 補助 alias | URL |
|---|---|---|---|---|
| Temporal split artifact | `disease-temporal-split` | `213.221.225.228`, `production` | `214.221.225.228` | `https://wandb.ai/wandb-healthcare/bioreason-pro-custom/artifacts/dataset/disease-temporal-split` |
| Reasoning dataset | `disease-temporal-reasoning` | `213.221.225.228`, `production` | `214.221.225.228` | `https://wandb.ai/wandb-healthcare/bioreason-pro-custom/artifacts/dataset/disease-temporal-reasoning` |

運用ルール:

- `213.221.225.228` はベンチマーク版を表す共通 alias として使う
- 今回主に使う main variant には `production` alias を付ける
- `214.221.225.228` は comparison variant 用 alias として使う
- comparison variant には `production` を付けない

公開 model artifact の現状:

| 用途 | Artifact family | 主 alias | URL |
|---|---|---|---|
| BioReason-Pro SFT | `bioreason-pro-sft` | `production`, `213.221.225.228` | `https://wandb.ai/wandb-healthcare/bioreason-pro-custom/artifacts/model/bioreason-pro-sft` |
| BioReason-Pro RL | `bioreason-pro-rl` | `production`, `213.221.225.228` | `https://wandb.ai/wandb-healthcare/bioreason-pro-custom/artifacts/model/bioreason-pro-rl` |

`BioReason-Pro base`, `BLAST / Diamond`, `ESM 系単体` は source dir が埋まってから publish する。

### 0.4 W&B Registry path manifest

ここでいう `registry` は **W&B Registry** のことである。  
repo 内の JSON は registry そのものではなく、**ユーザーが W&B Registry path を貼る manifest** として扱う。

使う manifest は次で固定する。

- data-bundle manifest: [data_registry.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/data_registry.json)
- evaluation-target manifest: [eval_target_registry.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/eval_target_registry.json)
- asset publish manifest: [artifact_publish_registry.json](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/artifact_publish_registry.json)
- path を貼る env template: [wandb_registry_paths.env.example](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_registry_paths.env.example)
- 実際に path を入れた local env file: [wandb_registry_paths.env](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_registry_paths.env)
- source を貼る env template: [wandb_asset_sources.env.example](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_asset_sources.env.example)

役割は次のとおりである。

- env file: temporal split artifact, dataset, prediction baseline, model の **W&B Registry path** を貼る
- source env file: public Hugging Face repo ID、private checkpoint の local dir、baseline prediction dir を貼る
- data-bundle manifest: benchmark version と、どの dataset registry path を使うかを束ねる
- evaluation-target manifest: `base-family`, `tuned-family`, `spec-comparison` の target group と、各 model / prediction source を束ねる

ここで貼る path は、browser URL ではなく `entity/project/artifact_name:alias` 形式の **W&B Registry ref** を使う。

現時点で [wandb_registry_paths.env](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/configs/disease_benchmark/wandb_registry_paths.env) に貼ってある ref は次である。

| env var | 用途 | 現在の W&B Registry ref |
|---|---|---|
| `BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH` | 213-start main variant の temporal split artifact | `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:production` |
| `BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH` | 213-start main variant の reasoning dataset | `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:production` |
| `BIOREASON_COMPARISON_TEMPORAL_SPLIT_REGISTRY_PATH` | 214-start comparison variant の temporal split artifact | `wandb-healthcare/bioreason-pro-custom/disease-temporal-split:214.221.225.228` |
| `BIOREASON_COMPARISON_REASONING_DATASET_REGISTRY_PATH` | 214-start comparison variant の reasoning dataset | `wandb-healthcare/bioreason-pro-custom/disease-temporal-reasoning:214.221.225.228` |
| `BIOREASON_SFT_MODEL_REGISTRY_PATH` | SFT model artifact | `wandb-healthcare/bioreason-pro-custom/bioreason-pro-sft:production` |
| `BIOREASON_RL_MODEL_REGISTRY_PATH` | RL model artifact | `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production` |

未設定の ref は次である。

| env var | 用途 | 状態 |
|---|---|---|
| `BIOREASON_BASE_MODEL_REGISTRY_PATH` | base model artifact | 未設定 |
| `BIOREASON_BLAST_DIAMOND_PRED_REGISTRY_PATH` | BLAST / Diamond baseline prediction artifact | 未設定 |
| `BIOREASON_ESM_STANDALONE_PRED_REGISTRY_PATH` | ESM standalone baseline prediction artifact | 未設定 |

公開 checkpoint は、まず一度 Hugging Face から materialize し、その後 **W&B model artifact に publish** して使う。  
private / internal checkpoint と dataset は、**貼り付けた W&B Registry path** から取得する。  
downstream の eval / SFT / RL は、source dir や HF repo ではなく **W&B Registry ref** を正本として解決する。

## 1. データの準備

この工程は **ローカル Mac** で行う。

### 1.1 uv 用の依存ファイルを使ってローカル Mac 環境を作る

ローカル Mac 側の lightweight 環境は、process の中で package を列挙せず、準備済みの uv 用ファイルを使う。

- data prep 用: [uv-local-data.txt](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/requirements/uv-local-data.txt)
- contract test 用: [uv-contract-tests.txt](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/requirements/uv-contract-tests.txt)

まず data prep 用の環境を作る。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv venv .venv-mac-data --python 3.11
source .venv-mac-data/bin/activate
uv pip install -r requirements/uv-local-data.txt
```

### 1.2 temporal split artifact の準備・生成・sanity check・W&B upload を一気に実行する

local storage preparation、temporal split build、sanity check、W&B upload は、次の 1 本のコマンドでまとめて回す。

まず `.env` を読み込む。

```bash
set -a
source .env
set +a
```

main variant と comparison variant をまとめて回す。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv run --active python scripts/run_temporal_split_artifact_pipeline.py \
  --variant all \
  --shortlist-mode high-confidence \
  --use-shell-filter \
  --upload-to-wandb
```

main variant だけに絞る場合は次でよい。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv run --active python scripts/run_temporal_split_artifact_pipeline.py \
  --variant main \
  --shortlist-mode high-confidence \
  --use-shell-filter \
  --upload-to-wandb
```

### 1.3 上の 1 コマンドが内部でやること

この pipeline script は variant ごとに次を自動で行う。

1. `data/artifacts`, `data/artifacts/eval`, variant ごとの `temporal_split/` 保存先を作る
2. `scripts/build_disease_temporal_split_artifact.py` を実行する
3. `summary.json` と `report.md` を読んで sanity check を行う
4. sanity check が通った場合だけ W&B artifact upload に進む
5. 実行結果を `pipeline_status.json` に保存する

reasoning dataset directory に中身が既にある場合は、temporal split artifact に加えて reasoning dataset artifact も同じ run で upload する。  
reasoning dataset directory が無い、または空である場合は、temporal split artifact だけを upload する。

sanity check で見ている項目は次である。

- `summary.json` に `split_validation.time_order_valid == true`
- `summary.json` に `split_validation.protein_disjoint_valid == true`
- `summary.json` に split ごとの protein 数が入っている
- `report.md` に split summary table がある
- temporal split artifact の必須成果物がそろっている

### 1.4 生成先と alias

variant ごとの release anchor と出力先は次で固定する。

| Variant | Window | Output dir | Artifact aliases |
|---|---|---|---|
| `main` | `213 -> 221 -> 225 -> 228` | `data/artifacts/benchmarks/213_221_225_228/temporal_split` | `213.221.225.228`, `production` |
| `comparison` | `214 -> 221 -> 225 -> 228` | `data/artifacts/benchmarks/214_221_225_228/temporal_split` | `214.221.225.228` |

各 `temporal_split/` ディレクトリには、少なくとも次が入る。

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

### 1.5 contract test は必要なときだけ別で回す

実装を触ったあとに contract test もローカル Mac で確認したい場合だけ、別の uv 環境で回す。

```bash
cd /Users/keisuke/Project/learning/drug_discovery/BioReason-Pro

uv venv .venv-contract-tests --python 3.11
source .venv-contract-tests/bin/activate
uv pip install -r requirements/uv-contract-tests.txt

uv run --active python -m unittest discover -s test -v
```

## 2. Slurm job の送信

評価と学習は **CoreWeave GPU クラスター** に移ってから行う。  
この README では、GPU node へ直接入るのではなく、**login node から `srun` で job を送る**運用に揃える。

### 2.1 login node に入る

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
```

このあとに実行する `uv`, `wandb`, `srun` の操作は login node 上で行う。  
長時間の計算は login node では回さず、必ず Slurm job として送る。

### 2.2 ローカル Mac からコードだけ送る

ローカル Mac 側で実行する。

```bash
cd /Users/keisuke/Project/learning/drug_discovery

rsync -av --delete \
  --exclude 'data/artifacts/' \
  --exclude '.venv*/' \
  BioReason-Pro/ \
  kkamata+cwb607@sunk.cwb607-training.coreweave.app:~/BioReason-Pro/
```

`data/artifacts` はローカル生成物なので、この `rsync` には含めない。  
benchmark / dataset / prediction baseline は、CoreWeave 側で **W&B Registry path を manifest 経由で解決**する。

### 2.3 CoreWeave 上で uv 環境を作る

CoreWeave 側ではフル実装用の環境を作る。

```bash
cd ~/BioReason-Pro

uv venv .venv-gpu --python 3.11
source .venv-gpu/bin/activate

uv sync
uv pip install esm --no-deps
uv pip install flash-attn --no-build-isolation --no-cache-dir
uv pip install unsloth
uv run --active wandb login
```

### 2.4 1 回だけ env file を作る

model path や dataset directory は毎回コマンドに書かず、**W&B Registry path** と source env file で解決する。

```bash
set -a
source .env
set +a

cp configs/disease_benchmark/wandb_registry_paths.env.example \
  configs/disease_benchmark/wandb_registry_paths.env

cp configs/disease_benchmark/wandb_asset_sources.env.example \
  configs/disease_benchmark/wandb_asset_sources.env

# temporal split / dataset / published model artifact の W&B Registry path をここに貼る
$EDITOR configs/disease_benchmark/wandb_registry_paths.env

# model / baseline prediction の source をここに貼る
$EDITOR configs/disease_benchmark/wandb_asset_sources.env

export BIOREASON_GO_EMBEDDINGS_PATH="/path/to/go-embeddings"
export BIOREASON_IA_FILE_PATH="/path/to/IA.txt"
export BIOREASON_STRUCTURE_DIR="$HOME/BioReason-Pro/data/structures"
export BIOREASON_DATASET_CACHE_DIR="$HOME/BioReason-Pro/data/artifacts/hf_cache"
```

`GO_OBO_PATH` は repo 内の `bioreason2/dataset/go-basic.obo` を既定で使うので、通常は設定不要である。
`WANDB_PROJECT` は run の記録先であり、dataset / model の取得元は `wandb_registry_paths.env` に書いた W&B Registry ref が source-of-truth になる。
`register_research_assets.py` は `wandb_asset_sources.env` を、`run_registered_eval.py` と `sh_train_protein_qwen_staged.sh` は `wandb_registry_paths.env` を既定で自動読込する。

### 2.5 model / baseline artifact を一度 publish する

比較対象は、**source から直接読むのではなく、最初に W&B Artifact に固定してから使う**。

固定ルール:

- `BioReason-Pro SFT` と `BioReason-Pro RL` は public Hugging Face source から一度 download し、W&B model artifact に publish する
- `BioReason-Pro base` は public source が固定されていないので、private checkpoint または local checkpoint dir を source にして publish する
- `BLAST / Diamond` と `ESM 系単体` は checkpoint ではなく **prediction artifact** として publish する
- publish が成功すると `configs/disease_benchmark/wandb_registry_paths.env` が自動更新される

まず public model を publish する。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

uv run --active python scripts/register_research_assets.py \
  --asset bioreason-pro-sft \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"

uv run --active python scripts/register_research_assets.py \
  --asset bioreason-pro-rl \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

`BioReason-Pro base` は source を埋めたあとに publish する。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

uv run --active python scripts/register_research_assets.py \
  --asset bioreason-pro-base \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

baseline prediction artifact も、prediction dir を埋めたあとに publish する。

```bash
cd ~/BioReason-Pro
source .venv-gpu/bin/activate

uv run --active python scripts/register_research_assets.py \
  --asset blast-diamond-baseline \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"

uv run --active python scripts/register_research_assets.py \
  --asset esm-standalone-baseline \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project "$WANDB_PROJECT"
```

補足:

- `configs/disease_benchmark/wandb_asset_sources.env` に `BIOREASON_PRO_SFT_HF_REPO="wanglab/bioreason-pro-sft"` と `BIOREASON_PRO_RL_HF_REPO="wanglab/bioreason-pro-rl"` を入れておけば、public model は自動で materialize される
- `BIOREASON_PRO_BASE_HF_REPO` は空のままでよい。base は current README では public HF source を固定しない
- `BLAST / Diamond` と `ESM 系単体` は予測 TSV 群を置いた directory を source にする
- 現在 publish 済みの registry ref は `wandb-healthcare/bioreason-pro-custom/bioreason-pro-sft:production` と `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production` である

### 2.6 srun の基本形

実際の partition 名や account 名は自分の環境に合わせて置き換える。  
以後の評価・学習コマンドは、すべてこの形で login node から送る。

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro
    source .venv-gpu/bin/activate
    <your command here>
  '
```

## 3. ベースモデルの評価

この工程は login node から `srun` で送る。  
高位 entry point は [run_registered_eval.py](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/run_registered_eval.py) とし、低位 wrapper の [sh_eval.sh](/Users/keisuke/Project/learning/drug_discovery/BioReason-Pro/scripts/sh_eval.sh) は直接触らない前提で進める。

### 3.1 仕様に沿った評価対象

スペシフィケーションに沿って、比較対象は次の 4 系統に固定する。

- `BLAST / Diamond`: prediction artifact として評価する
- `ESM 系単体`: prediction artifact として評価する
- `BioReason-Pro base`: W&B Registry path を解決して評価する
- `tuned model`: `bioreason-pro-sft`, `bioreason-pro-rl` を W&B Registry path に従って評価する

manifest 上の target group は次を使う。

- `base-family`: `blast-diamond-baseline`, `esm-standalone-baseline`, `bioreason-pro-base`
- `tuned-family`: `bioreason-pro-sft`, `bioreason-pro-rl`
- `spec-comparison`: 上記すべて

### 3.2 base-family を validation で一括評価する

まずは base-family を `validation` split で回す。

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target-group base-family \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

この command は target ごとに次を自動で行う。

1. data-bundle manifest から `temporal split artifact` と dataset の W&B Registry path を解決する
2. evaluation-target manifest から model source または prediction artifact source を解決する
3. data / model / prediction artifact は `wandb_registry_paths.env` に書いた W&B Registry ref から取得する
4. prediction artifact baseline は CAFA metric 経路で `F_max` を計算する
5. W&B に metric / summary table / sample table / result artifact を保存する
6. local eval scratch は debug 用に `--keep-local-eval-outputs` を付けない限り保持しない

### 3.3 個別 target を評価する

`base-family` 全体ではなく、1 target だけ回したい場合は `--target` を使う。

`BioReason-Pro base`:

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target bioreason-pro-base \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

`BLAST / Diamond` baseline:

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target blast-diamond-baseline \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

`ESM 系単体` baseline:

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target esm-standalone-baseline \
      --data-bundle main_production \
      --split validation \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 3.4 tuned-family を test で評価する

`train_sft` や `train_rl` の checkpoint が用意できたら、`tuned-family` を `test` split で回す。

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 1 \
  --cpus-per-task 8 \
  --mem 128G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    uv run --active python scripts/run_registered_eval.py \
      --target-group tuned-family \
      --data-bundle main_production \
      --split test \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT"
  '
```

### 3.5 各 eval run で自動保存されるもの

F_max は **各 eval run の中で自動計算され、そのまま W&B に保存される**。  
別ステップで `evals/cafa_evals.py` を手で回す想定ではない。

確認先は W&B を正本とする。

- W&B run page
- W&B artifact page
- Weave evaluation page

W&B では少なくとも次が見えていることを確認する。

- `fmax_mf`
- `fmax_bp`
- `fmax_cc`
- `overall_mean_fmax`
- `eval_summary` table
- `eval_samples` table

既定では、eval の JSON / TSV / metric export は **W&B artifact に保存されたあと local から cleanup される**。  
ローカルに残して目視確認したいときだけ、`scripts/run_registered_eval.py` に `--keep-local-eval-outputs` を付ける。

## 4. SFT

この工程も login node から `srun` で送る。

### 4.1 先に埋めるもの

`scripts/sh_train_protein_qwen_staged.sh` は、固定値をかなり持った low-level wrapper である。  
毎回 arbitrary な path を書くのではなく、次だけを確認する。

- `WANDB_ENTITY`
- `configs/disease_benchmark/wandb_registry_paths.env`
- `BIOREASON_GO_EMBEDDINGS_PATH`
- `BIOREASON_STRUCTURE_DIR`
- `BIOREASON_DATASET_CACHE_DIR`
- `BASE_CHECKPOINT_DIR`

既定値は次のとおりで、スペシフィケーション側に寄せてある。

- `BASE_CHECKPOINT_DIR="data/artifacts/models/bioreason_pro_base"`
- `TEMPORAL_SPLIT_ARTIFACT="$BIOREASON_MAIN_TEMPORAL_SPLIT_REGISTRY_PATH"`
- `DATASET_ARTIFACT="$BIOREASON_MAIN_REASONING_DATASET_REGISTRY_PATH"`
- `BASE_CHECKPOINT="$BIOREASON_BASE_MODEL_REGISTRY_PATH"`
- `CAFA5_DATASET="wanglab/cafa5"`
- `STAGE1_DATASET_NAME="disease_temporal_hc_reasoning_v1"`
- `STAGE2_DATASET_NAME="disease_temporal_hc_reasoning_v1"`

SFT wrapper は、`BIOREASON_BASE_MODEL_REGISTRY_PATH` を見て base model artifact を local に materialize し、前半フェーズと後半フェーズの両方でその重みを初期値として使う。

### 4.2 実行コマンド

```bash
srun \
  --partition <gpu_partition> \
  --account <account_name> \
  --gpus 8 \
  --cpus-per-task 16 \
  --mem 256G \
  --time 12:00:00 \
  bash -lc '
    cd ~/BioReason-Pro &&
    source .venv-gpu/bin/activate &&
    bash scripts/sh_train_protein_qwen_staged.sh
  '
```

この wrapper は次を行う。

1. 前半フェーズ: projector / GO module warm-up
2. 後半フェーズ: full model fine-tuning

### 4.3 実行後に確認するもの

- checkpoint directory
- W&B run config
- train / validation loss
- sample generation table
- output checkpoint artifact

W&B 上で最低限確認するもの:

- `job_type=train_sft`
- `benchmark_version`
- `temporal_split_artifact`
- `dataset_config`
- `reasoning_dataset_config`
- `dataset_artifact`
- `model_artifact`
- `job_time_limit=12:00:00`

## 5. RL

RL は flow 上は最後に置くが、**現時点では repo に実行 entry point がまだ揃っていない**。  
そのため、今すぐ回す対象は temporal split artifact, eval, SFT までである。

RL 着手時の前提だけ先に固定する。

- benchmark version は `213 -> 221 -> 225 -> 228`
- benchmark alias は `213.221.225.228`
- main run にだけ `production` alias を使う
- RL 用 dataset は `train` split 由来のみ
- `validation` は checkpoint selection 用
- `test` は最終評価専用

RL 実装が入ったら、README には次を追加する。

- RL dataset の生成コマンド
- RL rollout の W&B / Weave tracking コマンド
- RL checkpoint artifact の upload / alias 付け

## 6. 最短実行順

迷ったら次の順に進める。

1. ローカル Mac で `uv` 環境を作る
2. `scripts/run_temporal_split_artifact_pipeline.py` を 1 回実行する
3. CoreWeave に repo を送る
4. CoreWeave 上で `uv` 環境を作り、`WANDB_*`, `wandb_registry_paths.env`, `wandb_asset_sources.env` を用意する
5. public model と private baseline source を一度 W&B artifact に publish する
6. `base-family` を `validation` で評価する
7. SFT を回す
8. `tuned-family` を `test` で評価する
9. RL entry point が入ったら RL に進む
