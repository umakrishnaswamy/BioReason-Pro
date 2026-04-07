# 2026-04-07: comparison model を `bioreason-pro-rl-paper` に整理

## 背景

- README / specification / PLAN / manifest の間で、公開論文モデルと、今後こちらで学習して作る tuned model が混在していた
- `bioreason-pro-base` という名前は、実際に比較したい「独自 tuning 前の公開モデル」を正確に表していなかった
- `BLAST / Diamond` と `ESM standalone` は比較対象として文書に残っていたが、現時点では再利用可能な公開 Artifact ref も、repo 内で即利用できる prediction source も確定していなかった

## 今回の決定

1. tuning 前の比較モデルは `bioreason-pro-rl-paper` に固定する
2. `bioreason-pro-rl-paper` は公開 Hugging Face source `wanglab/bioreason-pro-rl` を指す
3. W&B 上の実 ref は当面 `wandb-healthcare/bioreason-pro-custom/bioreason-pro-rl:production` を使う
4. `BLAST / Diamond` と `ESM standalone` は specification / PLAN / RESEARCH_README の current scope から外す
5. comparison model と、今後こちらで生成する `train_sft` / `train_rl` output は別名で管理する

## 変更した方針

- 比較対象:
  - `bioreason-pro-rl-paper`
  - `train_sft` output
  - `train_rl` output
- target group:
  - `comparison-family`
  - `tuned-family`
  - `spec-comparison`
- SFT の初期 checkpoint:
  - `BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH` を既定入力として使う

## 反映先

- `domain/specification/busiless-rules/specification.md`
- `domain/specification/PLAN.md`
- `RESEARCH_README.md`
- `configs/disease_benchmark/eval_target_registry.json`
- `configs/disease_benchmark/artifact_publish_registry.json`
- `configs/disease_benchmark/wandb_registry_paths.env`
- `configs/disease_benchmark/wandb_registry_paths.env.example`
- `configs/disease_benchmark/wandb_asset_sources.env`
- `configs/disease_benchmark/wandb_asset_sources.env.example`
- `scripts/sh_train_protein_qwen_staged.sh`

## 補足

- W&B の Artifact family 自体は現時点では `bioreason-pro-rl` のままだが、repo 内の論理名は `bioreason-pro-rl-paper` として扱う
- これは「公開論文モデル」と「こちらで後から作る custom RL output」を混同しないための整理である
