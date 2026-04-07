# 2026-04-07: dataset iteration notes

本メモは、最終仕様から外した dataset iteration の学びを残すためのものである。  
`domain/specification/busiless-rules/specification.md` には、ここに書く比較版や候補案を持ち込まない。

## 1. comparison benchmark

- `214 -> 221 -> 225 -> 228` 版も一度作成した
- 実測は train `932 proteins / 1,898 unique labels`、validation `662 proteins`、test `969 proteins`
- これは final benchmark ではなく、archive choice と件数感度を見る comparison variant として扱う

## 2. broader filter

- `cc_disease:*` を主条件にした broader query も確認した
- live count は `5,093 proteins`
- final benchmark では、`MIM/Orphanet` を含む high-confidence filter を採用する

## 3. small-data 判断

- current benchmark は small-data 版である
- ただし、time-independent benchmark と strict split rule があるため、追加学習の検証対象としては成立すると判断した
- `3,000 unique labels` は安心材料としての目安であり、hard gate ではないという整理を一度行った

## 4. local scratch と W&B

- 以前は local `data/artifacts/...` をやや強く書いていた
- 現在は、local 出力は scratch とし、W&B Artifact ref を source-of-truth にする方針へ整理した
- final spec では「必要のない local dataset を残さない」を明示した

## 5. NK / LK

- NK / LK は補助解析として一度 EDA 対象に含めた
- ただし final spec では、benchmark の必須成果物や go/no-go 条件には置かない
- 必要なら将来の analysis artifact として別途扱う
