#!/usr/bin/env bash
# 依序建四個實驗資料庫並評估（一次只改一個變因）。結果在 evals/rag/experiments/<name>.json
set -euo pipefail
cd "$(dirname "$0")/../.."
export CHROMA_DB_PATH="${CHROMA_DB_PATH:-$(grep ^CHROMA_DB_PATH .env | cut -d= -f2-)}"

run() {  # run <name> <embed-model> [build 參數...]
  local name=$1 model=$2; shift 2
  local bf16=""; [[ " $* " == *" --bf16 "* ]] && bf16="--bf16"
  if [[ ! -d data/experiments/$name ]]; then
    echo "=== build $name ==="
    uv run python evals/rag/build_experiment_index.py --name "$name" --embed-model "$model" "$@"
  fi
  echo "=== eval $name ==="
  uv run python evals/rag/eval_rag_retrieval.py --configs vec_top10 \
    --db-path "data/experiments/$name" --embed-model "$model" $bf16 \
    --out "evals/rag/experiments/$name.json"
}

# 基準線（只按標題切、全部 metadata、bge-base）就是正式資料庫本身，不用重建：
#   uv run python evals/rag/eval_rag_retrieval.py --configs vec_top10
# 依「最可能有效」排序：先解決截斷，再換模型，最後才拆解 metadata 和切小各自的貢獻
run base_slim_split  BAAI/bge-base-zh-v1.5 --metadata slim --split tokens
run m3_slim_split    BAAI/bge-m3           --metadata slim --split tokens --batch-size 8
run base_slim        BAAI/bge-base-zh-v1.5 --metadata slim
