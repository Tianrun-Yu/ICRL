set -euo pipefail
ROOT=/home/jovyan/ICRL
export HF_HOME="$ROOT/models"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

CONFIG="$ROOT/configs/icrl.yaml"
CSV_PATH="$ROOT/results/icrl/metrics.csv"

# 读取 YAML 缺省 batch & rounds
read_cfg () {
  python - <<PY
import yaml, sys
cfg=yaml.safe_load(open("$CONFIG"))
print(cfg.get("$1"))
PY
}
BATCH=$(read_cfg batch)
ROUNDS=$(read_cfg rounds)

single_run () {
  local MODEL="$1"; local TASK="$2"
  local OUTDIR="$ROOT/results/icrl/${TASK}_${MODEL}"
  echo "[RUN] $MODEL | $TASK | batch=$BATCH | rounds=$ROUNDS"
  python -m icrl.icrl_runner \
    --model_path "$ROOT/models/$MODEL" \
    --task_dir   "$ROOT/data/$TASK"   \
    --output_dir "$OUTDIR"            \
    --csv_path   "$CSV_PATH"          \
    --batch "$BATCH" --rounds "$ROUNDS"
}

if [[ "${1:-}" == "all" ]]; then
  mapfile -t MODELS   < <(read_cfg models | tr -d '[],' | xargs -n1)
  mapfile -t DATASETS < <(read_cfg datasets | tr -d '[],' | xargs -n1)
  for m in "${MODELS[@]}";   do
    for d in "${DATASETS[@]}"; do single_run "$m" "$d"; done
  done
else
  [[ $# -eq 2 ]] || { echo "Usage: $0 MODEL DATASET | $0 all"; exit 1; }
  single_run "$1" "$2"
fi 