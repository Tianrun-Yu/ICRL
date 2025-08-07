set -e
ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"  # /home/jovyan/ICRL
export HF_HOME="$ROOT/models"          # optional central cache
export HUGGINGFACE_HUB_TOKEN="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"  # <-- your token

PY_SCRIPT="$ROOT/models/download.py"

if [[ $# -gt 0 ]]; then
  python "$PY_SCRIPT" "$@"
else
  python "$PY_SCRIPT"
fi