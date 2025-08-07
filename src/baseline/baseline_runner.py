# =============================================================
# File: src/baseline/baseline_runner.py
# =============================================================
"""
Baseline runner (batch, progress bar, dual outputs)

额外保存：
    predictions_answer.jsonl
        {
          "prompt":       …,
          "all_answers":  "204,204,235,…",   # k 个规范化答案
          "source":       dataset['source'],
          "id":           dataset['id']
        }
"""
from __future__ import annotations
import argparse, json, datetime, sys, csv, re
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import yaml                                   # ← 读取 YAML

from evaluation.evaluate_pass1 import compute_pass1, _normalize

ROOT_DIR   = Path(__file__).resolve().parents[2]
CONFIG_YML = ROOT_DIR / "configs" / "baseline.yaml"      # ★ 路径

# ------------------------------------------------------------------
# 读取 YAML 默认值 ---------------------------------------------------
# ------------------------------------------------------------------
_yaml_gen_defaults: Dict[str, Any] = {}
if CONFIG_YML.is_file():
    with CONFIG_YML.open() as fp:
        y = yaml.safe_load(fp)
        _yaml_gen_defaults = {k: y.get(k) for k in
                              ("k", "batch", "temperature", "top_p", "context_len")}
else:
    print(f"[WARN] Cannot read config {CONFIG_YML}, fallback to code defaults", file=sys.stderr)

def _cfg(key: str, fallback: Any) -> Any:
    """Get default from YAML, else fallback."""
    val = _yaml_gen_defaults.get(key)
    return fallback if val is None else val

# ------------------------------------------------------------------
# helpers -----------------------------------------------------------
# ------------------------------------------------------------------
_THINK_RE = re.compile(r"</think>", re.I)
def strip_think(txt: str) -> str:
    return _THINK_RE.split(txt, 1)[-1].lstrip() if "</think>" in txt.lower() else txt

def append_row(csv_path: str | Path, row: Dict[str, Any]):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    with csv_path.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)

def batch_generate(model, tok, prompts: List[str],
                   k: int, max_new: int,
                   temp: float, top_p: float) -> List[List[str]]:
    """Return k generations for each prompt."""
    inp = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=True, temperature=temp, top_p=top_p,
                             num_return_sequences=k, max_new_tokens=max_new)
    out = out.view(len(prompts), k, -1).cpu()
    ret: List[List[str]] = []
    for i, p in enumerate(prompts):
        dec = tok.batch_decode(out[i], skip_special_tokens=True)
        ret.append([strip_think(t[len(p):].strip()) for t in dec])
    return ret

# ------------------------------------------------------------------
# main --------------------------------------------------------------
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",  required=True, help="folder or HF repo")
    ap.add_argument("--task_dir",    required=True, help="dataset folder")
    ap.add_argument("--output_dir",  required=True)

    # ---------- generation hyper-parameters ----------
    ap.add_argument("--k",           type=int,   default=_cfg("k", 16))
    ap.add_argument("--batch",       type=int,   default=_cfg("batch", 4))
    ap.add_argument("--temperature", type=float, default=_cfg("temperature", 0.6))
    ap.add_argument("--top_p",       type=float, default=_cfg("top_p", 0.95))
    ap.add_argument("--context_len", type=int,   default=_cfg("context_len", 3072))

    ap.add_argument("--csv_path", default=f"{ROOT_DIR}/results/baseline/metrics.csv")
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- load model ---
    local_only = Path(args.model_path).is_dir()
    tok   = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                          local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
        local_files_only=local_only).eval()

    print(">>> cuda available :", torch.cuda.is_available())
    print(">>> device map      :", model.hf_device_map)
    print(f">>> generation args : k={args.k}  batch={args.batch}  "
          f"temp={args.temperature}  top_p={args.top_p}  context_len={args.context_len}")

    # --- load dataset ---
    ds_path = Path(args.task_dir)
    file_name = "test.parquet" if (ds_path / "test.parquet").exists() else "test.json"
    dataset = load_dataset(
        "parquet" if file_name.endswith(".parquet") else "json",
        data_files=str(ds_path / file_name),
        split="train"
    )

    preds, refs, ans_records = [], [], []
    pbar = tqdm(range(0, len(dataset), args.batch), desc="Generating", unit="batch")

    for start in pbar:
        end = min(start + args.batch, len(dataset))
        sub = dataset.select(range(start, end))

        prompts = [ex.get("prompt") or ex["problem"] for ex in sub]
        refs.extend(ex.get("answer") or ex.get("solution") or "" for ex in sub)

        # 计算本 batch 的 max_new_tokens
        max_new = max(16, args.context_len - max(len(tok(p).input_ids) for p in prompts))
        k_eff   = 4 if args.context_len > 30_000 else args.k

        batch_pred = batch_generate(model, tok, prompts, k_eff, max_new,
                                    args.temperature, args.top_p)
        preds.extend(batch_pred)

        # 保存规范化答案
        for ex, ans_list in zip(sub, batch_pred):
            ans_records.append({
                "prompt": ex.get("prompt") or ex["problem"],
                "all_answers": ", ".join(_normalize(a) for a in ans_list),
                "source": ex.get("source", ""),
                "id": ex.get("id", "")
            })
        pbar.set_postfix(done=len(preds))

    # --- pass@1 & outputs ---
    pass1 = compute_pass1(preds, refs)
    out_dir = Path(args.output_dir)

    with (out_dir / "predictions.jsonl").open("w") as fp:
        for row in preds:
            fp.write(json.dumps({"gens": row}) + "\n")

    with (out_dir / "predictions_answer.jsonl").open("w") as fp:
        for rec in ans_records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metric = {
        "model": Path(args.model_path).name,
        "task": ds_path.name,
        "pass@1": round(pass1 * 100, 2),
        "k": args.k, "batch": args.batch,
        "temp": args.temperature, "top_p": args.top_p,
        "context_len": args.context_len,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(metric, open(out_dir / "metrics.json", "w"), indent=2)
    append_row(args.csv_path, metric)
    print(f"\n[✓] {metric['model']} on {metric['task']} → pass@1 {metric['pass@1']}%")

if __name__ == "__main__":
    main()
