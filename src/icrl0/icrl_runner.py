# =============================================================
# File: src/icrl/icrl0_runner.py          (initial 2025‑07‑04)
# =============================================================
"""
ICRL‑0  —  Iterative Chain‑of‑Thought Reinforcement Learning **压缩版**。

算法区别（vs 原 ICRL）：
  1. 每轮只保留随机抽样回答的「推理逻辑摘要」(Si)，不携带原始数字计算细节。
  2. 依据 pseudo‑GT 判定 reward∈{0,1}；摘要与 reward 一并存入
     good (reward=1) / bad (reward=0) 两个集合。
  3. 构造下一轮 prompt 时，格式固定：
        {Q,
         不好的回答reward=0{S_bad1, S_bad2, …},
         好的回答reward=1{S_good1, S_good2, …}}
  4. Bootstrap 迭代 R 轮后，再以最终 prompt 生成 k 个候选答案做评测。

本实现在原 ICRL runner 基础上改动：
  • 新增 _compress_answer() (调用同一模型快速摘要);
  • 将 history 结构改为 {"good": [...], "bad": [...]} ;
  • 重写 build_prompt() ;
  • 其它核心流程保持一致。
"""
from __future__ import annotations
import argparse, datetime, json, csv, random, re
from pathlib import Path
from typing import List, Dict, Any

import torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from icrl.prompt_cache import PromptCache

ROOT_DIR   = Path(__file__).resolve().parents[2]
CONFIG_YML = ROOT_DIR / "configs" / "icrl.yaml"
EXAMPLE    = ROOT_DIR / "results" / "icrl0" / "example.txt"

_cfg: Dict[str, Any] = yaml.safe_load(CONFIG_YML.open()) if CONFIG_YML.is_file() else {}
def cfg(k, d): return _cfg.get(k, d)

# ---------- 基础正则 ----------
_THINK_TAGS = re.compile(r"</?think>", re.I)
_BOXED      = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
_NUMBER     = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")
_LETTER     = re.compile(r"\b([A-E])\b", re.I)

def _extract_numeric(txt: str) -> str:
    """从答案全文中提取最后出现的数字/选项 (与 ICRL 一致)."""
    txt = _THINK_TAGS.sub(" ", txt).strip()
    if not txt: return ""
    if (m := _BOXED.search(txt)):
        return m.group(1)
    nums = _NUMBER.findall(txt.replace(",", ""))
    if nums: return nums[-1]
    lets = _LETTER.findall(txt)
    if lets: return lets[-1]
    return txt

def _normalize(ans: str) -> str:
    ans = _extract_numeric(ans)
    return ans.replace("$", "").replace(" ", "").replace(",", "").rstrip(".").lower()

def _majority_raw(lst: List[str]) -> str:
    cnt, raw = {}, {}
    for a in lst:
        k = _normalize(a)
        cnt[k] = cnt.get(k, 0) + 1
        raw.setdefault(k, a)
    return raw[max(cnt, key=cnt.get)]

# ---------- prompt 构造 ----------
SYS_PROMPT = (
    "You are an AI mathematician.  Below are compressed solution ideas "
    "from previous attempts; each idea is marked with reward 1 (correct) "
    "or 0 (incorrect).  Use the question and these ideas to reason out the "
    "correct numeric answer.  **Respond with exactly one number**.\n"
)

def build_prompt(q: str, hist: Dict[str, List[str]]) -> str:
    """
    hist = {"good": [...], "bad": [...]}
    """
    lines = [SYS_PROMPT.rstrip(), "", q.rstrip(), ""]
    if hist["bad"]:
        lines.append("不好的回答reward=0{")
        lines += [s.strip() for s in hist["bad"]]
        lines.append("}")
    if hist["good"]:
        lines.append("好的回答reward=1{")
        lines += [s.strip() for s in hist["good"]]
        lines.append("}")
    return "\n".join(lines).rstrip()

# ---------- LLM utility ----------
def _gen(model, tok, prompt: str, max_new: int,
         temp: float = 0.6, top_p: float = 0.95) -> str:
    """单条生成（方便压缩、选答）"""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=temp, top_p=top_p,
                             pad_token_id=tok.eos_token_id)[0]
    return tok.decode(out[enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

def generate_batch(model, tok, prompts: List[str], n: int, max_new: int,
                   temp: float, top_p: float) -> List[List[str]]:
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    base = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, num_return_sequences=n, do_sample=True,
                             temperature=temp, top_p=top_p,
                             max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id)
    out = out.view(len(prompts), n, -1).cpu()
    res = []
    for i in range(len(prompts)):
        res.append([tok.decode(out[i, j, base:], skip_special_tokens=True).strip()
                    for j in range(n)])
    return res

# ---------- 压缩函数 ----------
_SUMMARY_PROMPT = (
    "生成解答的核心思路，可以忽略具体数值计算，仅保留推理逻辑，"
    "请用 1‑2 句话压缩以下内容：\n\n【回答开始】\n{}\n【回答结束】"
)
def _compress_answer(ans_raw: str, model, tok) -> str:
    prompt = _SUMMARY_PROMPT.format(ans_raw.strip())
    summary = _gen(model, tok, prompt, max_new=48, temp=0.1, top_p=0.9)
    # 若模型返回空或过短，fallback
    return summary.strip() or "<BLANK‑SUMMARY>"

# ---------- 评测指标 ----------
def mean_at_k(preds, refs):
    return sum(sum(_normalize(c) == _normalize(r) for c in cand)/len(cand)
               for cand, r in zip(preds, refs)) / len(preds)

# =============================================================
def main():
    ag = argparse.ArgumentParser()
    ag.add_argument("--model_path", required=True)
    ag.add_argument("--task_dir",   required=True)
    ag.add_argument("--output_dir", required=True)
    ag.add_argument("--csv_path",   default=f"{ROOT_DIR}/results/icrl0/metrics.csv")
    ag.add_argument("--k",      type=int,   default=cfg("k", 16))
    ag.add_argument("--batch",  type=int,   default=cfg("batch", 4))
    ag.add_argument("--temp",   type=float, default=cfg("temperature", 0.6))
    ag.add_argument("--top_p",  type=float, default=cfg("top_p", 0.95))
    ag.add_argument("--ctx",    type=int,   default=cfg("context_len", 3072))
    ag.add_argument("--rounds", type=int,   default=cfg("rounds", 3))
    args = ag.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True).eval()

    ds_path = Path(args.task_dir)
    file = "test.parquet" if (ds_path / "test.parquet").exists() else "test.json"
    dataset = load_dataset("parquet" if file.endswith(".parquet") else "json",
                           data_files=str(ds_path / file), split="train")

    cache = PromptCache(out_dir / ".prompt_cache")
    preds, refs, answer_log = [], [], []
    wrote_example = False
    pbar = tqdm(range(0, len(dataset), args.batch), unit="batch", desc="ICRL‑0")

    for st_idx in pbar:
        ed_idx = min(st_idx + args.batch, len(dataset))
        sub = dataset.select(range(st_idx, ed_idx))
        qs  = [ex.get("prompt") or ex["problem"] for ex in sub]
        refs += [ex.get("answer") or ex.get("solution") or "" for ex in sub]

        hist = [{"good": [], "bad": []} for _ in sub]

        # ---------- bootstrap 迭代 ----------
        for r in range(1, args.rounds + 1):
            prompts = []
            for q, h in zip(qs, hist):
                key = f"{hash(q)}-r{r}-g{len(h['good'])}-b{len(h['bad'])}"
                p    = cache.get(key) or build_prompt(q, h)
                cache.put(key, p); prompts.append(p)

            max_new = max(16, args.ctx - max(len(tok(p).input_ids) for p in prompts))

            # k 个候选
            k_batch = generate_batch(model, tok, prompts, args.k,
                                     max_new, args.temp, args.top_p)
            pseudo  = [_majority_raw(lst) for lst in k_batch]

            # 随机选 1 个回答
            rand_idx = [random.randrange(args.k) for _ in prompts]
            picked   = [lst[i] for lst, i in zip(k_batch, rand_idx)]

            for h, ans_raw, pseudo_raw in zip(hist, picked, pseudo):
                reward = int(_normalize(ans_raw) == _normalize(pseudo_raw))
                summary = _compress_answer(ans_raw, model, tok)
                tgt = "good" if reward == 1 else "bad"
                h[tgt].append(summary)

        # ---------- 评估轮 ----------
        final_prompts = [build_prompt(q, h) for q, h in zip(qs, hist)]
        max_new = max(16, args.ctx - max(len(tok(p).input_ids)
                                         for p in final_prompts))
        final_k = generate_batch(model, tok, final_prompts, args.k,
                                 max_new, args.temp, args.top_p)
        preds += final_k

        # 保存示例
        if not wrote_example:
            EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
            EXAMPLE.write_text(f"PROMPT:\n{final_prompts[0]}\n\nANSWER:\n{final_k[0][0]}",
                               encoding="utf-8")
            wrote_example = True

        # 记录 numeric 答案日志
        for prmpt, klist, ex in zip(final_prompts, final_k, sub):
            nums = [n for n in (_normalize(a) for a in klist) if n]
            answer_log.append({"prompt": prmpt, "numeric_answers": nums,
                               "source": ex.get("source", ""), "id": ex.get("id", "")})

        pbar.set_postfix(done=len(preds))

    # ---------- 结果输出 ----------
    mean_k = round(mean_at_k(preds, refs) * 100, 2)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps({"gens": g}, ensure_ascii=False) for g in preds),
        encoding="utf-8")
    (out_dir / "predictions_answer.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in answer_log),
        encoding="utf-8")

    metric = {"model": Path(args.model_path).name, "task": ds_path.name,
              "mean@k": mean_k, "k": args.k, "rounds": args.rounds,
              "batch": args.batch, "temp": args.temp, "top_p": args.top_p,
              "context_len": args.ctx,
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    (out_dir / "metrics.json").write_text(json.dumps(metric, indent=2),
                                          encoding="utf-8")

    csv_path = Path(args.csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    head = not csv_path.exists()
    with csv_path.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(metric.keys()))
        if head: w.writeheader()
        w.writerow(metric)

    print(f"\n[✓] {metric['model']} on {metric['task']} → mean@k {metric['mean@k']}%")

if __name__ == "__main__":
    main()
