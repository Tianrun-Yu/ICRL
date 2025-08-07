# =============================================================
# File: src/evaluation/evaluate_pass1.py
# =============================================================
"""
pass@1 evaluator with robust answer normalization.

规则
-----
1. 去除 LLM 输出中的 <think>…</think> 推理块
2. 若存在 \boxed{…}（任意位置）→ 提取括号内容
3. 否则：
      • 先抓最后出现的 **数字/分数/小数**  
      • 若无数字，则抓最后出现的 **单字母 A-E**（多选题）
4. 删除 $、空格、句号、逗号，数字去前导 0，统一小写
"""

from __future__ import annotations
import re
from typing import List

# ---------- 预编译正则 ----------
_BOXED   = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
_NUMBER  = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")
_LETTER  = re.compile(r"\b([A-E])\b", re.I)
_THINK   = re.compile(r"</think>", re.I)

# ------------------------------------------------------------------
def _extract_ans(text: str) -> str:
    """提取候选答案（不做大小写/符号处理）"""
    # 1) 去掉 <think>…>
    text = _THINK.split(text, 1)[-1]

    # 2) \boxed{...}
    m = _BOXED.search(text)
    if m:
        return m.group(1)

    # 3) 最后数字/分数/小数
    nums = _NUMBER.findall(text.replace(",", ""))
    if nums:
        return nums[-1]

    # 4) 最后出现的单字母 (A-E)
    letters = _LETTER.findall(text)
    if letters:
        return letters[-1]

    # 5) fallback: 原串
    return text.strip()

# ------------------------------------------------------------------
def _normalize(ans: str) -> str:
    ans = _extract_ans(ans)
    ans = ans.replace("$", "").replace(" ", "").replace(",", "").rstrip(".")
    # 若全为数字/分数/小数 → 去前导 0
    if re.fullmatch(r"-?\d+(?:/\d+)?(?:\.\d+)?", ans):
        ans = ans.lstrip("0") or "0"
    return ans.lower()

# ------------------------------------------------------------------
def compute_pass1(preds: List[List[str]], refs: List[str]) -> float:
    """pass@1 估计：每题取 k 生成中正确占比，再对题目平均"""
    assert len(preds) == len(refs)
    acc_sum = 0.0
    for cand_list, ref in zip(preds, refs):
        ref_norm = _normalize(ref)
        correct  = sum(_normalize(c) == ref_norm for c in cand_list)
        acc_sum += correct / len(cand_list)
    return acc_sum / len(preds)
