# ICRL – Information-Guided Chain-of-Thought Reinforcement Loop

This repo contains our reference implementation of **ICRL** (Entropy-Minimisation variant) together with evaluation scripts and YAML-driven experiment management.

---

## 1  Quick start

```bash
# create / activate the conda env
$ conda activate ICRL           

# run ONE model on ONE dataset
$ bash scripts/run_icrl.sh Qwen2.5-Math-7B AIME-TTT

