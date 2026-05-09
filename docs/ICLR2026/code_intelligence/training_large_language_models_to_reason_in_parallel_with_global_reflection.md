---
title: >-
  [Paper Note] Training Large Language Models to Reason in Parallel with Global Forking Tokens
description: >-
  [ICLR2026][parallel reasoning] This paper proposes Set Supervised Fine-Tuning (SSFT), which introduces global forking tokens and a set-based loss via bipartite matching to train LLMs to produce diverse and correct reasoning patterns triggered by a single control token, outperforming standard SFT+GRPO on both Pass@1 and Cons@k.
tags:
  - ICLR2026
  - parallel reasoning
  - global forking token
  - set supervised fine-tuning
  - bipartite matching
  - diverse reasoning
date: 2026-05-08
content_hash: c8854566b467439a
---

# Training Large Language Models to Reason in Parallel with Global Forking Tokens

**Conference**: ICLR2026
**arXiv**: [2510.05132](https://arxiv.org/abs/2510.05132)
**Code**: [GitHub](https://github.com/Sheng-J/SSFT)
**Area**: Code Intelligence
**Keywords**: parallel reasoning, global forking token, set supervised fine-tuning, bipartite matching, diverse reasoning

## TL;DR

This paper proposes Set Supervised Fine-Tuning (SSFT), which introduces global forking tokens and a set-based loss via bipartite matching to train LLMs to produce diverse and correct reasoning patterns triggered by a single control token, outperforming standard SFT+GRPO on both Pass@1 and Cons@k.

## Background & Motivation

- Test-time parallel sampling (e.g., self-consistency, Best-of-N) is an effective approach for improving LLM reasoning, but relies on generating diverse and correct reasoning paths.
- For challenging problems, the "forking tokens" that trigger different reasoning modes are typically buried deep in the sampling tree; increasing temperature improves diversity but degrades accuracy.
- Theoretical studies show that simply raising temperature cannot guarantee diversity unless the model is explicitly trained for coverage.
- Standard SFT on multiple reasoning trajectories leads to mode collapse — different control tokens produce nearly identical reasoning.

## Core Problem

How to train an LLM such that simple control tokens, placed at the beginning of inference, can trigger diverse and correct reasoning modes?

## Method

### Global Forking Tokens

- A set of $N$ special tokens $\{$`<think i>`$\}_{i=1}^N$ is reserved (with $N=6$ in experiments).
- Objective: given a question $\mathbf{x}$, different `<think i>` tokens initiate $M$ distinct reasoning trajectories ($M=4$).

### SSFT Loss Function

Parallel reasoning is formulated as a **set prediction problem**, solved via bipartite matching:

**Step 1: Compute the cost matrix**

For each pair of (forking token $g^{(i)}$, reasoning trajectory $\mathbf{r}^{(j)}$), compute the NTP loss:
$$\mathcal{L}_{\text{matching}}(g^{(i)}, \mathbf{r}^{(j)}) = -\text{sg}\left(\frac{1}{T_\mathbf{r}} \sum_{t=1}^{T_\mathbf{r}} \log \pi_\theta(r_t^{(j)}|\mathbf{x}, g^{(i)}, \mathbf{r}_{<t}^{(j)})\right)$$

where $\text{sg}(\cdot)$ denotes the stop-gradient operation to reduce memory usage.

**Step 2: Solve for the optimal matching via the Hungarian algorithm**

$$\hat{\boldsymbol{\sigma}} = \arg\min_{\boldsymbol{\sigma} \in \mathfrak{S}_P} \sum_{j=1}^M \mathcal{L}_{\text{matching}}(g^{(\boldsymbol{\sigma}(j))}, \mathbf{r}^{(j)})$$

**Step 3: Train under the optimal matching**

$$\mathcal{L}_{\text{Hungarian}}(\theta) = -\mathbb{E}_{\mathbf{x}, \mathbf{R} \sim \mathcal{D}}\left[\sum_{j=1}^M \sum_t \log \pi_\theta(r_t^{(j)}|\mathbf{x}, g^{(\hat{\sigma}(j))}, \mathbf{r}_{<t}^{(j)})\right]$$

### Computational Efficiency

- Only the first $L < T_\mathbf{r}$ tokens are used to compute the matching cost (with $L \approx 1000$ in experiments).
- Time complexity is reduced by a factor of $k = T_\mathbf{r}/L$; backpropagation is performed only over the $M$ matched sequences.

### GFPO: Global Forking Policy Optimization

- A small number of RL steps are applied after SSFT, optimizing only the output distribution of the global forking tokens $g^{(i)}$.
- Implementation is minimal: only a few slicing lines need to be added to existing GRPO code.

### Inference Strategy

- **Cons@k**: Each `<think i>` token is used to prompt the model separately, followed by majority voting.
- **Pass@1**: A graph-based heuristic selects the $g^{(i^\star)}$ that covers the most reasoning trajectories, or GFPO is used to let the model sample the optimal $g^{(i)}$ autonomously.

## Key Experimental Results

### Pass@1 Comparison (Qwen2.5-32B-Instruct backbone)

| Method | AIME24 | AIME25 | MATH-500 | GPQA-D | LCB(v5) |
|--------|--------|--------|----------|--------|---------|
| Qwen2.5-32B-Instruct | 15.80 | 10.40 | 80.40 | 47.00 | 23.35 |
| s1.1-32B (single-traj.) | 54.79 | 44.27 | 92.16 | 62.12 | – |
| SFT-mixed-32B (multi-traj.) | 58.23 | 51.96 | 88.49 | 59.96 | 32.34 |
| **SSFT-32B** | **64.06** | **58.13** | **90.02** | 60.39 | **38.92** |
| SSFT-32B-GFPO | 64.22 | **58.80** | 89.90 | **62.48** | **42.10** |

### Cons@6 and Cons@32 (Parallel Reasoning)

| Method | AIME24 Cons@6 | AIME25 Cons@6 | AIME25 Cons@32 |
|--------|--------------|--------------|----------------|
| s1.1-32B | 70.30 | 53.33 | 63.33 |
| SFT-mixed-32B | 73.94 | 70.00 | 76.67 |
| **SSFT-32B** | **75.45** | **73.94** | **86.67** |
| SSFT-32B-GFPO | **76.67** | **78.48** | 83.33 |

- SSFT outperforms SFT-mixed on Pass@1 by 8.33% on AIME24 and 6.57% on AIME25.
- Cons@32 reaches 86.67% on AIME25.
- Substantial improvements are also observed on the OOD code generation benchmark LCB.

### Ablation: Optimal Matching vs. Random Matching

| Matching | AIME24 Pass@1 | AIME25 Cons@6 |
|----------|--------------|--------------|
| Random matching | 61.77 | 67.58 |
| **Optimal matching** | **64.06** | **73.94** |

## Highlights & Insights

1. **First application of DETR-style set loss to language modeling**: The bipartite matching idea from object detection is elegantly transferred to the problem of reasoning diversity.
2. **Emergent behavior of global forking tokens**: Different `<think i>` tokens automatically learn distinct reasoning lengths and strategies without manual specification.
3. **Elegant resolution of mode collapse**: Standard SFT causes all control tokens to collapse to the same mode, whereas SSFT preserves diversity.
4. **Training data efficiency**: Significant gains on the 32B model are achieved using only 1K questions, each with 4 reasoning trajectories.
5. **OOD generalization to code generation**: The method transfers effectively to code tasks.

## Limitations & Future Work

- Thorough validation is limited to the Qwen2.5 series; performance gains on other architectures such as Llama are smaller.
- The matching cost computation introduces additional overhead, though the paper reports its impact to be minimal.
- The diversity of reasoning trajectories depends on the quality of the teacher model.
- The behavior under larger values of $N$ and $M$ remains unexplored.
- The GFPO stage uses only a limited number of RL steps; more extensive RL training may yield further gains.

## Related Work & Insights

| Method | Mechanism | Diversity Guarantee | Extra Annotation | Pass@1 |
|--------|-----------|---------------------|-----------------|--------|
| Self-consistency | Temperature sampling + voting | None | No | Low |
| Multiverse | Parallel CoT rewriting | Implicit | Teacher model | Medium |
| Standard SFT (multi-traj.) | Repeated training | Mode collapse | No | Medium |
| **SSFT** | **Set matching loss** | **Explicit** | **Teacher trajectories** | **High** |

The bipartite matching + language modeling paradigm has the potential to generalize to other tasks requiring diverse outputs (e.g., translation, multi-solution code generation). Global forking tokens can be viewed as learnable prompts or mode selectors, and bear a conceptual resemblance to mixture-of-experts, but operating at the level of reasoning paths.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing DETR-style set loss into language modeling; the global forking token concept is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple benchmarks and sufficient ablations, but limited model diversity.
- Writing Quality: ⭐⭐⭐⭐ — Clear formalization and intuitive figures.
- Value: ⭐⭐⭐⭐ — Provides a novel training paradigm for parallel test-time computation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_instruction_fol.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)
- [\[ICLR 2026\] DiaBlo: Diagonal Blocks Are Sufficient For Finetuning](diablo_diagonal_blocks_are_sufficient_for_finetuning.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](../../AAAI2026/code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)

<!-- RELATED:END -->
