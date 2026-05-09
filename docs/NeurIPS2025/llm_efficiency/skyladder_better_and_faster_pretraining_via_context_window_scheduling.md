---
title: >-
  [Paper Note] SkyLadder: Better and Faster Pretraining via Context Window Scheduling
description: >-
  [NeurIPS 2025][LLM Efficiency][Context window scheduling] SkyLadder, a progressive short-to-long context window scheduling strategy, achieves superior pretraining efficiency (22% training time saved) and improved model performance (+3.7%) under a fixed compute budget, challenging the prevailing belief that "longer context = better performance."
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - Context window scheduling
  - pretraining efficiency
  - long context
  - attention mechanism
  - training stability
date: 2026-05-08
content_hash: a7e3d7cc458d6cc8
---

# SkyLadder: Better and Faster Pretraining via Context Window Scheduling

**Conference**: NeurIPS 2025
**arXiv**: [2503.15450](https://arxiv.org/abs/2503.15450)
**Code**: [https://github.com/sail-sg/SkyLadder](https://github.com/sail-sg/SkyLadder)
**Area**: LLM Efficiency
**Keywords**: Context window scheduling, pretraining efficiency, long context, attention mechanism, training stability

## TL;DR
SkyLadder, a progressive short-to-long context window scheduling strategy, achieves superior pretraining efficiency (22% training time saved) and improved model performance (+3.7%) under a fixed compute budget, challenging the prevailing belief that "longer context = better performance."

## Background & Motivation

**Background**: The context windows of LLMs have continuously expanded (GPT 512→Llama-3 8K→128K+), yet fair comparisons under a fixed token budget remain lacking.

**Limitations of Prior Work**: Under strictly controlled conditions, short-context models *consistently* outperform long-context models on standard benchmarks—contradicting widely held assumptions in the field.

**Key Challenge**: Models must simultaneously achieve strong performance on standard tasks and long-sequence processing capability, yet strategies for balancing the two under a fixed budget have not been systematically explored.

**Goal**: How can one fully exploit the training efficiency advantage of short contexts while ensuring the model ultimately acquires long-sequence understanding capability?

**Key Insight**: Drawing an analogy to curriculum learning—progressively advancing from short context (simple) to long context (complex), allowing the model to master fundamentals before extending its capabilities.

**Core Idea**: A progressive short-to-long window scheduling strategy implemented via modified attention masks, requiring no changes to data or architecture—scheduling alone yields dual gains.

## Method

### Overall Architecture
SkyLadder divides pretraining into progressive stages: starting from a very short window (32 tokens) and linearly expanding to the target window (8K/32K), with a block-wise attention mask $M_{ij}$ controlling the effective context length at each step.

### Key Designs

1. **Linear Window Expansion Strategy**:

    - Function: Gradually increases the effective context window at a fixed rate.
    - Mechanism: $w(t) = \min(w_e, w_s + \lfloor\alpha t\rfloor)$, where $w_s=32$ (initial), $w_e$ = target window, $\alpha=1/8$ (expansion rate). Implemented via block-wise attention masks without modifying data packing.
    - Design Motivation: Linear scheduling achieves the best performance among six evaluated strategies (linear / sinusoidal / exponential / staircase / continual fine-tuning / Dataset Decomposition).

2. **Block-wise Attention Mask**:

    - Function: Restricts each position to attending only to preceding tokens within the current window.
    - Mechanism: $M_{ij} = 0$ if $\lfloor i/w \rfloor \cdot w \leq j \leq i$, else $-\infty$. Orthogonally composable with strategies such as IntraDoc Mask.
    - Design Motivation: Mask-based implementation avoids data re-packing and introduces no domain bias (vs. Dataset Decomposition, which segments data by length and thereby over-represents long-text domains such as books).

3. **Training Stability Gain**:

    - Function: The short-window phase provides more stable training dynamics.
    - Mechanism: Loss variance during short-window training is 0.023 vs. 0.041 for long windows (×1.78), with lower and more consistent gradient norms.
    - Design Motivation: Attention logits tend to explode on long sequences; progressive expansion allows the model to adapt smoothly.

### Loss & Training
Standard language modeling loss with no additional loss terms. Key hyperparameters: $w_s=8$ (larger scheduling range), $\alpha=1/8$ (balancing short and long tasks), linear scheduling. Orthogonal to learning rate scheduling.

## Key Experimental Results

### Main Results (1B model, 100B tokens)

| Method | Avg. Accuracy | ARC-E | HellaSwag | MMLU | Gain |
|------|----------|-------|----------|------|------|
| Random Baseline | 46.3% | 58.0 | 43.0 | 29.9 | - |
| **+ SkyLadder** | **50.0%** | **65.4** | **47.0** | **32.4** | **+3.7%** |
| IntraDoc Baseline | 47.4% | 61.8 | 45.6 | 30.5 | - |
| + SkyLadder | 49.3% | 64.8 | 47.9 | 31.8 | +1.9% |

### Ablation Study

| Expansion Rate $\alpha$ | Standard Tasks | Long Tasks | Training Time Saved |
|---|---|---|---|
| 1/12 (slowest) | 46.8 | 13.1 | 15% |
| **1/8 (default)** | **48.6** | **14.1** | **13.1%** |
| 1 (fastest) | 47.2 | 12.3 | 8% |

### Scaling Behavior

| Model Size | Baseline | + SkyLadder | Gain |
|---------|------|-----------|------|
| 120M | 40.1% | 41.2% | +1.1% |
| 360M | 47.2% | 49.6% | +2.4% |
| **3B** | **57.0%** | **60.5%** | **+3.5%** |

Gains increase with model size. For the 32K context setting: 22.2% training time saved and 26.3% FLOPs saved.

### Key Findings
- **Counter-intuitive**: Under a fixed token budget, short-context models consistently outperform long-context models on standard benchmarks.
- **Dual gain**: SkyLadder simultaneously improves both performance and efficiency—+3.5% accuracy and 22% time saved for the 3B model.
- **Nature of IntraDoc**: Its success stems not from denoising but from implicitly inducing a shorter effective window length distribution.
- **Training stability**: Short-window training is more stable (44% lower loss variance, 20% lower gradient norms).

## Highlights & Insights
- **Challenging conventional wisdom**: Rigorous controlled experiments demonstrate that "larger context ≠ better performance," with fundamental implications for pretraining strategy.
- **Extreme simplicity**: Only the attention mask is modified; data, architecture, and loss function remain unchanged. The method requires only three hyperparameters.
- **Training stability perspective**: This work is the first to quantify the stability advantage of short-window training (loss variance, gradient norms, attention logits), providing a mechanistic explanation for the observed performance gains.

## Limitations & Future Work
- The largest evaluated model is only 3B; validation at the 13B/70B scale is absent.
- No theoretical explanation is provided for why linear scheduling is optimal.
- Long-task evaluation relies on synthetic benchmarks (RULER); evaluation on naturally occurring long documents is insufficient.
- The relationship between hyperparameters and model size or data quality has not been systematically explored.

## Related Work & Insights
- **vs. Dataset Decomposition**: Segmenting data by length introduces domain bias (long documents ≈ books); SkyLadder leaves the data distribution unchanged.
- **vs. continual fine-tuning approaches** (YaRN, ProLong): Training from scratch vs. fine-tuning an existing model; SkyLadder avoids fine-tuning costs.
- **vs. GrowLength (Jin 2023)**: Validated only on 400M models; SkyLadder provides the first systematic validation at the 3B scale.

## Rating
- Novelty: ⭐⭐⭐⭐ The curriculum learning inspiration is not entirely new, but the empirical finding that challenges the long-context assumption carries genuine innovative value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous variable control, comprehensive ablations, cross-scale validation from 120M to 3B, and public code release.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure, high information density in figures and tables, and in-depth quantitative analysis of training stability.
- Value: ⭐⭐⭐⭐⭐ A directly deployable engineering solution with dual benefits of 22% speedup and +3.7% accuracy gain, immediately applicable to industrial-scale pretraining.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](advancing_expert_specialization_for_better_moe.md)
- [\[NeurIPS 2025\] Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[NeurIPS 2025\] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)
- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)

<!-- RELATED:END -->
