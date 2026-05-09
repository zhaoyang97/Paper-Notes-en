---
title: >-
  [Paper Note] Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
description: >-
  [ICLR 2026][LLM Alignment][MoE] This paper systematically investigates how sparsity in MoE language models differentially affects memorization and reasoning tasks: memorization tasks favor higher sparsity (more parameters), while reasoning tasks peak near TPP≈20, and this trend remains consistent after GRPO post-training and increased test-time compute.
tags:
  - ICLR 2026
  - LLM Alignment
  - MoE
  - scaling laws
  - sparsity
  - reasoning
  - memorization
  - tokens per parameter
  - GRPO
  - test-time compute
date: 2026-05-08
content_hash: f5bc3048954d0624
---

# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

**Conference**: ICLR 2026
**arXiv**: [2508.18672](https://arxiv.org/abs/2508.18672)
**Code**: [GitHub](https://github.com/rioyokotalab/optimal-sparsity)
**Area**: LLM Alignment
**Keywords**: MoE, scaling laws, sparsity, reasoning, memorization, tokens per parameter, GRPO, test-time compute

## TL;DR

This paper systematically investigates how sparsity in MoE language models differentially affects memorization and reasoning tasks: memorization tasks favor higher sparsity (more parameters), while reasoning tasks peak near TPP≈20, and this trend remains consistent after GRPO post-training and increased test-time compute.

## Background & Motivation

Classical scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) establish power-law relationships between pretraining loss and model size, data volume, and compute budget, forming the cornerstone of model planning. However, these laws have important limitations:

**Non-universal coefficients**: Re-estimation is required whenever the architecture or data pipeline changes.

**MoE introduces a new dimension**: MoE models achieve high capacity at fixed FLOPs through sparse routing, becoming the standard configuration for flagship models such as Gemini 2.5 Pro, DeepSeek-V3, and Qwen3. Yet the scaling frontier of dense models cannot cover the sparsity dimension.

**Loss ≠ performance**: Models with identical pretraining loss can differ substantially on downstream reasoning benchmarks (an observation also noted by the GLM-4.5 Team).

**Unknown effects of post-training and TTC**: It remains unclear whether GRPO and test-time compute can compensate for suboptimal sparsity choices made during pretraining.

## Method

### Overall Architecture

A family of Mixtral-architecture MoE models is trained with controlled variable sweeps:
- Model width $d \in \{512, 1024, 2048\}$
- Number of experts per layer $E \in \{8, 16, 32, 64, 128, 256\}$
- Top-k experts $k \in \{2, 4, 8, 16\}$
- Fixed 16 layers; all models trained on 125B tokens

Sparsity is defined as: $\text{sparsity} = 1 - \frac{\text{Top-}k}{\text{Experts}}$

### Key Designs

#### 1. Decoupling Pretraining Loss from Downstream Performance

Three levels of measurement are taken for each model:
- Train/val loss on pretraining data
- Task loss on downstream benchmarks (cross-entropy computed only on answer tokens)
- Accuracy on downstream benchmarks

This enables separate quantification of the generalization gap from training to test distribution, and the mapping gap from loss to accuracy.

#### 2. Two Primary Axes of Key Findings

**Active FLOPs axis**: At equal pretraining loss, models with more active compute (larger top-k) perform better on reasoning tasks. This indicates that reasoning quality is determined not only by loss but also by the active FLOPs available during training and inference.

**Total Tokens Per Parameter (TPP) axis**:
- Memorization tasks (TriviaQA, HellaSwag): parameter-hungry; lower TPP (more parameters) is consistently better.
- Reasoning tasks (GSM8K, GSM-Plus): data-hungry; performance peaks near TPP≈20, degrading at both extremes.

#### 3. MoE Training Details

- Optimizer: AdamW, peak learning rate $4 \times 10^{-4}$, 2k-step linear warmup followed by cosine decay
- Auxiliary losses: load-balancing loss ($\alpha = 10^{-2}$) + router z-loss ($\beta = 10^{-3}$)
- Total loss: $\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{LB} + \beta \mathcal{L}_{RZ}$
- Data: 125B token balanced mixture (web 43B, math 32B, STEM 49B, code 1B)

#### 4. Post-Training and TTC Experiments

- GRPO: Fine-tuned on the GSM8K training set using DeepSeek-R1's GRPO algorithm.
- TTC: Zero-shot Self-Consistency decoding, generating $2^7 = 128$ candidate answers per question and taking majority vote.

### Loss & Training

Standard MoE training loss:

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{LB} + \beta \mathcal{L}_{RZ}$$

where $\mathcal{L}_{LB}$ prevents expert collapse and $\mathcal{L}_{RZ}$ penalizes excessively large router logits to maintain numerical stability.

## Key Experimental Results

### Main Results

**Divergence between memorization and reasoning tasks as total parameters increase (Figures 1–3)**:

| Dimension | TriviaQA/HellaSwag (Memorization) | GSM8K/GSM-Plus (Reasoning) |
|-----------|----------------------------------|---------------------------|
| Pretraining loss | Monotonically decreases with total parameters ✓ | Monotonically decreases with total parameters ✓ |
| Task loss | Monotonically improves as pretraining loss decreases ✓ | U-shaped curve: decreases then increases ✗ |
| Accuracy | Monotonically improves as pretraining loss decreases ✓ | Non-monotonic: over-optimization is harmful ✗ |

**Optimal sparsity under iso-FLOP analysis (Figure 5)**:

| Task Type | Low FLOPs Budget | High FLOPs Budget |
|-----------|-----------------|------------------|
| Memorization | Higher sparsity is better | Higher sparsity is better (consistent) |
| Reasoning | Higher sparsity is better | **Denser models overtake** (reversal) |

**Effect of TPP on performance (Figure 7)**:

| Task Type | TPP Trend | Optimal TPP |
|-----------|----------|-------------|
| TriviaQA/HellaSwag | Monotonic: lower TPP is always better | As low as possible |
| GSM8K/GSM-Plus | Non-monotonic inverted U-shape | **≈20** |

### Ablation Study

**Effect of top-k at fixed active parameters**:
- Varying top-k at fixed active parameter count has negligible impact on pretraining loss.
- However, on reasoning tasks, larger top-k consistently outperforms smaller top-k, even at fixed TPP.

**GRPO post-training effect (Figure 6, right)**:
- All models improve in performance, but the non-monotonic relationship between pretraining loss and accuracy **remains unchanged**.
- Sparser models continue to underperform denser models after GRPO.

**TTC effect (Figure 6, left; Self-Consistency with $2^7$ samples)**:
- Performance scales with model size, but the loss–accuracy trade-off **remains unchanged**.
- TTC cannot compensate for deficiencies introduced by suboptimal pretraining sparsity.

**Hyperparameter control experiments**:
- Sweeping learning rate and initialization schemes produces effects strikingly consistent with the generalization gap induced by sparsity.
- This confirms that the memorization/reasoning performance gap is not caused solely by sparsity; conventional hyperparameters can reproduce the same patterns.

**Code task ablation (HumanEval, MBPP)**:
- Code generation exhibits trends similar to mathematical reasoning: denser models are superior under high FLOPs budgets.

### Key Findings

1. A decrease in pretraining loss does not necessarily lead to improved reasoning performance — in MoE models it may in fact be harmful.
2. Optimal sparsity must be jointly determined by Active FLOPs and TPP, not by compute budget alone.
3. Neither GRPO nor TTC can eliminate reasoning performance losses caused by pretraining sparsity choices.
4. Memorization favors sparsity (more parameters); reasoning favors moderate density (more data per parameter).

## Highlights & Insights

1. **Challenges classical scaling wisdom**: Reveals the counterintuitive finding that "more parameters are always better" does not hold for MoE reasoning tasks.
2. **Elegant experimental design**: Systematic sweeps over top-k, width, and expert count cleanly disentangle multiple confounding factors.
3. **Practical guidance**: Provides a clear decision framework for pretraining MoE models — stack parameters for memorization tasks; target TPP≈20 for reasoning tasks.
4. **Post-training cannot compensate**: Neither GRPO nor TTC alters the underlying trade-off, underscoring the critical importance of sparsity selection during pretraining.
5. **Fully open-sourced**: Checkpoints, code, and training logs are all publicly released, enabling high reproducibility.

## Limitations & Future Work

1. All models are trained on only 125B tokens; larger datasets may shift the optimal sparsity (acknowledged by the authors).
2. Only the Mixtral architecture is used; modern variants such as shared experts and QK-norm are not evaluated.
3. The evaluation benchmarks are limited (GSM8K / TriviaQA / HellaSwag); harder reasoning benchmarks such as MATH and ARC-C are absent.
4. The maximum width of d=2048 limits extrapolation to truly large-scale model behavior.
5. Depth is fixed at 16 layers; the interaction between depth and sparsity is not thoroughly explored.

## Related Work & Insights

Unlike Abnar et al. (2025), who analyze the parameter–FLOPs frontier of MoE models, this paper further distinguishes optimal strategies for memorization versus reasoning tasks. The findings are empirically consistent with Jelassi et al. (2025), whose theoretical analysis shows that adding experts in MoE improves memorization more than reasoning. The results are also closely aligned with Roberts et al. (2025), who characterize TPP and find that memorization is parameter-hungry while reasoning is data-hungry.

**Core insight**: When planning large-scale MoE training, it is insufficient to monitor only the perplexity curve. Downstream reasoning benchmarks must be tracked simultaneously, and the sparsity strategy should be chosen based on the target task type (memorization vs. reasoning). TPP≈20 represents a "sweet spot" for reasoning tasks and is worth adopting as an engineering rule of thumb.

## Rating

- Novelty: ⭐⭐⭐⭐ (Fills an important gap by analyzing sparsity through the lens of MoE reasoning performance)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive systematic sweeps, multiple ablations, GRPO+TTC validation, and extension to code tasks)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, information-dense figures, honest discussion)
- Value: ⭐⭐⭐⭐⭐ (Directly actionable for MoE model engineering and scaling law research)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)

<!-- RELATED:END -->
