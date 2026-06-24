---
title: >-
  [Paper Note] Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
description: >-
  [ICLR 2026 Oral][LLM Alignment][MoE] This study systematically investigates how the sparsity of Mixture-of-Experts (MoE) language models affects memory-intensive and reasoning-intensive tasks differently: memory tasks prefer higher sparsity (more total parameters), whereas reasoning tasks reach optimality near $\text{TPP} \approx 20$. This trend remains invariant even after GRPO post-training and increased test-time compute.
tags:
  - "ICLR 2026 Oral"
  - "LLM Alignment"
  - "MoE"
  - "scaling laws"
  - "sparsity"
  - "reasoning"
  - "memory"
  - "tokens per parameter"
  - "GRPO"
  - "test-time compute"
date: 2026-05-08
content_hash: 010a758027ffa928
---

# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

**Conference**: ICLR 2026 Oral  
**arXiv**: [2508.18672](https://arxiv.org/abs/2508.18672)  
**Code**: [GitHub](https://github.com/rioyokotalab/optimal-sparsity)  
**Area**: LLM Alignment  
**Keywords**: MoE, scaling laws, sparsity, reasoning, memory, tokens per parameter, GRPO, test-time compute

## TL;DR

This study systematically investigates how the sparsity of Mixture-of-Experts (MoE) language models affects memory-intensive and reasoning-intensive tasks differently: memory tasks prefer higher sparsity (more total parameters), whereas reasoning tasks reach optimality near $\text{TPP} \approx 20$. This trend remains invariant even after GRPO post-training and increased test-time compute.

## Background & Motivation

Classic scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) establish power-law relationships between pre-training loss and model scale/data volume/compute budget, serving as a cornerstone for model planning. However, these laws have significant limitations:

**Non-universal coefficients**: Constants must be re-estimated when architectures or data pipelines change.

**MoE introduces new dimensions**: MoE models achieve high capacity with fixed FLOPs through sparse routing, becoming the standard configuration for flagship models like Gemini 2.5 Pro, DeepSeek-V3, and Qwen3. However, the scaling frontier of dense models does not cover the dimension of sparsity.

**Loss $\neq$ Performance**: Models with the same pre-training loss may perform very differently on downstream reasoning benchmarks (an observation also noted by the GLM-4.5 team).

**Unknown effects of post-training and TTC**: Can GRPO and test-time compute (TTC) compensate for suboptimal sparsity choices made during pre-training?

## Method

### Overall Architecture

This paper does not propose a new model but uses a set of **controlled variable scanning experiments** to answer two questions: whether MoE sparsity affects "memory" and "reasoning" capabilities identically, and whether pre-training sparsity can be rectified post-hoc. The approach follows three steps. First, a family of MoE models based on the Mixtral architecture is trained to systematically scan model width $d \in \{512, 1024, 2048\}$, experts per layer $E \in \{8, 16, 32, 64, 128, 256\}$, and active experts top-$k \in \{2, 4, 8, 16\}$ (fixed at 16 layers, trained on 125B tokens). Sparsity is defined as $\text{sparsity} = 1 - k/E$ (lower activation indicates higher sparsity). Second, three levels of metrics are measured for each model to decouple "good loss" from "correct answers." Third, the observed trends are explained along the axes of Active FLOPs and TPP, followed by experiments with GRPO post-training and test-time compute to test if these trends diminish.

To ensure conclusions are attributable to sparsity itself rather than hyperparameter tuning, all models share the same training recipe: AdamW optimizer, peak learning rate $4 \times 10^{-4}$, 2k-step linear warmup followed by cosine decay, and weight decay of 0.1. The data consists of a 125B token balanced mix (web 43B, math 32B, STEM 49B, code 1B). This unified recipe makes the scans of width, expert count, and top-$k$ clean controlled experiments.

> As this is a scaling-law/analytical paper where the method focuses on "scan design and attribution" rather than multi-stage pipelines or multi-module data flows, **no Mermaid architecture diagram is provided**. The three key designs below are aligned with the "measurement $\rightarrow$ attribution $\rightarrow$ robustness check" steps.

### Key Designs

**1. Three-tier measurement: Decoupling "good loss" and "correct answers"**

Classic scaling laws focus solely on pre-training loss. This study specifically queries why continued loss reduction does not necessarily yield stronger reasoning. Thus, three levels are measured: train/val loss on pre-training data, task loss on downstream benchmarks (cross-entropy on answer tokens only), and accuracy on downstream benchmarks. This decoupling allows for the quantification of the "training $\rightarrow$ testing" generalization gap and the "loss $\rightarrow$ accuracy" mapping gap. It reveals that on reasoning tasks, while train loss decreases monotonically, the task loss follows a U-shaped curve, and accuracy degrades with over-optimization; conversely, memory tasks (TriviaQA, HellaSwag) show monotonic improvement across all three tiers.

**2. Two-axis attribution: Explaining opposing effects via Active FLOPs and TPP**

This forms the analytical framework of the paper. **Active FLOPs axis** refers to the actual compute activated during training and inference (determined by top-$k$): at the same pre-training loss, models with higher active compute consistently show higher reasoning accuracy, indicating that reasoning quality is determined by active FLOPs, not just loss. The **Total Tokens Per Parameter (TPP) axis** is defined as:

$$\text{TPP} = N_{\text{tokens}} / N_{\text{params}}$$

With $N_{\text{tokens}}$ fixed at 125B, TPP varies inversely with parameter count. It characterizes whether a model is "parameter-hungry" or "data-hungry." Memory tasks are parameter-hungry (lower TPP/more parameters are better), while reasoning tasks are data-hungry, reaching optimality at $\text{TPP} \approx 20$, with performance degrading if TPP is too low or too high.

**3. GRPO + TTC Robustness Check: Verifying if pre-training choices can be fixed post-hoc**

Can suboptimal pre-training sparsity be compensated for by post-training or extra compute at test time? The models were re-evaluated using **GRPO** (following the DeepSeek-R1 algorithm) on the GSM8K training set and **TTC** (test-time compute) using zero-shot Self-Consistency decoding with $2^7 = 128$ samples. While both methods improved absolute performance, the non-monotonic "loss-accuracy" tradeoff remained unchanged, and sparser models could not close the gap with denser models. This indicates that reasoning losses caused by sparsity cannot be smoothed over by post-training, making pre-training sparsity choices critical.

### Loss & Training

The standard MoE training loss used throughout all scans is:

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{LB} + \beta \mathcal{L}_{RZ}$$

where $\mathcal{L}_{CE}$ is the main cross-entropy term, the load-balancing loss $\mathcal{L}_{LB}$ ($\alpha = 10^{-2}$) prevents expert collapse, and the router z-loss $\mathcal{L}_{RZ}$ ($\beta = 10^{-3}$) penalizes large router logits for numerical stability.

## Key Experimental Results

### Main Results

**Divergence between memory vs. reasoning tasks when increasing total parameters (Figure 1-3)**:

| Dimension | TriviaQA/HellaSwag (Memory) | GSM8K/GSM-Plus (Reasoning) |
|------|--------------------------|---------------------|
| Pre-training Loss | Monotonic decrease ✓ | Monotonic decrease ✓ |
| Task Loss | Monotonic improvement ✓ | U-shaped: Decrease then increase ✗ |
| Accuracy | Monotonic improvement ✓ | Non-monotonic: Over-optimization hurts ✗ |

**Optimal Sparsity under Iso-FLOP Analysis (Figure 5)**:

| Task Type | Low FLOPs Budget | High FLOPs Budget |
|---------|-------------|-------------|
| Memory | Higher sparsity is better | Higher sparsity is better (Consistent) |
| Reasoning | Higher sparsity is better | **Denser models overtake** (Reversal) |

**Impact of TPP on Performance (Figure 7)**:

| Task Type | TPP Trend | Optimal TPP |
|---------|---------|---------|
| TriviaQA/HellaSwag | Monotonic: Lower TPP is better | As low as possible |
| GSM8K/GSM-Plus | Non-monotonic (Inverted U) | **$\approx 20$** |

### Ablation Study

**Impact of Top-$k$ with fixed active parameters**:
- Changing top-$k$ while fixing active parameter count has a negligible impact on pre-training loss.
- However, on reasoning tasks, larger top-$k$ consistently outperforms smaller top-$k$ even when TPP is fixed.

**GRPO Post-training Effects (Figure 6 Right)**:
- All models show performance gains, but the non-monotonic relationship between pre-training loss and accuracy **remains unchanged**.
- Sparser models still underperform denser models after GRPO.

**TTC Effects (Figure 6 Left, Self-Consistency $2^7$ sampling)**:
- Performance scales with model size, but the loss-accuracy tradeoff **remains unchanged**.
- TTC cannot compensate for deficiencies in pre-training sparsity.

**Hyperparameter Control Experiments**:
- Scans of learning rates and initialization schemes show generalization gaps strikingly similar to those caused by sparsity.
- Confirms that the memory/reasoning performance gap is not unique to MoE; traditional hyperparameters can replicate it.

**Code Task Ablation (HumanEval, MBPP)**:
- Code generation exhibits trends similar to mathematical reasoning: denser models are superior under high FLOP budgets.

### Key Findings

1. Decreasing pre-training loss does not necessarily improve reasoning performance—in MoE, it can sometimes be detrimental.
2. Optimal sparsity must be determined jointly by Active FLOPs and TPP, rather than compute budget alone.
3. Neither GRPO nor TTC can eliminate reasoning performance losses caused by pre-training sparsity.
4. Memory tasks prefer sparsity (more parameters), while reasoning tasks prefer moderate density (more data relative to parameters).

## Highlights & Insights

1. **Challenging classic scaling wisdom**: Reveals the counter-intuitive conclusion that "more parameters are always better" does not hold for MoE reasoning tasks.
2. **Elegant experimental design**: Decouples confounding factors through systematic scans of top-$k$, width, and expert counts.
3. **Practical guidance**: Provides a clear decision framework for MoE pre-training—scale parameters for memory, but maintain $\text{TPP} \approx 20$ for reasoning.
4. **Post-training cannot compensate**: GRPO and TTC do not change the underlying tradeoffs, emphasizing the criticality of sparsity selection during pre-training.
5. **Completely open-source**: Checkpoints, code, and training logs are released, ensuring high reproducibility.

## Limitations

1. All models were trained on 125B tokens; larger datasets might shift the optimal sparsity (acknowledged by authors).
2. Only the Mixtral architecture was used; modern variants like shared experts or QK-norm were not validated.
3. Evaluation benchmarks are limited (GSM8K/TriviaQA/HellaSwag); harder benchmarks like MATH or ARC-C are missing.
4. Max width $d=2048$ limits extrapolation to truly large-scale model behavior.
5. Depth was fixed at 16 layers; the interaction between depth and sparsity was not fully explored.

## Related Work & Insights

Unlike the MoE parameter-FLOPs frontier analysis by Abnar et al. (2025), this study further distinguishes different optimal strategies for memory and reasoning. It empirically corroborates the theoretical analysis of Jelassi et al. (2025) (MoE improves memory more than reasoning) and aligns with the TPP analysis of Roberts et al. (2025) (memory is parameter-hungry, reasoning is data-hungry).

**Core Insight**: When planning large-scale MoE training, one cannot rely solely on perplexity curves. Downstream reasoning benchmarks must be monitored simultaneously, and sparsity strategies should be chosen based on target task types. $\text{TPP} \approx 20$ serves as a valuable "sweet spot" rule of thumb for reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Fills a critical gap in MoE reasoning performance analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive systematic scans, multiple ablations, GRPO/TTC validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, high information density in charts)
- Value: ⭐⭐⭐⭐⭐ (Direct practical utility for MoE engineering and scaling law research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging](../../ACL2025/llm_alignment/upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)
- [\[ICLR 2026\] Multi-objective Large Language Model Alignment with Hierarchical Experts](multi-objective_large_language_model_alignment_with_hierarchical_experts.md)
- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ICLR 2026\] Cognitive models can reveal interpretable value trade-offs in language models](cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)

</div>

<!-- RELATED:END -->
