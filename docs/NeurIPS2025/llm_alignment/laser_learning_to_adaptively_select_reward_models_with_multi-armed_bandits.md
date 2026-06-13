---
title: >-
  [Paper Note] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits
description: >-
  [NeurIPS 2025][LLM Alignment][reward model selection] This work frames the selection of multiple reward models (RMs) as a contextual multi-armed bandit (LinUCB) problem…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "reward model selection"
  - "multi-armed bandits"
  - "iterative training"
  - "DPO"
  - "multi-RM alignment"
date: 2026-05-08
content_hash: 15d62d9ada7968d0
---

# LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2410.01735](https://arxiv.org/abs/2410.01735)  
**Code**: [https://github.com/duykhuongnguyen/LASeR-MAB](https://github.com/duykhuongnguyen/LASeR-MAB)  
**Area**: Alignment / RLHF
**Keywords**: reward model selection, multi-armed bandits, iterative training, DPO, multi-RM alignment

## TL;DR

This work frames the selection of multiple reward models (RMs) as a contextual multi-armed bandit (LinUCB) problem, adaptively choosing the most suitable RM for each training batch during iterative LLM training. LASeR comprehensively outperforms RM ensemble and single-RM baselines on reasoning, instruction-following, and long-context tasks, while achieving a 2–3× efficiency advantage.

## Background & Motivation

**Background**: RLHF/DPO iterative training pipelines rely on RM scoring of LLM outputs to construct preference data. RewardBench provides a large pool of candidate RMs, yet the generalization of individual RMs across tasks and domains remains unknown.

**Limitations of Prior Work**:
   - **Single-RM risk**: A single RM may fail to generalize across all task domains, and prolonged use tends to induce reward hacking.
   - **Multi-RM ensemble inefficiency**: Simultaneously loading and running inference over multiple large models multiplies computational costs; moreover, preference conflicts among RMs (Qwen vs. OLMo agreement is only 0.43) introduce noise when naively aggregated.
   - **Manual selection is infeasible**: The space of RM combinations grows exponentially, making exhaustive search impractical.

**Key Challenge**: Diversity across multiple RMs is needed to avoid the limitations of any single RM, yet the computational overhead and signal conflicts of multi-RM usage degrade training quality.

**Goal**: Design an efficient mechanism that automatically selects the most appropriate RM for each batch throughout training.

**Key Insight**: RM selection is analogous to the multi-armed bandit (MAB) problem—each RM is an "arm," and the LinUCB algorithm balances exploration (trying different RMs) with exploitation (using the known best RM).

**Core Idea**: Select only one RM per training step to save computation, while using the MAB algorithm to ensure adaptive and globally optimal selection.

## Method

### Overall Architecture

The iterative training pipeline proceeds as follows each round: (1) generate multiple responses for a batch of prompts using the LLM → (2) LinUCB selects one RM based on prompt embeddings → (3) the selected RM scores responses and constructs preference pairs → (4) train the LLM with DPO (+NLL) loss → (5) use the negative training loss as the MAB reward to update LinUCB parameters.

### Key Designs

1. **LinUCB Contextual Bandit for RM Selection**

    - **Function**: At each training batch, select one RM based on the embedding representation of the prompts.
    - **Mechanism**: $j = \arg\max_k (c(t)^\top \hat{\theta}_k + \alpha \sqrt{c(t)^\top A_k^{-1} c(t)})$, where $c(t)$ is the mean last-token embedding of prompts in the batch, $\hat{\theta}_k$ is the learned weight vector for each RM, and $\alpha$ controls the degree of exploration. $A_k$ and $b_k$ are updated after each step using the MAB reward.
    - **Design Motivation**: LinUCB leverages contextual information to make RM selection domain-adaptive—math problems may favor one RM while creative writing favors another. The UCB exploration term prevents premature commitment to a single RM.

2. **MAB Reward Design: Negative Training Loss**

    - **Function**: Use the negative DPO training loss after each update as the reward signal for the selected RM.
    - **Mechanism**: $-\hat{\mathcal{L}}^m(t)$. A lower DPO loss indicates that the model more clearly learned to distinguish preferred from non-preferred responses under the selected RM, implying that RM provided a more informative preference signal.
    - **Design Motivation**: No additional evaluation data or human annotation is required to assess RM quality—the training signal itself serves as direct feedback.

3. **Single RM Per Step**

    - **Function**: Only one RM is loaded and used per mini-batch, rather than multiple RMs simultaneously.
    - **Design Motivation**: This directly eliminates the computational overhead and signal conflicts of multi-RM usage. Experiments confirm this is both faster and more effective than ensembling all RM scores.

4. **Best-of-N Inference Variant**

    - **Function**: For settings where fine-tuning is unsuitable (e.g., long-context tasks), the MAB-learned RM selection policy is applied at inference time during best-of-N sampling.
    - **Design Motivation**: Extends LASeR beyond the training phase to broader applicable scenarios.

### Loss & Training

- Reasoning tasks: $\mathcal{L} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{NLL}}$ (DPO with NLL regularization on chosen responses)
- Instruction-following tasks: $\mathcal{L} = \mathcal{L}_{\text{DPO}}$
- 30 responses sampled per prompt, forming 10 preference pairs
- LoRA fine-tuning with temperature 0.8

## Key Experimental Results

### Main Results (Reasoning Tasks, Llama-3-8B)

| Method | StrategyQA | GSM8K | MMLU | Average |
|--------|-----------|-------|------|---------|
| SFT baseline | 80.41 | 69.43 | 65.66 | 71.83 |
| Best RM | 84.29 | 73.16 | 67.15 | 74.87 |
| Random RM | 84.37 | 71.99 | 67.85 | 74.74 |
| RM Score Ensemble | 82.96 | 70.94 | 67.04 | 73.65 |
| RM Agreement Ensemble | 84.03 | 73.85 | 68.35 | 75.41 |
| **LASeR** | **85.96** | **74.75** | **68.24** | **76.32** |

### Ablation Study / Efficiency Analysis

| Method | Avg. Acc | Training Time (relative to LASeR) |
|--------|---------|----------------------------------|
| LASeR | 76.32 | **1×** |
| Sequential RM | 74.95 | 3× |
| RM Score Ensemble | 73.65 | 2× |
| RM Online Ensemble | 74.05 | 2× |

### Key Findings

- **LASeR leads across all three domains**: reasoning (+2.67% vs. ensemble), instruction-following (72.69% win rate vs. ensemble), and long-context (+2.96 F1 vs. ensemble).
- **Multi-RM ensemble fails due to signal conflicts**: Qwen vs. OLMo preference agreement on MMLU is only 0.43; naive ensembling is dragged down by conflicting signals.
- **2–3× training efficiency gain**: Using only one RM per step avoids the GPU memory and compute overhead of simultaneously loading multiple RMs.
- **The exploration–exploitation balance of MAB is critical**: Both Random (pure exploration) and Best RM (pure exploitation) underperform LASeR.
- **LASeR is robust to noisy RMs**: When the RM pool includes low-quality RMs, the MAB naturally learns to reduce their selection frequency.

## Highlights & Insights

- **The MAB framework offers an elegant middle ground between selection and ensembling**: Rather than "use all RMs" or "use only one RM," LASeR adopts "always select the most suitable one." This paradigm transfers naturally to any setting requiring selection among multiple judges or reward signals (e.g., multi-evaluator scoring, multi-objective optimization).
- **Using negative training loss as the MAB reward requires no additional evaluation**: This avoids the chicken-and-egg problem of needing a validation set to assess RM quality, instead forming a self-contained feedback loop from the training process itself.
- **Explains why RM ensembling sometimes underperforms a single RM**: The quantitative analysis of RM preference conflicts (agreement F1) provides the first systematic explanation of this phenomenon.

## Limitations & Future Work

- Only four 7B-scale RMs are used; behavior with larger or more numerous RMs remains unverified.
- LinUCB assumes the MAB reward is a linear function of contextual features, which may be overly simplistic for complex tasks.
- RM selection operates at the batch level rather than the instance level (equivalent only when batch size = 1, while experiments use batch size > 1).
- Integration with recent RL methods (e.g., GRPO, DisCO) has not been validated.
- Long-context tasks are evaluated only via best-of-N inference; end-to-end long-context fine-tuning is not explored.

## Related Work & Insights

- **vs. RM Score Ensemble**: Averaging ensemble scores is easily dragged down by low-quality RMs; LASeR avoids conflicts by selecting a single RM.
- **vs. WARM (Weight-Averaged RM)**: WARM averages in the RM weight space, while LASeR performs adaptive selection in the decision space—the two approaches are conceptually distinct.
- **vs. DisCO**: DisCO improves upon GRPO from an optimization objective perspective, while LASeR improves training from a reward signal quality perspective; the two are orthogonal and potentially complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using MAB for RM selection is simple yet effective; the first work to apply bandits to multi-RM alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers reasoning, instruction-following, and long-context domains with 8+ baselines and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear and experimental analysis is well-grounded.
- Value: ⭐⭐⭐⭐ Directly practical for multi-RM deployment scenarios with clear computational efficiency advantages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[NeurIPS 2025\] Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models](generalizing_while_preserving_monotonicity_in_comparison-based_preference_learni.md)

</div>

<!-- RELATED:END -->
