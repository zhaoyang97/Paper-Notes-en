---
title: >-
  [Paper Note] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study
description: >-
  [ACL 2026][Reinforcement Learning][Paper Note] This paper presents the first systematic study of scaling behaviors in LLM reinforcement learning (RL) post-training. Conducted on the Qwen2.5 series (0.5B-72B), the study reveals that performance follows a power-law relationship with training resources, and learning efficiency tends toward saturation as model scale in
tags:
  - ACL 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: 58e3913d0b2877c6
---
# Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study

**Conference**: ACL 2026  
**arXiv**: [2509.25300](https://arxiv.org/abs/2509.25300)  
**Code**: [GitHub](https://github.com/reasoning360/rl-scaling)  
**Area**: Reinforcement Learning / Scaling Laws  
**Keywords**: Reinforcement Learning Post-Training, Scaling Laws, Mathematical Reasoning, Learning Efficiency, Data Reuse

## TL;DR

This paper presents the first systematic study of scaling behaviors in LLM reinforcement learning (RL) post-training. Conducted on the Qwen2.5 series (0.5B-72B), the study reveals that performance follows a power-law relationship with training resources, and learning efficiency tends toward saturation as model scale increases.

## Background & Motivation

**Background**: Scaling laws during the pre-training phase have been extensively studied—Kaplan et al. and Chinchilla established power-law relationships between loss and model size, data volume, and compute. However, while RL post-training (e.g., GRPO, RLHF) has become the dominant paradigm for enhancing LLM reasoning capabilities, its scaling behavior has remained almost entirely unexplored systematically.

**Limitations of Prior Work**: Practitioners lack guidance when performing RL post-training: What model size should be chosen? How many computational resources should be allocated? Should data be reused when volume is insufficient? These critical questions lack quantitative answers, leading to significant trial-and-error and resource waste.

**Key Challenge**: Do the scaling laws from the pre-training phase apply to RL post-training? RL post-training possesses unique characteristics—utilizing verifiable rewards instead of cross-entropy loss and employing on-policy sampling rather than i.i.d. data—differences that may lead to distinct scaling behaviors.

**Goal**: To systematically characterize the relationship between model scale, data volume, compute, and performance through large-scale experiments, establishing predictable scaling formulas for RL post-training.

**Core Idea**: Test loss in RL post-training follows a log-linear (power-law) relationship with resource consumption: $\log L(N,X) = -k(N) \cdot \log X + E(N)$, where the learning efficiency $k(N)$ improves with model size but approaches a saturation limit $K_{\max}$.

## Method

### Overall Architecture

This paper is a systematic empirical study. By training 63 LLMs (covering the full Qwen2.5 series from 0.5B to 72B, including both base and instruct variants) using the GRPO algorithm on mathematical reasoning tasks, the study measures performance under three resource-constrained scenarios to fit scaling formulas.

### Key Designs

**1. Power-Law Scaling Formula: Quantitative performance-resource relationships**

Practitioners require quantitative answers regarding resource investment versus performance gain. This study finds that RL post-training test loss and resource consumption exhibit a linear relationship in log-space: $\log L(N,X) = -k(N) \cdot \log X + E(N)$. Here, $X$ can represent compute $C$ or data volume $D$, and $N$ represents model parameters. The critical component is the slope $k(N)$, which measures learning efficiency. Rather than treating it as a constant, it is modeled using a saturation function $k(N) = \frac{K_{\max}}{1+N_0/N}$. This captures the "diminishing returns of scale" unique to RL post-training, implying that infinitely increasing parameters is not an optimal strategy.

**2. Inter-model and Intra-model Prediction Protocols**

The value of a scaling law lies in its predictive capability. Two protocols verify the formula's extrapolation: inter-model prediction uses parameters fitted from 0.5B–32B models to predict the 72B model's performance; intra-model prediction uses early training steps to predict the remainder of the training trajectory. Both protocols achieved a goodness-of-fit of $R^2 > 0.99$, indicating reliable extrapolation across both scales and training stages.

**3. Data Reuse Analysis**

Given the scarcity of high-quality reasoning data, the study quantifies the cost of reuse. By fixing the total training volume $D_{\mathrm{total}}$ and varying the reuse factor $\tau$ (where $D_{\mathrm{unique}} \times \tau = D_{\mathrm{total}}$), the study observes performance changes as unique samples decrease and repetitions increase. Results show that within the range of $\tau \leq 25$, performance is nearly insensitive to the reuse factor—total training volume, rather than sample uniqueness, determines performance until significant overfitting occurs at $\tau=100$.

### Loss & Training

The standard GRPO algorithm is used with a binary reward signal (Correct=1, Incorrect=0). The primary metric is test loss defined as $L = 1 - R/R_{\max}$. Training data utilizes the mathematical subset of guru-RL-92k (~50k problems), sorted by increasing difficulty for curriculum learning. Each configuration is repeated 3 times to ensure robustness.

## Key Experimental Results

### Main Results

| Model Scale | Compute Efficiency $k_C(N)$ | Data Efficiency $k_D(N)$ | Note |
|-------------|----------------------------|-------------------------|------|
| 0.5B-3B     | Rapid Growth               | Rapid Growth            | Significant benefits of scale in small models |
| 7B-32B      | Slowing Growth             | Slowing Growth          | Efficiency gains begin to saturate |
| 72B         | Approaching $K_{\max}$      | Approaching $K_{\max}$   | Saturation trend becomes obvious |

### Prediction Accuracy

| Prediction Type | $R^2$ | Note |
|-----------------|-------|------|
| Inter-model (0.5B-32B → 72B) | >0.99 | Accurately predicts large model performance |
| Intra-model (Early → Late) | >0.99 | Accurately predicts training trajectories |
| Cross-architecture (Llama 3) | >0.99 | Formula exhibits architectural independence |

### Key Findings
- **Larger models are consistently superior in compute and data efficiency**, but marginal returns diminish, with gains significantly reducing beyond 32B.
- **Data reuse is highly effective**: No significant performance degradation occurs for $\tau \leq 25$; overfitting only emerges at $\tau=100$.
- **Limited domain transfer**: RL post-training generalizes well within the math domain but offers almost no benefit for OOD tasks like code or logic, potentially even harming certain capabilities.
- **Crossover between 32B and 72B**: Under identical compute budgets, 32B models may outperform 72B models early on due to more training steps, revealing a trade-off between scale and steps.

## Highlights & Insights
- **The "Chinchilla" of RL Post-Training**: Establishes the first RL post-training scaling formula analogous to pre-training, filling a critical theoretical gap.
- **Discovery of Saturation Trends**: The saturation of learning efficiency implies that infinite scaling is not the optimal RL post-training strategy, providing an upper bound for resource allocation.
- **Practical Value of Data Reuse**: Validates that moderate data reuse is nearly lossless, providing direct guidance for data-scarce scenarios.
- **Cross-architecture Validation**: Validation on the Llama 3 series enhances the universality of the conclusions.
- **Warning on Domain Specificity**: The high specialization of RL post-training (and potential harm to other skills) serves as an important practical reminder.

## Limitations & Future Work
- Experiments cover only the mathematical reasoning domain; scaling behaviors for multi-domain RL remain unknown.
- The largest model is 72B; saturation trends beyond the hundred-billion parameter range cannot be empirically verified.
- Results are based only on the GRPO algorithm; whether other RL algorithms (e.g., DAPO, PPO) exhibit different behaviors requires exploration.
- Only dense models were studied; MoE architectures were not covered.
- Absolute coefficients of the scaling formula depend on the evaluation dataset and task difficulty, limiting universal interpretability.

## Related Work & Insights
- **vs Kaplan et al. (2020)**: While classic laws focus on cross-entropy loss, this work extends scaling to reward optimization in RL post-training.
- **vs Chinchilla (Hoffmann et al., 2022)**: Similar to Chinchilla's compute-optimal ratios, this provides resource allocation guidelines for RL.
- **vs Hilton et al. (2023)**: While prior work found power-law relationships in CNN+RL, this work validates similar patterns in the LLM+GRPO context.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of RL post-training scaling behavior.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments with 63 models across multiple architectures and scenarios, with 3 repetitions per config.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous derivation, and rich visualization.
- Value: ⭐⭐⭐⭐⭐ Provides highly practical quantitative guidance for resource allocation in RL post-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](../../ICML2026/reinforcement_learning/how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ACL 2026\] Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)

</div>

<!-- RELATED:END -->
