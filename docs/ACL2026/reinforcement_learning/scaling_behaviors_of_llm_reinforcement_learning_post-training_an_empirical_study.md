---
title: >-
  [Paper Note] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study
description: >-
  [ACL 2026][Reinforcement Learning][RL post-training] This work presents the first systematic study of scaling behaviors in LLM reinforcement learning post-training…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "RL post-training"
  - "scaling laws"
  - "mathematical reasoning"
  - "learning efficiency"
  - "data reuse"
date: 2026-05-08
content_hash: baf08de3b58a2e16
---

# Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study

**Conference**: ACL 2026
**arXiv**: [2509.25300](https://arxiv.org/abs/2509.25300)  
**Code**: [GitHub](https://github.com/reasoning360/rl-scaling)  
**Area**: Reinforcement Learning / Scaling Laws
**Keywords**: RL post-training, scaling laws, mathematical reasoning, learning efficiency, data reuse

## TL;DR

This work presents the first systematic study of scaling behaviors in LLM reinforcement learning post-training, revealing power-law relationships between performance and training resources across the Qwen2.5 family (0.5B–72B), with learning efficiency saturating as model scale increases.

## Background & Motivation

**Background**: Scaling laws for pre-training have been extensively studied—Kaplan et al. and Chinchilla established power-law relationships between loss and model size, data volume, and compute. However, RL post-training (e.g., GRPO, RLHF) has become the dominant paradigm for improving LLM reasoning, yet its scaling behavior remains almost entirely unexplored.

**Limitations of Prior Work**: Practitioners conducting RL post-training lack principled guidance: How large a model should be selected? How much compute should be allocated? Should data be reused when it is scarce? These critical questions lack quantitative answers, leading to extensive trial-and-error and resource waste.

**Key Challenge**: Do pre-training scaling laws transfer to RL post-training? RL post-training has distinct characteristics—using verifiable rewards rather than cross-entropy loss, and relying on on-policy sampling rather than i.i.d. data—which may give rise to fundamentally different scaling behavior.

**Goal**: To systematically characterize the relationships among model scale, data volume, compute, and performance in RL post-training through large-scale experiments, and to establish predictive scaling formulas.

**Core Idea**: The test loss of RL post-training follows a log-linear (power-law) relationship with resource consumption: $\log L(N,X) = -k(N) \cdot \log X + E(N)$, where the learning efficiency $k(N)$ increases with model scale but converges toward a saturation limit $K_{\max}$.

## Method

### Overall Architecture

This paper presents a systematic empirical study in which 63 LLMs spanning the full Qwen2.5 series (0.5B to 72B, both base and instruct variants) are trained on mathematical reasoning tasks using the GRPO algorithm. Performance is measured under three resource-constrained scenarios, and scaling formulas are fitted accordingly.

### Key Designs

1. **Power-Law Scaling Formulation**:

    - **Function**: Establishes a predictable relationship between test loss and model scale/resource consumption in RL post-training.
    - **Mechanism**: The formula $\log L(N,X) = -k(N) \cdot \log X + E(N)$ is derived, where $X$ denotes compute $C$ or data $D$. Learning efficiency $k(N)$ is modeled with a saturation function: $k(N) = \frac{K_{\max}}{1+N_0/N}$, capturing that larger models exhibit higher learning efficiency with diminishing marginal returns.
    - **Design Motivation**: Analogous to pre-training scaling laws, but augmented with a saturation term to capture the diminishing returns characteristic of RL post-training.

2. **Inter/Intra-model Prediction Protocol**:

    - **Function**: Validates the predictive capability of the scaling formula.
    - **Mechanism**: Inter-model prediction uses parameters fitted on 0.5B–32B models to predict 72B model performance; intra-model prediction uses early training steps to predict subsequent training trajectories. Both protocols achieve $R^2 > 0.99$.
    - **Design Motivation**: The core value of scaling laws lies in their predictive power; these two protocols validate generalization across model scales and training stages, respectively.

3. **Data Reuse Analysis**:

    - **Function**: Addresses the practical question of whether data should be reused when training data is limited.
    - **Mechanism**: Total data volume $D_{\mathrm{total}}$ is fixed while varying the reuse factor $\tau$ ($D_{\mathrm{unique}} \times \tau = D_{\mathrm{total}}$). Experiments show that performance is insensitive to $\tau$ for $\tau \leq 25$, with performance governed primarily by total training volume rather than sample uniqueness.
    - **Design Motivation**: High-quality reasoning data is a common bottleneck; validating the effectiveness of data reuse has direct practical significance.

### Loss & Training

Standard GRPO is employed with binary reward signals (correct = 1, incorrect = 0). The primary evaluation metric is test loss $L = 1 - R/R_{\max}$. Training data is drawn from the mathematical subset of guru-RL-92k (~50k problems), sorted by increasing difficulty for curriculum learning. Each configuration is repeated three times to ensure robustness.

## Key Experimental Results

### Main Results

| Model Scale | Compute Efficiency $k_C(N)$ | Data Efficiency $k_D(N)$ | Notes |
|---|---|---|---|
| 0.5B–3B | Rapid increase | Rapid increase | Small-model regime; scale benefits are pronounced |
| 7B–32B | Growth slows | Growth slows | Efficiency gains begin to saturate |
| 72B | Approaching $K_{\max}$ | Approaching $K_{\max}$ | Saturation trend clearly visible |

### Prediction Accuracy

| Prediction Type | $R^2$ | Notes |
|---|---|---|
| Inter-model (0.5B–32B → 72B) | >0.99 | Accurately predicts large-model performance |
| Intra-model (early → late) | >0.99 | Accurately predicts training trajectory |
| Cross-architecture (Llama 3) | >0.99 | Formula is architecture-agnostic |

### Key Findings

- **Larger models consistently achieve superior compute and data efficiency**, but with diminishing marginal returns; gains above 32B are substantially reduced.
- **Data reuse is highly effective**: no significant performance degradation is observed for $\tau \leq 25$; overfitting only emerges at $\tau = 100$.
- **Domain transfer is limited**: RL post-training generalizes well within the mathematical domain but yields little benefit—and may even impair certain capabilities—on OOD tasks such as code and logical reasoning.
- **An interesting crossover exists between 32B and 72B**: under the same compute budget, 32B models can outperform 72B models in early training (due to more feasible training steps), revealing a trade-off between model scale and training duration.

## Highlights & Insights

- **The "Chinchilla" of RL post-training**: This work is the first to establish scaling formulas for RL post-training analogous to those for pre-training, filling an important theoretical gap.
- **Discovery of saturation**: The saturation of learning efficiency implies that indefinitely scaling model size is not the optimal strategy for RL post-training, providing an upper-bound reference for resource allocation.
- **Practical value of data reuse**: Moderate data reuse is shown to incur negligible performance loss, offering direct guidance for data-scarce settings.
- **Cross-architecture validation**: Validation on the Llama 3 family strengthens the generality of the conclusions.
- **Warning on domain transfer**: The high specialization of RL post-training—which may degrade other capabilities—is an important practical caution.

## Limitations & Future Work

- Experiments are limited to mathematical reasoning; scaling behavior of multi-domain RL post-training remains unknown.
- The largest model evaluated is 72B; the saturation trend beyond this scale cannot be empirically verified.
- Only GRPO is studied; whether other RL algorithms (e.g., DAPO, PPO) exhibit different scaling behaviors warrants further investigation.
- Only dense models are considered; scaling behavior of RL post-training for MoE architectures is not addressed.
- The absolute coefficients of the scaling formula depend on the evaluation dataset and task difficulty, limiting universal interpretability.

## Related Work & Insights

- **vs. Kaplan et al. (2020)**: The classical pre-training scaling laws focus on cross-entropy loss; this work extends the scaling framework to reward optimization in RL post-training.
- **vs. Chinchilla (Hoffmann et al., 2022)**: Chinchilla provides compute-optimal model-data ratios; this work offers analogous resource allocation guidance for RL post-training.
- **vs. Hilton et al. (2023)**: That work identifies power-law relationships in CNN+RL settings; this work validates similar patterns in the LLM+GRPO regime.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of RL post-training scaling behavior, addressing an important open question.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale experiments with 63 models across multiple scales, architectures, and scenarios, each repeated three times.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical derivations, and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ — Provides quantitative guidance for resource allocation in RL post-training, with exceptionally high practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[CVPR 2026\] Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](../../CVPR2026/reinforcement_learning/rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)
- [\[ICLR 2026\] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](../../ICLR2026/reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)
- [\[ACL 2026\] Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)

</div>

<!-- RELATED:END -->
