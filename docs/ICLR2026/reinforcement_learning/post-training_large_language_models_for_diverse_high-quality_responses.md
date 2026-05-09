---
title: >-
  [Paper Note] Post-training Large Language Models for Diverse High-Quality Responses
description: >-
  [ICLR 2026][Reinforcement Learning][diversity] This paper proposes DQO (Diversity Quality Optimization), which defines a diversity metric in semantic embedding space via determinantal point processes (DPP), and jointly optimizes it with reward signals to simultaneously improve semantic diversity and response quality during LLM post-training. DQO can be stacked on top of GRPO/PPO.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - diversity
  - determinantal point process
  - GRPO
  - post-training
  - quality-diversity trade-off
date: 2026-05-08
content_hash: 9921539252ce6660
---

# Post-training Large Language Models for Diverse High-Quality Responses

**Conference**: ICLR 2026
**arXiv**: [2509.04784](https://arxiv.org/abs/2509.04784)
**Code**: [https://github.com/fairytale9/diversity-quality-optimization](https://github.com/fairytale9/diversity-quality-optimization)
**Area**: Reinforcement Learning
**Keywords**: diversity, determinantal point process, GRPO, post-training, quality-diversity trade-off

## TL;DR
This paper proposes DQO (Diversity Quality Optimization), which defines a diversity metric in semantic embedding space via determinantal point processes (DPP), and jointly optimizes it with reward signals to simultaneously improve semantic diversity and response quality during LLM post-training. DQO can be stacked on top of GRPO/PPO.

## Background & Motivation

**Background**: LLM post-training (RLHF/GRPO, etc.) significantly improves downstream task performance, but as a side effect severely reduces output diversity—models tend to generate narrow "canonical answers," losing the ability to explore diverse solution paths and personalized styles.

**Limitations of Prior Work**: Existing diversity-promoting methods focus on the inference side (temperature scaling, top-k sampling) or only address token-level differences (token entropy regularization), which cannot recover missing modes in the base model distribution or capture semantic-level diversity.

**Key Challenge**: How to define a semantically meaningful diversity metric that is both computationally efficient and theoretically principled, and balance it against quality objectives? Simple pairwise distance metrics are prone to degeneracy—the model may learn only two widely separated clusters.

**Goal**: (a) Define a semantic-level diversity metric; (b) avoid the cluster degeneracy of pairwise distances; (c) jointly optimize quality and diversity during training.

**Key Insight**: DPP determinants are used to define diversity—the larger the volume of the parallelepiped spanned by embedding vectors, the higher the diversity. The determinant naturally penalizes linear dependence (clustering), overcoming the degeneracy of pairwise distances.

**Core Idea**: Use the DPP determinant as a semantic diversity measure, with rewards serving as scaling factors for embedding vectors, and employ leave-one-out gradient estimation to stabilize training.

## Method

### Overall Architecture
A DPP diversity term is added to the standard RL post-training objective. For each prompt $x$, $k$ responses $y_{1:k}$ are sampled and mapped to semantic space via a pretrained embedding model $\phi$, constructing a Gram matrix $L_\phi(y_{1:k})[i,j] = \langle \phi(y_i), \phi(y_j) \rangle$. The diversity score is defined as $\text{Div}(y_{1:k}) = \det(L_\phi(y_{1:k}))$.

### Key Designs

1. **DPP Determinant Diversity Metric**:

    - Function: Measures the "volume" spanned by a set of responses in semantic embedding space.
    - Mechanism: $\det(L)$ equals the squared volume of the parallelepiped spanned by the embedding vectors. The more linearly independent the vectors (i.e., the more semantically distinct), the larger the determinant; clustering (linear dependence) drives the determinant toward zero.
    - Design Motivation: Pairwise distance metrics are susceptible to pseudo-diversity via two clusters. The determinant is sensitive to linear dependence and can detect degenerate cases where responses appear distant yet lie in a low-dimensional subspace.

2. **Joint Quality-Diversity Objective**:

    - Function: $J_{Div}(\pi_\theta) = \mathbb{E}[\sum_i r(x,y_i) + \alpha \log\det(L_\phi(y_{1:k})) - \beta \text{KL}(\pi_\theta || \pi_{ref})]$
    - Mechanism: The optimal policy can be expressed as $\pi_{div}(y_{1:k}|x) \propto \det(L_\psi(x,y_{1:k}))$, where $\psi(x,y) = \sqrt{\exp(r/\alpha)\pi_{ref}(y|x)} \cdot \phi(y)$ is the reward-augmented embedding. Rewards serve as scaling factors (norms) of embedding vectors, while semantics determine their directions.
    - Design Motivation: Provides a geometric interpretation of the quality-diversity trade-off—maximizing volume requires vectors to be both large (high quality) and orthogonal (high diversity), consistent with D-optimal experimental design theory.

3. **Leave-one-out Gradient Estimator**:

    - Function: Stabilizes training and reduces gradient variance.
    - Mechanism: Replaces the raw $\log\det(L)$ with $\log\frac{\det(L(y_{1:k})+I_k)}{\det(L(y_{-i})+I_{k-1})}$. Adding $I_k$ ensures a bounded range $[0, \log(1+k)]$; the leave-one-out term subtracts a baseline excluding the $i$-th response.
    - Design Motivation: The raw $\log\det$ diverges to negative infinity as the determinant approaches zero, causing training instability. Adding the identity regularization combined with the LOO baseline simultaneously addresses both stability and variance.

### Loss & Training
- Compatible as an add-on to GRPO (reasoning tasks) or PPO (non-reasoning tasks).
- Hyperparameter $\alpha$ controls the quality-diversity trade-off; $k$ controls the number of samples per prompt.
- A reward model (rather than outcome reward) is used for scoring to prevent reward hacking, where the model provides a correct answer followed by random content to inflate diversity scores.

## Key Experimental Results

### Main Results

| Method | Dolly distinct-4↑ | Dolly self-rouge↑ | Dolly pass@1↑ | Dolly pass@10↑ |
|------|------------------|-------------------|---------------|----------------|
| PPO | 0.64 | 0.49 | 5.65 | 8.39 |
| GRPO-likelihood | 0.70 | 0.54 | 5.86 | 8.50 |
| GRPO-entropy | 0.75 | 0.57 | 4.71 | 7.70 |
| **DQO** | **0.69** | **0.54** | **5.92** | **8.74** |

| Method | GSM8K distinct-4↑ | GSM8K self-rouge↑ | GSM8K pass@1↑ | GSM8K pass@10↑ |
|------|------------------|-------------------|---------------|----------------|
| GRPO | 0.32 | 0.21 | 76.8 | 87.9 |
| GRPO-likelihood | 0.86 | 0.59 | 50.9 | 80.4 |
| GRPO-entropy | 0.38 | 0.25 | 77.0 | 92.6 |
| **DQO** | **0.42** | **0.31** | **76.3** | **91.2** |

### Ablation Study

| $\alpha$ | $k$ | distinct-4↑ | pass@1↑ | pass@10↑ |
|----------|-----|------------|---------|----------|
| 0 (PPO) | - | 0.64 | 5.65 | 8.39 |
| 0.5 | 4 | 0.69 | 5.84 | 8.79 |
| 1.0 | 4 | 0.69 | 5.92 | 8.74 |
| 2.0 | 4 | 0.75 | 5.27 | 7.86 |

### Key Findings
- DQO is the only method that consistently maintains high quality and high diversity across all tasks. GRPO-entropy achieves good diversity on GSM8K but poor quality on Dolly.
- DPP determinant vs. pairwise distance: in a city recommendation experiment, pairwise distance leads to two clusters, whereas the determinant produces genuinely broad diversity.
- DQO's advantage becomes more pronounced as $n$ increases in pass@$n$—higher diversity translates to a greater probability of finding good answers at larger $n$.
- Excessively large $\alpha$ (e.g., 2.0) sacrifices pass@1 quality.

## Highlights & Insights
- **The DPP determinant as a diversity metric** resolves the degeneracy of pairwise distances and is theoretically connected to D-optimal experimental design. This metric is transferable to any scenario requiring set-level diversity (recommendation systems, active learning, etc.).
- **The leave-one-out gradient estimator** provides a bounded guarantee (Lemma 1) that stabilizes training and ensures robustness to varying $k$, representing a key engineering contribution.
- Outcome rewards are found to be susceptible to reward hacking (answering correctly then generating random content); a reward model must be used instead.

## Limitations & Future Work
- Diversity depends on the quality of the pretrained embedding model; different embeddings may yield different results.
- Sampling $k$ responses simultaneously and computing the determinant introduces additional GPU overhead during training.
- Diversity gains on reasoning tasks (GSM8K) are limited, likely because the space of correct answers is inherently constrained.

## Related Work & Insights
- **vs. GRPO-entropy (Yao et al.)**: Token-level entropy regularization fails to capture semantic diversity and leads to significant quality degradation on non-reasoning tasks.
- **vs. GRPO-likelihood (He et al.)**: Generation-probability-based diversity performs poorly on reasoning tasks.
- **vs. Chung et al.**: Pairwise embedding distance weighting based on DPO is prone to cluster degeneracy.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines DPP with LLM post-training and establishes a theoretical connection to experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four task types, multiple diversity metrics, complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ Geometric interpretation is clear; the connection to D-optimal design is insightful.
- Value: ⭐⭐⭐⭐ Offers a practical contribution to the diversity problem in LLM post-training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](../../NeurIPS2025/reinforcement_learning/repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_diverse_behaviors.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)

<!-- RELATED:END -->
