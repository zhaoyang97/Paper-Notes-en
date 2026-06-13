---
title: >-
  [Paper Note] Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs
description: >-
  [AAAI 2026][Model Compression][Soft Prompt Transfer] This paper proposes PUMA, a framework that leverages lightweight adapters and a grouped user selection strategy to efficiently migrate personalized soft prompts from a…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Soft Prompt Transfer"
  - "Personalization"
  - "LLM"
  - "Parameter-Efficient Adapter"
  - "User Selection Strategy"
date: 2026-05-08
content_hash: fbd91982f32e6960
---

# Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs

**Conference**: AAAI 2026 Oral  
**arXiv**: [2601.12034](https://arxiv.org/abs/2601.12034)  
**Code**: [github](https://github.com/Kimagure7/Dont-Start-Over)  
**Area**: Model Compression
**Keywords**: Soft Prompt Transfer, Personalization, LLM, Parameter-Efficient Adapter, User Selection Strategy

## TL;DR
This paper proposes PUMA, a framework that leverages lightweight adapters and a grouped user selection strategy to efficiently migrate personalized soft prompts from a source LLM to a target LLM with a different architecture. PUMA matches or surpasses from-scratch training on three large-scale datasets while reducing computational cost by up to 98%.

## Background & Motivation

As LLMs are increasingly deployed in recommendation systems, personal assistants, and adaptive education, **personalization** has become a core requirement. Soft prompts, as a lightweight personalization technique, give rise to a distinctive "1+N" application architecture: one general-purpose backbone model ("1") paired with thousands of user-specific soft prompts ("N").

However, this architecture harbors a **critical vulnerability**:

**Model upgrades are inevitable**: the backbone model will be upgraded (to a more powerful version) or replaced (with a smaller, more efficient one).

**Semantic alignment breaks**: the new model typically has a different embedding dimension and semantic space ($d_t \neq d_s$), rendering existing soft prompts entirely ineffective.

**Retraining is prohibitively expensive**: retraining soft prompts for tens of thousands or even millions of users from scratch incurs enormous computational cost.

**Core Problem**: Can the personalized soft prompts of large-scale users be migrated from a source model to a target model at minimal computational cost while preserving personalization performance?

**Distinction from prior work**: Existing soft prompt transfer research focuses on **task-level** transfer (transferring a single public prompt from NLI to text classification), a "one-to-one" problem. This paper is the first to address **user-level** migration—transferring thousands of private user prompts from one model to another—an "N-to-N" challenge.

**Problem decomposition**: The problem is decomposed into two coupled sub-problems: (1) semantic incompatibility—how to make the target model understand prompts trained on the source model; and (2) migration efficiency—how to scale to tens of thousands of users.

## Method

### Overall Architecture

PUMA (Prompt-level User Migration Adapter) consists of two core components (as shown in Figure 2):

1. **Migration Adapter**: An end-to-end trained lightweight feedforward network that bridges the semantic spaces of the source and target models.
2. **Grouped User Selection Strategy**: A representative training subset selected via K-Means clustering and variance-stratified sampling.

### Key Designs

#### 1. **Migration Adapter**

The migration function $\Phi$ is implemented as a feedforward network with residual connections and Layer Normalization:

$$p'_u = \Phi_\theta(p_u), \quad p_u \in \mathbb{R}^{l \times d_s}, \; p'_u \in \mathbb{R}^{l \times d_t}$$

Key design choices:
- **End-to-end training**: Optimized directly via task loss to ensure the transformation preserves downstream task utility.
- **Parameter efficiency**: Only adapter parameters are trained; the target model and source prompts are frozen.
- **Simple architecture**: Residual connections and LayerNorm balance expressiveness and computational cost.

Optimization objective:

$$\theta^* = \arg\min_\theta \sum_{(u,i,y) \in D} \mathcal{L}_{\text{task}}\left(M_t(T(\Phi_\theta(p_u), \phi(i))), y\right)$$

#### 2. **Grouped User Selection Strategy (Core Efficiency Innovation)**

Training the adapter over tens of thousands of users remains costly. The core insight is that an ideal user subset must simultaneously capture **preference diversity** and **complexity spectrum**.

**Two-stage selection process**:

**Stage 1: K-Means clustering to capture diversity**
- K-Means clustering is applied to source prompts $\{p_u\}_{u \in \mathcal{U}}$, partitioning users into $k$ groups.
- Ensures the selection covers diverse learning preference patterns.

**Stage 2: Variance-stratified sampling to capture complexity**
- Within each cluster, users are stratified by the variance of their historical outputs.
- Low-variance users = consistent preferences (e.g., always rating highly), easy to model.
- High-variance users = complex preferences, difficult to model.
- Normal-distribution weighting assigns higher weight to the medium-variance group.

#### 3. **Advanced Migration Topologies**

**Chained Migration**: $M_A \rightarrow M_B \rightarrow M_C$
- Handles sequential model replacements.
- Migrated prompts serve as the source for the next step.

**Aggregated Migration**: $[M_A, M_B] \rightarrow M_C$
- Fuses personalization from multiple source models.
- Achieved by concatenating a user's source prompts $[p_u^A; p_u^B]$ and mapping them to the target model.
- Multi-source integration produces richer user representations.

### Loss & Training

**Task-specific loss functions**:

- **Rating prediction** (Amazon/Yelp): Mixed loss $0.8 \cdot \mathcal{L}_{MSE} + 0.2 \cdot \mathcal{L}_{CE}$
    - Logits for five discrete rating tokens are extracted from LLM outputs.
    - Cross-entropy handles classification; an MLP head regresses continuous rating values.
- **CTR prediction** (MIND): Standard binary cross-entropy $\mathcal{L}_{BCE}$ computed on the logit of the "yes" token.

**Training details**:
- Source prompt length $l=1$; pre-trained for 15 epochs with learning rate $5 \times 10^{-4}$.
- PUMA adapter trained for 4 epochs using FusedAdam, learning rate $10^{-4}$, batch size 32.
- Hardware: NVIDIA A100 GPU; framework: PyTorch 2.5.

## Key Experimental Results

### Main Results

**PUMA vs. from-scratch training** (LLaMA-2-1B → LLaMA-2-3B):

| Method | Amazon RMSE↓ | Amazon MAE↓ | MIND AUC↑ | MIND uAUC↑ | Yelp RMSE↓ | Yelp MAE↓ |
|--------|-------------|------------|----------|-----------|-----------|----------|
| Random Init | 1.2352 | 1.1168 | 0.4917 | 0.4883 | 1.6671 | 1.4981 |
| From Scratch | 0.9414 | 0.6296 | 0.5778 | 0.5289 | 1.1994 | 0.9269 |
| **PUMA** | **0.9135** | **0.5701** | **0.6546** | **0.6552** | **1.1073** | **0.8493** |

**Efficiency comparison**:

| Method | Time/Epoch | Epochs | Total Time | Speedup |
|--------|-----------|--------|-----------|---------|
| From Scratch | 3.00h | 8 | 24.0h | 1× |
| PUMA (2k users) | 0.16h | 3 | 0.48h | **50×** |

### Ablation Study

**User selection strategy comparison** (fixed budget: 2,000 users for Amazon/Yelp, 1,500 for MIND):

| Strategy | Amazon RMSE↓ | MIND uAUC↑ | Yelp RMSE↓ |
|----------|-------------|-----------|-----------|
| Random Sampling | 0.9419 | 0.5861 | 1.1146 |
| Random Sampling (6k) | 0.9320 | 0.6636 | 1.1128 |
| Variance Bucketing | 0.9508 | 0.5888 | 1.1171 |
| K-Means Stratified | 0.9546 | 0.5927 | 1.1152 |
| K-Means + FPS | 0.9355 | 0.5966 | 1.1147 |
| **K-Means + Variance Stratified (PUMA)** | **0.9315** | **0.6344** | **1.1111** |

PUMA with only 2,000 users outperforms random sampling with 6,000 users.

**Cross-architecture migration (Figure 4 heatmap)**:

| Source → Target | Performance Gain | Notes |
|----------------|-----------------|-------|
| LLaMA → Qwen | >1.0 | Matches or exceeds from-scratch |
| LLaMA → Phi-3 | >1.0 | Effective cross-family transfer |
| Gemma → Phi-3 | <1.0 | Weak source → strong target slightly underperforms |
| Phi-3 → LLaMA | >1.0 | Strong source → weak target exceeds from-scratch |

**Chained migration stability** (Llama→Qwen→Gemma→StableLM→Phi-3):
- Initial RMSE: 0.9348; final RMSE: 0.9277.
- The full chain outperforms from-scratch training at every step.

**Aggregated migration** (dual-source → Phi-3):
- Llama+StableLM→Phi-3 RMSE: 0.9217 (vs. single-source Llama: 0.9293, StableLM: 0.9380).
- Multi-source fusion yields "knowledge synergy," with different models capturing complementary user preferences.

### Key Findings

1. **PUMA surpasses from-scratch training**: The shared mapping function learned by the adapter generalizes better than independently learning $k$ user representations.
2. **98% computational savings**: Training the adapter on only 2,000 out of 30,000 users achieves a 50× speedup.
3. **Cross-architecture migration is effective**: Transfer between distinct model families (e.g., LLaMA→Qwen) remains effective.
4. **Chained migration is stable**: Performance does not degrade after five sequential migrations.
5. **Aggregated migration enhances personalization**: Multi-source fusion outperforms single-source migration, revealing a "knowledge synergy" effect.
6. **Migration quality is bounded by source prompt quality**: A weak source → strong target scenario may slightly underperform from-scratch training.

## Highlights & Insights

1. **Precise and practically valuable problem formulation**: Model replacement for large-scale personalized prompts is a genuine pain point in industry.
2. **The finding that PUMA surpasses from-scratch training** is insightful—the shared mapping function may discover a superior semantic transformation space through a regularization effect.
3. **The grouped user selection strategy** combines diversity (K-Means) and complexity (variance stratification), achieving full-dataset performance with only 1/15 of the data.
4. **The success of chained and aggregated migration** transforms prompt migration from a passive maintenance task into a strategic opportunity to actively enhance personalization.
5. **Comprehensive experimental design**: three datasets, five models, and two advanced migration topologies.

## Limitations & Future Work

1. Evaluation is limited to recommendation system tasks; applicability to other personalization scenarios (dialogue, writing assistants) remains unverified.
2. The user selection strategy is based on static heuristics; it could be further optimized via reinforcement learning.
3. Only user prompts are migrated; item embeddings are not considered—comprehensive knowledge transfer requires addressing both.
4. The cold-start problem is not addressed—handling new users absent from the source system remains open.
5. Prompt length is fixed at 1; whether longer prompts yield better migration performance warrants exploration.

## Related Work & Insights

- **Soft Prompt Tuning** (Lester et al., 2021): The foundational personalization technique whose model-binding limitation PUMA resolves.
- **SPoT** (Vu et al., 2022): Task-level prompt transfer; this paper extends the paradigm to the user level.
- **ATTRIPOTION** (Asai et al., 2022): Multi-prompt mixing mechanism that inspired the aggregated migration design.
- **Coreset Selection** methods (K-Means, FPS, gradient matching, etc.): This paper designs a variance-stratified approach better suited to the target scenario.
- Insight: Personalized assets should be treated as transferable, composable "digital assets" rather than one-time parameters bound to a specific model; prompt migration may become an infrastructure-level technology in the LLM ecosystem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to formally define and address user-level soft prompt cross-model migration)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three datasets, five models, multiple migration topologies)
- Writing Quality: ⭐⭐⭐⭐ (Well-organized with thorough problem articulation)
- Value: ⭐⭐⭐⭐⭐ (Addresses a real industrial pain point; 98% cost reduction has direct application value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Highly Efficient and Effective LLMs with Multi-Boolean Architectures](../../ICLR2026/model_compression/highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)
- [\[AAAI 2026\] Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection](explore_and_establish_synergistic_effects_between_weight_pruning_and_coreset_sel.md)
- [\[AAAI 2026\] LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph](loom_personalized_learning_informed_by_daily_llm_conversations_toward_long-term_.md)
- [\[AAAI 2026\] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)

</div>

<!-- RELATED:END -->
