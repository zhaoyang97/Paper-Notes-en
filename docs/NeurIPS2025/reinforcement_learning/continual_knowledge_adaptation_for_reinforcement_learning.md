---
title: >-
  [Paper Note] Continual Knowledge Adaptation for Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][continual RL] This paper proposes CKA-RL, which maintains a task-specific knowledge vector for each task and employs softmax-weighted dynamic knowledge adaptation along with an adaptive knowledge merging mechanism, achieving a 4.20% overall performance gain and 8.02% forward transfer improvement across three continual RL benchmarks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - continual RL
  - knowledge vector
  - catastrophic forgetting
  - forward transfer
  - knowledge merging
date: 2026-05-08
content_hash: f297748cd758bf4e
---

# Continual Knowledge Adaptation for Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.19314](https://arxiv.org/abs/2510.19314)
**Code**: [GitHub](https://github.com/Fhujinwu/CKA-RL)
**Area**: Reinforcement Learning / Continual Learning
**Keywords**: continual RL, knowledge vector, catastrophic forgetting, forward transfer, knowledge merging

## TL;DR
This paper proposes CKA-RL, which maintains a task-specific knowledge vector for each task and employs softmax-weighted dynamic knowledge adaptation along with an adaptive knowledge merging mechanism, achieving a 4.20% overall performance gain and 8.02% forward transfer improvement across three continual RL benchmarks.

## Background & Motivation
**Background**: Continual reinforcement learning (CRL) enables agents to sequentially learn multiple tasks in non-stationary environments. Existing approaches fall into four categories: regularization, experience replay, architectural expansion, and meta-learning.

**Limitations of Prior Work**: (1) Cross-task conflict — different tasks may share structure but have incompatible objectives, causing direct knowledge reuse to introduce interference; (2) Catastrophic forgetting — learning new tasks overwrites previously acquired knowledge; (3) Scalability — memory and computation costs grow linearly with the number of tasks (e.g., CompoNet).

**Key Challenge**: How can an agent retain knowledge of past tasks while efficiently leveraging historical knowledge to accelerate learning on new tasks, without incurring linear memory growth with the number of tasks?

**Goal**: To design a continual RL method that efficiently accumulates, reuses, and compresses historical knowledge.

**Key Insight**: Inspired by the concept of "task vectors" in model editing, the paper represents each task's learned parameter increment as a knowledge vector and dynamically combines historical knowledge vectors via learnable weights to adapt to new tasks.

**Core Idea**: A knowledge vector pool stores historical knowledge; softmax-weighted combinations adapt to new tasks; similar vectors are automatically merged to bound memory usage.

## Method

### Overall Architecture
The framework consists of three components: (1) a base parameter $\theta_{base}$ trained on the first task; (2) for each subsequent task $\tau_k$, a knowledge vector $v_k$ is learned alongside softmax-weighted coefficients $\alpha_k$ over historical vectors $\{v_1, \ldots, v_{k-1}\}$; (3) when the vector pool exceeds capacity $K_{max}$, the two most similar vectors are merged.

### Key Designs

1. **Knowledge Vectors and Dynamic Adaptation**:

   - Function: Upon completing training on each task, a knowledge vector $v_k = \theta_k - \theta_{base}$ is extracted and added to the pool $\mathcal{V}$.
   - Mechanism: The parameters for task $k$ are defined as $\theta_k = \theta_{base} + \sum_{j=1}^{k-1} \alpha_j^k v_j + v_k$, where $\alpha_j^k = \text{softmax}(\beta_j^k)$ are learnable scalars and $v_k$ is initialized to zero. During training, $\theta_{base}$ is frozen; only $\beta_k$ and $v_k$ are optimized.
   - Design Motivation: The softmax normalization ensures weights sum to one, and the inclusion of $v_1 = 0$ (null knowledge) allows the model to opt out of using historical knowledge (when $\alpha_1 = 1$), thereby preventing negative transfer.

2. **Adaptive Knowledge Merging**:

   - Function: When the pool size exceeds $K_{max}$, the most similar pair of vectors is merged.
   - Mechanism: Cosine similarity $S_{ij} = \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}$ is used to identify the most similar pair $(v_m, v_n) = \arg\max S_{ij}$, which are then merged as $v_{merge} = \frac{1}{2}(v_m + v_n)$.
   - Design Motivation: Similar knowledge vectors encode functionally analogous adaptation directions, so merging them minimizes information loss while keeping the pool compact and addressing scalability issues as the number of tasks grows.

3. **Base Model Construction**:

   - Function: $\theta_{base}$ is trained on the first task and serves as the foundation for all subsequent knowledge adaptation.
   - Mechanism: $\theta_{base}$ encodes general feature representations; setting $v_1 = 0$ includes it implicitly in the vector pool.
   - Design Motivation: Subsequent tasks need only learn incremental updates (knowledge vectors) rather than training from scratch.

### Training Procedure
- Task 1: Train $\theta_{base}$; add $v_1 = 0$ to the pool.
- Task $k$: Initialize $v_k = 0$, $\beta_k \sim \mathcal{N}(0,1)$; construct $\theta_k$; optimize $(v_k, \beta_k)$ via RL; add $v_k$ to the pool.
- If $|\mathcal{V}| > K_{max}$: identify the most similar pair and merge.

## Key Experimental Results

### Overall Performance (PERF.) and Forward Transfer (FWT.) Across Three Benchmarks

| Method | Meta-World PERF. | Meta-World FWT. | SpaceInvaders PERF. | Freeway FWT. | Avg. PERF. | Avg. FWT. |
|--------|-----------------|----------------|-------------------|-------------|-----------|----------|
| Baseline | 0.419 | 0.000 | 0.631 | 0.000 | 0.392 | 0.000 |
| PackNet | 0.584 | 0.019 | 0.773 | 0.197 | 0.504 | 0.145 |
| CompoNet | 0.639 | 0.161 | 0.859 | 0.403 | 0.547 | 0.274 |
| RECALL | 0.613 | 0.109 | 0.821 | 0.356 | 0.532 | 0.220 |
| **CKA-RL** | **0.673** | **0.223** | **0.897** | **0.481** | **0.576** | **0.355** |

### Ablation Study

| Configuration | Avg. PERF. Change | Note |
|--------------|-------------------|------|
| w/o knowledge adaptation (only $v_k$) | −3.2% | No historical knowledge; degenerates to independent task learning |
| w/o adaptive merging | −1.1% | Linear memory growth; marginal performance gain |
| Fixed $\alpha$ (uniform weights) | −2.5% | Cannot adaptively select historical knowledge |
| Full CKA-RL | Best | Dynamic adaptation + adaptive merging |

### Key Findings
- CKA-RL consistently outperforms 9 state-of-the-art methods across all three benchmarks, achieving a 4.20% overall performance gain and an 8.02% forward transfer improvement.
- The softmax weights over knowledge vectors differentiate automatically during training — concentrating on a single historical task when relevance is high, and approaching uniform distribution when relevance is low.
- The adaptive merging mechanism maintains a constant pool size of $K_{max}$ with negligible performance degradation.

## Highlights & Insights
- **Transferring model editing ideas to RL**: The paper elegantly adapts the "task vector" concept from NLP to continual RL; the linear composability of knowledge vectors makes cross-task transfer intuitive.
- **Null knowledge design**: Setting $v_1 = 0$ allows the model to selectively ignore historical knowledge, effectively preventing negative transfer — a simple yet critical design choice.
- **Similarity-guided merging**: This approach is more principled than random or rule-based merging, guaranteeing minimal information loss.

## Limitations & Future Work
- **Knowledge vector dimensionality equals model parameter dimensionality**: Memory overhead remains substantial for large models; low-rank knowledge vectors (e.g., LoRA-style decomposition) could be explored.
- **Simple averaging during merging** may lose directional information: the average of two vectors does not necessarily preserve the optimal adaptation direction of either; weighted or performance-guided merging strategies warrant investigation.
- **Evaluation limited primarily to discrete action spaces** (except Meta-World): further validation on continuous control and more complex task sequences is needed.
- **Sensitivity to base model quality**: The first task's training quality directly affects all subsequent tasks; an unrepresentative first task may yield a poor $\theta_{base}$.
- **Interpretability of knowledge vectors**: It remains unclear which dimensions encode which aspects of task knowledge; visualization analyses could provide further insight.
- **Manual selection of $K_{max}$**: An adaptive scheduling mechanism for the pool capacity would be preferable.
- **Task order sensitivity**: The paper does not analyze how different task presentation orders affect final performance.

## Related Work & Insights
- **vs. CompoNet**: CompoNet composes policies via modular architectures; CKA-RL achieves composition more compactly in vector space and additionally controls model scale through merging.
- **vs. PackNet**: PackNet isolates task parameters via binary masks within a fixed network; CKA-RL's continuous vector adaptation is more flexible.
- **vs. MAML**: MAML achieves fast adaptation through meta-learning; CKA-RL achieves transfer through explicit accumulation of knowledge vectors.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of knowledge vectors and adaptive merging is novel; the borrowing from model editing is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, nine baselines, detailed ablations, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; theoretical analysis is provided in the appendix.
- Value: ⭐⭐⭐⭐ Offers an elegant solution to the knowledge reuse problem in continual RL.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](../../ICLR2026/reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](temporal-difference_variational_continual_learning.md)
- [\[NeurIPS 2025\] Parameter Efficient Fine-tuning via Explained Variance Adaptation](parameter_efficient_fine-tuning_via_explained_variance_adaptation.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)

<!-- RELATED:END -->
