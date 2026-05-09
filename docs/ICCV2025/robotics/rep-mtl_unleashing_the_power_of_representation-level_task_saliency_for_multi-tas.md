---
title: >-
  [Paper Note] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning
description: >-
  [ICCV 2025][Robotics][Multi-task learning] This paper proposes Rep-MTL, a multi-task optimization method grounded in representation-level task saliency. It mitigates negative transfer and explicitly promotes cross-task complementarity via entropy-regularized task-specific saliency regulation (TSR) and sample-level cross-task saliency alignment (CSA), without modifying the optimizer or network architecture.
tags:
  - ICCV 2025
  - Robotics
  - Multi-task learning
  - task saliency
  - representation space
  - contrastive learning
  - negative transfer mitigation
date: 2026-05-08
content_hash: f311d72bd41b6ae3
---

# Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning

**Conference**: ICCV 2025
**arXiv**: [2507.21049](https://arxiv.org/abs/2507.21049)
**Code**: None (project page available)
**Area**: Robotics
**Keywords**: Multi-task learning, task saliency, representation space, contrastive learning, negative transfer mitigation

## TL;DR

This paper proposes Rep-MTL, a multi-task optimization method grounded in representation-level task saliency. It mitigates negative transfer and explicitly promotes cross-task complementarity via entropy-regularized task-specific saliency regulation (TSR) and sample-level cross-task saliency alignment (CSA), without modifying the optimizer or network architecture.

## Background & Motivation

Multi-task learning (MTL) improves efficiency and generalization through shared representations, yet conflicting gradient updates across tasks lead to negative transfer. Existing multi-task optimization (MTO) methods fall into two main categories:

**Loss scaling methods** (e.g., UW, DWA, FAMO): adjust task loss weights

**Gradient manipulation methods** (e.g., PCGrad, CAGrad, Nash-MTL): modify gradient directions in shared parameters

However, these approaches suffer from two fundamental problems:

- **Inconsistent effectiveness**: In demanding scenarios, many MTO methods fail to deliver consistent performance gains, with numerous methods yielding negative $\Delta p_{task}$ values.
- **Neglect of complementarity**: Existing methods focus almost exclusively on resolving conflicts while overlooking the equally important aspect of **complementary information sharing** across tasks, leaving it entirely to architectural design.

**Central Argument**: The shared representation space is where task interactions truly occur, harboring rich information and operational potential. Rather than modifying the optimizer, the authors propose regulating task saliency in the representation space to simultaneously achieve two goals: (1) preserving task-specific patterns to mitigate negative transfer; and (2) explicitly promoting cross-task complementarity sharing.

## Method

### Overall Architecture

Rep-MTL is incorporated as a regularization term added to the standard MTL objective and consists of two complementary modules:

1. **TSR (Task-specific Saliency Regulation)**: Entropy-based saliency regularization that preserves the discriminability of task-specific learning patterns.
2. **CSA (Cross-task Saliency Alignment)**: Contrastive learning-based cross-task alignment that promotes complementary information sharing.

The total loss is $\mathcal{L}_{Rep} = \sum_{t=1}^T \mathcal{L}_t(\theta_s, \theta_t) + \lambda_{tsr}\mathcal{L}_{tsr}(Z) + \lambda_{csa}\mathcal{L}_{csa}(Z)$

### Key Designs

1. **Representation-Level Task Saliency Definition**:

    - Function: Quantifies how different tasks interact in the shared representation space.
    - Mechanism: The saliency of task $\mathcal{T}_t$ is defined as the gradient of the loss with respect to the shared representation $Z$: $\mathcal{S}_t = \nabla_Z \mathcal{L}_t(\theta_s, \theta_t) \in \mathbb{R}^{B \times C \times H' \times W'}$, measuring each task objective's sensitivity to representational changes.
    - Design Motivation: Unlike parameter gradients used for direct model updates, representation-level saliency serves as a dynamic indicator that identifies and regulates inter-task dependencies, providing rich learning signals.

2. **TSR: Task-Specific Saliency Regulation**:

    - Function: Encourages each spatial location to maintain a clear task-specific learning pattern via entropy penalization.
    - Mechanism: Saliency is first aggregated across channels as $\hat{\mathcal{S}}_t = \frac{1}{|C|}\sum_c \mathcal{S}_{t,b,c,h,w}$, then normalized into a cross-task probability distribution $\mathcal{P}_{i,t} = \frac{|\hat{\mathcal{S}}_{i,t}|}{\sum_{k=1}^T |\hat{\mathcal{S}}_{i,k}|}$, and finally entropy is minimized as $\mathcal{L}_{tsr} = \frac{1}{BH'W'}\sum_i(-\sum_t \mathcal{P}_{i,t}\log\mathcal{P}_{i,t})$.
    - Design Motivation: High-entropy distributions indicate that a spatial location is equally important to all tasks (over-sharing), while low entropy indicates that the location is more critical to a specific task. Penalizing high entropy preserves task-specific learning patterns, mitigating negative transfer at its source rather than patching gradient conflicts post hoc.

3. **CSA: Cross-Task Saliency Alignment**:

    - Function: Explicitly promotes cross-task complementarity via contrastive learning in the sample dimension.
    - Mechanism: A saliency affinity matrix $\mathcal{M}_t = \mathcal{S}_t\mathcal{S}_t^\top \in \mathbb{R}^{B \times C \times C}$ is computed, and for each sample $b$, a cross-task mean anchor $\hat{\mathcal{A}_b} = \mathcal{A}_b\mathcal{A}_b^\top$ is obtained. Affinity matrices from different tasks of the same sample form positive pairs, while those from different samples form negative pairs, trained with the InfoNCE loss $\mathcal{L}_{csa} = \frac{1}{B}\sum_b -\log\frac{\exp(\text{sim}(z_b^a, z_b^t)/\tau)}{\sum_{k \neq b}\exp(\text{sim}(z_b^a, z_k^a)/\tau)}$.
    - Design Motivation: The MTO field has rarely explored how to explicitly promote task complementarity. CSA achieves this by encouraging the same sample to share consistent feature interaction patterns across tasks, while maintaining task discriminability through in-batch negative samples.

### Loss & Training

- Total loss = standard multi-task loss + $\lambda_{tsr} \cdot \mathcal{L}_{tsr}$ + $\lambda_{csa} \cdot \mathcal{L}_{csa}$
- As a pure regularization method, it does not modify the optimizer (and can be used with the basic equal-weighting strategy EW).
- Orthogonal to existing MTO methods and can be combined with them.
- Gradients flow naturally through all components, implicitly regulating model parameter updates.

## Key Experimental Results

### Main Results

**NYUv2 Dataset (3 tasks, DeepLabV3+):**

| Method | Semseg mIoU↑ | Depth Abs.Err↓ | Normal Mean↓ | $\Delta p_{task}$↑ |
|--------|-------------|---------------|-------------|-------------------|
| Single-Task | 53.50 | 0.3926 | 21.99 | 0.00 |
| EW | 53.93 | 0.3825 | 23.57 | -1.78 |
| GLS | 54.59 | 0.3785 | 22.71 | +0.30 |
| Nash-MTL | 53.41 | 0.3867 | 22.57 | -1.01 |
| DB-MTL | 53.92 | 0.3768 | 21.97 | +1.15 |
| **Rep-MTL (EW)** | **54.59** | **0.3750** | **21.91** | **+1.70** |

**Cityscapes Dataset (2 tasks):**

| Method | Semseg mIoU↑ | Depth Abs.Err↓ | $\Delta p_{task}$↑ |
|--------|-------------|---------------|-------------------|
| Single-Task | 69.06 | 0.01282 | 0.00 |
| EW | 68.93 | 0.01315 | -2.05 |
| Rep-MTL (EW) | Best | Best | Positive |

### Ablation Study

| Configuration | NYUv2 $\Delta p_{task}$↑ | Description |
|--------------|--------------------------|-------------|
| EW (baseline) | -1.78 | Standard equal weighting |
| + TSR only | Improved | Task-specific saliency regulation only |
| + CSA only | Improved | Cross-task contrastive alignment only |
| + TSR + CSA (Rep-MTL) | **+1.70** | Complementary collaboration of both modules |

### Key Findings

1. **Equal weighting alone with Rep-MTL outperforms most MTO methods**: Rep-MTL combined with the basic equal-weighting strategy achieves +1.70% on NYUv2, the highest among all methods.
2. **Efficiency advantage**: Approximately 26% faster than Nash-MTL and ~12% faster than FairGrad, as no second-order gradient computation is required.
3. **Most MTO methods yield negative results in practice**: Over 15 methods exhibit negative $\Delta p_{task}$ on NYUv2, suggesting that resolving gradient conflicts alone may not be the right direction.
4. Power Law exponent analysis confirms that Rep-MTL simultaneously improves both task-specific learning and cross-task sharing quality.

## Highlights & Insights

- **Paradigm shift**: From "resolving conflicts" to "maintaining effective training + explicitly promoting complementarity" — a meaningful shift in perspective.
- **TSR's entropy suppression**: Using entropy to measure a spatial location's task specificity as a regularization objective is both elegant and effective.
- **CSA's contrastive design**: Leveraging the consistency of per-sample saliency across tasks to promote complementarity represents the first explicit exploration of this direction in the MTO literature.
- **Abundance of negative results**: The paper honestly reports that 15+ methods fail to improve performance, serving as an important cautionary signal for the community.
- **Regularization as MTO**: Effective multi-task optimization without modifying the optimizer lowers the barrier to adoption.

## Limitations & Future Work

1. Saliency computation requires backpropagation to the representation layer for each task, leading to linearly increasing computational cost as the number of tasks grows.
2. Two hyperparameters $\lambda_{tsr}$ and $\lambda_{csa}$ require tuning, though the paper reports low sensitivity.
3. The affinity matrix $\mathcal{M}_t \in \mathbb{R}^{B \times C \times C}$ may incur memory issues when the channel dimension $C$ is large.
4. The construction of positive and negative pairs in CSA depends on batch size; small batches may degrade contrastive learning effectiveness.
5. The method is currently limited to Hard Parameter Sharing (HPS) architectures; applicability to soft sharing settings remains unexplored.

## Related Work & Insights

- Compared to RotoGrad (rotating the feature space) and SRDML (regularizing task similarity), Rep-MTL not only addresses conflict resolution but also explicitly promotes complementarity sharing.
- The entropy regularization idea in TSR can be generalized to other settings requiring representational discriminability (e.g., representation collapse in contrastive learning).
- The sample-level alignment mechanism in CSA may offer insights for multi-modal learning (e.g., vision-language alignment).

## Rating

- Novelty: ⭐⭐⭐⭐ — Approaching MTO from the representation space is a novel perspective; the TSR and CSA designs are conceptually clear and distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four benchmarks, extensive baselines, Power Law analysis, and in-depth validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, method derivation is coherent, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ — Practically strong (plug-and-play regularization) and provides an important critical reflection for the MTO community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICCV 2025\] Embodied Representation Alignment with Mirror Neurons](embodied_representation_alignment_with_mirror_neurons.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](../../NeurIPS2025/robotics/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)

<!-- RELATED:END -->
