---
title: >-
  [Paper Note] Soft Task-Aware Routing of Experts for Equivariant Representation Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][equivariant representation] This paper proposes STAR (Soft Task-Aware Routing), which employs a MoE routing mechanism to coordinate shared and task-specific information between invariant and equivariant representation learning objectives, reducing redundant feature learning and improving downstream transfer performance.
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "equivariant representation"
  - "mixture of experts"
  - "redundant feature learning"
  - "routing"
date: 2026-05-08
content_hash: df988ed845012393
---

# Soft Task-Aware Routing of Experts for Equivariant Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.27222](https://arxiv.org/abs/2510.27222)  
**Code**: [https://github.com/YonseiML/star](https://github.com/YonseiML/star)  
**Area**: Self-Supervised Learning
**Keywords**: equivariant representation, mixture of experts, self-supervised learning, redundant feature learning, routing

## TL;DR

This paper proposes STAR (Soft Task-Aware Routing), which employs a MoE routing mechanism to coordinate shared and task-specific information between invariant and equivariant representation learning objectives, reducing redundant feature learning and improving downstream transfer performance.

## Background & Motivation

Self-supervised learning (SSL) has become a dominant paradigm for learning representations from unlabeled data. **Invariant representation learning** (e.g., SimCLR) maps different augmented views of the same image to identical representations to preserve semantics, while **equivariant representation learning** (e.g., EquiMod) captures structured changes in representations induced by augmentation transformations. Recent work has shown that jointly learning both types of representations generally benefits downstream tasks.

**Limitations of Prior Work**:

**Redundant feature learning**: Existing methods (e.g., EquiMod) employ two separate projection heads for invariant and equivariant objectives, implicitly assuming the two tasks are independent. In practice, however, they are intrinsically correlated — understanding semantic categories aids in inferring illumination direction, and vice versa (illustrated by the "crater illusion": the same lunar surface appears as a crater or dome depending on lighting direction).

**Information waste**: Independent projection heads cause both branches to redundantly capture shared information, leading to inefficient use of model capacity.

**Poor gradient quality**: Redundant feature learning slows convergence of the projection heads, degrading the quality of gradient signals propagated to the backbone.

**Core Idea**: Treat projection heads as "experts" and introduce a MoE routing mechanism that adaptively assigns experts to invariant, equivariant, or shared tasks, thereby reducing redundancy, improving expert specialization, and enhancing gradient quality.

## Method

### Overall Architecture

Given an image $x$, two augmented views $v = T(x; a)$ and $v' = T(x; a')$ are generated. A shared encoder $f$ extracts features, which are then passed through the STAR projection module to produce an invariant embedding $z^{\text{inv}}$ and an equivariant embedding $z^{\text{eq}}$. The invariant branch applies an InfoNCE loss to align embeddings of different views of the same image, while the equivariant branch uses an equivariant predictor $\phi_T$ to predict the embedding shift induced by augmentation.

### Key Designs

1. **Single Shared Projection (STAR-SS)**:

    - **Function**: The simplest form of shared information modeling.
    - **Mechanism**: Three experts are defined — an invariant expert $E^{\text{inv}}$, an equivariant expert $E^{\text{eq}}$, and a shared expert $E^{\text{sh}}$. Embeddings are computed as: $z_i^{\text{inv}} = E^{\text{inv}}(f(v_i)) + E^{\text{sh}}(f(v_i))$, $z_i^{\text{eq}} = E^{\text{eq}}(f(v_i)) + E^{\text{sh}}(f(v_i))$.
    - **Design Motivation**: The shared expert naturally learns information required by both tasks, as it is jointly optimized under both objectives. However, the equal-weight summation is inflexible.

2. **MMoE Projection (STAR-MMoE)**:

    - **Function**: Adaptively routes experts to different tasks.
    - **Mechanism**: $N$ shared experts $\{E_k\}_{k=1}^N$ combined with two task-specific routers $R^{\text{inv}}$ and $R^{\text{eq}}$. The routers compute softmax weights: $s_{i,k}^{\text{inv}} = \text{softmax}_k(R^{\text{inv}}(f(v_i)))$. Embeddings are computed as weighted sums of expert outputs: $z_i^{\text{inv}} = \sum_k s_{i,k}^{\text{inv}} E_k(f(v_i))$.
    - **Design Motivation**: Different images and tasks have varying demands for shared versus task-specific information; soft routing allows dynamic allocation. STAR-SS is a degenerate case of STAR-MMoE with uniform weights.
    - **Key Constraint**: Soft routing must be used, as sparse (top-k) routing causes batch normalization instability.

3. **Equivariant Learning Design**:

    - **Function**: Models the embedding shift induced by augmentation.
    - **Mechanism**: $\hat{z}_i^{\text{eq}} = z_i^o + \phi_T(z_i^o, \psi(a_i))$, where $z_i^o$ is the equivariant embedding of the original image, $\psi$ projects augmentation parameters into the embedding space, and $\phi_T$ is a 3-layer MLP equivariant predictor.
    - The **residual connection** ensures semantic content is preserved while effectively modeling transformation shifts.
    - The **equivariant loss** is formulated as InfoNCE, with predicted embeddings $\hat{z}^{\text{eq}}$ and target embeddings $z^{\text{eq}}$ forming positive pairs.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}^{\text{inv}} + \lambda \mathcal{L}^{\text{eq}}$, with $\lambda = 1$ and temperature $\tau = 0.2$.
- The invariant loss follows the SimCLR InfoNCE formulation.
- After pretraining, the **projection heads are discarded**; only the encoder is transferred to downstream tasks, entirely bypassing the transferability limitations of MoE models.
- STL10 experiments use 16 experts trained for 200 epochs; ImageNet100 experiments use 8 experts trained for 500 epochs, with batch size 256.

## Key Experimental Results

### Main Results

**Cross-domain classification (ImageNet100 pretrained ResNet-50 → 11 downstream datasets)**:

| Method | CIFAR10 | CIFAR100 | Food | Flowers | Cars | Mean | Avg. Rank |
|--------|---------|----------|------|---------|------|------|-----------|
| SimCLR | 87.88 | 67.92 | 63.60 | 88.37 | 47.09 | 68.42 | 5.00 |
| EquiMod | 88.99 | 70.22 | 64.43 | 90.33 | 48.94 | 69.72 | 3.18 |
| **STAR-MMoE** | **90.09** | **72.31** | **67.05** | **91.45** | **51.54** | **71.23** | **1.18** |

STAR-MMoE ranks first on 10/11 datasets, with a mean improvement of 1.51% over EquiMod.

**STL10 pretrained ResNet-18 → downstream tasks**:

| Method | Mean | Avg. Rank |
|--------|------|-----------|
| SimCLR | 45.55 | 5.73 |
| EquiMod | 49.54 | 3.82 |
| **STAR-MMoE** | **53.07** | **1.36** |

STAR-MMoE achieves a mean improvement of 3.53% over EquiMod.

**Object detection (VOC07+12, Faster R-CNN, frozen ResNet-50-C4 backbone)**:

| Method | AP | AP50 | AP75 |
|--------|----|------|------|
| SimCLR | 47.96 | 76.35 | 51.62 |
| EquiMod | 48.52 | 76.55 | 52.82 |
| **STAR-MMoE** | **48.85** | **76.81** | **53.01** |

### Ablation Study

| No. of Experts | Mean Canonical Correlation | Mean Accuracy | Notes |
|----------------|---------------------------|---------------|-------|
| 2 (≈EquiMod) | ~0.55 | ~48.5% | Severe redundant feature learning |
| 4 | ~0.42 | ~50.5% | Reduced redundancy |
| 8 | ~0.35 | ~52.0% | Further improvement |
| 16 | ~0.30 | ~53.0% | Minimal redundancy, best performance |

**Equivariance evaluation**:

| Method | R-equiv. ↑ | P-equiv. ↓ |
|--------|------------|------------|
| SimCLR | 0.74 | 0.72 |
| EquiMod | 0.91 | 0.38 |
| **STAR-MMoE** | **0.98** | **0.27** |

STAR comprehensively outperforms all baselines on equivariance metrics.

### Key Findings

1. **Natural expert specialization**: Among 8 experts, Expert 1 is used equally by both routers (shared expert), Experts 2–6 primarily serve the invariant objective, and Experts 7–8 primarily serve the equivariant objective.
2. **Negative correlation between redundancy and generalization**: Lower canonical correlation (less redundancy) consistently corresponds to higher downstream accuracy.
3. **Faster convergence**: Experts in the MMoE projection converge faster than EquiMod's projection heads, providing higher-quality gradient signals.
4. **kNN retrieval using equivariant vs. invariant experts** confirms that each expert captures qualitatively different types of information.

## Highlights & Insights

1. **Diagnosis and resolution of redundant feature learning**: This work is the first to explicitly identify and quantify redundancy between invariant and equivariant projection heads using canonical correlation analysis.
2. **A new perspective on MoE for SSL pretraining**: By confining MoE to the projection heads and discarding them after training, the paper elegantly circumvents the transferability limitations of MoE models.
3. **Clear theoretical motivation**: The "crater illusion" example intuitively illustrates the intrinsic correlation between invariant and equivariant tasks.
4. **Consistent improvements**: Stable gains are demonstrated across classification, detection, and few-shot learning downstream tasks.

## Limitations & Future Work

- Only soft routing is applicable; sparse (top-k) routing cannot be used to improve efficiency and scalability, as batch normalization requires all experts to receive inputs in every batch.
- The number of experts requires hyperparameter search.
- Validation is currently limited to the SimCLR framework; compatibility with other SSL frameworks (BYOL, DINO, etc.) remains unexplored.
- Experiments are conducted only on ResNet-18/50; performance on larger models and ViTs is unknown.

## Related Work & Insights

- **EquiMod (2023)**: The direct baseline for STAR, using separate projection heads for invariant and equivariant objectives.
- **MMoE (2018)**: STAR draws on the multi-gate MoE paradigm for multi-task learning, but innovatively applies it to SSL projection heads.
- **V-MoE (2022) & Neural Experts (2024)**: Applications of MoE in vision; the key distinction of STAR is the discarding of the MoE structure after training.
- **Insight**: Applying multi-task learning tools (e.g., MoE) to the design of SSL projection heads is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The redundant feature learning perspective is novel, and applying MoE to SSL projection heads is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Classification / detection / few-shot learning + expert analysis + redundancy analysis + equivariance evaluation — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the analysis sections are substantive.
- Value: ⭐⭐⭐⭐ Meaningfully advances equivariant representation learning in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](../../ICML2026/self_supervised/scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)

</div>

<!-- RELATED:END -->
