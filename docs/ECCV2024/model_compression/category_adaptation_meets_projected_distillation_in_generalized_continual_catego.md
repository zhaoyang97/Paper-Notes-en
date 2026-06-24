---
title: >-
  [Paper Note] Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery
description: >-
  [ECCV2024][Model Compression][Generalized Continual Category Discovery] Proposes the CAMP method, which significantly improves the balance between learning new categories and retaining old knowledge in Generalized Continual Category Discovery (GCCD) scenarios through the cooperative combination of learnable projector distillation and category prototype adaptation networks.
tags:
  - "ECCV2024"
  - "Model Compression"
  - "Generalized Continual Category Discovery"
  - "knowledge distillation"
  - "continual learning"
  - "Category Adaptation"
  - "Projected Distillation"
date: 2026-05-08
content_hash: 864b7bc148e87a84
---

# Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery

**Conference**: ECCV2024  
**arXiv**: [2308.12112](https://arxiv.org/abs/2308.12112)  
**Code**: [GitHub](https://github.com/grypesc/CAMP)  
**Area**: Model Compression  
**Keywords**: Generalized Continual Category Discovery, knowledge distillation, continual learning, Category Adaptation, Projected Distillation

## TL;DR

Proposes the CAMP method, which significantly improves the balance between learning new categories and retaining old knowledge in Generalized Continual Category Discovery (GCCD) scenarios through the cooperative combination of learnable projector distillation and category prototype adaptation networks.

## Background & Motivation

Generalized Continual Category Discovery (GCCD) is a practical scenario combining Continual Learning and Generalized Category Discovery: the model must sequentially learn from partially labeled datasets, discovering new categories while maintaining recognition capability for old ones.

Existing methods widely adopt **Feature Distillation** (FD) to mitigate catastrophic forgetting. However, the authors point out a fundamental contradiction in FD:

- FD prevents forgetting by constraining the drift of category distributions in the feature space.
- But this rigid constraint severely degrades the **plasticity** of the model, making it difficult to distinguish new categories effectively.
- Existing category adaptation methods (such as SDC, Feature Adaptation) show limited performance under FD, as the drift patterns caused by FD are difficult to predict.

This motivates the researchers to investigate: Can a method be developed to allow the distribution of old categories to drift freely to enhance plasticity, while accurately predicting and compensating for this drift to prevent forgetting?

## Core Problem

1. **Plasticity-Stability Dilemma**: Although feature distillation reduces forgetting, it restricts the model's capacity to learn new categories.
2. **Unpredictable Drift**: Under standard FD, the drift of old categories in the feature space is irregular, making it difficult for adaptation networks to predict accurately.
3. **Exemplar-Free Setting**: Most methods rely on storing exemplar data from old categories, which is highly expensive in real-world scenarios.

## Method

CAMP (Category Adaptation Meets Projected distillation) consists of three training phases:

### Phase 1: Feature Extractor Training

The loss function consists of three components:

1. **Self-Supervised Learning Loss** $\mathcal{L}_{SSL}$: SimCLR contrastive learning loss + pseudo-label cross-entropy loss on all data (including unlabeled data).
2. **Supervised Learning Loss** $\mathcal{L}_{SL}$: SupCon supervised contrastive loss + cross-entropy loss on labeled data.
3. **Projected Distillation Loss** $\mathcal{L}_{KD}$: Maps current features back to the old feature space using a learnable MLP projector $\phi^{t \to t-1}$.

The key innovation of projected distillation lies in not directly constraining $\mathcal{F}^t(x) \approx \mathcal{F}^{t-1}(x)$, but instead learning a projector such that:

$$\mathcal{L}_{KD} = \sum_{i \in B} \| \phi^{t \to t-1}(\mathcal{F}^t(x_i)) - \mathcal{F}^{t-1}(x_i) \|^2$$

In this way, the model's feature space can evolve freely, only requiring the existence of a learnable mapping connecting the new and old spaces. The final loss is:

$$\mathcal{L} = (1-\alpha)((1-\beta)\mathcal{L}_{SSL} + \beta\mathcal{L}_{SL}) + \alpha\mathcal{L}_{KD}$$

### Phase 2: Semi-Supervised Clustering

Performs semi-supervised K-Means clustering on the current task data:
- Initializes the prototypes of known categories using labeled data.
- Discovers new category prototypes from unlabeled data using K-Means++.
- Estimates the number of categories using the elbow method.

### Phase 3: Category Prototype Adaptation

Trains an auxiliary adaptation network $\psi^{(t-1) \to t}$ to predict the drift of old category prototypes from the old space to the new space:

$$\mathcal{L}_{PA} = \sum_{i \in B} \| \mathcal{F}^t(x_i) - \psi^{(t-1) \to t}(\mathcal{F}^{t-1}(x_i)) \|^2$$

After training, old category prototypes are updated via: $p_i^t = \psi^{(t-1) \to t}(p_i^{t-1})$

**Key Insight**: Projected distillation makes the drift of old categories **regular and predictable**, which is key to how the two components of CAMP work synergistically. Using projected distillation alone increases forgetting (by permitting more drift), and using category adaptation alone has limited effect under standard FD. However, combining them significantly boosts performance.

### Network Architecture Choices

- Distillation projector $\phi$: 2-layer MLP (384-dim + ReLU)
- Adaptation network $\psi$: Linear layer (384-dim)
- Feature extractor: ViT-Small (DINO pre-trained, freezing the first 11 blocks)

## Key Experimental Results

### GCCD Settings (5 datasets, exemplar-free)

| Dataset | CAMP (All) | Second Place (All) | Gain |
|--------|-----------|-------------|------|
| CIFAR100 | **52.1%** | GCD+FD 36.6% | +15.5% |
| Stanford Cars | **48.8%** | GCD+EWC 30.4% | +18.4% |
| CUB200 | **58.9%** | GCD+EWC 50.3% | +8.6% |
| FGVCAircraft | **39.9%** | MetaGCD 28.6% | +11.3% |
| DomainNet | **36.7%** | SimGCD 36.5% | +0.2% |

### Exemplar-Free Class Incremental Learning

| Dataset | CAMP | Second Place (FeTrIL) | Gain |
|--------|------|----------------|------|
| CIFAR100 (5 tasks) | **65.0%** | 58.5% | +6.5% |
| CIFAR100 (10 tasks) | **56.7%** | 46.3% | +10.4% |
| ImageNet-Subset (5 tasks) | **73.1%** | 63.6% | +9.5% |

### Ablation Study (CUB200)

- Full CAMP: Known 62.6% / Novel 44.2%
- Without projected distillation: Performance drops significantly
- Without category adaptation: Performance decreases noticeably
- Without self-supervised loss: Known +2.0% but Novel -2.8%

### Combined Analysis of Projector and Adapter

- Using MLP projector only (without adaptation): Improves by 4.7% over the baseline.
- MLP projector + linear adapter: Improves by **20.2%** over the baseline.
- This demonstrates that the synergistic effect of the two components far exceeds the sum of their individual contributions.

## Highlights & Insights

1. **Synergistic effect insight**: Projected distillation and category adaptation show mediocre or even counterproductive performance when used in isolation, but yield a significant synergy when combined—which is the most core contribution of the paper.
2. **Exemplar-free**: CAMP outperforms many exemplar-based methods in an exemplar-free setting, significantly reducing memory overhead.
3. **Visual intuition**: Clearly demonstrates why projected distillation makes the drift more predictable through a 2D bottleneck experiment.
4. **Broad applicability**: The method is applicable to both GCCD and traditional CIL settings.
5. **Simple design**: Only needs to store one prototype vector per category, which is far more efficient than storing features or exemplars.

## Limitations & Future Work

1. **Poor accuracy of new classes on DomainNet**: GCD+FD achieves a novel accuracy of 39.7% on DomainNet, far exceeding CAMP's 29.7%, indicating that projected distillation may over-relax constraints in scenarios with large domain shifts.
2. **Dependence of category number estimation on the elbow method**: This is a heuristic method and may be inaccurate when category numbers vary widely.
3. **Task boundary assumption**: The method assumes explicit task transition boundaries, whereas real-world data streams are often continuous.
4. **Cumulative error of the adaptation network**: In multi-task sequences, prototype adaptation may accumulate errors over time ($\psi^{1\to2} \to \psi^{2\to3} \to \ldots$).
5. **Validated only on ViT-Small**: Generalizability of the method to larger models or different architectures has not been sufficiently explored.

## Related Work & Insights

| Method | Core Strategy | Needs Exemplars | GCCD Support |
|------|---------|---------|----------|
| GCD+FD | Feature distillation | No | Yes |
| PA | Proxy Anchor + FD + Exemplars | Yes | Yes |
| IGCD | Density support set + Replay | Yes | Yes |
| MetaGCD | Meta-learning + Task-1 data | Yes (Task-1) | Yes |
| GM | Dual model merger | No | Yes |
| FeTrIL | Frozen features + Classifier training | No | CIL Only |
| **CAMP** | **Projected distillation + Category adaptation** | **No** | **Yes** |

Compared with the most related category adaptation works: SDC estimates drift using a vector field but does not involve projected distillation; Feature Adaptation Network requires storing multiple features per class and needs exemplars. CAMP only stores one prototype vector per class, making it much more efficient.

## Insights & Connections

1. **The idea of "allowing drift to happen but making it predictable"** can be generalized to other continual learning scenarios: instead of restricting model updates, it is better to learn the mapping of changes.
2. **The concept of projected distillation** originates from transferability studies in representation learning (e.g., BYOL, SimSiam), indicating that self-supervised techniques can feed back into continual learning.
3. The category adaptation network essentially learns **affine transformations between feature spaces**, which shares a similar spirit with feature alignment in domain adaptation.
4. From a model-compression perspective, this method demonstrates how to achieve efficient knowledge transfer via auxiliary networks without expanding model capacity.

## Rating

- Novelty: ⭐⭐⭐⭐ (The synergistic combination of projected distillation and category adaptation is novel, though individual components have precedents)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 GCCD datasets + 3 CIL datasets + detailed ablation + visualization analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, intuitive 2D visualization, but some notations are lengthy)
- Value: ⭐⭐⭐⭐ (Comprehensive lead in GCCD and exemplar-free CIL, the underlying ideas are inspiring for the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery](../../CVPR2026/model_compression/talon_test-time_adaptive_learning_for_on-the-fly_category_discovery.md)
- [\[ECCV 2024\] Anytime Continual Learning for Open Vocabulary Classification](anytime_continual_learning_for_open_vocabulary_classification.md)
- [\[ECCV 2024\] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)
- [\[ECCV 2024\] Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)

</div>

<!-- RELATED:END -->
