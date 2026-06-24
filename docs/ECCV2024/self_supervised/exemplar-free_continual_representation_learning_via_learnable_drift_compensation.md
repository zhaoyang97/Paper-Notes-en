---
title: >-
  [Paper Note] Exemplar-Free Continual Representation Learning via Learnable Drift Compensation
description: >-
  [ECCV 2024][Self-Supervised Learning][Continual Learning] Proposes Learnable Drift Compensation (LDC), which trains a forward projector to map the old feature space to the new feature space. This effectively compensates for the semantic drift of class prototypes without needing to store old exemplars, achieving exemplar-free semi-supervised continual learning for the first time.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Continual Learning"
  - "Class Prototypes"
  - "Semantic Drift Compensation"
  - "Exemplar-Free"
  - "Semi-Supervised Learning"
date: 2026-05-08
content_hash: d66b2a9d03d43fc8
---

# Exemplar-Free Continual Representation Learning via Learnable Drift Compensation

**Conference**: ECCV 2024  
**arXiv**: [2407.08536](https://arxiv.org/abs/2407.08536)  
**Code**: [https://github.com/alviur/ldc](https://github.com/alviur/ldc)  
**Area**: Self-Supervised  
**Keywords**: Continual Learning, Class Prototypes, Semantic Drift Compensation, Exemplar-Free, Semi-Supervised Learning

## TL;DR

Proposes Learnable Drift Compensation (LDC), which trains a forward projector to map the old feature space to the new feature space. This effectively compensates for the semantic drift of class prototypes without needing to store old exemplars, achieving exemplar-free semi-supervised continual learning for the first time.

## Background & Motivation

Continual Learning requires models to sequentially learn multiple tasks on non-stationary data streams without forgetting old knowledge. Existing exemplar-free methods face a core challenge: when the backbone is updated on a new task, the feature representations of old classes undergo **semantic drift**, causing previously stored class prototypes to become invalid in the new feature space, which leads to catastrophic forgetting.

**Limitations of Prior Work**:
- Most exemplar-free methods are only evaluated under a "warm start" setting (i.e., starting from a large initial task or pre-trained model), thereby dodging the severe feature drift problem.
- Existing drift compensation methods like SDC assume that drift can be approximated by local translation, whereas actual drift may involve complex transformations like rotation and scaling.
- Methods that freeze the backbone (e.g., FeTrIL, FeCAM) suffer from insufficient representation capacity under cold start settings due to the small size of the initial task.

**Key Insight**: Through oracle experiments, the authors uncover a key insight: when using an oracle to perfectly compensate for prototype drift, performance can be largely restored. This indicates that the discriminative ability of the backbone does not degrade significantly due to incremental learning; forgetting is primarily caused by the shift in prototype positions. This finding directly motivates the approach: "rather than preventing forgetting, compensate for the drift."

**Core Idea**: Learn a simple forward projector using the current task's data to map prototypes from the old feature space to the new feature space without requiring labels or storing old data.

## Method

### Overall Architecture

LDC is applied after the training of each task. Given the old feature extractor $f_\theta^{t-1}$ and the new feature extractor $f_\theta^t$, a forward projector $p_F^t$ is learned to map the old feature space to the new feature space. Paired features are obtained by passing the current task data $D_t$ through both frozen extractors. After training the projector, all old class prototypes are updated. Finally, a Nearest Class Mean (NCM) classifier is used for inference.

### Key Designs

1. **Forward Projector**: The projector $p_F^t$ maps old features to the new feature space. During training, the MSE loss is minimized:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(p_F^t(f_\theta^{t-1}(x_i)) - f_\theta^t(x_i))^2$$

where both $f_\theta^t$ and $f_\theta^{t-1}$ are frozen. **Design Motivation**: Unlike the local translation assumption of SDC, LDC can capture arbitrary forms of drift (including rotation and scaling) through a learnable mapping. Empirical results show that a simple linear layer yields the best performance, suggesting that the mapping between feature spaces is approximately linear.

2. **Unlabeled Prototype Update**: After training the projector, all old class prototypes are updated via forward projection:

$$P_t^c = p_F^t(P_{t-1}^c)$$

**Key Features**: This process does not require class labels, utilizing only stored old prototypes and current task data. This naturally makes LDC applicable to unsupervised and semi-supervised settings.

3. **NCM Classifier**: A Nearest Class Mean classifier is used for inference, assigning test samples to the nearest class prototype:

$$y^* = \arg\min_{y=1,...,Y^t} \|f_\theta^t(x) - P_t^y\|$$

### Loss & Training

- **Supervised CL**: Can be combined with regularization methods like LwF, training the projector additionally at the end of each task. LwF utilizes a knowledge distillation loss: $\mathcal{L} = \mathcal{L}_{ce}(h_t(x), y) + \lambda \mathcal{L}_{ce}(h_{t-1}(x), h_t(x))$
- **Semi-Supervised CL**: Combined with self-supervised CL methods (PFR, CaSSLe, POCON), leveraging all unlabeled data to train the projector, while utilizing only a small amount of labels to compute the prototypes. This is the first exemplar-free semi-supervised continual learning method.
- The projector is trained using the Adam optimizer. In the supervised setting, it is trained for 20 epochs with a learning rate of lr=0.001. In the semi-supervised setting, it is trained for 100 epochs with a learning rate of lr=5e-3.

## Key Experimental Results

### Main Results (Supervised Setting)

| Dataset | Metric | LwF+LDC | EFC (Prev. SOTA) | LwF+SDC | Gain |
|--------|------|---------|----------------|---------|------|
| CIFAR-100 | $A_{last}$ | **45.4** | 43.6 | 40.6 | +1.8 vs EFC |
| CIFAR-100 | $A_{inc}$ | **59.5** | 58.6 | 56.2 | +0.9 vs EFC |
| Tiny-ImageNet | $A_{last}$ | **34.2** | 34.1 | 29.5 | +0.1 vs EFC |
| ImageNet100 | $A_{last}$ | **51.4** | 47.3 | 42.6 | +4.1 vs EFC |
| ImageNet100 | $A_{inc}$ | **69.4** | 59.9 | 65.3 | +9.5 vs EFC |
| Stanford Cars (ViT) | $A_{last}$ t10 | **62.9** | - | 53.3 | +9.6 vs SDC |

### Semi-Supervised Setting (CIFAR-100, 10 tasks)

| Method | Exemplar-Free | 0.8% Labels | 5% Labels | 25% Labels |
|------|--------|---------|--------|---------|
| CaSSLe+naive | ✓ | 19.4 | 23.2 | 23.6 |
| CaSSLe+SDC | ✓ | 22.1 | 25.8 | 26.5 |
| CaSSLe+LDC | ✓ | **27.0** | **32.8** | **35.0** |
| NNCSL (500 exemplars) | ✗ | 27.4 | 31.4 | 35.3 |

### Ablation Study

| Projector Architecture | LwF+LDC (CIFAR-100) | CaSSLe+LDC (25%) |
|-----------|---------------------|-------------------|
| Linear (w/o bias) | **45.4** | **35.0** |
| Linear + bias | 45.2 | 34.3 |
| Linear + ReLU | 41.2 | 29.1 |
| MLP (two layers) | 43.7 | 34.9 |

### Key Findings

1. **Forgetting primarily stems from prototype drift rather than feature degradation**: Oracle experiments show that perfect drift compensation recovers most of the performance, indicating that the discriminative power of the backbone is not significantly impaired.
2. **LDC significantly outperforms SDC under cold start**: On CIFAR-100, LDC achieves a 4.8% improvement over SDC, because SDC's local translation assumption fails in high-drift scenarios.
3. **LDC matches NME without requiring exemplars**: The performance of LDC is comparable to NME storing 20 exemplars per class.
4. **LDC matches exemplar-based methods for the first time in semi-supervised settings without needing exemplars**: Under 5% and 25% label settings, LDC performs on par with NNCSL.

## Highlights & Insights

- Highly valuable core insight: "forgetting $\neq$ feature degradation"; most forgetting can be recovered by correcting the prototype positions.
- Extremely simple method—requiring only a single linear layer for projection, with minimal training overhead (20 epochs).
- Unified framework: the same LDC method can be seamlessly applied to supervised, self-supervised, and semi-supervised CL.
- The first exemplar-free semi-supervised continual learning method, opening up a new research direction.

## Limitations & Future Work

- The projector is trained only on the current task's data, which may result in less precise updates for some old prototypes due to data bias (a gap still remains between the corrected prototypes and the oracle).
- As the number of tasks increases, the cumulative error of prototypes passing through multiple projections may gradually amplify.
- Only class-mean prototypes are stored, losing intra-class distribution information (such as covariance).
- Explored only standard architectures, leaving more complex backbone architectures (such as large-scale ViTs) and diverse task division methods uninvestigated.

## Related Work & Insights

- SDC [Yu et al.], which proposed the concept of drift compensation, is the direct predecessor of this work, but its translation assumption limits its applicability.
- FeCAM [Goswami et al.] uses Mahalanobis distance instead of Euclidean distance; the proposed LDC can be combined with it.
- MEA [Moon et al.] also learns a mapping but requires labels and a large amount of stored features, rendering it inapplicable to unsupervised settings.
- LDC could potentially be combined with adapter-based or prompt-based methods.

## Rating

- Novelty: ⭐⭐⭐⭐ Highly valuable core insight (forgetting $\neq$ degradation) and a clear, elegant methodological approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across three settings (supervised/self-supervised/semi-supervised) with multiple datasets and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Smooth logic flow from motivation analysis to method derivation, accompanied by clear illustrations.
- Value: ⭐⭐⭐⭐ Simple and highly practical paradigm that establishes a new path for exemplar-free semi-supervised CL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Revisiting Supervision for Continual Representation Learning](revisiting_supervision_for_continual_representation_learning.md)
- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)
- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/self_supervised/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[ACL 2025\] AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](../../ACL2025/self_supervised/analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)
- [\[CVPR 2026\] Dual-Estimator: Decoupling Global and Local Semantic Shift for Drift Compensation in Class-Incremental Learning](../../CVPR2026/self_supervised/dual-estimator_decoupling_global_and_local_semantic_shift_for_drift_compensation.md)

</div>

<!-- RELATED:END -->
