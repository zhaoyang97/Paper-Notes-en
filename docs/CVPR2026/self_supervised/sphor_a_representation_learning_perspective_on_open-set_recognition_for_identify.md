---
title: >-
  [Paper Note] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks
description: >-
  [CVPR 2026][Self-Supervised Learning][Open-set Recognition] This paper proposes SpHOR, a two-stage decoupled training framework for open-set recognition (OSR) that explicitly shapes the feature space via spherical repres…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Open-set Recognition"
  - "Representation Learning"
  - "Spherical Representation"
  - "von Mises-Fisher Distribution"
  - "Orthogonality Constraint"
date: 2026-05-08
content_hash: 1a4e90e8700a0070
---

# SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks

**Conference**: CVPR 2026 Findings  
**arXiv**: [2503.08049](https://arxiv.org/abs/2503.08049)  
**Code**: None  
**Area**: Self-supervised Learning
**Keywords**: Open-set Recognition, Representation Learning, Spherical Representation, von Mises-Fisher Distribution, Orthogonality Constraint

## TL;DR

This paper proposes SpHOR, a two-stage decoupled training framework for open-set recognition (OSR) that explicitly shapes the feature space via spherical representation learning (vMF distributions), orthogonal label embeddings, and integrated Mixup/Label Smoothing, achieving up to 5.1% OSCR improvement on the Semantic Shift Benchmark.

## Background & Motivation

**Background**: Deep neural networks are widely deployed in safety-critical applications, yet conventional closed-set classifiers assume all test-time classes were seen during training. OSR requires models to correctly flag unknown-class samples as "unknown" while maintaining high accuracy on known classes.

**Limitations of Prior Work**: Most OSR methods jointly train the feature extractor and classifier end-to-end, causing feature representations to adapt to unknown classes only implicitly. Vaze et al. showed that simple closed-set training strategies can outperform many dedicated OSR methods, indicating that classifier-level improvements are approaching a ceiling. Existing methods leveraging contrastive learning (e.g., SupCon) are also not specifically designed for OSR.

**Key Challenge**: Feature magnitudes in Euclidean space can grow without bound, leaving the open space unbounded and greatly increasing the risk of misclassifying unknown samples. Additionally, shared inter-class features (e.g., background textures) trap models in the *Familiarity Trap*, causing semantically related but unknown classes to be confidently misidentified as known ones.

**Goal**: Can OSR be improved by explicitly designing the feature representation itself—rather than relying on classifier-level improvements? How can class separation in feature space be made cleaner and open-space modeling more effective?

**Key Insight**: A two-stage decoupled training strategy is adopted—first learning feature representations, then training the classifier—with spherical constraints, orthogonal label embeddings, and data augmentation strategies introduced during the representation learning stage.

**Core Idea**: Features are projected onto a hypersphere and modeled as a mixture of vMF distributions; orthogonal label embeddings enforce inter-class subspace orthogonality; Mixup simulates semantically ambiguous samples to improve open-space modeling.

## Method

### Overall Architecture

SpHOR employs a two-stage decoupled training pipeline:

- **Stage 1 (Representation Learning)**: An encoder paired with a projection network extracts L2-normalized spherical features. Class-specific representations are learned via a vMF Alignment Loss combined with an Orthogonality Regularizer. Training data is augmented through RandAugment → Label Smoothing → Mixup.
- **Stage 2 (Classifier Training)**: The encoder is frozen and only a linear classifier is trained using standard cross-entropy loss.

### Key Designs

1. **Spherical Representation and vMF Alignment Loss**: Features are L2-normalized onto a hypersphere, and each class is modeled as a vMF distribution. The loss is defined as $\mathcal{L}_{\text{vMFAL}} = -\frac{1}{N}\sum_i \sum_k S_{ik} \log P_{ik}$, where $S_{ik}$ denotes label similarity and $P_{ik}$ denotes classification probability based on cosine similarity. Theoretical analysis shows this loss simultaneously promotes Alignment (pulling same-class representations toward their label embeddings) and Uniformity (dispersing representations across classes). For Mixup-generated ambiguous samples, the uniformity term dominates, pushing them away from class centers and effectively mitigating the Familiarity Trap.

2. **Orthogonality Regularization $\mathcal{R}_{\text{Ortho}}$**: Label embeddings are constrained to be mutually orthogonal, ensuring each class occupies an independent linear subspace and preventing embedding collapse. The formulation is $\log \frac{1}{|C|^2-|C|}\sum_{j\neq i}\exp(\frac{1}{\tau}(\mu_j \cdot \mu_i)^2)$, which is more stable than SVD- or Equiangular Tight Frame-based approaches and avoids negative correlation issues.

3. **Mixup and Label Smoothing Integrated into Representation Learning**: Unlike applying these techniques at the classifier stage, SpHOR incorporates Mixup and Label Smoothing directly into the representation learning stage. Mixup generates semantically ambiguous samples to simulate unknown classes, while Label Smoothing reduces overfitting. Two new metrics—Angular Separability (AS) and Norm Separability (NS)—are proposed to quantify the improvements these techniques bring to feature representations.

### Loss & Training

- Total training loss: $\mathcal{L} = \mathcal{L}_{\text{vMFAL}} + \mathcal{R}_{\text{Ortho}}$
- Stage 2 uses standard cross-entropy loss
- OSR scoring supports four post-processing strategies: MaxLogit, PostMax, KNN, and NNGuide
- Hyperparameters: Stage 1 uses a 1024-dimensional linear projection network; Stage 2 uses a linear classifier
- Hardware: 40 GB NVIDIA A100 GPU

## Key Experimental Results

### Main Results: Semantic Shift Benchmark (ImageNet-pretrained ResNet-50)

| Method | CUB Acc↑ | CUB AUROC(Easy/Hard)↑ | CUB OSCR(Easy/Hard)↑ | SCars Acc↑ | SCars AUROC(E/H)↑ | Aircraft Acc↑ | Aircraft AUROC(E/H)↑ |
|--------|----------|----------------------|---------------------|-----------|-------------------|--------------|---------------------|
| ARPL+ | 85.4 | 81.8/73.9 | 73.1/66.9 | 89.8 | 85.0/76.4 | 83.3 | 85.8/74.6 |
| MLS+Mixup+MaxLogit | 88.3 | 86.2/78.0 | 78.6/72.1 | 91.4 | 87.3/82.4 | 81.3 | 87.3/75.3 |
| SupCON+KNN | 78.2 | 88.6/75.3 | 72.8/63.1 | 91.8 | 92.1/81.2 | 88.9 | 89.9/81.4 |
| **SpHOR+MaxLogit** | **90.8** | **91.7/83.3** | **85.7/79.0** | **96.3** | **94.1/83.1** | **90.6** | **91.5/81.1** |

### Ablation Study: Legacy CNN-32 Benchmark A (AUROC)

| Method | SVHN | CIFAR10 | CIFAR+10 | CIFAR+50 | TIN |
|--------|------|---------|----------|----------|-----|
| ARPL | 95.3 | 91.0 | 97.1 | 95.1 | 78.2 |
| MLS | 97.1 | 93.6 | 97.9 | 96.5 | 83.0 |
| ConOSR | 99.1 | 94.2 | 98.1 | 97.3 | 80.9 |
| SpHOR (w/o $\mathcal{R}_{\text{Ortho}}$) | 98.9 | 94.2 | 98.0 | 96.9 | 83.8 |
| **SpHOR** | **99.1** | **94.5** | **98.2** | **97.2** | **84.1** |

### Key Findings

- SpHOR achieves comprehensive state-of-the-art results on the Semantic Shift Benchmark, with closed-set accuracy reaching up to 96.3% (SCars) while leading on both AUROC and OSCR.
- Without ImageNet pretraining, SpHOR still demonstrates significant advantages over MLS+Mixup (e.g., OSCR on CUB improves from 45.7/42.6 to 76.7/70.0).
- The orthogonality regularization $\mathcal{R}_{\text{Ortho}}$ contributes approximately 1–2% improvement and is more pronounced in the absence of pretraining.
- MaxLogit is the best-matched scoring rule for SpHOR.

## Highlights & Insights

- **Theoretical Elegance**: The alignment and uniformity properties are unified under a vMF distribution framework, with rigorous theoretical derivations clearly explaining why Mixup samples are pushed away from class centers.
- **Effectiveness of Decoupled Training**: The results demonstrate that explicitly designing the feature space is more effective than relying on classifier-level training, offering a new paradigm for OSR.
- **New Evaluation Metrics**: Angular Separability and Norm Separability provide objective tools for quantifying OSR-friendly feature representations.
- **Orthogonal Constraints Outperform Margin-based Methods**: No additional hyperparameters are introduced, and negative correlation issues are avoided.

## Limitations & Future Work

- Validation is limited to ResNet-50 and CNN-32; experiments on modern architectures such as ViT are absent.
- Mixup-generated ambiguous samples still differ from real unknown classes; more sophisticated synthesis strategies could be explored.
- The two-stage training introduces additional overhead; an end-to-end approach may be more practical.
- The orthogonality constraint may degrade when the number of classes greatly exceeds the feature dimensionality.

## Related Work & Insights

- **Relation to SupCon**: SupCon applies general contrastive learning, whereas SpHOR tailors label embeddings and spherical constraints specifically for OSR.
- **Relation to ARPL**: ARPL relies on reciprocal point learning, a prototype-based method constrained to Euclidean space; SpHOR operates on the hypersphere, which is better suited for OSR.
- **Insights**: Decoupled training combined with explicit feature space design is a promising general paradigm extensible to OOD detection, continual learning, and related areas.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of vMF distributions, orthogonal embeddings, and Mixup integrated into the representation learning stage is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both the fine-grained SSB and the legacy CNN-32 benchmarks with sufficient ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous, motivation is clear, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Provides both a theoretical foundation and a practical solution for feature learning in OSR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Maximizing Asynchronicity in Event-based Neural Networks](../../ICLR2026/self_supervised/maximizing_asynchronicity_in_event-based_neural_networks.md)
- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](../../AAAI2026/self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)
- [\[CVPR 2026\] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)

</div>

<!-- RELATED:END -->
