---
title: >-
  [Paper Note] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations
description: >-
  [ICCV 2025][Robotics][Multimodal Domain Generalization] This paper proposes URMMDG, a framework that constructs a cross-modal unified representation space via supervised contrastive learning and decouples class-generic i…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Multimodal Domain Generalization"
  - "Unified Representation"
  - "Supervised Contrastive Learning"
  - "Information Decoupling"
  - "Mixup"
date: 2026-05-08
content_hash: 18e609fe8ccca26b
---

# Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations

**Conference**: ICCV 2025
**arXiv**: [2507.03304](https://arxiv.org/abs/2507.03304)
**Code**: N/A
**Area**: Robotics / Multimodal Learning
**Keywords**: Multimodal Domain Generalization, Unified Representation, Supervised Contrastive Learning, Information Decoupling, Mixup

## TL;DR

This paper proposes URMMDG, a framework that constructs a cross-modal unified representation space via supervised contrastive learning and decouples class-generic information from modality/domain-specific information through mutual information minimization. This enables effective transfer of classical single-modal domain generalization methods (Mixup, JiGen, IBN-Net) to multimodal domain generalization (MMDG) settings, achieving state-of-the-art performance on the EPIC-Kitchens and HAC benchmarks.

## Background & Motivation

Domain generalization (DG) aims to train models on source domains that remain robust on unseen target domains. Existing DG methods—spanning data augmentation, learning strategies, and representation learning—achieve notable success on unimodal data, yet tend to perform poorly when directly transferred to MMDG settings.

The core challenge lies in **modality asynchrony**: the data distributions of different modalities (video, audio, optical flow) differ substantially. For instance, when Mixup is applied to the same pair of classes, the interpolated video representation may be semantically closer to "running," while the interpolated optical flow representation may lean toward "eating"—resulting in inconsistent generalization directions across modalities and causing joint multimodal training to underperform independent single-modal training.

The authors quantify this problem experimentally: JiGen improves performance by 2.87% on a single modality (Video), but yields only a 1.61% gain under three-modality joint training. This demonstrates that **inter-modal intrinsic discrepancies limit the direct transfer of DG methods to MMDG**.

## Method

### Overall Architecture

The URMMDG framework proceeds in two stages: (1) constructing a unified representation space via supervised contrastive learning and information decoupling; and (2) applying DG methods (Mixup/JiGen/IBN-Net) within the unified representation space to achieve synchronized cross-modal augmentation.

### Key Designs

1. **Supervised Contrastive Decoupling**:

    - For each modality $m$, two separate encoders extract generic information $\mathbf{z}_i^m = \Phi^m(\mathbf{x}_i^m)$ and specific information $\bar{\mathbf{z}}_i^m = \Psi^m(\mathbf{x}_i^m)$.
    - Generic information captures class-level semantics shared across modalities and domains; specific information captures domain/modality-distinctive features.
    - A multimodal supervised contrastive loss $\mathcal{L}_{scl}$ pulls together the generic representations of same-class samples across modalities:
    $\mathcal{L}_{scl} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$
    - **Design Motivation**: Construct a modality-agnostic unified semantic space so that DG methods can operate on all modalities synchronously within that space.

2. **Mutual Information Minimization**:

    - The CLUB estimator is used to minimize the upper bound of mutual information between generic information $\mathbf{z}_i^m$ and specific information $\bar{\mathbf{z}}_i^m$:
    $L_{club} = \frac{1}{N} \sum_{i=1}^{N} [\log q_\theta(\bar{\mathbf{z}}_i^m | \mathbf{z}_i^m) - \frac{1}{N} \sum_{j=1}^{N} \log q_\theta(\bar{\mathbf{z}}_j^m | \mathbf{z}_i^m)]$
    - A reconstruction loss $L_{rec} = \|\mathbf{x}_i^m - D(\mathbf{z}_i^m; \bar{\mathbf{z}}_i^m)\|_2^2$ is also introduced to preserve information completeness after decoupling.
    - **Design Motivation**: Ensure that the generic representation contains only class-level semantics, free from domain/modality noise.

3. **Transfer of DG Methods onto Unified Representations**:

    - **UR-Mixup**: Mixup is applied to the generic representations $\mathbf{z}^m$; the augmented samples are concatenated with specific information and passed through a decoder to reconstruct features for classification training.
    - **UR-JiGen**: The generic representation is partitioned into patches, which are **randomly sampled across modalities** and concatenated before being shuffled, forming a cross-modal jigsaw task.
    - **UR-IBN**: IBN-a normalization (half Instance Normalization + half Batch Normalization) is applied directly to the unified representations.
    - **Design Motivation**: Operating within the unified space ensures synchronized augmentation across all modalities, avoiding the divergent generalization directions caused by independently augmenting each modality.

### Loss & Training

The total loss is a weighted combination of multiple terms:
$$L = \alpha_1 L_{cls} + \alpha_2 L_{scl} + \alpha_3 L_{club} + \alpha_4 L_{rec}$$
UR-JiGen additionally incorporates $L_{jig}$ (with weight set to 1).

## Key Experimental Results

### Main Results (EPIC-Kitchens, Video + Audio + Flow)

| Method | D2,D3→D1 | D1,D3→D2 | D1,D2→D3 | Avg. |
|--------|----------|----------|----------|------|
| Base(VAF) | 54.71 | 67.20 | 61.70 | 61.20 |
| SimMMDG | 62.08 | 66.13 | 64.40 | 64.20 |
| CMRF | 61.84 | 70.13 | 70.12 | 67.36 |
| Mixup(VAF) | 57.95 | 67.95 | 64.37 | 63.42 |
| **UR-Mixup** | **61.72** | **70.89** | **70.76** | **67.79** |
| **UR-JiGen** | 62.20 | 71.14 | 67.78 | 67.04 |

### Ablation Study (Validating the Value of Unified Representations)

| Configuration | Video | Audio | Flow | V-A-F | Notes |
|---------------|-------|-------|------|-------|-------|
| Base(V) single-modal | 58.73 | - | - | - | Single-modal baseline |
| Base(VAF) multimodal | 57.13 | 37.96 | 56.65 | 61.20 | Per-modality performance drops under joint training |
| JiGen(V) | 61.60 | - | - | - | Single-modal DG gain: +2.87 |
| JiGen(VAF) | 59.23 | 39.58 | 57.18 | 62.81 | Multimodal DG gain: only +1.61 |
| UR-Mixup(VA) | 56.99 | 68.85 | - | 64.77 | Two-modality unified representation |
| UR-Mixup(VF) | 64.85 | - | 68.84 | 66.42 | Two-modality unified representation |
| UR-Mixup(VAF) | 61.72 | 70.89 | 70.76 | 67.79 | Three-modality unified representation (best) |

### Key Findings

- Under joint multimodal training, individual modality performance degrades relative to independent training (modality competition); the unified representation effectively alleviates this issue.
- Both UR-Mixup and UR-JiGen substantially outperform their counterparts applied independently per modality.
- The proposed approach essentially reformulates the MMDG problem as a single-modal DG problem in a unified representation space, making it more tractable.
- On the HAC dataset, UR-Mixup achieves 73.40% average accuracy, surpassing CMRF's 72.44%.

## Highlights & Insights

- **Clear problem formulation**: Table 1 precisely quantifies the phenomenon that DG methods suffer performance degradation when directly transferred to MMDG settings.
- **Methodological contribution outweighs technical novelty**: The paper proposes a general paradigm—first construct a unified representation, then apply any DG method within it—which offers strong extensibility.
- The cross-modal jigsaw design in UR-JiGen is notably elegant: randomly sampling patches from different modalities and reassembling them integrates multimodal information while preserving the difficulty of the self-supervised task.

## Limitations & Future Work

- Validation is limited to the video + audio + optical flow triplet; experiments on more common multimodal combinations such as image + text are absent.
- The quality of the unified representation is highly dependent on the effectiveness of contrastive learning and may degrade when modality discrepancies are extreme.
- Comparisons against large-scale pretrained multimodal models (e.g., CLIP) are lacking.
- Sensitivity analysis of the hyperparameters ($\alpha_1$ through $\alpha_4$) is insufficiently thorough.

## Related Work & Insights

- SimMMDG and CMRF are prior works in MMDG but do not explicitly propose the paradigm of "bridging via unified representations."
- The CLUB mutual information upper bound estimator originates from variational inference; its application here for decoupling generic and specific information is a well-motivated choice.
- This approach can inspire the transfer of other mature unimodal techniques—such as domain adaptation and data augmentation strategies—to multimodal settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically bridge DG methods to MMDG via unified representations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple DG methods × multiple modality combinations; well-structured experimental design.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; Figure 1 is intuitive.
- Value: ⭐⭐⭐⭐ Provides a general solution paradigm for MMDG with sound practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](../../NeurIPS2025/robotics/pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)

</div>

<!-- RELATED:END -->
