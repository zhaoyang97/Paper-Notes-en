---
title: >-
  [Paper Note] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning
description: >-
  [CVPR 2026][Self-Supervised Learning][Online Class-Incremental Learning] This paper proposes an online mixture model framework driven by optimal transport theory (MMOT), which maintains multiple adaptive centroids per class to capture the multimodal distribution of streaming data. Combined with a dynamic preservation strategy to mitigate catastrophic forgetting, the method substantially outperforms existing approaches in the OCIL setting.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Online Class-Incremental Learning
  - Optimal Transport
  - Gaussian Mixture Model
  - Catastrophic Forgetting
  - Latent Space Modeling
date: 2026-05-08
content_hash: 33896784b4f286a8
---

# An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning

**Conference**: CVPR 2026
**arXiv**: [2211.16780](https://arxiv.org/abs/2211.16780)
**Code**: None
**Area**: Continual Learning / Online Incremental Learning
**Keywords**: Online Class-Incremental Learning, Optimal Transport, Gaussian Mixture Model, Catastrophic Forgetting, Latent Space Modeling

## TL;DR

This paper proposes an online mixture model framework driven by optimal transport theory (MMOT), which maintains multiple adaptive centroids per class to capture the multimodal distribution of streaming data. Combined with a dynamic preservation strategy to mitigate catastrophic forgetting, the method substantially outperforms existing approaches in the OCIL setting.

## Background & Motivation

1. **Background**: Online Class-Incremental Learning (OCIL) is one of the most challenging continual learning scenarios — data distributions shift dynamically, the model can only be updated once per data batch, and no task ID is available at inference time. Existing methods typically employ a single classification head or a single centroid to represent each class.
2. **Limitations of Prior Work**: A single adaptive centroid fails to capture the multimodal nature of a class data stream (a class may comprise multiple clusters); while GMM-based methods use multiple centroids, these are fixed after training and not updated online.
3. **Key Challenge**: The backbone network continuously adapts to new data, causing feature drift, whereas fixed centroids cannot track this drift — a significant shift exists between latent representations at training and inference time.
4. **Goal**: Dynamically update multiple centroids during online learning while preserving inter-class separability and intra-class compactness.
5. **Key Insight**: Replace the conventional EM algorithm with OT theory, leveraging its continuity and geometric sensitivity. OT is differentiable, numerically stable, and respects the geometric structure of data.
6. **Core Idea**: Minimize the distance between the empirical distribution and a GMM using the entropy-regularized dual form of the Wasserstein distance; centroids are updated online via gradient descent, and Gumbel-Softmax enables differentiable sampling of mixture proportions.

## Method

### Overall Architecture

Given a batch of new-class data $X$ and buffered old-class data $\bar{X}$, latent representations are obtained via feature extractor $f_\theta$. The pipeline consists of three steps: (1) cross-entropy loss for initial class separation; (2) the MMOT framework to learn multiple adaptive centroids per class; (3) a dynamic preservation strategy to reinforce class discrimination. At inference, classification is performed using Mahalanobis distance.

### Key Designs

1. **MMOT (Multimodal Optimal Transport Framework)**:

    - **Function**: Online learning of multiple adaptive centroids and covariance matrices for each class.
    - **Mechanism**: For each class $c$, the empirical distribution is approximated by a GMM $\mathbb{Q}_c = \sum_{k=1}^K \pi_{k,c} \mathcal{N}(\mu_{k,c}, \text{diag}(\sigma_{k,c}^2))$. GMM parameters are learned by minimizing the entropy-regularized dual form of the Wasserstein distance. The reparameterization trick enables differentiable sampling, and Gumbel-Softmax allows differentiable sampling of mixture proportions. The expectation-form objective is naturally suited to online learning.
    - **Design Motivation**: The EM algorithm requires multiple iterations and is computationally expensive, making it unsuitable for online settings; KL divergence is unstable when distribution supports do not overlap. OT provides a continuously differentiable and numerically stable alternative.

2. **Dynamic Preservation Strategy**:

    - **Function**: Leverage the multi-centroid representations learned by MMOT to enhance class discrimination in representation learning.
    - **Mechanism**: A contrastive-style objective is employed, where the numerator encourages sample features to be close to all $K$ centroids of their own class, while the denominator pushes them away from centroids and features of other classes. Similarity values are scaled by a temperature parameter $\tau$.
    - **Design Motivation**: Centroids near class boundaries are particularly effective at enhancing inter-class separation; multiple centroids yield more informative representations.

3. **Memory Buffer Selection and Inference Strategy**:

    - **Function**: Use centroids to select representative samples for the memory buffer, and classify at inference using Mahalanobis distance.
    - **Mechanism**: For each centroid, the nearest samples are selected for the buffer; at inference, the minimum Mahalanobis distance from a test sample to all Gaussians of each class is used for classification.
    - **Design Motivation**: Centroid-based selection ensures diverse coverage of the class distribution in the buffer.

### Loss & Training

Per batch: cross-entropy loss for initial training → MMOT centroid update (alternating updates of the Kantorovich network and GMM parameters) → dynamic preservation strategy → buffer update.

## Key Experimental Results

### Main Results

| Dataset | Metric | OTC | BiC+AC (Prev. SOTA) | Gain |
|--------|------|-----|-------------------|------|
| CIFAR-10 (M=0.2k) | Avg Acc | 64.8 | 63.5 | +1.3 |
| CIFAR-100 (M=2k) | Avg Acc | 48.5 | 47.3 | +1.2 |
| CIFAR-100 (M=5k) | Avg Acc | 56.5 | 54.2 | +2.3 |
| Tiny-ImageNet (M=5k) | Avg Acc | 31.6 | 22.6 | +9.0 |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Centroid-based buffer selection (K=4) | 75.9% | MMOT centroid-selected samples |
| Random buffer selection (K=4) | 73.4% | Centroid selection consistently outperforms random |
| Number of centroids = 1 | 71.6% | Single centroid yields lowest performance |
| Number of centroids = 4 | 75.9% | Optimal centroid count (M=1k) |

### Key Findings

- The advantage of OTC is most pronounced when the buffer is smallest, with a lead of up to 9 percentage points on Tiny-ImageNet.
- CoPE's low forgetting metric is attributable to low initial accuracy rather than effective representation learning.
- Increasing the number of centroids consistently improves performance within a reasonable range; excessive centroids cannot be adequately supported by the buffer.

## Highlights & Insights

- **Elegant substitution of EM with OT**: This is the first work to apply OT to GMM learning in OCIL; the expectation-form objective is naturally suited to online scenarios.
- **Unified framework**: Centroids are jointly used for training (dynamic preservation), inference (Mahalanobis distance), and buffer selection, creating tight coupling across all three components.
- **Transferability**: The multi-centroid + OT paradigm is transferable to other tasks requiring online representation learning.

## Limitations & Future Work

- Validation is limited to relatively small datasets; large-scale experiments (e.g., ImageNet-1k) are absent.
- The number of centroids $K$ is a fixed hyperparameter, whereas different classes may require different numbers.
- The Kantorovich network introduces additional computational overhead, and a detailed cost analysis is lacking.

## Related Work & Insights

- **vs. OnPro**: OnPro employs a single adaptive centroid per class; OTC extends this to multiple centroids and updates them online using OT.
- **vs. MOSE**: MOSE uses fixed GMM centroids; in OTC, centroids are continuously updated via gradient descent.
- **vs. BiC+AC**: BiC+AC mitigates forgetting through bias correction and does not address the multimodal structure of the latent space.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of OT to GMM learning in OCIL, with rigorous theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐ — Dataset scale is limited; ablation is sufficient but large-scale validation is missing.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, mathematical derivations are complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Introduces a new theoretical tool for online incremental learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)

</div>

<!-- RELATED:END -->
