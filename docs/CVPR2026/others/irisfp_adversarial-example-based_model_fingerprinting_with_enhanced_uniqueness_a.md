---
title: >-
  [Paper Note] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness
description: >-
  [CVPR 2026][model fingerprinting] This paper proposes IrisFP, a model fingerprinting framework that simultaneously enhances fingerprint uniqueness and robustness through three innovations: placing fingerprints at the int…
tags:
  - "CVPR 2026"
  - "model fingerprinting"
  - "adversarial examples"
  - "intellectual property protection"
  - "ownership verification"
  - "decision boundary"
date: 2026-05-08
content_hash: 4c1aca3c517f184e
---

# IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness

**Conference**: CVPR 2026
**arXiv**: [2603.24996](https://arxiv.org/abs/2603.24996)  
**Code**: None  
**Area**: Other
**Keywords**: model fingerprinting, adversarial examples, intellectual property protection, ownership verification, decision boundary

## TL;DR

This paper proposes IrisFP, a model fingerprinting framework that simultaneously enhances fingerprint uniqueness and robustness through three innovations: placing fingerprints at the intersection of multi-class decision boundaries, constructing composite sample fingerprints, and performing statistically-guided fingerprint selection. IrisFP consistently achieves higher AUC than state-of-the-art methods across 5 datasets.

## Background & Motivation

Adversarial-example-based model fingerprinting techniques elicit model-specific response behaviors by adding imperceptible perturbations to clean inputs, serving as a mechanism for DNN intellectual property protection and ownership verification. Existing methods face a **fundamental tension between uniqueness and robustness**:

- **Uniqueness**: Fingerprints must lie near decision boundaries to capture model-specific behavior, but existing methods target only a single boundary, resulting in insufficient discriminative power.
- **Robustness**: Model modification attacks (fine-tuning, pruning, adversarial training, etc.) shift decision boundaries and invalidate fingerprints. To improve robustness, prior methods place fingerprints deep within the target class region, which in turn compromises uniqueness.

Root Cause: Existing methods achieve either weak uniqueness or weak robustness, but not both simultaneously.

The key insight of this paper is that samples located at the **intersection of multi-class decision boundaries** exhibit a larger predicted margin — i.e., high confidence for the target class while being close to all other class boundaries. This simultaneously preserves model sensitivity (uniqueness) and increases predicted margin (robustness), without requiring fingerprints to be embedded deep in the target class region.

## Method

### Overall Architecture

IrisFP consists of two main pipelines:
1. **Fingerprint Generation**: Three phases — fingerprint seed initialization → composite sample fingerprint generation → fingerprint set selection.
2. **Ownership Verification**: Two steps — ownership matching → decision aggregation.

### Key Designs

1. **Multi-boundary Intersection Fingerprint Seed Initialization (Phase I)**:
    - Function: Places fingerprints at the intersection of all decision boundaries of the protected model.
    - Mechanism: For each input $x_i^0$, a probability distribution $p_i$ biased toward target class $\hat{y}_i^0$ is defined, where the target class probability is $\frac{1}{C}+\tau$ and the remaining probability is distributed uniformly among all other classes. The loss $\mathcal{L}_{phase1} = KL(f_o(\hat{x}_i^0) || p_i) + \lambda_1\|\delta_i^0\|_1$ is minimized to align the model output distribution with this biased distribution.
    - Design Motivation: Unlike conventional methods that push fingerprints toward a single boundary, this strategy causes fingerprints to simultaneously approach all boundaries. The parameter $\tau$ controls the degree of bias; a smaller $\tau$ places fingerprints closer to the intersection center, thereby increasing the predicted margin.

2. **Composite Sample Fingerprint Generation (Phase II)**:
    - Function: Further enhances uniqueness through the collective behavior of multiple samples.
    - Mechanism: For each fingerprint seed $\hat{x}_i^0$, $T$ small trainable perturbations $\{\delta_i^1, ..., \delta_i^T\}$ are applied, each variant being assigned a different random target class. The same biased probability distribution and KL divergence optimization are used, such that all variants remain near multi-boundary intersections while producing distinct predicted outputs.
    - Design Motivation: The behavior of a single fingerprint sample may be coincidentally replicated by an independently trained model, whereas the collective behavioral pattern of a group of samples (predictions across $T+1$ samples) is extremely difficult to replicate, significantly enhancing discriminative capability.

3. **Fingerprint Set Selection and Adaptive Thresholding (Phase III)**:
    - Function: Retains the most discriminative fingerprints and assigns an optimal threshold to each.
    - Mechanism:
        - Two reference model sets are constructed: a piracy model set $\mathcal{V}_f$ (generated via FT/KD/AT) and an independent model set $\mathcal{I}_f$ (independently trained).
        - The matching rate distributions of each composite fingerprint over both sets are computed, and discriminative power is quantified via Cohen's d effect size: $d_i = (\mu_i^{\mathcal{V}} - \mu_i^{\mathcal{I}}) / \sqrt{\frac{1}{2}((\sigma_i^{\mathcal{V}})^2 + (\sigma_i^{\mathcal{I}})^2)}$.
        - The top-$K$ most discriminative fingerprints are selected.
        - An adaptive threshold $\theta_i$ is computed for each selected fingerprint as a weighted average of the mean matching rates from the piracy and independent sets, with weights inversely proportional to standard deviation.
    - Design Motivation: Existing methods entirely ignore the effects of model modification and independent training during fingerprint construction. IrisFP addresses this via reference model set-based quality evaluation. Adaptive thresholding avoids the suboptimality of a globally fixed threshold.

### Loss & Training

- Phase I: $\mathcal{L}_{phase1} = KL(f_o(\hat{x}_i^0) || p_i) + \lambda_1\|\delta_i^0\|_1$
- Phase II: $\mathcal{L}_{phase2} = \frac{1}{T}\sum_{t=1}^T [KL(f_o(\hat{x}_i^t) || p_i^t) + \lambda_2\|\delta_i^t\|_1]$
- Verification threshold: Two-step decision — a single fingerprint with matching rate $\geq \theta_i$ is considered a match; the model is flagged as pirated if the proportion of matched fingerprints $\geq \alpha$.

## Key Experimental Results

### Main Results — AUC Comparison

| Protected Model | Method | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Tiny-ImageNet |
|----------------|--------|----------|-----------|--------------|-------|---------------|
| ResNet-18 | IPGuard | 0.675 | 0.654 | 0.721 | 0.471 | 0.726 |
| ResNet-18 | ADV-TRA | 0.799 | 0.806 | 0.845 | 0.753 | 0.767 |
| ResNet-18 | AKH | 0.710 | 0.785 | 0.765 | 0.820 | 0.823 |
| ResNet-18 | **IrisFP** | **0.893** | **0.916** | **0.940** | **0.854** | **0.874** |
| MobileNet-V2 | **IrisFP** | **0.936** | **0.937** | **0.963** | **0.876** | **0.934** |
| ViT-B/16 | **IrisFP** | — | — | — | — | **0.887** |

### Robustness Against Model Modification Attacks (ResNet-18, CIFAR-10)

| Method | FT | PR | KD | AT | PFT | NFT |
|--------|-----|-----|-----|-----|-----|-----|
| IPGuard | 0.656 | 0.997 | 0.515 | 0.511 | 0.687 | 0.724 |
| ADV-TRA | 1.000 | 1.000 | 0.805 | 0.025 | 0.959 | 0.962 |
| AKH | 0.921 | 0.876 | 0.621 | 0.531 | 0.701 | 0.733 |
| **IrisFP** | 0.954 | **1.000** | 0.616 | **0.929** | **0.965** | **0.968** |

### Ablation Study

| Configuration | CIFAR-10 AUC | Description |
|--------------|-------------|-------------|
| Seed | 0.691 | Seeds only |
| Seed_s | 0.748 | + fingerprint selection |
| Com_ft | ~0.79 | + composite samples + fixed threshold |
| Com_s_ft | 0.812 | + composite samples + selection + fixed threshold |
| IrisFP | **0.893** | + composite samples + selection + adaptive threshold |

### Key Findings

- IrisFP is particularly effective against adversarial training (AT) attacks — ADV-TRA achieves an AUC of only 0.025 under AT on CIFAR-10 (near complete failure), while IrisFP reaches 0.929.
- The composite sample mechanism and fingerprint selection are each independently effective; adaptive thresholding yields the largest single contribution, improving AUC from 0.812 to 0.893.
- The method remains effective on the more complex ViT-B/16 architecture (AUC 0.887).

## Highlights & Insights

- The core insight of multi-boundary intersection placement is simple yet profound: proximity to all boundaries simultaneously confers greater robustness than embedding fingerprints deep in the target class region, owing to larger predicted margins.
- Composite sample fingerprints leverage collective behavioral patterns rather than single-sample matching, substantially improving uniqueness.
- Cohen's d effect size and adaptive thresholding provide statistically grounded quantitative methods for fingerprint quality evaluation.
- The method operates in a black-box verification setting — only model query outputs are required.

## Limitations & Future Work

- Performance is relatively weak under knowledge distillation (KD) attacks (e.g., AUC 0.616 on CIFAR-10), since KD can fundamentally alter the decision boundary structure of the model.
- Constructing reference piracy and independent model sets for fingerprint selection incurs additional upfront cost.
- Validation is limited to image classification tasks; applicability to detection, segmentation, and other tasks remains unexplored.
- The assumed query budget of 200 may be excessive in certain deployment scenarios.

## Related Work & Insights

- **vs. IPGuard**: IPGuard pushes fingerprints directly toward a single decision boundary, yielding the weakest uniqueness and robustness among compared methods.
- **vs. ADV-TRA**: ADV-TRA captures rich model characteristics through adversarial trajectories, achieving acceptable robustness but poor uniqueness; it also nearly completely fails under AT attacks.
- **vs. IBSF/SDBF**: Although these methods also exploit multi-boundary intersections, they are designed solely for tampering detection and exhibit extremely weak robustness; IrisFP addresses the robustness issue through composite samples and fingerprint selection.

## Rating

- Novelty: ⭐⭐⭐⭐ Three complementary innovations — multi-boundary intersection placement, composite samples, and statistical selection — jointly address both uniqueness and robustness.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 datasets, 3 architectures, 6 attack types, 4 baselines, and detailed ablations; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the methodology is presented in a well-structured, progressive manner, though notation is dense.
- Value: ⭐⭐⭐ Addresses a clear practical need for model IP protection, though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)
- [\[ICCV 2025\] Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training](../../ICCV2025/others/failure_cases_are_better_learned_but_boundary_says_sorry_facilitating_smooth_per.md)
- [\[AAAI 2026\] DECOR: Deep Embedding Clustering with Orientation Robustness](../../AAAI2026/others/decor_deep_embedding_clustering_with_orientation_robustness.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](../../ICLR2026/others/the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)

</div>

<!-- RELATED:END -->
