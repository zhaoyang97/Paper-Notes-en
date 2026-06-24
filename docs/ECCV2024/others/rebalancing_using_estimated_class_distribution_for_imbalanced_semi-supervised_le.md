---
title: >-
  [Paper Note] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch
description: >-
  [ECCV 2024][Class Imbalance] This paper proposes the RECD algorithm, which estimates the unknown class distribution of unlabeled data via Monte Carlo approximation. It rebalances the classifier based on the estimated distribution and introduces feature cluster compression to mitigate representation space imbalance, achieving SOTA performance in semi-supervised learning scenarios with labeled-unlabeled class distribution mismatch.
tags:
  - "ECCV 2024"
  - "Class Imbalance"
  - "Semi-Supervised Learning"
  - "Class Distribution Estimation"
  - "Distribution Mismatch"
  - "Feature Cluster Compression"
date: 2026-05-08
content_hash: 5663f3969543a94b
---

# Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Semi-supervised Learning  
**Keywords**: Class Imbalance, Semi-Supervised Learning, Class Distribution Estimation, Distribution Mismatch, Feature Cluster Compression

## TL;DR

This paper proposes the RECD algorithm, which estimates the unknown class distribution of unlabeled data via Monte Carlo approximation. It rebalances the classifier based on the estimated distribution and introduces feature cluster compression to mitigate representation space imbalance, achieving SOTA performance in semi-supervised learning scenarios with labeled-unlabeled class distribution mismatch.

## Background & Motivation

**Background**: Class-Imbalanced Semi-Supervised Learning (CISSL) is an important topic in semi-supervised learning, aiming to learn from a small amount of labeled data and a large amount of unlabeled data while tackling the challenges of imbalanced class distributions. Recent CISSL methods (e.g., DARP, DASO, ABC) have achieved significant progress.

**Limitations of Prior Work**: Existing CISSL algorithms typically assume, either explicitly or implicitly, that the class distribution of unlabeled data matches that of the labeled data (distribution match). However, in practical applications, this assumption often fails—labeled data might be heavily skewed towards certain classes (due to differing labeling costs), while unlabeled data may have an entirely different class distribution. When a distribution mismatch occurs, methods based on incorrect distribution assumptions assign incorrect weights to different classes during training, leading to a significant degradation in classification performance.

**Key Challenge**: To correctly rebalance the classifier, the true class distribution of the overall training data needs to be known—but the class distribution of the unlabeled data is precisely what is unknown. Existing methods either assume the distribution is known or estimate it using simple heuristics (such as predicted class frequencies of the model), but these estimations are highly unreliable in the early stages of model training, creating a chicken-and-egg problem.

**Goal**: (1) How to reliably estimate the class distribution of unlabeled data without relying on distribution assumptions? (2) How to effectively rebalance the classifier based on the estimated distribution? (3) How to simultaneously alleviate class imbalance in the representation space?

**Key Insight**: The authors propose to estimate the class distribution of unlabeled data using a Monte Carlo approximation based on the model's predicted class probabilities for unlabeled samples. This estimation becomes progressively accurate as the model improves, forming a positive feedback loop. Simultaneously, Feature Clusters Compression (FCC) is introduced in the feature space to alleviate the sparsity of minority class representations.

**Core Idea**: Estimate the class distribution of unlabeled data via Monte Carlo approximation, rebalance the classifier based on this estimation, and alleviate representation space imbalance with feature cluster compression.

## Method

### Overall Architecture

RECD adds two core components on top of a standard semi-supervised learning framework (such as FixMatch): (1) a class distribution estimation module, which utilizes the model's prediction probabilities on unlabeled data to estimate its class distribution via Monte Carlo approximation; (2) a feature cluster compression module, which clusters and compresses the feature representations of minority classes to increase their density. Finally, the classifier decision boundary is rebalanced based on the estimated mixture class distribution (labeled + unlabeled).

### Key Designs

1. **Monte Carlo Class Distribution Estimation**:

    - **Function**: Estimate the sample proportion of each class in the unlabeled data
    - **Mechanism**: For a batch of unlabeled data $\{u_i\}_{i=1}^{N_u}$, the model outputs the predicted probability of each sample belonging to each class, $p(y=c|u_i)$. The proportion of class $c$ in the unlabeled data can be approximated via Monte Carlo: $\hat{\pi}_c = \frac{1}{N_u} \sum_{i=1}^{N_u} p(y=c|u_i)$. As training progresses, the model predictions become increasingly accurate, thereby improving the distribution estimation. To enhance estimation stability, exponential moving average (EMA) is used to smooth the estimates across multiple mini-batches.
    - **Design Motivation**: Relying directly on hard labels predicted by the model to count class frequencies exhibits high volatility when the model is insufficiently accurate, whereas the Monte Carlo approximation with soft probabilities is much smoother and more stable. It also avoids making any prior assumptions about the unlabeled data distribution.

2. **Classifier Rebalancing Based on Estimated Distribution**:

    - **Function**: Adjust the decision boundary of the classifier according to the estimated true class distribution
    - **Mechanism**: In notebook training, the classifier tends to favor majority classes. Knowing the true class proportion of the data, posterior correction (logit adjustment) can be performed at the logic layer. Specifically, the model's logits are adjusted to $\tilde{z}_c = z_c - \tau \log(\hat{\pi}_c^{total})$, where $\hat{\pi}_c^{total}$ is the estimated mixture class proportion of both labeled and unlabeled data. This adjustment is equivalent to modifying the prior from a uniform distribution to the estimated true distribution under a Bayesian framework.
    - **Design Motivation**: Conventional rebalancing methods (such as logit adjustment) require the class distribution to be known priori. This paper provides the required distribution information through the estimation module, making logit adjustment applicable to semi-supervised scenarios with unknown distributions.

3. **Feature Clusters Compression (FCC)**:

    - **Function**: Alleviate the sparsity and dispersion of minority class features in the representation space
    - **Mechanism**: Because minority classes have few training samples, their features in the embedding space are often sparse and scattered, leading to loose decision boundaries learned by the classifier. FCC performs clustering (such as K-means) on the features of each class and then compresses the features within each cluster towards their cluster centers, increasing the density of minority class features. Without adding real samples, this operation makes minority classes occupy a tighter regional space in the feature space, approximating an oversampling effect without introducing the overfitting risk of repeated samples.
    - **Design Motivation**: Rebalancing at the classifier level adjusts the decision boundary, but representation space imbalance remains. FCC addresses this issue from the perspective of representation learning, allowing rebalancing at both levels to work synergistically.

### Loss & Training

RECD is built upon the FixMatch framework. Its total loss consists of three components: (1) cross-entropy loss for labeled data (with logit adjustment); (2) consistency regularization loss for unlabeled data (cross-entropy between predictions of strongly augmented images and pseudo-labels of weakly augmented images); (3) auxiliary regularization loss of FCC. The class distribution estimate is updated via EMA, and the temperature coefficient $\tau$ of logit adjustment serves as a hyperparameter to control the adjustment strength.

## Key Experimental Results

### Main Results

| Dataset | Imbalance Ratio | Distribution | RECD | DARP | DASO | ABC |
|--------|---------|------|------|------|------|-----|
| CIFAR-10 | 100 | Match | SOTA | Sub-optimal | - | - |
| CIFAR-10 | 100 | Mismatch | SOTA | Significant Drop | Significant Drop | Drop |
| CIFAR-100 | 50 | Match | SOTA | Sub-optimal | - | - |
| CIFAR-100 | 50 | Mismatch | SOTA | Significant Drop | Drop | Drop |
| STL-10 | 20 | Mismatch | SOTA | Drop | Drop | Drop |
| Food-101 | 50 | Mismatch | SOTA | Drop | Drop | - |

### Ablation Study

| Configuration | Accuracy | Description |
|------|-------|------|
| Full RECD | Optimal | Full model |
| w/o Distribution Estimation | Drop 2-5% | Replaced by labeled data distribution, degrades significantly under mismatch |
| w/o FCC | Drop 1-2% | Feature space imbalance unresolved |
| w/o Logit Adjustment | Drop 3-4% | Decision boundary of classifier shifts |
| Hard Label Estimation instead of MC | Drop 1-2% | Soft probability estimation is more stable |

### Key Findings

- Distribution mismatch is the "Achilles' heel" of existing CISSL methods: when the assumption fails, almost all methods degrade significantly, whereas RECD remains robust.
- The class distribution estimated by Monte Carlo converges as training proceeds, with minimal estimation errors in the later stages.
- FCC benefits minority classes most significantly, especially under extreme imbalance ratios (e.g., 100:1).
- In distribution match scenarios, RECD also achieves competitive results, demonstrating the generality of the method.
- EMA smoothing is critical for estimation stability; directly using single-batch estimations results in excessive fluctuations.

## Highlights & Insights

- **Distribution-Agnostic Design Philosophy**: RECD makes no assumptions about the distribution of unlabeled data but adaptively estimates and utilizes it, making it effective in both "distribution match" and "distribution mismatch" scenarios. This design philosophy can be extended to other learning paradigms that leverage unlabeled data with unknown distributions.
- **Simple Estimation Scheme of Monte Carlo Approximation + EMA**: It does not require complex distribution modeling or generative models, effectively estimating class distributions solely using the model's own predicted probabilities. This "bootstrap-style" estimation strategy is simple yet effective.
- **Two-Level Rebalancing**: Addressing the imbalance at both the classifier level (logit adjustment) and the feature space level (FCC), which are complementary to each other.

## Limitations & Future Work

- The Monte Carlo distribution estimation may be unreliable in the early stages of training (when the model is still inaccurate), possibly requiring a warm-up strategy.
- The clustering operation of FCC adds computational overhead, and its efficiency on large-scale datasets needs to be validated.
- The performance under open-set scenarios (where unlabeled data contains novel classes not covered in the labeled set) is not discussed.
- The temperature coefficient $\tau$ of logit adjustment is a hyperparameter, and different datasets may require different settings.
- Comparison with the latest semi-supervised methods based on contrastive learning is omitted.

## Related Work & Insights

- **vs DARP**: DARP alleviates imbalance by aligning pseudo-label distributions with estimated prior distributions, but assumes the prior is known or can be inferred from labeled data. RECD directly estimates the unlabeled data distribution, making it more flexible.
- **vs CReST**: CReST handles class imbalance through class-rebalanced self-training iterations, but also assumes that the unlabeled data distribution is known. RECD addresses the more general distribution mismatch scenario.
- **vs Logit Adjustment (Menon et al.)**: Classical logit adjustment requires a known class prior, while RECD extends it to semi-supervised scenarios where the prior is unknown.

## Rating

- Novelty: ⭐⭐⭐⭐ CISSL under distribution mismatch has received insufficient attention; the combination of Monte Carlo estimation and two-level rebalancing is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four datasets with various imbalance ratios and distribution setups.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problem with well-motivated methodology.
- Value: ⭐⭐⭐⭐ Resolves key assumption issues in the practical deployment of CISSL, demonstrating strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ECCV 2024\] ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)
- [\[ECCV 2024\] Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)
- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [\[CVPR 2025\] Sufficient Invariant Learning for Distribution Shift](../../CVPR2025/others/sufficient_invariant_learning_for_distribution_shift.md)

</div>

<!-- RELATED:END -->
