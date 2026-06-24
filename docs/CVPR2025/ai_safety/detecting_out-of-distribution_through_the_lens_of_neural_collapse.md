---
title: >-
  [Paper Note] Detecting Out-of-Distribution through the Lens of Neural Collapse
description: >-
  [CVPR 2025][AI Safety][OOD detection] Based on Neural Collapse theory, this paper discovers that centered in-distribution (ID) features cluster near the weight vectors of their predicted classes and far from the origin (forming a simplex ETF). Guided by this, the NCI detector is designed by combining the angular proximity (pScore) between features and weight vectors with a feature norm filter. NCI achieves the best overall OOD detection performance on CIFAR-10/100 and ImageNe…
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "OOD detection"
  - "neural collapse"
  - "feature geometry"
  - "weight vector proximity"
  - "equiangular tight frame"
date: 2026-05-08
content_hash: 1756a8bb6e2e0fd9
---

# Detecting Out-of-Distribution through the Lens of Neural Collapse

**Conference**: CVPR 2025  
**arXiv**: [2311.01479](https://arxiv.org/abs/2311.01479)  
**Code**: [https://github.com/litianliu/NCI-OOD](https://github.com/litianliu/NCI-OOD)  
**Area**: Other  
**Keywords**: OOD detection, neural collapse, feature geometry, weight vector proximity, equiangular tight frame

## TL;DR
Based on Neural Collapse theory, this paper discovers that centered in-distribution (ID) features cluster near the weight vectors of their predicted classes and far from the origin (forming a simplex ETF). Guided by this, the NCI detector is designed by combining the angular proximity (pScore) between features and weight vectors with a feature norm filter. NCI achieves the best overall OOD detection performance on CIFAR-10/100 and ImageNet across multiple architectures while maintaining inference latency on par with the softmax baseline.

## Background & Motivation

**Background**: Post-hoc out-of-distribution (OOD) detection approaches are mainly divided into two paradigms: (1) output-space methods (MSP, Energy, ODIN, etc.) that design OOD scores utilizing model logits/probabilities; (2) feature-space methods (KNN, Mahalanobis, etc.) that leverage the clustering of ID features—where ID samples cluster in the feature space while OOD samples deviate.

**Limitations of Prior Work**: (1) **Inconsistent Generalization**: Methods that perform strongly on CIFAR-10 (e.g., KNN, Mahalanobis) perform poorly on ImageNet and vice versa (e.g., Energy, ASH)—no single method ranks in the top three on both benchmarks. (2) **Efficiency Concerns**: Feature-space methods (e.g., KNN) require storing training set features and computing $k$-nearest neighbor distances, which leads to high inference latency. (3) **Lack of Unified Theoretical Explanation**: Why do ID features cluster? Why do OOD features deviate? Why do some methods excel in specific scenarios but fail in others?

**Key Challenge**: Existing methods excel either in feature clustering scenarios (CIFAR-10) or feature diffusion scenarios (ImageNet), and there lacks a unified framework to explain and bridge this generalization gap.

**Goal**: (1) Provide a unified explanation for the geometric structures of ID/OOD features using Neural Collapse theory; (2) Design an efficient OOD detector that generalizes robustly across datasets and architectures.

**Key Insight**: Neural Collapse (NC) reveals four properties in the penultimate layer of well-trained neural networks: class-wise variance collapse, class means forming an ETF, weight-mean alignment, and classification simplifying to the nearest class center behavior. This work exploits two key corollaries from this: (a) centered ID features cluster along the direction of their predicted class's weight vector; (b) ID features are driven away from the origin to support the ETF structure.

**Core Idea**: Leverage NC properties where ID features align with weight vectors and stay away from the origin to design an OOD detector based on angular proximity and feature norm filtering.

## Method

### Overall Architecture
NCI is a post-hoc detector applied to a pre-trained classifier. Given an input $x$, it extracts the penultimate layer feature $h$ and computes two metrics: (1) the angular proximity (pScore) between the centered feature and the predicted class’s weight vector; (2) the $L_1$ norm of the feature. Their linear combination yields the final OOD score: $\text{NCI} = \text{pScore} + \alpha \|h\|_1$. A lower score implies a higher likelihood of being OOD.

### Key Designs

1. **Feature-Weight Proximity (pScore)**:

    - **Function**: Measures the directional alignment between centered features and the predicted class's weight vector.
    - **Mechanism**: Theorem 3.1 proves that under Neural Collapse conditions, $(h_c^i - \mu_G) \to \lambda w_c$, meaning that centered features align with the direction of the weight vector. Accordingly, pScore is defined as the projection magnitude of the weight vector along the direction of the centered feature: $\text{pScore} = \cos(w_c, h-\mu_G) \cdot \|w_c\|_2$, where $c$ is the predicted class and $\mu_G$ is the global mean of training features. The projection magnitude is adopted instead of pure cosine similarity because weight vectors of different classes have varying magnitudes—classes with larger weight vectors occupy larger decision domains and should thus have wider detection 'cones'.
    - **Design Motivation**: Avoids using Euclidean distance, which requires estimating the scale factor $\lambda$ (which is inaccurate in partially collapsed models). Angular metrics only require directional information, making them more robust to incomplete Neural Collapse in real-world models. Furthermore, it incorporates class-specific information through the weight vectors, which is missing in methods like KNN.

2. **Feature Norm Filtering**:

    - **Function**: Complements pScore in scenarios where ID clustering is weak.
    - **Mechanism**: Neural Collapse requires features to form a simplex ETF—a maximally separated structure featuring equal angles and equal norms, meaning that ID features require sufficient norm length to span the space. In contrast, OOD features are not driven by the training process to stay away from the origin, hence they tend to cluster near the origin. The $L_1$ norm $\|h\|_1$ is utilized to filter out OOD samples close to the origin, yielding the final score of $\text{NCI} = \text{pScore} + \alpha \|h\|_1$.
    - **Design Motivation**: On CIFAR-10, ID clustering is strong, making pScore sufficient on its own. However, on ImageNet, clustering is weaker (smaller inter-class margins), which degrades the effectiveness of pScore. Here, norm filtering becomes a critical complement. This explains why KNN performs well on CIFAR-10 but poorly on ImageNet—it only measures clustering while ignoring the feature norm.

3. **Relaxed Application of Neural Collapse Theory**:

    - **Function**: Ensures the method remains effective on real-world models that are not fully collapsed.
    - **Mechanism**: Full Neural Collapse requires training to zero training error, but real-world models typically stop training early. The authors cite results from [He & Su 2023] showing that Neural Collapse trends are established early in training—thus, full collapse is unnecessary as the trend suffices to separate ID and OOD samples. On CIFAR-10 with ResNet-18, at epoch 50, pScore already detects SVHN effectively (AUROC 94.44 vs baseline 91.27).
    - **Design Motivation**: Avoids constraining the method's applicability with overly strict theoretical assumptions, ensuring direct usability on various off-the-shelf models.

### Loss & Training
NCI is a purely post-hoc method that requires no training. It only requires a one-time calculation of the global mean of training features, $\mu_G$. The hyperparameter $\alpha$ is selected from four scales $\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ using a validation set, showing low sensitivity to this choice.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 AUROC↑ | ImageNet AUROC↑ | Average AUROC | Inference Latency (ms/img) |
|------|----------------|----------------|-----------|---------------|
| MSP* | 91.3 | 80.9 | 86.1 | 0.09 |
| KNN | **96.3** | 81.2 | 88.8 | 70.0 |
| Energy | 91.7 | 85.5 | 88.6 | 0.12 |
| ASH | 92.1 | **87.1** | 89.6 | 0.13 |
| Scale | 92.0 | 86.8 | 89.4 | 0.12 |
| **NCI** | **95.7** | **86.2** | **91.0** | **0.09** |

### Ablation Study

| Configuration | ImageNet AUROC | Description |
|------|---------------|------|
| pScore only (w/o filter) | 82.5 | Only angular proximity |
| L1 norm only | 84.1 | Norm filtering only |
| NCI (pScore + L1) | 86.2 | Two are complementary |
| pScore + L2 norm | 85.5 | L1 outperforms L2 |
| KNN + L1 filter | 84.8 | Filtering also improves KNN |

### Key Findings
- NCI is the only method that ranks within the top three on both CIFAR-10 and ImageNet benchmarks, achieving the highest average AUROC of 91.0.
- The inference latency is merely 0.09ms/image, on par with the simplest MSP baseline and 778 times faster than KNN (70ms).
- Norm filtering contributes +3.7 to AUROC on ImageNet but yields almost no gain on CIFAR-10, demonstrating the complementary relationship between clustering strength and the importance of filtering.
- It is also effective on ViT and Swin v2 architectures, demonstrating cross-architecture generalization.
- Applying L1 filtering to KNN also brings a significant improvement, validating the universality of the norm filtering concept.

## Highlights & Insights
- **Unified explanation of two major OOD detection paradigms using Neural Collapse**: Clustering-based methods (like KNN) exploit the intra-class collapse property of NC, while energy-based methods (Energy/ASH) implicitly leverage the norm properties of ETF. NCI explicitly combines both, resolving the cross-scenario generalization problem. This research paradigm of "identifying the common theoretical foundations of existing methods and unifying them" is highly instructive.
- **Class-aware detection with $O(P)$ complexity**: pScore requires only one vector dot product and one norm calculation. The time complexity is bounded by the dimension of the penultimate layer, $P$, which is far lower than KNN's $O(NP)$ where $N$ is the size of the training set. Introducing class information via weight vectors is a key elegant design—replacing "neighbors" with "anchors".
- **Relaxing theoretical assumptions rather than strictly adhering to them**: The authors clearly state that full Neural Collapse is not necessary; the trend alone suffices, making the method applicable to any off-the-shelf classifier.

## Limitations & Future Work
- The strength parameter $\alpha$ for L1 norm filtering still requires selection via a validation set; although there are only 4 candidate values, it is not completely parameter-free.
- Under extreme class-imbalanced long-tailed scenarios, the Neural Collapse trend might be less pronounced, and the method's effectiveness warrants further evaluation.
- The computation of the global mean $\mu_G$ relies on the representativeness of the training data; if the training distribution is biased, the mean estimate may be inaccurate.
- Adversarial scenarios are not discussed; carefully crafted adversarial examples might simultaneously exhibit a high pScore and high norm.

## Related Work & Insights
- **vs KNN**: KNN focuses on clustering but ignores feature norms, performing well on CIFAR-10 but poorly on ImageNet. NCI resolves this generalization gap through norm filtering while achieving 778x faster inference.
- **vs Energy/ASH/Scale**: These methods implicitly utilize feature norm information (e.g., log-sum-exp corresponds to LogSumExp of projections) but fail to account for class clustering, which underperforms on CIFAR-10. NCI replenishes this by introducing clustering information through pScore.
- **vs NECO**: Also inspired by Neural Collapse, but NECO conducts analysis strictly in the feature space, which requires expensive matrix multiplications. NCI leverages the interaction between weight vectors and features, making it highly efficient while seamlessly introducing class-specific information.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Neural Collapse perspective for explaining OOD detection is innovative, although the core components (cosine similarity and norm) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Conducted across multiple datasets, multiple architectures (ResNet/DenseNet/ViT/Swin), 13 baseline methods, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous and clear theoretical derivations, in-depth experimental analyses, and excellent discussions connecting existing methods.
- **Value**: ⭐⭐⭐⭐ Provides a unified theoretical framework and implements an efficient and practical OOD detector.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[CVPR 2025\] Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)
- [\[CVPR 2025\] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)
- [\[CVPR 2026\] Learning Latent Concepts for Detecting Out-of-Distribution Objects](../../CVPR2026/ai_safety/learning_latent_concepts_for_detecting_out-of-distribution_objects.md)

</div>

<!-- RELATED:END -->
