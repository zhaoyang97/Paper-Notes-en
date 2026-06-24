---
title: >-
  [Paper Note] Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels
description: >-
  [ECCV2024][Social Computing][noisy label] Proposes the DaSC framework, which simultaneously addresses the joint problem of long-tailed distribution and noisy labels through distribution-aware class centroid estimation (DaCC) and confidence-aware contrastive learning (SBCL + MIDL), achieving SOTA results on CIFAR and real-world noisy datasets.
tags:
  - "ECCV2024"
  - "Social Computing"
  - "noisy label"
  - "long-tailed learning"
  - "contrastive learning"
  - "sample selection"
  - "semi-supervised learning"
date: 2026-05-08
content_hash: 4e7f8daa68a4ba58
---

# Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels

**Conference**: ECCV2024  
**arXiv**: [2407.16802](https://arxiv.org/abs/2407.16802)  
**Code**: [GitHub](https://github.com/JaesoonBaik1213/DaSC)  
**Area**: Social Computing  
**Keywords**: noisy label, long-tailed learning, contrastive learning, sample selection, semi-supervised learning

## TL;DR
Proposes the DaSC framework, which simultaneously addresses the joint problem of long-tailed distribution and noisy labels through distribution-aware class centroid estimation (DaCC) and confidence-aware contrastive learning (SBCL + MIDL), achieving SOTA results on CIFAR and real-world noisy datasets.

## Background & Motivation
Real-world data often suffers from both **long-tailed distribution** and **label noise**. While extensive work has addressed either problem individually, jointly handling them (NL-LT) remains challenging:

- **Long-tailed distribution**: Severe imbalance in class sample sizes causes the model to bias towards head classes.
- **Noisy labels**: Some samples are incorrectly labeled, and models memorizing the noise leads to degraded generalization performance.
- The superposition of both makes it more difficult to accurately identify the true data distribution.

Existing methods (such as RoLT, PCL, SFA) use feature-based noisy sample selection strategies to determine whether samples are noisy by computing the centroid of each class. However, they suffer from three key limitations:

1. **Only using Seals/high-confidence samples within the target class** to estimate class centroids—insufficient samples in tail classes lead to unreliable centroids.
2. **Assigning equal weights to all samples**—ignoring the fact that some samples might be incorrectly labeled.
3. **Lacking active mechanisms to improve representation quality**—where the representation quality of tail classes requires enhancement even more.

## Core Problem
How to simultaneously cope with long-tailed distributions and noisy labels? Specifically:

- How to accurately estimate the feature centroid of each class under the condition of insufficient sample sizes and the presence of noise?
- How to learn balanced and robust feature representations in a noisy label environment?

## Method

### Overall Architecture DaSC
DaSC comprises four core components:

1. **DaCC** (Distribution-aware Class Centroid Estimation): Distribution-aware class centroid estimation.
2. **Noisy Sample Selection**: Distinguishing clean/noisy samples based on GMM.
3. **SBCL** (Semi-supervised Balanced Contrastive Loss): Semi-supervised balanced contrastive loss for high-confidence samples.
4. **MIDL** (Mixup-enhanced Instance Discrimination Loss): Mixup-enhanced instance discrimination loss for low-confidence samples.

The model architecture includes a shared feature extractor $f$, a normal classifier $g^c$, a balanced classifier $g^b$ (using Balanced Softmax), a pseudo-label generator, and an MLP projection head $q$.

### DaCC: Distribution-Aware Class Centroid Estimation
Unlike traditional methods that only use samples within the target class, DaCC leverages **samples from all classes** to estimate each class centroid. The key is to assign weights according to the prediction confidence of the model:

$$c_k = \text{Norm}\left(\sum_{x(i) \in \mathcal{D}^I} \hat{p}_k^c(i) \cdot z'(i)\right)$$

Where weights are obtained through temperature-scaled softmax:

$$\hat{p}_k^c(i) = \frac{\exp(p_k^c(i) / \tau_T)}{\sum_{k=1}^K \exp(p_k^c(i) / \tau_T)}$$

- The temperature parameter $\tau_T = 0.1$ assigns larger weights to high-confidence samples and suppresses weights of low-confidence samples.
- Only using the sample set $\mathcal{D}^I$ where predictions from both classifiers (normal + balanced) exceed the threshold $\tau$.
- The threshold $\tau$ grows gradually with training: $\tau = \phi^t \hat{\tau}$, where $\phi=1.005$ and $\hat{\tau}=1/K$.

After computing the centroids, the cosine similarity between sample features and centroids is calculated, and GMM is used to fit the allocation probabilities, thereby partitioning the samples into clean and noisy samples.

### SBCL: Semi-supervised Balanced Contrastive Loss
An improved supervised contrastive learning is applied to high-confidence samples (where the maximum pseudo-label probability is $> \tau_c$):

- In the denominator of the contrastive loss, the contribution of each class is **homogenized** (taking the class-wise average) to prevent head classes from dominating the contrastive loss due to their large sample sizes.
- Reliable pseudo-label information is utilized to penalize head classes, promoting the learning of balanced representations.

### MIDL: Mixup-enhanced Instance Discrimination Loss
A self-supervised approach is applied to low-confidence samples (where pseudo-labels are unreliable and unsuitable for using label information):

- Generating two augmented views $x^1, x^2$ for the same sample to serve as the query and positive key.
- Generating $x^{mix} = \lambda x^1 + (1-\lambda) x^2$ via Mixup to serve as a **hard negative sample**.
- Mixup samples are pushed into a memory bank to increase the diversity of negative samples.
- Improving the representation quality of low-confidence samples through the instance discrimination task.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{MixMatch} + \mathcal{L}_{BMixMatch} + \lambda_{SBCL} \mathcal{L}_{SBCL} + \lambda_{MIDL} \mathcal{L}_{MIDL}$$

The training consists of two stages:
- **Warmup Stage** (first 30 epochs): Using cross-entropy + Balanced Softmax + SBCL (introduced after 10 epochs).
- **Main Training Stage**: DaCC sample selection + SSL (MixMatch) + SBCL + MIDL.

During inference, the average predictions of the normal classifier and the balanced classifier are utilized. A co-training framework is employed to counteract confirmation bias.

## Key Experimental Results

### CIFAR-10 (Long-tailed + Symmetric Noise, Imbalance Ratio 0.1)

| Noise Ratio | SFA | TABASCO | **DaSC** |
|--------|-----|---------|----------|
| 0.4 | 86.81 | 85.53 | **89.04** |
| 0.6 | 82.89 | 84.83 | **87.12** |

### CIFAR-100 (Long-tailed + Symmetric Noise, Imbalance Ratio 0.1)

| Noise Ratio | SFA | TABASCO | **DaSC** |
|--------|-----|---------|----------|
| 0.4 | 56.70 | 56.52 | **61.85** |
| 0.6 | 47.71 | 45.98 | **54.40** |

### CIFAR-10 (Long-tailed + Asymmetric Noise, Imbalance Ratio 0.1)

| Noise Ratio | SFA | TABASCO | **DaSC** |
|--------|-----|---------|----------|
| 0.2 | 87.70 | 82.10 | **89.89** |
| 0.4 | 78.13 | 80.57 | **88.85** |

It significantly outperforms previous SOTA across all configurations, with particularly pronounced improvements in scenarios with high noise ratios and asymmetric noise.

## Highlights & Insights
1. **Cross-class Weighted Centroid Estimation of DaCC**: Breaks the limitation of traditional methods that only use intra-class samples. By utilizing temperature-scaled prediction probabilities as weights, it effectively leverages all samples, which is particularly crucial for estimating tail class centroids.
2. **Confidence-Aware Dual-Path Contrastive Learning**: High-confidence samples employ supervised contrastive learning (SBCL) to exploit label information for learning balanced representations, while low-confidence samples adopt self-supervised contrastive learning (MIDL) to avoid noisy label interference—exhibiting a well-designed strategy.
3. **Mixup as Hard Negative Samples**: Utilizing samples generated by Mixup as negative samples in the memory bank simultaneously enhances representation learning and data diversity.
4. **Experimental Thoroughness**: Covers symmetric/asymmetric noise, different imbalance ratios, and real-world noisy datasets.

## Limitations & Future Work
1. **Computational Overhead**: DaCC requires forward propagation through all samples to estimate centroids, and employing a co-training framework (two networks) leads to a high computational cost.
2. **Abundant Hyperparameters**: Hyperparameters such as $\tau_T, \tau_c, \tau_s, \tau_m, \lambda_{SBCL}, \lambda_{MIDL}$ require fine-tuning.
3. **Classification Domain Assumption**: The method is primarily tailored for classification tasks, and its transferability to detection/segmentation tasks has not been explored.
4. **Threshold Strategy of DaCC**: The growth strategy of the confidence threshold $\tau$ ($\phi=1.005$) is relatively empirical.
5. **Validation Only on Small-Scale Datasets**: The maximum experimental scale is Red mini-ImageNet, leaving effectiveness on large-scale datasets unverified.

## Related Work & Insights
- **vs RoLT/PCL**: Only use high-confidence samples within the target class to estimate centroids; DaSC uses weighted estimation of all samples, leading to more accurate tail class centroids.
- **vs SFA**: Models class centroid uncertainty using Gaussian distributions; DaSC directly weights by prediction probabilities, which is much simpler.
- **vs TABASCO**: Employs two metrics + GMM to select clean samples; DaSC's DaCC provides higher quality centroids and incorporates contrastive learning to enhance representations.
- **vs DivideMix/UNICON**: Only handles noisy labels without considering long-tailed distributions; DaSC explicitly handles class imbalance through Balanced Softmax and SBCL.
- **vs BCL/PaCo**: Only handles long-tailed distributions without considering noise; DaSC selects clean samples via DaCC + GMM.

## Inspirations & Connections
- The cross-class weighted estimation idea of DaCC can be extended to other scenarios requiring class prototype estimation (such as few-shot learning and open-set recognition).
- The idea of adopting different learning strategies for high/low confidence samples can be applied to semi-supervised learning and active learning.
- The approach of utilizing Mixup as hard negative samples has generality and can be extended to other contrastive learning frameworks.

## Rating
- Novelty: 7/10 (The cross-class weighted centroid estimation of DaCC and the dual-path contrastive learning design are relatively novel, but the technologies of each component are not entirely new.)
- Experimental Thoroughness: 8/10 (Includes multiple noise types, multiple datasets, and comprehensive ablation studies.)
- Writing Quality: 7/10 (Clear structure, but containing many equations, making readability average.)
- Value: 7/10 (Features clear contributions to the specific NL-LT problem, but validation in practical scenarios remains insufficient.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [\[ICML 2025\] Learning Survival Distributions with the Asymmetric Laplace Distribution](../../ICML2025/social_computing/learning_survival_distributions_with_the_asymmetric_laplace_distribution.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](../../NeurIPS2025/social_computing/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](../../NeurIPS2025/social_computing/precise_information_control_in_long-form_text_generation.md)

</div>

<!-- RELATED:END -->
