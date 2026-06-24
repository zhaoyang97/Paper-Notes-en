---
title: >-
  [Paper Note] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling
description: >-
  [CVPR 2025][Self-Supervised Learning][MAE Curriculum Learning] Proposes a prototype-driven curriculum learning for MAE, which identifies "prototype" samples (representative images close to cluster centroids) in the dataset using K-means clustering. By using a temperature-controlled sampling strategy, the training smoothly transitions from prototypes to the full distribution, achieving an up to $8\times$ training acceleration (a 200-epoch prototype curriculum performs comparab…
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "MAE Curriculum Learning"
  - "Prototype Samples"
  - "K-means Clustering"
  - "Temperature Sampling"
  - "Training Acceleration"
date: 2026-05-08
content_hash: c84abdd8a272818f
---

# From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling

**Conference**: CVPR 2025  
**arXiv**: [2411.10685](https://arxiv.org/abs/2411.10685)  
**Code**: None  
**Area**: Self-Supervised Learning  
**Keywords**: MAE Curriculum Learning, Prototype Samples, K-means Clustering, Temperature Sampling, Training Acceleration

## TL;DR
Proposes a prototype-driven curriculum learning for MAE, which identifies "prototype" samples (representative images close to cluster centroids) in the dataset using K-means clustering. By using a temperature-controlled sampling strategy, the training smoothly transitions from prototypes to the full distribution, achieving an up to $8\times$ training acceleration (a 200-epoch prototype curriculum performs comparably to an 800-epoch standard MAE).

## Background & Motivation

**Background**: Masked Image Modeling (MAE) is a powerful self-supervised pre-training method, but it requires a large number of training epochs (800–1600). Curriculum learning (easy-to-hard) has been shown to be effective in classification but remains unexplored in MIM.

**Limitations of Prior Work**: (1) MAE requires over 800 epochs to achieve satisfactory linear probe performance. (2) Hard-samples-first curricula (anti-curriculum) are detrimental rather than helpful in MIM. (3) There is no consensus on how to define "easy" and "hard" in MIM.

**Key Challenge**: The training efficiency of MIM is low—the early stages spend a large number of steps exploring the high-dimensional representation space, resulting in slow learning. If the model can focus on training "typical" samples in the early stage to quickly establish a framework for representations, and then extend to complex samples later, convergence could be accelerated.

**Goal**: Design a curriculum strategy tailored for MIM—transitioning from prototype samples (representative easy samples) to the full distribution to accelerate training while maintaining or improving representation quality.

**Key Insight**: Define "prototypes" using K-means clustering—samples closer to cluster centers are more typical and simpler. A temperature-controlled softmax sampling enables a smooth transition from prototypes to the full distribution.

**Core Idea**: Use distance to K-means centroids to define sample "prototypicality," and gradually expand from prototypes to the full distribution using a temperature-annealed sampling strategy, achieving an $8\times$ training acceleration for MAE.

## Method

### Overall Architecture
Pre-computation: Perform K-means clustering in the feature space (DINO/SIFT) $\rightarrow$ Calculate the distance $\hat{d}_i$ of each sample to its cluster center $\rightarrow$ During training: a temperature parameter $\tau$ controls the sampling probability $P(x_i, \tau) \propto \exp(-\hat{d}_i/\tau)$ $\rightarrow$ low $\tau$ concentrates on prototypes, while high $\tau$ tends toward a uniform distribution $\rightarrow$ the effective dataset size scales up from small to large via cosine annealing $\rightarrow$ binary search is used to solve for the corresponding $\tau$.

### Key Designs

1. **Prototype Identification (K-means Clustering)**:
    - **Function**: Quantify the "prototypicality" of each sample.
    - **Mechanism**: Perform K-means clustering ($K \approx 978$) on ImageNet-1K in the SIFT or DINO feature space, and compute the distance $\hat{d}_i$ of each sample to its cluster center. The smaller the distance, the more "prototypical" the sample, representing typical visual patterns of that class.
    - **Design Motivation**: The Davies-Bouldin index is used to automatically determine $K=978$. Surprisingly, self-supervised clustering (DINO) identifies prototypes better than using supervised labels (ImageNet 1000 classes).

2. **Temperature-Controlled Sampling**:
    - **Function**: Transition seamlessly from prototypes to the full distribution.
    - **Mechanism**: $P(x_i, \tau) = \frac{\exp(-\hat{d}_i/\tau)}{\sum_j \exp(-\hat{d}_j/\tau)}$. At low $\tau$, only prototypes are sampled (concentrating training), while at high $\tau$, all samples are equally probable (full distribution training). Instead of tuning $\tau$ directly, the "effective dataset size" $|D_\tau|/|D|$ is adjusted, following a cosine annealing schedule from its initial value to $(1-1/e)$.
    - **Design Motivation**: Fixed $\tau$ is inferior to annealing: a fixed $\tau=0.2$ achieves at best 41.57% NN accuracy, while annealing reaches 47.40% NN.

3. **Feature Space Selection**:
    - **Function**: Determine which feature space to use for clustering.
    - **Mechanism**: Various feature spaces such as DINO, SimCLR, SIFT, and ImageNet labels were tested. DINO performed the best (40.15% NN), but SIFT (traditional and requiring no pre-training) also performed reasonably well (36.85%).
    - **Design Motivation**: Eliminates the need for extra pre-training—SIFT acts as an effective zero-cost alternative.

### Loss & Training
Standard MAE loss (MSE pixel reconstruction). Only modifications are made to the data sampling strategy without altering the model architecture or loss functions.

## Key Experimental Results

### Main Results

| Method | Epoch | NN↑ | LP↑ | FT↑ |
|------|-------|-----|-----|-----|
| MAE baseline | 800 | 30.25 | 64.25 | 83.08 |
| Hard-first curriculum | 800 | 24.63 | 62.09 | 82.95 |
| **Prototype curriculum** | **800** | **47.40** | **68.84** | **83.31** |
| **Prototype curriculum** | **200** | **34.92** | **63.74** | 82.75 |

### Ablation Study

| Configuration | NN↑|
|------|-----|
| Uniform sampling (baseline) | 30.25 |
| Fixed $\tau=0.2$ | 41.57 |
| **Temperature annealing** | **47.40** |
| DINO features | 40.15 (single-step) |
| SIFT features | 36.85 (single-step) |
| ImageNet labels | 37.89 (single-step) |

### Key Findings
- **200-epoch prototype curriculum > 800-epoch baseline**: NN of 34.92 vs 30.25, representing a $4\times$ acceleration.
- **Self-supervised clustering > Supervised labels**: Prototypes defined by DINO clustering perform better than those defined by ImageNet 1,000-class labels.
- **Hard-first is detrimental**: The NN of the anti-curriculum (24.63) is worse than the baseline (30.25), indicating that MIM needs to establish basic representations first before dealing with complex samples.
- **Greater advantages in few-shot scenarios**: For 5-shot learning, the 200-epoch curriculum matches or outperforms the 1600-epoch baseline.

## Highlights & Insights
- **The insight that "MIM needs to learn prototypes first"** draws an analogy to human learning—first establishing conceptual frameworks with typical samples, then expanding to variations.
- **The discovery that self-supervised clustering outperforms supervised labels** is surprising—indicating that the optimal prototype definition for MIM does not perfectly align with classification categories.
- **SIFT as a zero-cost alternative** makes the method completely free of additional pre-training overhead.

## Limitations & Future Work
- K-means clustering requires pre-computing features (DINO/SIFT), introducing initialization overhead.
- Evaluated only on ImageNet-1K with ViT-B; performance on larger datasets/models remains unknown.
- Specific schedules for temperature annealing (cosine, initial value) require some hyperparameter tuning.

## Related Work & Insights
- **vs Standard MAE**: 800 epochs. Prototype curriculum surpasses its NN performance at only 200 epochs.
- **vs Hard-sample curriculum**: Anti-curriculum is detrimental in MIM (24.63 vs 30.25). The prototype curriculum is the correct direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Prototype curriculum learning for MIM is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple feature spaces, multiple temperature strategies, multiple epoch comparisons, and few-shot analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly explains the theoretical motivation of curriculum learning.
- Value: ⭐⭐⭐⭐ Possesses practical value for accelerating self-supervised pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CheXWorld: Image World Modeling for Radiograph Representation Learning](chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[ECCV 2024\] Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](../../ECCV2024/self_supervised/efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)
- [\[ICML 2026\] Riemannian Metric Matching for Scalable Geometric Modeling of Distributions](../../ICML2026/self_supervised/riemannian_metric_matching_for_scalable_geometric_modeling_of_distributions.md)
- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](mos_modeling_object-scene_associations_in_generalized_category_discovery.md)

</div>

<!-- RELATED:END -->
