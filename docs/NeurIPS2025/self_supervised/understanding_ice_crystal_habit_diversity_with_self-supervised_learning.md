---
title: >-
  [Paper Note] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][ice crystal habit] This paper presents the first application of self-supervised learning (SSL) to latent representation learning for ice crystal images. By pre-training a ViT on a…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "ice crystal habit"
  - "climate science"
  - "Vision Transformer"
  - "data curation"
date: 2026-05-08
content_hash: ce900d50d0e67f93
---

# Understanding Ice Crystal Habit Diversity with Self-Supervised Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.07688](https://arxiv.org/abs/2509.07688)  
**Code**: None  
**Area**: Self-Supervised Learning / AI for Science
**Keywords**: self-supervised learning, ice crystal habit, climate science, Vision Transformer, data curation

## TL;DR

This paper presents the first application of self-supervised learning (SSL) to latent representation learning for ice crystal images. By pre-training a ViT on a large-scale cloud particle image dataset, the method learns continuous latent representations of ice crystal habits and quantifies habit diversity using the vMF concentration parameter, achieving a state-of-the-art classification accuracy of 84.39% with a 30× reduction in computational cost.

## Background & Motivation

**Background**: Clouds represent one of the largest sources of uncertainty in climate models, and ice-containing clouds are particularly difficult to model due to the high diversity of ice crystal habits. The microphysical properties of ice crystals govern particle–radiation interactions and aerodynamics, thereby influencing global radiative forcing, precipitation, and the spatiotemporal distribution of clouds across multiple scales.

**Limitations of Prior Work**: Research on ice crystal habits has primarily relied on millions of images captured by Cloud Particle Imagers (CPI). Conventional approaches extract geometric features (e.g., aspect ratio, circularity) via image processing techniques, or apply supervised machine learning for classification. However, these methods suffer from two fundamental problems: (1) they require extensive manual annotation at high cost; and (2) they depend on predefined habit categories, rendering them unable to capture continuous morphological variation and intra-class diversity.

**Key Challenge**: Ice crystal habits are intrinsically continuously distributed, yet existing analytical methods either rely on discrete categories or require expensive manual annotation, fundamentally limiting the understanding of habit diversity.

**Goal**: How can meaningful representations of ice crystal habits be learned without manual annotation? How can the morphological diversity of ice crystals be quantified in a data-driven manner?

**Key Insight**: The authors observe that CPI images exhibit a natural clustering structure governed by ice crystal habit, which closely aligns with the assumptions of clustering-based SSL methods (the DINO family). This motivates the use of SSL to learn physically meaningful representations without any labels.

**Core Idea**: Apply DINO-family self-supervised ViTs pre-trained on a large-scale CPI dataset to learn continuous latent representations of ice crystal habits, replacing conventional discrete classification and geometric feature extraction pipelines.

## Method

### Overall Architecture

The input consists of 3.2 million unlabeled CPI images (the CPI-3M dataset). A ViT-Small model is pre-trained using the iBOT-vMF self-supervised method, producing 384-dimensional latent embeddings that can be used for downstream tasks such as habit classification and diversity quantification. The full pipeline comprises three stages: data curation → efficient pre-training → downstream application.

### Key Designs

1. **vMF-Based SSL Pre-training (iBOT-vMF)**:

    - **Function**: Learns latent representations of ice crystal habits from unlabeled CPI images.
    - **Mechanism**: Employs a teacher–student self-distillation framework in which the student learns to match the cluster assignments of the teacher. The key contribution is the introduction of von Mises-Fisher (vMF) distribution normalization, which naturally constrains embedding vectors to a hypersphere. Domain-specific data augmentation is applied: saturation and hue jitter are removed (images are monochromatic), random vertical flipping is added (ice crystals can rotate freely), and the range of random crop aspect ratio variation is reduced (to preserve the needle-like structure of crystals).
    - **Design Motivation**: The vMF distributional assumption naturally aligns with the clustering structure of ice crystal habits, and the vMF concentration parameter $\kappa$ can be directly used to quantify diversity.

2. **Hierarchical Sampling Data Curation**:

    - **Function**: Addresses the severe class imbalance in the CPI dataset.
    - **Mechanism**: Hierarchical sampling is performed in the learned latent space to curate a more uniformly distributed subset of 1.2 million images (CPI-H-1M) from the original 3.2 million, yielding a more balanced representation of habit classes in latent space.
    - **Design Motivation**: DINO-family methods are known to perform poorly when pre-trained on imbalanced data. Although the curated dataset is only one-third the size of the original, it yields superior training outcomes.

3. **Efficient Pre-training Strategy (ImageNet Initialization + Short Training)**:

    - **Function**: Achieves optimal performance with approximately 30× less computation.
    - **Mechanism**: The model is initialized with ImageNet pre-trained iBOT weights and fine-tuned on CPI-H-1M for only 10 epochs, rather than training from scratch for 100 epochs. This exploits the finding that ImageNet pre-trained features transfer across domains.
    - **Design Motivation**: Training from scratch on CPI-3M for 100 epochs incurs substantial computational overhead, while ImageNet features already transfer well to CPI images, requiring only minimal domain adaptation.

### Loss & Training

Training follows the standard iBOT cross-entropy loss. The student network is updated via gradient descent, while the teacher network is updated as an exponential moving average (EMA) of the student. Pre-training uses a batch size of 1024. Ice crystal diversity is estimated using the vMF concentration parameter:

$$\hat{\kappa} = \frac{\bar{R}(p - \bar{R}^2)}{1 - \bar{R}^2}$$

where $\bar{R}$ denotes the mean length of the normalized embedding vectors.

## Key Experimental Results

### Main Results

The primary evaluation task is classification on CPI-21K (21,000 manually annotated test images) using the learned representations:

| SSL Method | Pre-training Data | Epochs | ImageNet Init | kNN (%) | Logistic Reg. (%) |
|-----------|------------------|--------|---------------|---------|-------------------|
| DINOv3 | LVD-1689M | 1000 | ✗ | 74.83 | 81.83 |
| iBOT | ImageNet | 800 | ✗ | 78.33 | 82.00 |
| iBOT-vMF | CPI-3M | 100 | ✗ | 75.05 | 81.00 |
| iBOT-vMF | CPI-H-1M | 100 | ✗ | 77.67 | 83.17 |
| iBOT-vMF | CPI-H-1M | 10 | ✓ | **81.56** | **84.39** |

Baseline comparison: a logistic regression classifier using 13 geometric features achieves only 65% accuracy, far below the 84.39% attained by SSL representations.

### Ablation Study

| Configuration | Accuracy (%) | Notes |
|--------------|-------------|-------|
| Geometric feature baseline | 65.00 | Traditional image processing features |
| ImageNet SSL directly applied | 82.00 | Strong cross-domain transferability |
| From-scratch training on CPI-3M | 81.00 | Degraded by class imbalance |
| Training on curated CPI-H-1M | 83.17 | +2.17% from curation |
| Curation + initialization + short training | **84.39** | 30× computational efficiency gain |

### Key Findings

- **Data curation contributes most**: Training on CPI-H-1M (one-third the data) outperforms training on CPI-3M (83.17 vs. 81.00), confirming that class imbalance is the primary bottleneck for SSL.
- **ImageNet transfer is surprisingly effective**: A purely ImageNet pre-trained model achieves 82% on CPI classification, indicating strong transferability of natural image features to CPI images.
- **PCA projections reveal linear separability**: The 384-dimensional embeddings form three clearly separated clusters under PCA projection, indicating that the learned features are approximately linearly separable.
- **Crystal diversity varies with environmental conditions**: Higher temperature leads to greater diversity (lower $\kappa$); larger particles exhibit lower diversity (higher $\kappa$). Significant variation is observed across different field campaigns.

## Highlights & Insights

- **Natural alignment between vMF distributions and ice crystal clustering**: Using the vMF concentration parameter $\kappa$ to directly quantify habit diversity is more natural and continuous than conventional Shannon entropy. This idea generalizes to other scientific image domains with clustering structure.
- **"Curate first, train briefly" efficiency paradigm**: Performing hierarchical sampling in the latent space of a large dataset, followed by ImageNet initialization and short-epoch fine-tuning, achieves a 30× computational saving — a valuable recipe for science domains with limited computational resources.
- **Assumption-free diversity quantification via SSL**: Diversity can be quantified without predefined habit categories, avoiding information loss introduced by manual classification schemes.

## Limitations & Future Work

- **Limited dataset scale**: 3.2 million CPI images is not large by SSL standards and may constrain representation quality.
- **Only ViT-Small is evaluated**: Larger models may yield better representations but were precluded by computational constraints.
- **Downstream validation limited to classification**: Diversity quantification is presented qualitatively, without quantitative comparison against ground-truth references.
- **Anomaly detection and rare habit discovery not explored**: The authors identify these as future directions — leveraging SSL representations to detect mislabeled samples or discover rare ice crystal habits.
- **Insufficient comparison with other SSL methods**: The performance of methods such as MAE and SimCLR on CPI data remains unknown.

## Related Work & Insights

- **vs. traditional geometric feature methods**: Conventional methods using 13 hand-crafted features (e.g., aspect ratio) achieve only 65% accuracy, while SSL representations reach 84.39%, demonstrating the superiority of end-to-end learned features over hand-designed ones.
- **vs. DINOv3 large-scale pre-training**: DINOv3, pre-trained on 1.7 billion natural images, underperforms domain-specific curated training on CPI (81.83 vs. 84.39), suggesting that domain adaptation is more important than data scale.
- **vs. supervised CNN classification (Przybylo et al.)**: The prior VGG16 supervised approach requires extensive annotation, whereas the SSL method achieves comparable performance with no labels whatsoever.

## Rating

- **Novelty**: ⭐⭐⭐ — First application of SSL to ice crystal habit analysis, though the SSL methodology itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — Both classification validation and diversity analysis are conducted, but more quantitative comparisons are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; the connection between problem and method is natural.
- **Value**: ⭐⭐⭐ — Offers practical value to the climate science community, though methodological innovation is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)
- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](../../ICML2026/self_supervised/understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)

</div>

<!-- RELATED:END -->
