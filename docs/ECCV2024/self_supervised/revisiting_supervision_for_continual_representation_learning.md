---
title: >-
  [Paper Note] Revisiting Supervision for Continual Representation Learning
description: >-
  [ECCV 2024][Self-Supervised Learning][continual learning] This work challenges the common belief that "self-supervised learning outperforms supervised learning in continual representation learning." It reveals that **supervised learning with an MLP projector** can build stronger representations than SSL in continual learning scenarios. The key lies not in the presence or absence of labels, but in the enhancement of feature transferability by the MLP projector.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "continual learning"
  - "Representation Learning"
  - "MLP Projector"
  - "Feature Transferability"
date: 2026-05-08
content_hash: 88467ac9e39cce93
---

# Revisiting Supervision for Continual Representation Learning

**Conference**: ECCV 2024  
**arXiv**: [2311.13321](https://arxiv.org/abs/2311.13321)  
**Code**: [GitHub](https://github.com/danielm1405/sl-vs-ssl-cl)  
**Area**: Self-Supervised Learning / Continual Learning  
**Keywords**: continual learning, Representation Learning, Self-Supervised Learning, MLP Projector, Feature Transferability

## TL;DR

This work challenges the common belief that "self-supervised learning outperforms supervised learning in continual representation learning." It reveals that **supervised learning with an MLP projector** can build stronger representations than SSL in continual learning scenarios. The key lies not in the presence or absence of labels, but in the enhancement of feature transferability by the MLP projector.

## Background & Motivation

Continual Learning requires models to incrementally learn over a sequence of tasks without forgetting old knowledge. Recent studies generally believe that Self-Supervised Learning (SSL) outperforms Supervised Learning (SL) in continual representation learning, as SSL produces more robust representations with less forgetting. While widely accepted, this view is puzzling: **why would additional annotation information (supervisory signals) lead to inferior representations?**

The authors notice that SSL methods (SimCLR, BarlowTwins, BYOL, etc.) commonly employ an **MLP projector**, whereas standard SL directly connects to a linear classification head. Recent transfer learning research (Wang et al. 2021, Sariyildiz et al. 2023) has shown that the MLP projector is a key component in enhancing the feature transferability of supervised models, narrowing the transferability gap between SL and SSL.

This paper introduces this insight into continual learning: in a continual fine-tuning scenario, is adding an MLP projector to SL sufficient to outperform SSL?

Core Argument: The advantage of SSL in continual learning does not stem from "unsupervised" training itself, but rather from the improvement in feature transferability brought by the MLP projector. By simply adding an MLP projector (used during training, discarded during testing) to SL, higher-quality representations can be constructed in continual learning.

## Method

### Overall Architecture

The method is extremely simple: inserting an MLP projector between the backbone and the classification head in standard supervised learning:

- **SL**: Backbone → Linear Head → Cross-Entropy Loss
- **SL+MLP**: Backbone → MLP Projector → Linear Head → Cross-Entropy Loss

The MLP projector only exists during training and is discarded during testing. Evaluation is performed using k-NN classification on the representations output by the backbone.

### Key Designs

#### 1. MLP Projector Structure

**Function**: Add a non-linear projection layer between the backbone and the classification head to enhance feature transferability.

**Core Structure**: Each block contains Linear → BatchNorm → ReLU, followed by an output linear layer. It adopts the MLPP (Wang et al. 2021) architecture, with a hidden dimension of $d_h=4096$ and an output dimension of $d_o=512$.

**Design Motivation**: The MLP projector isolates the classification-task-specific information within the projection layers, enabling the backbone to generate more general and transferable representations. Discarding the projector allows the backbone features to maintain stronger adaptability to unseen tasks. Ablation studies show that **BatchNorm is the most critical component**, although the complete Linear+BN+ReLU combination yields the best results.

#### 2. Synergy with Continual Learning Strategies

**Function**: Verify the compatibility of SL+MLP with existing CL methods.

**Experimental Combinations**:

- SL+MLP + LwF (Knowledge Distillation)
- SL+MLP + PFR (Projection-based Feature Regularization)
- SupCon + CaSSLe / PFR

**Findings**: The combination of SL+MLP and PFR achieves the best performance, because PFR utilizes a learnable projection to enhance feature distillation, aligning with the design philosophy of the MLP projector.

#### 3. Representation Quality Analysis

The authors analyze the reasons for the superior representation quality of SL+MLP from multiple perspectives:

- **Lower Forgetting**: SL+MLP achieves the lowest representation forgetting rate (C10→C100: 4.5%), significantly lower than SL (12.4%) and SSL (9.7%).
- **Preservation of Task-Specific Features (EXC)**: SL+MLP achieves the highest EXC score (4.3), indicating its ability to retain old task features. Surprisingly, SSL scored negative (-1.8), meaning that after training on old tasks and fine-tuning on new ones, the representations of old tasks actually deteriorated.
- **Increasing Feature Diversity**: Singular value spectrum analysis reveals that SL+MLP is the only method where feature diversity continuously increases during continual learning, while SL exhibits neural collapse and SSL's diversity remains largely unchanged.

### Loss & Training

- Standard Cross-Entropy Loss is used.
- SL is trained for 100 epochs/task, while SSL is trained for 500 epochs/task.
- SGD optimizer with a cosine schedule is adopted.
- SL+MLP **randomly re-initializes the projector** for each new task (ablation studies show that re-initialization outperforms weight inheritance under the SL scheme).

## Key Experimental Results

### Main Results

k-NN accuracy in continual fine-tuning scenarios (%, ResNet-18 backbone):

| Method | C10/5 | C100/5 | C100/20 | IN100/5 |
|------|:---:|:---:|:---:|:---:|
| SL (without projector) | 59.8 | 45.3 | 23.1 | 40.4 |
| **SL+MLP** | **65.9** | **61.9** | **47.1** | **62.4** |
| t-ReX | 69.3 | 59.2 | 50.8 | 59.2 |
| SupCon | 60.4 | 49.4 | 30.0 | 57.6 |
| BarlowTwins (SSL) | 76.2 | 54.1 | 40.0 | 57.0 |
| SimCLR (SSL) | 72.4 | 48.9 | 33.4 | 54.7 |

SL+MLP outperforms SSL comprehensively on C100/5, C100/20, and IN100/5. Although SSL leads on C10/5, the gap is closed when integrated with CL methods.

### Ablation Study

Ablation of MLP Projector components (CIFAR100/5, k-NN accuracy %):

| Linear | ReLU | BatchNorm | Acc (%) |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 46.74 |
| ✗ | ✓ | ✗ | 49.06 |
| ✓ | ✗ | ✗ | 47.59 |
| ✓ | ✓ | ✗ | 53.87 |
| ✗ | ✗ | ✓ | 57.33 |
| ✗ | ✓ | ✓ | 59.15 |
| ✓ | ✗ | ✓ | 58.47 |
| ✓ | ✓ | ✓ | **61.46** |

BatchNorm contributes most (+10.6 percentage points compared to not using any components), and the full block configuration performs best. Ablation on projector depth reveals that **one block is sufficient**, with deeper layers providing almost no additional gains.

Effect of combining with CL strategies (k-NN accuracy %):

| Method | CL Strategy | C100/5 | IN100/5 |
|------|---------|:---:|:---:|
| SL+MLP | Finetune | 61.9 | 62.4 |
| SL+MLP | PFR | **63.6** | **65.2** |
| BarlowTwins | Finetune | 54.1 | 57.0 |
| BarlowTwins | CaSSLe | 58.6 | 64.9 |

### Key Findings

1. **All supervised methods using an MLP projector** (SL+MLP, t-ReX, SupCon) significantly outperform SL without a projector, despite using different loss functions (CE, cosine softmax CE, contrastive loss). This indicates that **the performance key is the projector rather than the loss function**.
2. SL+MLP can outperform BarlowTwins trained on the full dataset using only **30% of the data**.
3. SL+MLP still outperforms SSL under **40% label noise**, demonstrating good robustness against noisy annotations.
4. Transfer learning evaluation (average of 8 downstream datasets): SL+MLP 41.9% vs SSL 36.1% vs SL 23.3%.

## Highlights & Insights

- **Counter-intuitive conclusion**: Annotations should not degrade continual learning representations; the MLP projector is the actual source of SSL's advantages.
- **Extremely simple method**: Simply add a standard MLP between the backbone and the classification head without changing the training pipeline.
- **SL+MLP is the only method that continuously accumulates knowledge across sequential tasks**: For other methods, the task-agnostic accuracy either decreases (SL) or stagnates (SSL), whereas SL+MLP steadily improves.
- **Negative EXC for SSL**: This indicates that pre-training on old tasks with SSL is worse than training from scratch, which challenges the common belief that "SSL is naturally resilient to forgetting."

## Limitations & Future Work

1. **Annotation cost**: SL+MLP still requires labeled data. Although it outperforms SSL with only 30% of the data, the cost of labeling itself might be higher than acquiring a larger amount of unlabeled data.
2. **Only representation quality evaluated**: The class-incremental learning problem—namely, the continuous updating of the classification head—is not directly addressed.
3. **Unaligned training epochs**: SL is trained for 100 epochs vs SSL for 500 epochs. Although the authors explain the reasoning behind this, it remains a point of contention for a fair comparison.
4. **Evaluation restricted to ResNet-18**: It remains unclear whether the conclusions hold consistently on larger backbones or ViT architectures.

## Related Work & Insights

- **Wang et al. (CVPR 2021)**: Discovered that MLP projectors can enhance the transfer learning ability of supervised models; this work generalizes this finding to continual learning.
- **CaSSLe (CVPR 2022)**, **PFR (CVPR Workshop 2021)**: Continual learning methods for SSL that use learnable projections for feature distillation.
- **Madaan et al. (ICLR 2022)**: Argued that SSL outperforms SL in continual representation learning; this paper directly refutes this claim.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Very simple method but deep insights. The conclusion that "simply adding an MLP can reverse the SL vs. SSL comparison" is highly disruptive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 datasets × various CL and learning methods × 8 downstream tasks × rich ablation studies and analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized, with a clear intuitive illustration in Figure 1 and in-depth analytical sections.
- **Value**: ⭐⭐⭐⭐ — Holds significant cognitive-correction value for the continual learning community; the method is simple and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)
- [\[AAAI 2026\] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](../../AAAI2026/self_supervised/improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)
- [\[ECCV 2024\] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)
- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](../../ICLR2026/self_supervised/fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)
- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/self_supervised/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)

</div>

<!-- RELATED:END -->
