---
title: >-
  [Paper Note] Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation
description: >-
  [CVPR 2025][Segmentation][Weakly-Supervised Semantic Segmentation] This paper proposes a soft pseudo-label-based self-labeling method. By systematically evaluating multiple formulations of Potts relaxations and cross-entropy variants, it achieves segmentation performance close to or even exceeding fully supervised levels on standard network architectures using only scribble (3% of pixels) supervision, without any network structure modifications.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Weakly-Supervised Semantic Segmentation"
  - "Soft Pseudo-Labels"
  - "Potts Relaxation"
  - "Collision Cross-Entropy"
  - "Scribble Supervision"
date: 2026-05-08
content_hash: 44b2a28ad522eded
---

# Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2507.01721](https://arxiv.org/abs/2507.01721)  
**Code**: [https://vision.cs.uwaterloo.ca/code](https://vision.cs.uwaterloo.ca/code)  
**Area**: Image Segmentation / Weakly-Supervised Learning  
**Keywords**: Weakly-Supervised Semantic Segmentation, Soft Pseudo-Labels, Potts Relaxation, Collision Cross-Entropy, Scribble Supervision

## TL;DR

This paper proposes a soft pseudo-label-based self-labeling method. By systematically evaluating multiple formulations of Potts relaxations and cross-entropy variants, it achieves segmentation performance close to or even exceeding fully supervised levels on standard network architectures using only scribble (3% of pixels) supervision, without any network structure modifications.

## Background & Motivation

**Background**: Fully supervised semantic segmentation requires a large number of pixel-level annotations, which is extremely costly. Weakly-supervised methods (image-level labels, scribbles, bounding boxes) are important directions to reduce annotation costs. Among them, scribble supervision provides slightly more information than image-level labels with similar annotation costs, and has previously been shown to approach fully supervised performance without modifying the segmentation model. The ideological core of scribble supervision is designing unsupervised/self-supervised loss functions and stronger optimization algorithms.

**Limitations of Prior Work**: (1) Directly optimizing Potts relaxation (the most common unsupervised segmentation loss) via gradient descent has limited effectiveness, and even convex relaxations are challenging when combined with concave entropy terms; (2) existing self-labeling methods use **hard pseudo-labels** (one-hot distributions), which cannot represent the uncertainty and errors in class estimation, leading to the propagation of error signals; (3) there lacks a systematic comparison of different Potts relaxation formulations and cross-entropy variants in scribble-supervised fields.

**Key Challenge**: In the self-labeling framework, pseudo-labels need to be certain enough to guide network training while also being able to express uncertainty to prevent forcing erroneous labels onto the network. Hard pseudo-labels cannot achieve this balance.

**Goal**: (1) Introduce soft pseudo-labels into a self-labeling framework with principled convergence guarantees; (2) systematically evaluate different Potts relaxations and cross-entropy variants; (3) demonstrate that soft self-labeling can outperform complex and specialized weakly-supervised methods on standard architectures.

**Key Insight**: Based on the Alternating Direction Method (ADM) splitting method, the original weakly-supervised loss is decomposed into two subproblems—the network training subproblem and the pseudo-label optimization subproblem. In the latter, pseudo-labels are allowed to take soft values (probability distributions) rather than being restricted to one-hot vectors.

**Core Idea**: Utilize soft categorical distributions as pseudo-labels in conjunction with newly proposed Potts relaxation formulations, such as collision cross-entropy and normalized quadratic/collision divergence, to achieve improved weakly-supervised segmentation within a self-labeling framework that possesses convergence guarantees.

## Method

### Overall Architecture

The overall framework revolves around the alternating optimization of a joint loss function. This joint loss consists of three terms: (1) the negative log-likelihood (NLL) loss on labeled pixels (scribbles); (2) the cross-entropy term $H(\sigma_i, y_i)$ connecting network predictions $\sigma_i$ and soft pseudo-labels $y_i$ on unlabeled pixels; and (3) the Potts relaxation regularization $P(y_i, y_j)$ between pseudo-labels on unlabeled pixels. During alternating optimization: fixing the pseudo-labels to update network parameters (the standard training subproblem), and fixing network parameters to update soft pseudo-labels (the Potts optimization subproblem). Solving these two subproblems iteratively guarantees the convergence of the joint loss.

### Key Designs

1. **Soft Pseudo-Label Self-Labeling Framework**:

    - **Function**: Extends pseudo-labels from one-hot vectors to general probability distributions, enabling the representation of uncertainty in class estimation.
    - **Mechanism**: Employs ADM splitting to transform the weakly-supervised loss $-\sum_{i \in S} \ln \sigma_i^{\bar{y}_i} + \eta \sum_{i \notin S} H(\sigma_i) + \lambda \sum_{ij \in \mathcal{N}} P(\sigma_i, \sigma_j)$ into a joint loss by introducing auxiliary variables $y_i \in \Delta^K$. The pseudo-labels $y_i$ are general distributions defined on the $K$-class probability simplex, constrained to match the ground truth on seed pixels ($y_i = \bar{y}_i$) and freely optimized on unlabeled pixels. By using a combination of KL divergence and entropy to incorporate the constraint $\sigma_i \approx y_i$ into the loss, a concisely formulated joint loss is eventually obtained.
    - **Design Motivation**: Hard pseudo-labels irreversibly propagate classification errors to the network. In contrast, soft pseudo-labels naturally maintain high uncertainty at boundaries, reducing erroneous supervisory signals near boundaries. Compared to hard-label graph cut solvers, this better handles ambiguous regions.

2. **Systematic Study of Potts Relaxations (6 Formulations)**:

    - **Function**: Provides multiple choices for optimizing the Potts model in continuous domains, addressing the vanishing gradient and local optima issues associated with different relaxation formulations.
    - **Mechanism**: Investigates six relaxation formulations. Basic formulations: bilinear $P_{BL} = 1 - p^\top q$ (tight but non-convex), and quadratic $P_Q = \frac{1}{2}\|p-q\|^2$ (convex but not tight). A newly proposed normalized quadratic $P_{NQ} = 1 - \frac{p^\top q}{\|p\|\|q\|}$ combines the advantages of both. Logarithmic variants: collision cross-entropy $P_{CCE} = -\ln p^\top q$, collision divergence $P_{CD} = -\ln \frac{p^\top q}{\|p\|\|q\|}$, and log-quadratic $P_{LQ} = -\ln(1-\frac{\|p-q\|^2}{2})$. The logarithmic variants resolve the vanishing gradient problem present in the basic forms.
    - **Design Motivation**: By analyzing two representative "movement" scenarios (re-classification within the same region and label switching in boundary regions), it is revealed that bilinear relaxation yields local optima in the former scenario, whereas quadratic relaxation yields local optima in the latter. Normalization eliminates both issues, and the logarithmic transformation resolves the vanishing gradient in flat regions.

3. **Collision Cross-Entropy**:

    - **Function**: A loss function that links network predictions and soft pseudo-labels, demonstrating robustness to label uncertainty.
    - **Mechanism**: Defined as $H_{CCE}(y_i, \sigma_i) = -\ln \sum_k \sigma_i^k y_i^k = -\ln \sigma^\top y$. The dot product $\sigma^\top y$ can be interpreted as the probability that the predicted class $C$ and the true class $T$ are identical: $\Pr(C=T) = \sum_k \Pr(C=k)\Pr(T=k)$. This loss maximizes the "collision probability" rather than forcing the distributions to be identical. Furthermore, it is symmetric with respect to both arguments—neither forcing predictions to mimic the uncertainty of pseudo-labels, nor forcing pseudo-labels to mimic the uncertainty of predictions.
    - **Design Motivation**: Standard cross-entropy $H_{CE}$ forces the network to replicate pseudo-label uncertainty (e.g., when $y=(0.5,0.5)$, the network learns to output $(0.5,0.5)$); reverse cross-entropy $H_{RCE}$ resolves this direction but causes pseudo-labels to mimic predictions instead. Due to its symmetry, collision cross-entropy permanently resolves mutual uncertainty imitation issues.

### Loss & Training

Uses the DeepLabv3+ architecture with ResNet-101/MobileNetV2/ViT backbones. The network parameters are first warmed up using the cross-entropy loss on scribbles, followed by 60 epochs of alternating optimization under the joint loss. SGD optimizer is utilized with an initial learning rate of 0.0007 under polynomial decay. Optimal hyperparameters are $\eta=0.3$ and $\lambda=6$, using collision cross-entropy $H_{CCE}$ and collision divergence $P_{CD}$. The pseudo-label subproblem is solved using more GPU-friendly gradient descent.

## Key Experimental Results

### Main Results

| Method | Architecture | Supervision | PASCAL VOC mIoU |
|------|------|------|-----------------|
| Full supervision | V3+ (R101) | Full Pixel | 76.6 |
| Full supervision | V3+ (R101, bs16) | Full Pixel | 78.9 |
| Full supervision | ViT-linear | Full Pixel | 81.4 |
| **Soft SL (ours)** | V3+ (R101) | **Scribble** | **76.7** |
| **Soft SL (ours)** | ViT-linear | **Scribble** | **81.6** |
| Hard SL [29] | V3+ (R101) | Scribble | 69.6 |
| GD baseline [38] | V3+ (R101) | Scribble | 69.5 |
| BPG [41] | V2 (△) | Scribble | 76.0 |

### Ablation Study (Comparison of Potts Relaxations using MobileNetV2)

| Relaxation Formulation | scribble=0 | scribble=0.5 | scribble=1.0 |
|----------|-----------|-------------|-------------|
| $P_{BL}$ (Bilinear) | 56.42 | 63.81 | 67.24 |
| $P_Q$ (Quadratic) | 58.92 | 67.81 | 71.05 |
| $P_{NQ}$ (Normalized Quadratic) | 59.01 | 67.80 | 71.12 |
| $P_{CCE}$ (Collision Cross-Entropy) | 56.40 | 63.81 | 67.41 |
| $P_{CD}$ (Collision Divergence) | **59.04** | **67.84** | **71.22** |
| $P_{LQ}$ (Log-Quadratic) | 59.03 | 67.81 | 71.21 |

### Key Findings

- **Soft self-labeling with scribbles can outperform full-pixel supervision**: On the ViT-linear architecture, scribble-supervised performance (81.6 mIoU) exceeds fully supervised performance (81.4 mIoU), which is a remarkable outcome.
- **Collision cross-entropy consistently outperforms**: Across different supervision levels, $H_{CCE}$ consistently outperforms standard cross-entropy and reverse cross-entropy, validating its robustness to label uncertainty.
- **Logarithmic relaxation variants surpass basic ones**: The logarithmic transformation consistently improves the performance of all relaxation formulations by addressing the vanishing gradient problem and encouraging smoother transitions at boundaries.
- **Nearest Neighbor (NN) outperforms Dense Neighborhood (DN)**: NN (71.1%) vs. DN (67.9%), because a large neighborhood degrades the Potts model into volumetric potentials, which is disadvantageous for boundary alignment.
- **Normalized relaxations eliminate local optima of basic formulations**: $P_{NQ}$ and $P_{CD}$ successfully avoid the respective issues associated with bilinear and quadratic relaxations.

## Highlights & Insights

- **3% pixel annotation outperforms 100% annotation**: This represents a major methodological breakthrough, showing that proper unsupervised loss design and optimization methods can be more effective than simple full supervision, as scribbles plus regularization prevent annotation noise inherent in full supervision.
- **Probabilistic interpretation of collision cross-entropy**: Framing alignment between soft labels as the "probability that two random variables are equal" rather than "equal distributions" is an elegant perspective shift that can be widely applied to other scenarios involving soft/pseudo labels.
- **Generality of the method**: Contributing purely at the optimization methodology level, the proposed approach does not rely on specific architectures or training tricks, allowing it to be directly transferred to other weakly-supervised tasks (such as point or bounding-box supervision).

## Limitations & Future Work

- The pseudo-label subproblem is solved via gradient descent, which might fall into local optima for non-convex relaxations; developing specialized convex or combinatorial optimization solvers could yield further improvements.
- Evaluation is limited to PASCAL VOC, Cityscapes, and ADE20k; validation on larger-scale datasets is currently missing.
- The choice of neighborhood system (NN vs. DN) significantly affects results and currently relies on manual selection.
- While robust, collision cross-entropy is not the most rigorous information-theoretic metric; its theoretical foundations could be further strengthened.

## Related Work & Insights

- **vs. GCRF [28]**: An early scribble-based segmentation method using graph cuts to solve hard pseudo-labels. This work extends from hard to soft pseudo-labels with a joint loss optimization scheme that possesses convergence guarantees.
- **vs. ADM/Trust-Region [30,29]**: These methods already utilize ADM splitting but are limited to hard pseudo-labels. In contrast, this represents the first work to utilize soft pseudo-labels within a principled framework and study their performance under Potts relaxations.
- **vs. BPG [41]**: BPG uses modified architectures and soft pseudo-label proposals but lacks convergence guarantees. This work achieves better performance on standard architectures using mathematically proven methods.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Collision cross-entropy represents an entirely new loss design, and the systematic study of Potts relaxations fills theoretical gaps.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically evaluates multiple combinations, though the scope of datasets could be expanded.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical derivations are rigorous and clear, with tight integration of theory and experiments, making it an excellent methodological paper.
- Value: ⭐⭐⭐⭐⭐ Achieving scribble performance that outperforms full supervision is a milestone result, and collision cross-entropy has broad potential for application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2025\] DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation](dformerv2_geometry_self-attention_for_rgbd_semantic_segmentation.md)
- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](../../NeurIPS2025/segmentation/exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)

</div>

<!-- RELATED:END -->
