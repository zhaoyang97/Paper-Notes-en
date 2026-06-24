---
title: >-
  [Paper Note] Streamlined Knowledge Distillation
description: >-
  [CVPR 2026][Model Compression][logit distillation] This paper points out that the increasing complexity of recent logit distillation (stacking multiple knowledge alignments and relationship modeling) leads to redundant objectives and improper losses. It proposes a minimalist approach, SKD, which transfers only two types of knowledge: "instance-level" semantics via KL divergence and "direction-level" relations via the Gram matrix of normalized logits. For the latter…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "logit distillation"
  - "directional relations"
  - "Gram matrix"
  - "Mahalanobis distance"
  - "covariance whitening"
date: 2026-05-08
content_hash: cff23092817f6416
---

# Streamlined Knowledge Distillation

**Conference**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Jeong_Streamlined_Knowledge_Distillation_CVPR_2026_paper.html)  
**Code**: https://github.com/HyunJunSik/StreamLined  
**Area**: Model Compression / Knowledge Distillation  
**Keywords**: logit distillation, directional relations, Gram matrix, Mahalanobis distance, covariance whitening

## TL;DR
This paper points out that the increasing complexity of recent logit distillation (stacking multiple knowledge alignments and relationship modeling) leads to redundant objectives and improper losses. It proposes a minimalist approach, SKD, which transfers only two types of knowledge: "instance-level" semantics via KL divergence and "direction-level" relations via the Gram matrix of normalized logits. For the latter, a Mahalanobis distance loss stabilized by Tikhonov regularization and Cholesky decomposition is designed (provably equivalent to the L2 norm in a covariance-whitened space). SKD not only outperforms all logit distillation methods but also exceeds feature distillation on CIFAR-100/ImageNet/COCO, while being the fastest to train.

## Background & Motivation
**Background**: Knowledge Distillation (KD) transfers knowledge from a large teacher model to a small student model. It is categorized into feature distillation, which aligns intermediate features, and logit distillation, which aligns only the output logits. Logit distillation has gained attention for being more lightweight, as it does not requires access to intermediate features, and for avoiding security risks like backdoors being implanted in intermediate layers of heterogeneous teacher-student pairs.

**Limitations of Prior Work**: Historically, logit distillation performance has been inferior to feature distillation. Thus, recent methods (such as DKD decoupling instance knowledge, MLKD augmenting both instance and directional knowledge, and SDD using multi-scale pooling for directional knowledge) have continuously stacked "multi-knowledge alignment + relational structure modeling" to catch up. However, this complexity introduces three issues: (1) alignment objectives are **redundant and overlapping**, slowing down training and increasing complexity; (2) relational structures in the output space are often destroyed by **improper transformations** (e.g., non-linear distortions from softmax or temperature scaling); (3) using the **L2 norm** as a loss function treats all relations equally, ignoring the uneven variance of different directional relations, which allows high-variance directions to dominate training and distort meaningful structures.

**Key Challenge**: The complexity added to match feature distillation has become a burden—"transferring more" does not necessarily mean "transferring better." Instead, students are overwhelmed by excessive, overlapping knowledge.

**Goal**: Return to the essence—transfer only two complementary and non-redundant types of knowledge (instance-level semantics + direction-level relations) while ensuring the direction-level loss is numerically stable and variance-sensitive in the output space.

**Key Insight**: The authors observe that the Gram matrix effectively characterizes directional relations between samples. However, applying it directly to the output space encounters issues with scale instability and the unsuitability of the L2 norm. The solution is to "purify" directional information through normalization and then utilize a distance that "whitens" the variance.

**Core Idea**: Use a single, original KL divergence for instance-level knowledge (avoiding multiple distributions) and use the "Gram matrix of normalized logits + Mahalanobis distance loss" for direction-level knowledge. The latter is essentially an L2 norm in a covariance-whitened space, maintaining the simplicity of L2 while automatically correcting for variance and correlation.

## Method

### Overall Architecture
SKD is a pure logit distillation framework with a total loss consisting of only two terms: $L_{\text{SKD}} = L_{\text{INS}} + L_{\text{DIR}}$. It introduces no projection layers, auxiliary modules, or multiple alignment objectives. Given a batch of $B$ samples, the teacher and student output logits $z_t, z_s \in \mathbb{R}^{B\times C}$. The **instance branch** directly applies KL divergence to the temperature-softened outputs to align per-sample class distributions. The **direction branch** first normalizes each row of logits into unit vectors to calculate a $B\times B$ Gram matrix (i.e., pairwise cosine similarities), then aligns the difference between the teacher and student Gram matrices using a Mahalanobis distance loss. The method follows a "two parallel losses" structure without a multi-stage serial pipeline. The three key designs are: single instance knowledge, Gram construction for directional knowledge, and a stabilized directional loss.

### Key Designs

**1. Instance-level Knowledge: Returning to Single KL**

Recent methods use two sets of $P$ distributions (DKD) or augmented logits (MLKD) to create multiple instance-level knowledge sources. The authors argue this is the source of redundancy. SKD reverts to Hinton’s original form using a single soft target:
$$L_{\text{INS}} = \text{KL}\!\left(\text{softmax}(z_t/\tau),\ \text{softmax}(z_s/\tau)\right),$$
where $\tau$ is the temperature. This term allows the student to align with the teacher's per-sample output distribution and capture inter-class similarities, but it naturally focuses on individual samples and ignores structural relationships between samples—a gap that the direction branch fills. Reducing instance knowledge back to "one" rather than "many" is the first step in streamlining SKD.

**2. Direction-level Knowledge: Constructing Gram Matrix via EBN**

The Gram matrix, originally used to characterize feature correlations between CNN channels, effectively expresses directional relations between representations and has been widely used in feature distillation. MLKD brought it to the output space, but direct application faces two issues: the output space lacks BN to control scale, leading to large batch fluctuations, and transformations like softmax distort directional structures. Ours uses **Euclidean Batch Normalization (EBN)** to normalize each row of logits into unit vectors, retaining pure direction and removing scale:
$$\hat z_i = \frac{z_i}{\|z_i\|_2},\qquad G = \hat z\,\hat z^\top,\quad G_{ij} = \langle \hat z_i, \hat z_j\rangle.$$
Each element of $G\in\mathbb{R}^{B\times B}$ is the cosine similarity of the normalized logits of samples $i$ and $j$, characterizing the pairwise directional relations within the batch. This step ensures that the subsequent alignment targets "pure directional structure" rather than scale-contaminated values.

**3. Direction Loss: Mahalanobis Distance + Tikhonov/Cholesky Stabilization**

EBN preserves directional relations, but the strength of relations varies across sample pairs; some directions are far less stable than others. If a standard L2 norm treats all equally, high-variance directions will dominate. SKD thus builds the direction loss on the **Mahalanobis distance**. Defining the difference between teacher and student Gram matrices as $D = G_s - G_t \in \mathbb{R}^{B\times B}$, each row $D_{i,:}$ is treated as a directional relation vector for that sample across the batch. The empirical covariance is estimated as $\Sigma = \text{Cov}(\{D_{i,:}\})$, and the loss is:
$$L_{\text{DIR}} = \frac{1}{B}\sum_{i=1}^B \sqrt{D_{i,:}^\top\,\Sigma^{-1}\,D_{i,:}}.$$
To address the potential non-positive definiteness of $\Sigma$ (causing numerical instability) and the $\mathcal{O}(B^3)$ bottleneck of inversion, the authors apply two stabilizations: **Tikhonov regularization** $\Sigma' = \Sigma + \lambda I$ ($\lambda>0$) to ensure positive definiteness, and **Cholesky decomposition** $\Sigma' = LL^\top$ to stabilize the inversion. The final loss is:
$$L_{\text{DIR}} = \frac{1}{B}\sum_{i=1}^B \sqrt{D_{i,:}^\top\,\Sigma'^{-1}\,D_{i,:}}.$$
The authors provide a formal proof: since $(LL^\top)^{-1} = (L^\top)^{-1}L^{-1}$, the equation can be rewritten as $L_{\text{DIR}} = \frac{1}{B}\sum_i \|L^{-1}D_{i,:}\|_2$. Thus, **this Mahalanobis loss is exactly equivalent to performing the L2 norm in a covariance-whitened space**. This equivalence is the highlight of the design: it maintains the simplicity of the L2 form while inherently perceiving variance and correlation.

### Loss & Training
- Total loss $L_{\text{SKD}} = L_{\text{INS}} + L_{\text{DIR}}$, with both terms equally weighted (no tuning needed).
- Key hyperparameters are only temperature $\tau$ and Tikhonov factor $\lambda$. Implementation is simple (~15 lines of PyTorch-style pseudocode: `kl_div` + `normalize` + `mm` for Gram + `torch.cov` with $\lambda I$ + `cholesky`/`cholesky_inverse` + `einsum` for Mahalanobis + `sqrt().mean()`).
- CIFAR-100 trained for 240 epochs using SGD (lr 0.1, decayed at 150/180/210); ImageNet for 100 epochs (lr 0.2, decayed every 30 epochs); COCO using Faster R-CNN-FPN. Experiments are averaged over 5 runs.

## Key Experimental Results

### Main Results
Top-1 Accuracy for homogeneous teacher-student pairs on CIFAR-100 (selected):

| Teacher → Student | KD [16] | DKD [42] | MLKD [17] | RLD [30] | **SKD (Ours)** |
|------|------|------|------|------|------|
| ResNet32x4 → ResNet8x4 | 74.75 | 76.51 | 75.59 | 76.64 | **78.33** |
| WRN-40-2 → WRN-16-2 | 76.04 | 76.41 | 76.83 | 76.06 | **76.60** |
| VGG13 → VGG8 | 74.08 | 74.41 | 74.25 | 74.93 | **75.75** |
| ResNet56 → ResNet20 | 71.74 | 71.17 | 72.21 | 72.00 | **72.50** |

Heterogeneous pairs (Table 2, selected): SKD achieves 77.91 on ResNet32x4→ShuffleNetV1, outperforming MLKD (77.57) and DKD (76.75). On VGG13→MobileNetV1, it reaches 68.60. ImageNet (Table 3): R34→R18 reached 71.13 and R50→MV1 reached 71.53. For COCO detection (Table 4), SKD’s AP is on par with or exceeds MLKD and **surpasses some feature distillation methods**.

### Ablation Study
CIFAR-100, ResNet32x4 → ResNet8x4 (Table 5):

| Configuration | Top-1 | Description |
|------|------|------|
| Baseline (softmax + L2 direction loss) | 76.61 | Starting point |
| + EBN (Normalized Gram) | 77.28 | Pure direction info, +0.67 |
| + Mahalanobis Loss (Replacing L2) | 77.76 | Variance awareness, +0.48 |
| + Stabilization (Tikhonov + Cholesky) | **78.33** | Numerical robustness, +0.57 |

### Key Findings
- **Every component contributes positively and cumulatively**: EBN → Mahalanobis → Stabilization add +0.67/+0.48/+0.57 respectively, proving that "purifying direction + whitening variance + numerical robustness" are all essential.
- **Students often surpass the Teacher** (Table 6): e.g., WRN-40-2→WRN-16-2 gap of -1.01, VGG13→VGG8 gap of -1.11 (negative indicates student > teacher). This is attributed to directional knowledge encouraging more structured representation spaces.
- **Can be used as an "add-on" for feature distillation**: Stacking SKD on FitNet/RKD yields gains of 0.11%–3.55% (Table 7), as it requires no projection layers and integrates seamlessly.
- **Generalization to ViT**: On RegNetY-16GF→ViT-Tiny, SKD reaches 68.68, exceeding DeiT (68.03) and Swin (67.88), showing effectiveness for non-CNN architectures.
- **Fastest Training**: The per-batch training time for ResNet56→ResNet20 is the shortest among representative logit and feature distillation methods (Fig. 5).

## Highlights & Insights
- **The "whitened L2" equivalence proof is ingenious**: Proving a seemingly complex Mahalanobis loss is an "L2 norm in a covariance-whitened space" provides both theoretical grounding and an explanation for its simplicity and variance sensitivity. This approach of "changing the coordinate system to simplify complex losses" is transferable to any relation-alignment task with unequal variance.
- **Success through Subtraction**: In a trend of "increasingly complex" logit distillation, Ours goes against the grain. By keeping only two non-redundant knowledge types, it outperforms complex methods, reminding the field that "transferring correctly" is better than "transferring more."
- **Identifying the hurdles of direction knowledge in logit space**: The authors pinpoint scale instability (solved by EBN) and the unsuitability of L2 (solved by Mahalanobis). This paradigm of "normalize to purify, then whiten to align" is valuable for other Gram/relation matrix distillation methods.
- **Plug-and-play**: Requires no additional projections or modules, making it engineering-friendly and capable of augmenting existing feature distillation.

## Limitations & Future Work
- The authors acknowledge that the covariance $\Sigma'$ in $L_{\text{DIR}}$ might be unstable under **large batches or noisy labels**, affecting training stability; future work plans to improve covariance estimation.
- Covariance inversion is $\mathcal{O}(B^3)$. Although mitigated by Cholesky, the cost of directional loss is still bounded by $B$ as batches grow very large.
- Evaluation focused on classification and detection; the effectiveness in sequential and multimodal domains remains to be verified.
- The sensitivity and specific values for temperature $\tau$ and Tikhonov $\lambda$ were not fully explored in the main text.

## Related Work & Insights
- **vs MLKD [17]**: MLKD augments both instance and direction knowledge and uses Gram matrices in output space but relies on L2. SKD points out redundancy and the unsuitability of L2, using single KL + whitened Mahalanobis for superior results.
- **vs DKD [42]**: DKD decouples multiple instance distributions for alignment; SKD proves "multiple instance knowledge" is unnecessary by using only a single KL.
- **vs SDD [36]**: SDD uses multi-scale pooling for directional knowledge; SKD uses a single normalized Gram matrix, being lighter yet stronger.
- **vs Feature Distillation (FitNet/RKD/ReviewKD)**: Feature distillation requires projection modules and is computationally heavy with potential backdoor risks. SKD is pure logit, requires no projection, and can even serve as a performance booster for them.

## Rating
- Novelty: ⭐⭐⭐⭐ The "whitened L2" proof and minimalist two-knowledge design are insightful, though individual components (Gram, Mahalanobis) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers CIFAR/ImageNet/COCO, homogeneous/heterogeneous, ViT, feature distillation add-ons, visualization, and training time.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with formal proofs; some formula formatting (cache) in the draft was messy but the original is likely well-organized.
- Value: ⭐⭐⭐⭐⭐ Provides a simple, strong, and plug-and-play baseline for logit distillation with high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs](selectkd_selective_token-weighted_knowledge_distillation_for_llms.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[CVPR 2026\] QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](qkd_quantum_gated_incremental_learning.md)
- [\[ICLR 2026\] Exploring Knowledge Purification in Multi-Teacher Knowledge Distillation for LLMs](../../ICLR2026/model_compression/exploring_knowledge_purification_in_multi-teacher_knowledge_distillation_for_llm.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](distilling_balanced_knowledge_from_a_biased_teacher.md)

</div>

<!-- RELATED:END -->
