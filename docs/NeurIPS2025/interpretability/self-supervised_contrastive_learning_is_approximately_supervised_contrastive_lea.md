---
title: >-
  [Paper Note] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning
description: >-
  [NeurIPS 2025][Contrastive Learning] This paper theoretically proves that self-supervised contrastive learning (DCL) is approximately equivalent to a supervised contrastive loss (NSCL), with the gap vanishing at rate $O(1/C)$ as the number of classes increases. It further proves that the global optimum of NSCL satisfies Neural Collapse (augmentation collapse + within-class collapse + Simplex ETF), and proposes a tighter few-shot error bound based on directional CDNV.
tags:
  - NeurIPS 2025
  - Contrastive Learning
  - Self-Supervised Learning
  - Supervised Contrastive Loss
  - Neural Collapse
  - Few-Shot Learning
date: 2026-05-08
content_hash: 6c1273a90cff607c
---

# Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.04411](https://arxiv.org/abs/2506.04411)
**Code**: Available (project page)
**Area**: Interpretability
**Keywords**: Contrastive Learning, Self-Supervised Learning, Supervised Contrastive Loss, Neural Collapse, Few-Shot Learning

## TL;DR
This paper theoretically proves that self-supervised contrastive learning (DCL) is approximately equivalent to a supervised contrastive loss (NSCL), with the gap vanishing at rate $O(1/C)$ as the number of classes increases. It further proves that the global optimum of NSCL satisfies Neural Collapse (augmentation collapse + within-class collapse + Simplex ETF), and proposes a tighter few-shot error bound based on directional CDNV.

## Background & Motivation

**Background**: Self-supervised contrastive learning methods (e.g., SimCLR, MoCo) learn representations from unlabeled data that match or exceed supervised learning on downstream tasks. However, the theoretical foundation remains incomplete — why does a loss function with no access to labels learn representations that cluster semantically by class?

**Limitations of Prior Work**:
   - Existing theoretical analyses (e.g., Arora et al., Saunshi et al.) rely on strong assumptions such as conditional independence of augmented views given class labels, limiting the generality of their conclusions.
   - Alignment and uniformity analyses characterize properties of representations but do not explain why CL organizes class structure.
   - Neural Collapse is well understood for supervised classification losses but has not been connected to self-supervised contrastive learning.

**Key Challenge**: In self-supervised CL, same-class samples are treated as negatives in the denominator and pushed apart, which should theoretically hurt within-class compactness — yet representations still cluster by class. This apparent contradiction demands explanation.

**Goal**: (1) Formally connect SSL CL to supervised CL; (2) characterize the geometric structure of representations learned by CL; (3) explain why CL representations support few-shot transfer.

**Key Insight**: When the number of classes $C$ is large, the fraction of same-class samples among negatives is small ($\sim 1/C$), so the extra terms in the DCL denominator from same-class samples become negligible, making DCL approximately equivalent to NSCL, which excludes same-class negatives.

**Core Idea**: Self-supervised contrastive learning is essentially optimizing a supervised contrastive loss approximately, with the gap vanishing as the number of classes grows.

## Method

### Overall Architecture
This is a theory-driven work. Three core contributions build on each other: first proving DCL ≈ NSCL (Theorem 1) → then proving the global optimum of NSCL satisfies Neural Collapse (Theorem 2) → then proposing a tighter few-shot error bound (Proposition 1), together providing a complete explanation of why CL works.

### Key Designs

1. **DCL–NSCL Duality (Theorem 1)**:

    - **Function**: Proves an upper bound on the gap between the DCL and NSCL losses.
    - **Mechanism**: The DCL denominator sums over all $j \neq i$, while the NSCL denominator sums only over $y_j \neq y_i$. The extra same-class terms number at most $K(n_{\max}-1)$, which is small relative to the total negative count $K(N - n_{\max})$. Formally: $\mathcal{L}^{\text{NSCL}} \leq \mathcal{L}^{\text{DCL}} \leq \mathcal{L}^{\text{NSCL}} + \log\!\left(1 + \frac{n_{\max} e^2}{N - n_{\max}}\right)$. For balanced classification, $\frac{n_{\max}}{N - n_{\max}} = \frac{1}{C-1}$.
    - **Design Motivation**: The bound is **label-agnostic** and **architecture-agnostic** — no assumptions on model architecture or data distribution are required.
    - **Key Corollary**: As $C \to \infty$, DCL and NSCL become identical; self-supervised learning equals supervised learning in the limit.

2. **Neural Collapse at the Global Optimum of NSCL (Theorem 2)**:

    - **Function**: Characterizes the geometric structure of NSCL global optima under the unconstrained features model.
    - **Mechanism**: Any global optimum of NSCL simultaneously satisfies three properties:
        - **Augmentation collapse**: All augmented views of the same sample map to the same point, $z_i^{l_1} = z_i^{l_2}$.
        - **Within-class collapse**: All samples of the same class share the same representation, $z_i = z_j$ if $y_i = y_j$.
        - **Simplex ETF**: Class means form an equiangular tight frame, $\langle \mu_c, \mu_{c'} \rangle = -\frac{\|\mu_c\|^2}{C-1}$.
    - **Design Motivation**: These properties are identical to the global optima of cross-entropy, MSE, and supervised contrastive losses — NSCL also induces Neural Collapse.

3. **Directional CDNV-based Few-Shot Error Bound (Proposition 1)**:

    - **Function**: Proposes a tighter upper bound on few-shot classification error than existing CDNV-based bounds.
    - **Mechanism**: Introduces the directional CDNV $\tilde{V}_f$, which measures variance only along the direction connecting class means (rather than total variance). The new bound is $\text{err}^{\text{NCC}}_{m,D}(f) \leq (C'-1)\!\left[8\tilde{V}_f + \frac{8}{\sqrt{m}}V_f^s + \frac{8}{\sqrt{m}}V_f + \frac{4}{m}V_f\right]$.
    - **Key Advantage**: Since $\tilde{V}_f \leq V_f$, and for isotropic distributions $\tilde{V}_f = \frac{1}{d} V_f$, the directional CDNV can be small even when the full CDNV is large — explaining why SSL representations transfer effectively even when full CDNV appears unfavorable.

## Key Experimental Results

### Main Results

| Dataset | Method | NCC 100-shot Acc | LP 100-shot Acc |
|--------|------|-------------------|-----------------|
| CIFAR-10 | DCL | 85.3% | 86.3% |
| CIFAR-10 | NSCL | 95.7% | 95.6% |
| CIFAR-100 | DCL | 57.3% | 61.7% |
| CIFAR-100 | NSCL | 70.8% | 73.7% |
| mini-ImageNet | DCL | 69.0% | 72.9% |
| mini-ImageNet | NSCL | 79.8% | 81.3% |

### Ablation Study

| Validation | Result |
|---------|------|
| DCL–NSCL loss gap vs. $C$ | Decays as $O(1/C)$, consistent with theory |
| Correlation between two losses | Highly correlated throughout training (correlation > 0.99) |
| Whether minimizing DCL also minimizes NSCL | Yes — NSCL values are close to those from directly optimizing NSCL |
| Whether directional CDNV is tighter than full CDNV | Yes — influence of full CDNV diminishes as $m$ increases |
| Tightness of error bound | Optimized Cor. 1 bound closely tracks actual error in practice |

### Key Findings
- **DCL essentially optimizes NSCL**: Even without label access, the NSCL value achieved during DCL training is close to that of directly optimizing NSCL.
- **More classes → DCL ≈ NSCL is more accurate**: The loss gap on CIFAR-100 ($C=100$) is an order of magnitude smaller than on CIFAR-10 ($C=10$).
- **Directional CDNV dominates few-shot performance**: As the number of labeled shots increases, the influence of full CDNV diminishes, while directional CDNV remains the primary factor.
- **Results hold for ViT and MoCo**: Conclusions are not limited to ResNet/SimCLR architectures.

## Highlights & Insights
- **Elegant theoretical connection**: A single concise inequality (Theorem 1) formally connects SSL and supervised learning — two seemingly disparate paradigms — without requiring any assumption on model architecture or data distribution, which no prior theoretical work achieved.
- **Neural Collapse extended to SSL**: Previously, Neural Collapse was known only for the optima of CE/MSE/SCL. This paper proves it also holds for NSCL, thereby indirectly explaining the semantic clustering behavior of SSL via DCL ≈ NSCL.
- **Introduction of directional CDNV**: This metric is more fine-grained than full CDNV and explains the common yet puzzling phenomenon where representations have large total variance but still achieve strong few-shot performance.

## Limitations & Future Work
- **Limitation of the unconstrained features model**: Theorem 2 holds under the unconstrained features model; actual neural network parameterization may result in incomplete collapse.
- **Suboptimal constants in the bound**: The coefficients (8, 8, 4) in Proposition 1 are not optimal; Cor. 1 provides an improved version but is still not the tightest possible.
- **Optimization dynamics not analyzed**: The paper characterizes global optima but does not analyze whether or how quickly SGD converges to these structures.
- **DCL cannot achieve the NSCL optimum**: Since DCL is label-agnostic, it cannot achieve perfect within-class collapse (which requires label information) — how the residual gap affects practical performance warrants further investigation.

## Related Work & Insights
- **vs. Arora et al. (2019)**: Their CL theory requires an augmentation conditional independence assumption; the bound in this paper is architecture-agnostic and label-agnostic, making it more general.
- **vs. Neural Collapse literature**: This paper is the first to extend NC properties to unsupervised/self-supervised loss functions.
- **vs. Alignment & Uniformity analysis**: The latter characterizes low-order statistics of representations; this paper reveals deeper class-oriented distributional structure.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal connection between SSL CL and supervised CL; strong theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and architectures; theoretical predictions align well with experiments.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](../../ACL2026/interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)

<!-- RELATED:END -->
