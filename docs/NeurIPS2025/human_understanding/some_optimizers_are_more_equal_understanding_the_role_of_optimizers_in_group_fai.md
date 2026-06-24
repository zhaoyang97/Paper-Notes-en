---
title: >-
  [Paper Note] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness
description: >-
  [NeurIPS 2025][Human Understanding][group fairness] This paper presents the first systematic study on how the choice of optimization algorithm affects group fairness in deep learning. Through stochastic differential equation (SDE) analysis and two novel theorems, it demonstrates that adaptive optimizers (RMSProp/Adam) are more likely to converge to fair minima than SGD, particularly under severe data imbalance.
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "group fairness"
  - "optimizers"
  - "adaptive gradients"
  - "stochastic differential equations"
  - "fairness in deep learning"
date: 2026-05-08
content_hash: f7a591a088ca2799
---

# Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness

**Conference**: NeurIPS 2025
**arXiv**: [2504.14882](https://arxiv.org/abs/2504.14882)  
**Code**: [GitHub](https://github.com/Mkolahdoozi/Some-Optimizers-Are-More-Equal)  
**Area**: Human Understanding
**Keywords**: group fairness, optimizers, adaptive gradients, stochastic differential equations, fairness in deep learning

## TL;DR

This paper presents the first systematic study on how the choice of optimization algorithm affects group fairness in deep learning. Through stochastic differential equation (SDE) analysis and two novel theorems, it demonstrates that adaptive optimizers (RMSProp/Adam) are more likely to converge to fair minima than SGD, particularly under severe data imbalance.

## Background & Motivation

- **Background**: The widespread deployment of machine learning in socially sensitive domains—such as decision-making systems and risk assessment—has made fairness a critical concern. Existing techniques for promoting group fairness fall into three categories: pre-processing (data augmentation), in-processing (loss function modification), and post-processing (output calibration). However, these approaches often introduce additional computational overhead or disrupt the training pipeline.

- **Limitations of Prior Work**: The authors identify a completely overlooked question: **does the choice of optimization algorithm itself affect group fairness?** Optimizers are a fundamental component of every deep learning training pipeline. If certain optimizers are inherently more conducive to fairness, no additional fairness-enhancing mechanisms would be necessary. Prior work (Rebuffi et al.) has shown that SGD-trained models are more robust than those trained with adaptive methods, and since fairness and robustness are often competing objectives, this suggests adaptive optimizers may hold an advantage in fairness. Yet this relationship has never been formally investigated.

## Method

### Overall Architecture

The authors adopt a three-stage methodology: *theoretical analysis → simulation validation → empirical validation*. They first analyze the fairness behavior of SGD and RMSProp in an analytically tractable setting using SDEs to derive closed-form solutions, then prove two general theorems, and finally validate their findings experimentally on three datasets.

### Key Designs

1. **SDE Analysis (Theorem 1)**: Consider a simple loss function for two subgroups: $\mathcal{L}_0(w)=\frac{1}{2}(w-1)^2$ and $\mathcal{L}_1(w)=\frac{1}{2}(w+1)^2$, with the fairest minimum at $w^*_{pop}=0$. With subgroup sampling probabilities $p_0$ and $p_1$, the stationary distributions of SGD and RMSProp are derived via the Fokker–Planck equation:

   SGD stationary distribution: $p_{sgd}(w)=\sqrt{\frac{\vartheta}{\pi}}\exp(-\vartheta(w-(p_0-p_1))^2)$, where $\vartheta=\frac{1}{8\eta p_0 p_1}$

   RMSProp stationary distribution: $p_{rms}(w)=\sqrt{\frac{\kappa}{\pi}}\exp(-\kappa(w-(p_0-p_1))^2)$, where $\kappa=\frac{1}{4\eta\Theta\sqrt{p_0 p_1}}$

   **Key finding**: When the sampling bias $|p_0-p_1|$ exceeds a threshold $\Delta(p_1p_2,\eta)$, RMSProp converges to the fair minimum with higher probability than SGD, i.e., $\frac{p_{rms}(w^*_{pop})}{p_{sgd}(w^*_{pop})}>1$.

2. **Fairer Parameter Updates (Theorem 2)**: Under the assumption of isotropic gradient noise, the inter-subgroup parameter update discrepancy $\|D(\nabla\mathcal{L}_0-\nabla\mathcal{L}_1)\|$ under RMSProp is upper-bounded by the corresponding discrepancy $\|\nabla\mathcal{L}_0-\nabla\mathcal{L}_1\|$ under SGD. This follows because the diagonal entries of RMSProp's normalization matrix satisfy $D_{jj}<1$ (when $\Theta^2>\mu^2$), effectively "compressing" inter-subgroup gradient differences and preventing subgroups with large gradients from dominating training dynamics.

3. **Demographic Parity Guarantee (Theorem 3)**: In a single optimization step, the worst-case increase in the demographic parity gap induced by RMSProp is upper-bounded by the corresponding bound for SGD. This implies that RMSProp's adaptive learning rates—by scaling updates based on historical squared gradients—help mitigate demographic parity gaps that arise during training.

### Fairness Metrics

Three widely adopted fairness criteria are used:
- **Equalized Odds** $F_{EOD}$: requires equal true positive rates and false positive rates across subgroups
- **Equal Opportunity** $F_{EOP}$: requires equal true positive rates across subgroups
- **Demographic Parity** $F_{DPA}$: requires that predicted label distributions are independent of the sensitive attribute

Higher values indicate greater fairness for all metrics.

## Key Experimental Results

### Main Results (ViT backbone, three datasets)

| Dataset | Sensitive Attr. | Metric | Adam | RMSProp | SGD | Gain (vs SGD) |
|--------|---------|------|------|---------|-----|-------------|
| CelebA | Gender (G) | $F_{EOD}$ | 65.21 | 65.18 | 62.66 | +2.55 |
| CelebA | Gender (G) | $F_{EOP}$ | 99.90 | 99.91 | 96.60 | +3.31 |
| CelebA | Gender (G) | $F_{DPA}$ | 73.50 | 73.68 | 60.80 | +12.88 |
| CelebA | Age (A) | $F_{EOD}$ | 72.34 | 71.99 | 68.40 | +3.59 |
| FairFace | Race (R) | $F_{EOD}$ | — | +9% (vs SGD) | baseline | RMSProp advantage clear |

### Accuracy Comparison (No Sacrifice in Classification Performance)

| Dataset | SGD Acc | RMSProp Acc | Adam Acc | SGD F1 | RMSProp F1 | Adam F1 |
|--------|---------|-------------|----------|--------|------------|---------|
| CelebA | 91.23 | 91.54 | 92.08 | 92.12 | 91.17 | 92.09 |
| MS-COCO | 89.62 | 89.71 | 90.03 | 68.35 | 71.03 | 74.10 |
| FairFace | 89.41 | 91.37 | 92.20 | 91.13 | 92.07 | 92.17 |

### Statistical Significance (Wilcoxon Test p-values)

| Metric | SGD vs RMSProp (Gender) | SGD vs Adam (Gender) | SGD vs RMSProp (Age) | SGD vs Adam (Age) |
|------|---------------------|-------------------|---------------------|-------------------|
| $F_{EOD}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $1\times10^{-3}$ |
| $F_{EOP}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $1\times10^{-3}$ | $5\times10^{-3}$ |
| $F_{DPA}$ | $2\times10^{-3}$ | $1\times10^{-3}$ | $7\times10^{-3}$ | $3\times10^{-3}$ |

### Complementarity with Fairness-Enhancing Methods

| Setting | Metric | Adam | RMSProp | SGD |
|------|------|------|---------|-----|
| With fairness enhancement | Equal Opportunity Gap↓ | 0.45 | 0.48 | 0.71 |
| With fairness enhancement | Demographic Parity Gap↓ | 0.86 | 0.86 | 2.60 |
| Without fairness enhancement | Equal Opportunity Gap↓ | 13.99 | 13.90 | 15.19 |
| Without fairness enhancement | Demographic Parity Gap↓ | 11.49 | 11.45 | 11.80 |

### Key Findings

- The more imbalanced the dataset (e.g., racial minority groups comprising only 0.9% in FairFace), the more pronounced the fairness advantage of adaptive optimizers, consistent with the theoretical predictions of Theorem 1.
- As the male proportion in CelebA decreases from 42% to 2%, the $F_{DPA}$ gap between RMSProp and SGD continues to widen.
- The fairness improvements from adaptive optimizers are independent of overall classification performance; higher F1 scores do not necessarily imply greater fairness.
- The fairness benefits of adaptive optimizers are complementary to existing fairness-enhancing methods.

## Highlights & Insights

- **Novelty and Significance**: The finding that optimizer choice affects fairness is simple yet profound, with strong practical implications—in fairness-sensitive scenarios, switching to Adam/RMSProp alone can yield meaningful improvements.
- **Theoretical Rigor**: The SDE analysis yields closed-form solutions; the derivation via the Fokker–Planck equation is rigorous, and the two subsequent theorems extend the results to the general case.
- **Comprehensive Experimental Design**: Ten repeated runs with Wilcoxon significance testing, evaluated across multiple backbone architectures, datasets, and sensitive attribute combinations.

## Limitations & Future Work

- The theoretical analysis is conducted under a simple quadratic loss; although simulations and experiments validate the generality of the findings, extending the theory to high-dimensional settings remains challenging.
- The paper focuses exclusively on group fairness and does not address individual fairness.
- All theorems rely on the isotropic noise assumption (an anisotropic extension is provided in the appendix but requires stronger conditions).
- The paper does not provide guidance on selecting fairness-optimal hyperparameters for adaptive optimizers.

## Related Work & Insights

- The findings complement those of Rebuffi et al. (SGD yields greater robustness): the trade-off between robustness and fairness warrants further investigation.
- Zeng et al. (2024) showed that balanced datasets lead to tighter fairness guarantees; the SDE analysis in this paper provides an alternative pathway for imbalanced data settings.
- This work motivates further exploration of the optimizer design space toward developing optimization algorithms explicitly tailored for fairness.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First work to understand fairness through the lens of optimizer choice—a unique and practically impactful insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, multiple sensitive attributes, statistical significance testing, ablation studies, and complementarity experiments with fairness methods.
- **Writing Quality**: ⭐⭐⭐⭐ Theory is built progressively from simple examples; the pedagogical style is commendable.
- **Value**: ⭐⭐⭐⭐⭐ Direct guidance for practitioners—fairness can be improved simply by switching the optimizer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)
- [\[CVPR 2025\] SocialGesture: Delving into Multi-Person Gesture Understanding](../../CVPR2025/human_understanding/socialgesture_delving_into_multi-person_gesture_understanding.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[CVPR 2026\] MOFA-VTON: More Fashion Possibilities with Fine-Grained Adaptations in Virtual Try-On](../../CVPR2026/human_understanding/mofa-vton_more_fashion_possibilities_with_fine-grained_adaptations_in_virtual_tr.md)
- [\[CVPR 2026\] Bridging Facial Understanding and Animation via Language Models](../../CVPR2026/human_understanding/bridging_facial_understanding_and_animation_via_language_models.md)

</div>

<!-- RELATED:END -->
