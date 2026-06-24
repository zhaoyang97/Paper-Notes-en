---
title: >-
  [Paper Note] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation
description: >-
  [NeurIPS 2025][Human Understanding][cross-domain gaze estimation] This paper formulates cross-domain gaze estimation (CDGE) as a generalized label shift (GLS) problem, demonstrating that existing domain-invariant representation learning methods are theoretically insufficient under label shift. It proposes continuous importance reweighting based on truncated Gaussian distributions and a Probability-aware Conditional Operator Discrepancy (PCOD) to jointly correct label shift an…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "cross-domain gaze estimation"
  - "generalized label shift"
  - "importance reweighting"
  - "conditional distribution alignment"
  - "kernel embedding"
date: 2026-05-08
content_hash: a6fb02219ec32a32
---

# A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2505.13043](https://arxiv.org/abs/2505.13043)  
**Code**: None  
**Area**: Other
**Keywords**: cross-domain gaze estimation, generalized label shift, importance reweighting, conditional distribution alignment, kernel embedding

## TL;DR
This paper formulates cross-domain gaze estimation (CDGE) as a generalized label shift (GLS) problem, demonstrating that existing domain-invariant representation learning methods are theoretically insufficient under label shift. It proposes continuous importance reweighting based on truncated Gaussian distributions and a Probability-aware Conditional Operator Discrepancy (PCOD) to jointly correct label shift and conditional shift, achieving an average error reduction of 12%–27% across multiple backbones.

## Background & Motivation
Appearance-based deep gaze estimation has broad applications in human–computer interaction, VR, and medical analysis, yet model performance degrades severely under cross-domain deployment. Existing CDGE methods fall into two categories: domain generalization (DG) methods, which extract domain-invariant features by removing gaze-irrelevant factors, and unsupervised domain adaptation (UDA) methods, which achieve generalization by aligning feature distributions. Both categories fundamentally focus on domain-invariant representation learning.

However, GLS theory has established that **invariant representation learning alone is insufficient to minimize domain shift when label shift exists**. In CDGE, different datasets exhibit distinct gaze ranges and concentration regions (i.e., different label distributions), and the visual appearance of the same gaze direction varies significantly across acquisition environments (i.e., different conditional distributions). This precisely constitutes a GLS problem. Existing methods completely neglect label shift correction, rendering them theoretically insufficient at a fundamental level.

## Core Problem
How can GLS correction be performed in a **continuous regression** task such as gaze estimation? Existing GLS correction methods target classification problems with finite discrete categories, relying on class-level distribution ratio estimation and class-level conditional alignment — approaches that are entirely infeasible for regression problems with continuous, infinite-valued label variables. This paper must address two key challenges: (1) how to perform importance reweighting over continuous label distributions; and (2) how to embed the reweighted source distribution into conditional invariant learning.

## Method

### Overall Architecture
The GLSGE framework proceeds in three steps: (1) estimate an importance weight function $\omega(y)$ to approximate the target label distribution using the reweighted source label distribution; (2) learn a conditionally invariant feature transformation $g$ based on the reweighted source distribution; (3) train a gaze predictor $h$ on the reweighted source domain. The inputs are labeled source domain data, unlabeled target domain data, and a source-domain pretrained model; the outputs are $(g, h)$ that generalize to the target domain.

### Key Designs

1. **Truncated Gaussian Label Distribution Modeling and Continuous Importance Reweighting**: Gaze variables possess two special properties — a finite range (compact support) and a concentrated human fixation distribution. The authors therefore model the label distribution as a bivariate truncated Gaussian. Target distribution mean and covariance are estimated from target domain pseudo-labels $\hat{y}_t = h(g(x_t))$, and the importance weight is computed as $\omega(y) = f_{\mathrm{TGau}}(y;\,\hat{\mu}_t, \hat{\sigma}_t, a, b) / p_Y^s(y)$. The reweighted source label probability then directly equals the estimated target truncated Gaussian distribution. This parametric approach elegantly circumvents the infeasibility of iterating over all categories as required by classification-based methods.

2. **Probability-aware Conditional Operator Discrepancy (PCOD)**: Existing conditional distribution alignment method COD aligns $P(Z|Y)$ holistically via kernel embedding, but its empirical estimator assumes a uniform label distribution and cannot incorporate a reweighted label distribution. The authors re-derive the empirical estimator of COD, replacing the original equal-weight $1/n$ summation with an $\omega(y_i) \cdot \hat{p}(y_i)$-weighted summation. This involves probability-aware estimation of the conditional mean operator and conditional covariance operator in the RKHS, requiring extensive matrix analysis (e.g., the Woodbury identity). The resulting PCOD naturally integrates label shift correction information.

3. **Marginal Alignment Auxiliary Term**: In the early training phase, large domain shift leads to low-quality pseudo-labels from the predictor, potentially causing convergence to a suboptimal solution. An additional marginal distribution alignment term (using DAGE-GRAM) is therefore incorporated to improve pseudo-label reliability. The final conditional alignment loss is: $L_{\mathrm{cond}} = \mathrm{PCOD} + \mathrm{marginal\ alignment}$.

### Loss & Training
The overall objective is $\min_{g,h} L_{\mathrm{src}}^{\omega} + \lambda \cdot L_{\mathrm{cond}}$, where $L_{\mathrm{src}}^{\omega}$ is a reweighted L1 loss (each source sample scaled by its reweighting probability), and $L_{\mathrm{cond}} = \mathrm{PCOD} + \mathrm{marginal\ alignment}$. Training adopts alternating optimization: an outer loop of $N_1$ steps updates the label distribution estimate (updating pseudo-labels → re-estimating truncated Gaussian parameters → updating weights), and an inner loop of $N_2 = 5$ epochs updates $(g, h)$. Deep backbone layers are frozen; only a shallow MLP feature extractor and a linear predictor are trained.

## Key Experimental Results

Four standard CDGE benchmarks: ETH-XGaze (E) → MPIIFaceGaze (M), E → EyeDiap (D), Gaze360 (G) → M, G → D.

| Backbone | Method | E→M | E→D | G→M | G→D | Avg |
|----------|--------|------|------|------|------|------|
| ResNet-18 | Baseline | 8.05 | 9.03 | 7.41 | 8.83 | 8.33 |
| ResNet-18 | PnP-GA+ (UDA SOTA) | 5.34 | 5.73 | 6.10 | 7.62 | 6.20 |
| ResNet-18 | **GLSGE** | **5.31** | 6.21 | **5.43** | 7.30 | **6.06** |
| ResNet-50 | Baseline | 8.03 | 8.06 | 7.75 | 8.79 | 8.16 |
| ResNet-50 | PnP-GA+ | 6.49 | 6.61 | 5.64 | 7.09 | 6.46 |
| ResNet-50 | **GLSGE** | 5.54 | 6.10 | **5.27** | 7.14 | **6.01** |

Generalizability across models (as a plug-and-play module):

| Model | Original Avg | +GLSGE Avg | Reduction |
|-------|-------------|------------|-----------|
| ResNet-18 | 8.33 | 6.06 | **27.2%** |
| ResNet-50 | 8.16 | 6.01 | **26.3%** |
| GazeTR (ViT) | 8.99 | 7.27 | **19.1%** |
| FSCI (DG SOTA) | 6.95 | 6.11 | **12.1%** |

### Ablation Study
- Label shift correction and conditional distribution alignment each independently yield significant error reductions, and their combination (GLS correction) achieves the best performance.
- PCOD outperforms the original COD, confirming the effectiveness of probability-aware estimation.
- Without label shift correction, both COD and PCOD are negatively affected, validating the theoretical necessity claim of GLS.
- The method exhibits strong hyperparameter robustness, with a standard deviation of only 0.05° in prediction error across a wide parameter range.

## Highlights & Insights
- **Novel theoretical perspective**: This is the first work to formulate CDGE as a GLS problem, identifying at a theoretical level the deficiency of existing DG and UDA methods — they fundamentally perform only invariant representation learning while ignoring label shift.
- **GLS correction for continuous regression**: The method cleverly exploits the properties of gaze variables (compact support and concentrated distribution) to parameterize the continuous label distribution with a truncated Gaussian, bypassing the infeasibility of classification-based GLS approaches.
- **Plug-and-play**: As a general framework, it can be integrated into different backbones (CNN, ViT) and existing SOTA methods (FSCI), yielding consistent improvements across all models.
- **PCOD derivation**: Embedding importance weight information into a kernel-based conditional distribution discrepancy measure is technically non-trivial.

## Limitations & Future Work
- **Requires target domain samples**: As a UDA method, it requires unlabeled target domain data and cannot generalize to completely unseen domains, making it less flexible than DG methods.
- **Computational cost of kernel matrices**: PCOD involves $O(n^2)$ kernel matrix computation, which becomes prohibitive at large sample sizes. Random feature approximations could reduce complexity.
- **Label distribution assumption**: The truncated Gaussian is a general but simplistic assumption; certain scenarios (e.g., driving, where gaze concentrates in a few directions) may require more precise modeling such as Gaussian mixture models.
- **Pseudo-label dependency**: Both label shift correction and PCOD rely on target domain pseudo-label quality; poor pseudo-labels in early training may impair convergence.
- **Extensibility to other regression tasks**: The authors suggest the framework can generalize to tasks such as pose estimation, but no experimental validation on such tasks is provided.

## Related Work & Insights
- **vs. PnP-GA+** (UDA SOTA): PnP-GA+ employs marginal/conditional alignment but ignores label shift. GLSGE achieves marginally better average error (6.06 vs. 6.20), yet PnP-GA+ relies on extensive data augmentation and up to 10 auxiliary models, while GLSGE uses no data augmentation.
- **vs. FSCI** (DG SOTA): FSCI achieves generalization through de-confounding without using target domain data. GLSGE further reduces error by 12.1% on top of FSCI using only a small amount of unlabeled target domain data, demonstrating that label shift correction addresses a blind spot of DG methods.
- **vs. COD** (conditional alignment method): COD, proposed at ECCV 2024, measures conditional operator discrepancy but cannot handle label shift. GLSGE's PCOD is a probability-aware extension of COD, and ablation experiments directly demonstrate PCOD's advantage over COD.

The GLS-theoretic insight that "invariant representation learning is insufficient under label shift" has broad implications — any cross-domain regression task (e.g., depth estimation, pose estimation, 3D reconstruction) where source and target label ranges or distributions differ may be inadequately addressed by existing domain adaptation methods. The idea of performing label shift correction via parametric continuous distributions is generalizable to other tasks where prior knowledge constrains the label space. The PCOD derivation strategy of embedding importance weights into kernel-based conditional distribution measures also has general applicability.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to introduce GLS theory into CDGE; the perspective is genuinely novel, though GLS theory itself is pre-existing.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four tasks, four backbones, complete ablations and visualizations; validation on other regression tasks is absent.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and the framework is unified; the PCOD derivation is lengthy but necessary.
- Value: ⭐⭐⭐⭐ — Plug-and-play improvements across multiple models with both theoretical and practical contributions; however, gaze estimation itself has a relatively narrow application scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Through the Frequency Lens: Cross-Domain Generalisable Gaze Estimation with Adaptive Modulation](../../CVPR2026/human_understanding/through_the_frequency_lens_cross-domain_generalisable_gaze_estimation_with_adapt.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)
- [\[CVPR 2026\] See Through the Noise: Improving Domain Generalization in Gaze Estimation](../../CVPR2026/human_understanding/see_through_the_noise_improving_domain_generalization_in_gaze_estimation.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](../../CVPR2025/human_understanding/3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[ICCV 2025\] Multi-view Gaze Target Estimation](../../ICCV2025/human_understanding/multi-view_gaze_target_estimation.md)

</div>

<!-- RELATED:END -->
