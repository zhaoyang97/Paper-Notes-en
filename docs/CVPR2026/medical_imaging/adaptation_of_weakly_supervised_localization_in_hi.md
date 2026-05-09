---
title: >-
  [Paper Note] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions
description: >-
  [CVPR 2026][Medical Imaging][weakly supervised localization] This paper proposes SFDA-DeP, a source-free domain adaptation (SFDA) framework inspired by machine unlearning that models adaptation as an iterative process of identifying and correcting prediction bias. The method selectively reduces confidence on uncertain samples from the dominant class, retains reliable predictions, and jointly trains a pixel-level classifier to recover localization discriminability. It consistently outperforms SFDA baselines in both classification and localization across cross-organ and cross-center histopathology benchmarks.
tags:
  - CVPR 2026
  - Medical Imaging
  - weakly supervised localization
  - source-free domain adaptation
  - prediction debiasing
  - machine unlearning
  - histopathology
date: 2026-05-08
content_hash: 50054f33caf52ff7
---

# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**Conference**: CVPR 2026
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)
**Code**: [anonymous.4open.science/r/SFDA-DeP-1797](https://anonymous.4open.science/r/SFDA-DeP-1797/)
**Area**: Medical Imaging / Computational Pathology / Domain Adaptation
**Keywords**: weakly supervised localization, source-free domain adaptation, prediction debiasing, machine unlearning, histopathology

## TL;DR

This paper proposes SFDA-DeP, a source-free domain adaptation (SFDA) framework inspired by machine unlearning that models adaptation as an iterative process of identifying and correcting prediction bias. The method selectively reduces confidence on uncertain samples from the dominant class, retains reliable predictions, and jointly trains a pixel-level classifier to recover localization discriminability. It consistently outperforms SFDA baselines in both classification and localization across cross-organ and cross-center histopathology benchmarks.

## Background & Motivation

**State of the Field**: Deep WSOL models have achieved successful simultaneous classification and ROI localization in pathology images using only image-level labels (e.g., NEGEV, PixelCAM, SAT). However, when deployed across domains (different organs, centers, or staining/scanning protocols), distribution shift causes severe performance degradation.

**Limitations of Prior Work**:

1. Under large domain shifts (especially cross-organ), WSOL predictions become heavily biased toward the dominant class — e.g., when transferring from GlaS (colon) to CAMELYON16/17 (breast lymph nodes), models may assign 90%+ predictions to the cancer class.
2. Conventional SFDA methods (SFDA-DE, ERL, CDCL) rely on self-training and implicitly assume the source classifier retains sufficient discriminability on the target domain — an assumption that breaks down under severe domain shift.
3. Biased pseudo-labels are iteratively reinforced rather than corrected during self-training, causing a bias amplification effect that degrades both classification and localization.

**Root Cause**: The self-training mechanism of SFDA is precisely what amplifies WSOL prediction bias — the more training, the greater the bias.

**Paper Goals**: To correct class prediction imbalance caused by domain shift and simultaneously recover localization discriminability, without access to source data.

**Starting Point**: Drawing on the concept of machine unlearning — not to "forget" a class, but to make the model "unlearn" its biased decision boundary and establish a new, balanced one.

**Core Idea**: Apply a "forgetting" loss to high-entropy samples in the dominant class to shift the decision boundary, while retaining low-entropy samples to maintain stable predictions.

## Method

### Overall Architecture

On the target domain, the method dynamically repartitions a "forget set" (high-entropy samples from the dominant class) and a "retain set" (all remaining samples) every $m$ epochs. Three losses are jointly optimized: a retain loss to preserve reliable predictions, a forget loss to correct bias, and a pixel-level localization loss to anchor spatially discriminative features.

### Key Designs

1. **Dynamic Partition of Forget/Retain Sets and Corresponding Losses**

   - Let $\mathbb{B}$ denote the set of samples predicted as the dominant class $\mathcal{B}$. The top-$\rho$ most uncertain samples (by normalized entropy) form the forget set $\mathbb{B}_f$ ($\rho \in \{5\%, 15\%, 25\%\}$); the remainder constitute the retain set $\mathbb{B}_r = \mathbb{T} - \mathbb{B}_f$.
   - Retain loss: standard cross-entropy $\mathcal{L}_{\text{retain}} = -\log(p_i(\hat{y}))$, encouraging the model to continue predicting pseudo-labels for retained samples.
   - Forget loss: $\mathcal{L}_{\text{forget}} = -\log(1 - p_i(\hat{y}))$, minimized to prevent the model from predicting the dominant class for forget-set samples, thereby forcing the decision boundary to shift.
   - **Key design**: The partition is recomputed every $m$ epochs using the current model state, preventing irreversible accumulation of erroneous forgetting decisions — this is dynamic debiasing, not a one-shot operation.

2. **Pixel-Level Localization Supervision**

   - A lightweight pixel-level classifier $h$ is jointly trained to perform foreground/background binary classification on feature maps.
   - Source model CAMs are used as pixel-level pseudo-labels, extracted only from the top-$\rho_{\text{loc}}$ lowest-entropy samples per class.
   - Training uses a BCE loss: $\mathcal{L}_{\text{loc}} = -(1-Y_p)\log(h(z_p)_0) - Y_p\log(h(z_p)_1)$
   - Purpose: to anchor spatially discriminative features during classification debiasing, preventing localization capability from drifting throughout adaptation.

### Loss & Training

Total loss: $\mathcal{L} = \lambda_{\text{retain}} \mathcal{L}_{\text{retain}} + \lambda_{\text{forget}} \mathcal{L}_{\text{forget}} + \lambda_{\text{loc}} \mathcal{L}_{\text{loc}}$

- $\lambda_{\text{retain}}, \lambda_{\text{forget}} \in \{0.2, 0.5, 1.0, 2.0\}$; $\lambda_{\text{loc}} \in \{0.5, 1.0, 5.0\}$
- Learning rate selected from $\{10^{-5}, 10^{-4}, 10^{-3}\}$
- CNN backbone: ResNet-50; Transformer backbone: DeiT-Tiny; validated on three WSOL models (PixelCAM, SAT, DeepMIL)
- Datasets: GlaS (colon), CAMELYON16 (breast), CAMELYON17 (5 centers)

## Key Experimental Results

### Main Results

Average metrics across 6 target domains in GlaS→CAMELYON cross-organ/cross-center adaptation:

| WSOL Model | Method | Avg. PxAP | Avg. CL | vs SFDA-DE |
|------------|--------|-----------|---------|-----------|
| PixelCAM | Source-only | 36.9 | 49.3 | — |
| PixelCAM | SFDA-DE | 28.0 | 54.6 | baseline |
| PixelCAM | ERL | 25.4 | 59.9 | -2.6 PxAP |
| PixelCAM | RGV | 34.7 | 52.1 | +6.7 PxAP |
| PixelCAM | **SFDA-DeP** | **44.1** | **67.1** | **+16.1 PxAP, +12.5 CL** |
| DeepMIL | Source-only | 20.9 | 49.8 | — |
| DeepMIL | SFDA-DE | 20.5 | 53.9 | baseline |
| DeepMIL | **SFDA-DeP** | **40.7** | **73.4** | **+20.2 PxAP, +19.5 CL** |
| SAT | Source-only | 21.3 | 52.1 | — |
| SAT | SFDA-DE | 21.6 | 68.7 | baseline |
| SAT | **SFDA-DeP** | **30.3** | **69.2** | **+8.7 PxAP, +0.5 CL** |

### Ablation Study

| Ablation | Key Finding |
|----------|-------------|
| Dynamic repartitioning vs. static partition | Dynamic partitioning significantly outperforms static; prevents accumulation of erroneous forgetting decisions |
| With/without $\mathcal{L}_{\text{loc}}$ | Adding pixel-level loss yields notable PxAP gains |
| Forget ratio $\rho$ (5%–25%) | Method is insensitive to this hyperparameter |

### Key Findings

- SFDA-DE collapses to CL ≈ 50% (random guessing) on multiple centers; SFDA-DeP recovers to 80%+: e.g., PixelCAM on C17-0 improves CL from 50.0% to 86.2%, DeepMIL on C17-0 from 50.0% to 82.8%.
- Existing SFDA methods yield PxAP even lower than Source-only (SFDA-DE: 28.0 vs. Source-only: 36.9), confirming the bias amplification effect.
- SFDA-DeP achieves the largest gains on DeepMIL (+20.2 PxAP), indicating greater benefit for architectures with weaker baseline discriminability and more severe bias.
- Qualitative visualizations show SFDA-DeP CAM activations concentrate on tumor tissue, whereas SFDA baselines frequently highlight background regions.

## Highlights & Insights

- Precisely diagnoses the root cause of SFDA failure on WSOL (prediction bias amplification) rather than broadly attributing failure to domain shift.
- The analogy of "unlearning old decision boundaries to establish new, balanced ones" is intuitive, and the method is concise (only three loss terms).
- Consistently effective across three WSOL architectures (CNN/Transformer/MIL) and six target domains, demonstrating strong generalization.
- Recovers CL to 80%+ even in scenarios where SFDA-DE completely fails (CL collapses to 50%), demonstrating robustness.

## Limitations & Future Work

- Validated only on binary classification (tumor/normal); dominant class identification and forgetting strategies require extension to multi-class pathology settings.
- The forget/retain partition relies solely on prediction entropy, without exploiting feature-space structure (e.g., clustering density).
- Pixel-level CAM pseudo-label quality is bounded by source model quality; CAMs themselves may be unreliable under cross-domain settings.
- Dataset scale is limited (GlaS contains only 67 training images); effectiveness on larger-scale datasets remains to be verified.
- Open-set scenarios where the target domain contains novel classes absent from the source domain are not explored.

## Related Work & Insights

- **vs. SFDA-DE (CVPR'22)**: A classical distribution estimation SFDA method; PxAP degrades under severe prediction bias (28.0 vs. Source-only 36.9), with CL collapsing to 50% across multiple centers.
- **vs. RGV (CVPR'25)**: An uncertainty-controlled SFDA method; its conservative strategy approximates Source-only performance (PxAP 34.7 vs. 36.9), yielding almost no adaptation gain.
- **vs. ERL (ICLR'23)**: Addresses domain shift via noisy label learning; CL improves but PxAP frequently declines (25.4), failing to resolve localization degradation.
- **Insight**: Prediction bias amplification is a general problem in self-training-based domain adaptation; the debiasing paradigm is broadly transferable to SFDA for detection and segmentation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introduces machine unlearning into SFDA for prediction debiasing with a clear motivation and concise, effective design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three WSOL models × multiple cross-domain settings; ablations and visualizations are thorough, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; the bias visualization in Fig. 1 is highly intuitive.
- **Value**: ⭐⭐⭐⭐ Identifies and addresses a core bottleneck in SFDA+WSOL, with practical deployment implications for computational pathology.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)

<!-- RELATED:END -->
