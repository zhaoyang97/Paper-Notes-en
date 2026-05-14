---
title: >-
  [Paper Note] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions
description: >-
  [CVPR 2026][Medical Imaging][WSOL] This paper proposes SFDA-DeP, a method inspired by machine unlearning that reformulates SFDA as an iterative process of "identifying and correcting prediction bias." It applies a forget…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "WSOL"
  - "Source-Free Domain Adaptation"
  - "Prediction Debiasing"
  - "Machine Unlearning"
  - "Histopathology"
date: 2026-05-08
content_hash: 9f4f66d7347fed8c
---

# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**Conference**: CVPR 2026
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)
**Code**: [anonymous.4open.science/r/SFDA-DeP-1797/](https://anonymous.4open.science/r/SFDA-DeP-1797/)
**Authors**: Alexis Guichemerre et al. (ÉTS Montréal, Sorbonne Université, University of York, McGill University)
**Area**: Medical Imaging / Histopathological Image Analysis
**Keywords**: WSOL, Source-Free Domain Adaptation, Prediction Debiasing, Machine Unlearning, Histopathology

## TL;DR

This paper proposes SFDA-DeP, a method inspired by machine unlearning that reformulates SFDA as an iterative process of "identifying and correcting prediction bias." It applies a forgetting operation to high-entropy uncertain samples from the dominant class to force the model to abandon biased predictions, maintains self-training on reliable samples, and anchors localization capacity via a pixel-level classifier. The method consistently outperforms existing SFDA approaches on cross-organ and cross-center histopathology benchmarks.

## Background & Motivation

### State of the Field

Weakly supervised object localization (WSOL) models have attracted considerable attention in digital pathology — requiring only image-level labels (e.g., "tumor/normal") to simultaneously perform classification and ROI localization, substantially reducing the reliance on pixel-level annotations. Representative methods include CAM-based PixelCAM, attention-based DeepMIL, and Transformer-based SAT.

### Limitations of Prior Work

WSOL models suffer severe performance degradation under **cross-domain deployment** (different organs, medical centers, staining protocols, or scanning equipment). More critically, this degradation is not solely attributable to low-level appearance shifts — experiments reveal that when transferring from GlaS (colon glands) to CAMELYON16/17 (breast lymph node metastasis detection), models classify nearly all samples as cancer, inducing extreme **prediction bias**.

### Root Cause

Source-Free Domain Adaptation (SFDA) is the predominant framework for cross-domain deployment without access to source data, relying solely on unlabeled target data. However, existing SFDA methods (e.g., SFDA-DE, CDCL, ERL) fundamentally depend on self-training (pseudo-labeling + self-training), implicitly assuming that the source classifier still produces reasonable predictions on the target domain. When predictions are already severely biased toward the dominant class, self-training **amplifies the bias** — pseudo-labels are dominated by the dominant class, causing the model to become increasingly skewed. Fig. 1 clearly illustrates this vicious cycle: SFDA-DE's bias worsens after adaptation, nearly collapsing to a single class.

### Starting Point

The authors draw inspiration from **machine unlearning**: rather than making the model forget a specific class or source knowledge, the goal is to make the model "unlearn" erroneous class boundaries. Specifically, if the model's predictions for certain dominant-class samples are themselves uncertain (high entropy), those predictions should be actively suppressed to force the decision boundary to readjust.

### Core Idea

A dual-set mechanism that periodically corrects prediction bias by "forgetting high-entropy dominant samples while retaining reliable samples" replaces the indiscriminate self-training of conventional SFDA.

## Method

### Overall Architecture

SFDA-DeP takes as input a WSOL model $f$ pretrained on the source domain and an unlabeled target dataset $\mathbb{T}$. Adaptation proceeds iteratively:

1. Use the current model to predict all target samples → detect the dominant class (the class predicted with excessive frequency).
2. Select the high-entropy (uncertain) subset from dominant-class samples as the **forget set** $\mathbb{B}_f$.
3. The remaining samples form the **retain set** $\mathbb{B}_r$.
4. Apply a forgetting loss to the forget set and a retention loss to the retain set, jointly with a pixel-level localization loss.
5. Rebuild the forget/retain sets every $m$ epochs to dynamically track boundary shifts.

### Key Designs

**1. Forget Set Construction**

- **Function**: Select the most uncertain top-$\rho$ subset from dominant-class predicted samples.
- **Mechanism**: Normalized entropy $H(x)$ measures prediction uncertainty. Define $\mathbb{B} = \{x \in \mathbb{T}: \hat{y}(x) \in \mathcal{B}\}$ (dominant-class sample set), then $\mathbb{B}_f = \text{top}_\rho(\mathbb{B}, H(x))$.
- **Design Motivation**: High-entropy samples already reside near the decision boundary, and the model lacks confidence in forcing them into the dominant class. Forgetting these samples is most efficient — they are most likely to have been incorrectly assigned to the dominant class.

**2. Forget Loss**

- **Function**: Force the model to abandon dominant-class predictions for forget-set samples.
- **Core Formula**: $\mathcal{L}_{\text{forget}} = \mathbb{E}_{x_i \in \mathbb{B}_f}[-\log(1 - p_i(\hat{y}))]$
- Minimizing this loss is equivalent to maximizing the cross-entropy of the model's prediction against the current pseudo-label $\hat{y}$, causing the model to "unlearn" its prior predictions on these samples.
- **Interaction with Retain Loss**: $\mathcal{L}_{\text{retain}} = \mathbb{E}_{x_i \in \mathbb{B}_r}[-\log(p_i(\hat{y}))]$ is the standard cross-entropy that preserves predictions on reliable samples. The two losses jointly redefine class boundaries.

**3. Pixel-Level Localization Loss**

- **Function**: Train a lightweight pixel-level classifier $h$ to classify each pixel as foreground (ROI) or background.
- **Mechanism**: For each predicted class $k$, select the low-entropy (most reliable) subset $D_{\text{loc}}$; extract CAMs from the source model to generate pixel-level pseudo-labels $\bm{Y}$; train $h$ using binary cross-entropy:
  $$\mathcal{L}_{\text{loc}} = -(1-\bm{Y}_p)\log(h(z_p)_0) - \bm{Y}_p\log(h(z_p)_1)$$
- **Design Motivation**: Classification-level debiasing alone is insufficient — target ROI appearance may differ substantially after domain shift, requiring pixel-level supervision to anchor localization features and prevent localization capacity from drifting during adaptation.

**4. Dynamic Resampling**

- Every $m$ epochs, the current model recomputes the prediction distribution and entropy to rebuild the forget/retain sets.
- This prevents irreversible early-stage forgetting errors — as boundaries shift, previously forgotten samples may become reliable, and previously retained samples may require forgetting.

### Loss & Training

Total loss:

$$\mathcal{L} = \lambda_{\text{retain}}\mathcal{L}_{\text{retain}} + \lambda_{\text{forget}}\mathcal{L}_{\text{forget}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}}$$

Hyperparameter search ranges: $\lambda_{\text{retain}}, \lambda_{\text{forget}} \in \{0.2, 0.5, 1.0, 2.0\}$, $\lambda_{\text{loc}} \in \{0.5, 1.0, 5.0\}$, $\rho \in \{5\%, 15\%, 25\%\}$.

## Key Experimental Results

### Main Results (GlaS → CAMELYON16/17, Cross-Organ + Cross-Center)

| WSOL Model | Method | Avg PxAP | Avg CL |
|-----------|------|----------|--------|
| PixelCAM | Source only | 36.9 | 49.3 |
| PixelCAM | SFDA-DE | 28.0 | 54.6 |
| PixelCAM | ERL | 25.4 | 59.9 |
| PixelCAM | RGV | 34.7 | 52.1 |
| PixelCAM | **SFDA-DeP** | **44.1** | **67.1** |
| SAT | Source only | 21.3 | 52.1 |
| SAT | SFDA-DE | 21.6 | 68.7 |
| SAT | ERL | 22.2 | 68.9 |
| SAT | **SFDA-DeP** | **30.3** | **69.2** |
| DeepMIL | Source only | 20.9 | 49.8 |
| DeepMIL | SFDA-DE | 20.5 | 53.9 |
| DeepMIL | ERL | 16.2 | 57.8 |
| DeepMIL | **SFDA-DeP** | **40.7** | **73.4** |

### Ablation Study

| Configuration | Key Effect | Remarks |
|------|---------|------|
| w/o $\mathcal{L}_{\text{loc}}$ | Notable PxAP drop | Absence of pixel-level anchoring causes localization drift |
| Static sampling (no forget/retain rebuild) | Clearly inferior to dynamic | Early forgetting errors become irreversible |
| Too-low resampling frequency | Performance degradation | Boundary shifts are not tracked in time |
| Too-high resampling frequency | Slight performance drop | Unstable sets lead to training oscillation |

### Key Findings

1. **SFDA baselines fail comprehensively under strong bias**: SFDA-DE stalls at ~50% CL (equivalent to random guessing) across multiple centers, and its PxAP frequently falls below source-only, confirming that self-training genuinely amplifies bias.
2. **Largest gains on DeepMIL** (+20.2 PxAP, +19.5 CL vs. SFDA-DE), indicating that debiasing benefits models with weaker baseline localization capacity the most.
3. **Most significant gains on C17-0** (PixelCAM CL jumps from 50.0% to 86.2%), the center exhibiting the most severe initial bias.
4. **Dynamic resampling is critical**: Static forget/retain partitioning causes irreversible error accumulation.
5. **Qualitative analysis**: SFDA-DeP's CAM activations focus on tumor tissue, whereas SFDA baselines activate background regions under strong domain shift.

## Highlights & Insights

1. **Precise problem diagnosis**: The paper clearly identifies that SFDA failure stems not from "insufficient adaptation" but from "bias amplification." The experimental analysis in Fig. 1 is intuitive and compelling. This research paradigm of first pinpointing the bottleneck before designing a solution is methodologically instructive.
2. **Creative appropriation of machine unlearning**: Rather than truly "forgetting a class," the forgetting mechanism reshapes decision boundaries — the forget loss $-\log(1-p(\hat{y}))$ is concise and elegant, with minimal implementation overhead yet substantial effect.
3. **Cross-architecture consistency**: The method is effective across CNN-based (PixelCAM, DeepMIL) and Transformer-based (SAT) WSOL architectures, demonstrating that the approach is decoupled from the underlying architecture and is broadly generalizable.
4. **Dual-set dynamic resampling**: Rebuilding sets every $m$ epochs mitigates the cumulative effects of pseudo-label noise and effectively constitutes an implicit curriculum learning strategy.

## Limitations & Future Work

1. **Binary classification constraint**: All experiments involve binary classification (tumor vs. normal). Dominant-class bias under multi-class settings exhibits more complex patterns, and the forget-set construction strategy would require adjustment.
2. **Sensitivity to $\rho$**: The forgetting ratio $\rho$ requires manual search; too small a value provides insufficient correction, while too large a value risks incorrectly forgetting accurate predictions. Adaptive $\rho$ warrants exploration.
3. **CAM pseudo-label quality**: Pixel-level localization depends on the quality of source-model CAMs. If source-domain CAMs are themselves inaccurate — a well-known issue — $\mathcal{L}_{\text{loc}}$ may introduce noise.
4. **Computational overhead**: Every $m$ epochs require full-dataset inference on the target domain to recompute entropy and rebuild sets, which may become a bottleneck at large scale.
5. **Absence of UDA comparison**: Comparisons are limited to SFDA methods; the performance gap introduced by the source-free constraint cannot be assessed without comparison against UDA methods with access to source data.

## Related Work & Insights

- **vs. SFDA-DE**: SFDA-DE performs adaptation via distribution estimation but lacks a debiasing mechanism, failing completely under dominant bias (CL often stalls at 50%, PxAP drops below source-only). SFDA-DeP's forgetting mechanism directly addresses this root cause.
- **vs. ERL/RGV**: ERL uses regularization to stabilize training; RGV employs generative replay. Neither explicitly handles class prediction imbalance, and both remain ineffective under strong domain shift.
- **vs. Machine Unlearning (Basak & Yin, ECCV'24)**: Conventional machine unlearning removes knowledge of specific classes or data. SFDA-DeP innovatively repurposes the unlearning mechanism to reshape decision boundaries rather than erase knowledge.
- **Insights**: The forget/retain dual-set strategy can be generalized to other self-training scenarios with pseudo-label bias, such as long-tailed distributions in semi-supervised learning or class imbalance in self-supervised pretraining.

## Rating

- Novelty: ⭐⭐⭐⭐ The transfer of machine unlearning to prediction debiasing is novel, though the overall framework is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, three WSOL models, four SFDA baselines, with thorough ablation and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; Fig. 1 diagnosis is compelling; mathematical notation is rigorous.
- Value: ⭐⭐⭐⭐ Identifies the core failure mode of SFDA in pathology and provides an effective solution with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[AAAI 2026\] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation](../../AAAI2026/medical_imaging/graph-theoretic_consistency_for_robust_and_topology-aware_semi-supervised_histop.md)
- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](../../NeurIPS2025/medical_imaging/match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)

</div>

<!-- RELATED:END -->
