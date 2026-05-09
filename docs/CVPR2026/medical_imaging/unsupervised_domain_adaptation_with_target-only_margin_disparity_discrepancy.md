---
title: >-
  [Paper Note] Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy
description: >-
  [CVPR 2026][Medical Imaging][Unsupervised Domain Adaptation] This paper addresses unsupervised domain adaptation (UDA) for CT→CBCT liver segmentation. It identifies a contradictory term in the classical MDD objective—where the feature extractor is optimized to *maximize* the discrepancy between $f$ and $f'$ on the source domain—and proposes Target-Only MDD, which removes this contradiction and minimizes prediction discrepancy on both domains. The method achieves state-of-the-art UDA performance in both 2D and 3D experiments.
tags:
  - CVPR 2026
  - Medical Imaging
  - Unsupervised Domain Adaptation
  - Margin Disparity Discrepancy
  - CBCT
  - Liver Segmentation
  - Interventional Imaging
date: 2026-05-08
content_hash: 471f4e41b9e342da
---

# Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy

**Conference**: CVPR 2026
**arXiv**: [2603.09932](https://arxiv.org/abs/2603.09932)
**Area**: Medical Imaging
**Keywords**: Unsupervised Domain Adaptation, Margin Disparity Discrepancy, CBCT, Liver Segmentation, Interventional Imaging
**Code**: None

## TL;DR

This paper addresses unsupervised domain adaptation (UDA) for CT→CBCT liver segmentation. It identifies a contradictory term in the classical MDD objective—where the feature extractor is optimized to *maximize* the discrepancy between $f$ and $f'$ on the source domain—and proposes Target-Only MDD, which removes this contradiction and minimizes prediction discrepancy on both domains. The method achieves state-of-the-art UDA performance in both 2D and 3D experiments.

## Background & Motivation

- **Clinical Context**: CBCT provides real-time intraoperative 3D guidance in interventional radiology; automatic liver segmentation is critical for surgical planning.
- **Data Scarcity**: CT benefits from abundant publicly annotated datasets, whereas interventional CBCT data are scarce and unannotated.
- **Sources of Domain Shift**:
  1. Scatter artifacts, limited dynamic range, and reconstruction geometry differences in CBCT
  2. Intra-arterial contrast enhancement causing bright regions inside the liver
  3. Limited CBCT field of view differing from CT
- **Limitations of Existing Methods**:
  - Foundation models (SAM-MED 2D/3D, MA-SAM): primarily trained on natural images; generalization to CBCT is limited.
  - Image alignment methods (SIFA): rely on a shared field-of-view assumption, inapplicable to CT/CBCT pairs.
  - Self-training methods (BDCL): pseudo-label quality degrades under large domain shifts.
- **Problem with MDD**: In the classical MDD formulation, the feature extractor $\psi$ is optimized to **maximize** the discrepancy between $f$ and $f'$ on the source domain (the red-boxed term in Eq. 3), which directly contradicts the goal of domain alignment.

## Method

### Background: Classical MDD

A U-Net is decomposed into a feature extractor $\psi$, a segmentation head $f$, and an adversarial segmentation head $f'$. The classical MDD objective is:

$$\min_{f,\psi} \max_{f'} \mathcal{L}^{\text{task}}(f(z^S), y^S) + \alpha \mathcal{L}_{CE}(f'(z^T), f(z^T)) - \gamma \mathcal{L}_{CE}(f'(z^S), f(z^S))$$

**Issue**: In practice, $f$ is excluded from the discrepancy terms, and $\psi$ is optimized to maximize the discrepancy between $f$ and $f'$ on the source domain (the negative sign on the last term), contradicting the objective of aligning features across both domains.

### Target-Only MDD (Proposed Method)

The contradictory term is removed and the optimization is restructured into three alternating steps:

**Step 1** — Optimize segmentation head $f$:
$$\min_f L^{\text{task}}(f(z^S), y^S)$$

**Step 2** — Optimize adversarial head $f'$:
$$\min_{f'} \left[ \mathcal{L}_{CE}(f'(z^S), f(z^S)) - \gamma \mathcal{L}_{CE}(f'(z^T), f(z^T)) \right]$$

$f'$ is trained to imitate $f$ on the source domain while diverging from $f$ on the target domain.

**Step 3** — Optimize feature extractor $\psi$:
$$\min_\psi \left[ L^{\text{task}}(f(z^S), y^S) + \alpha \mathcal{L}_{CE}(f'(z^S), f(z^S)) + \gamma \mathcal{L}_{CE}(f'(z^T), f(z^T)) \right]$$

The key change: $\psi$ now **minimizes** the discrepancy between $f$ and $f'$ on **both domains** (the source domain term changes from negative to positive), eliminating the contradiction.

### Few-Shot Extension

After UDA training, $f \circ \psi$ is retained and $f'$ is discarded; the model is then fine-tuned with a small number of annotated target-domain samples.

### Implementation Details

- Backbone: U-Net, 5 stages, 64 channels in the first stage
- Hyperparameters: $\alpha = 7.5 \times 10^{-2}$, $\gamma = 3 \times 10^{-1}$
- Data: 573 CBCT cases + 678 CT cases, patient-level segmentation

## Key Experimental Results

### 2D CT→CBCT Liver Segmentation

| Type | Method | F1 (%) |
|------|------|--------|
| Source Only | U-Net | 54.1 |
| Foundation | SAM-MED 2D (5pt) | 67.7 |
| Self-Training | BDCL | 60.0 |
| Feature Align | DANN | 68.3 |
| Feature Align | MDD | 70.0 |
| **Feature Align** | **Ours** | **74.4** |
| Few-shot | Ours + 50 vol | 84.6 |
| Upper Bound | Target Only (100%) | 85.5 |

### 3D CT→CBCT Liver Segmentation

| Type | Method | F1 (%) |
|------|------|--------|
| Source Only | U-Net | 80.1 |
| Foundation | SAM-MED 3D (5pt) | 65.3 |
| Foundation | MA-SAM | 61.8 |
| Image Align | SIFA | 64.7 |
| Feature Align | DANN | 84.6 |
| **Feature Align** | **Ours** | **86.6** |
| Few-shot | Ours + 5 vol | 90.9 |
| Upper Bound | Target Only (100%) | 93.7 |

### Key Findings

1. **UDA without annotations outperforms 5-shot target-domain training**: Ours (86.6%) > Target Only 5-vol (84.7%).
2. **Ours + 5 vol (90.9%) ≈ Target Only 20-vol (89.6%)**.
3. **Hyperparameter robustness**: Performance remains stable across different $\alpha$ and $\gamma$ combinations.
4. **Lowest variance**: The proposed method achieves a standard deviation of 9.4%, substantially lower than MA-SAM (18.3%) and SAM-MED 3D (28.8%).

## Highlights & Insights

1. **Theory-driven improvement**: Rather than stacking complex modules, the paper identifies and corrects a contradictory term in the MDD objective, yielding a principled and concise fix.
2. **Sober assessment of foundation models**: SAM-MED performs substantially worse than UDA on CBCT, demonstrating that domain shift in medical imaging still requires dedicated treatment.
3. **In-depth analysis of CBCT-specific challenges**: Contrast-enhanced bright regions cause under-segmentation of the liver; the proposed method mitigates this effectively via 3D contextual information.
4. **Natural few-shot extension**: A small number of annotations after UDA suffices to approach fully supervised performance.

## Limitations & Future Work

- Validated on liver segmentation only; generalization to other organs or segmentation tasks remains unexplored.
- The dataset is proprietary, precluding reproducibility.
- Common image translation baselines (e.g., CycleGAN) are not compared in the 2D experiments.
- The MDD theoretical framework is primarily designed for classification; theoretical guarantees for segmentation tasks remain incomplete.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)

<!-- RELATED:END -->
