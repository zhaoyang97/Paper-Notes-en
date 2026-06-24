---
title: >-
  [Paper Note] Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading
description: >-
  [NeurIPS 2025 Workshop (GenAI for Health)][Medical Imaging][diabetic retinopathy] This paper proposes CAP-WAE (Constrained Asymmetric Prior Wasserstein Autoencoder), which addresses the challenges of long-tailed distribution and ordinal structure in diabetic retinopathy (DR) grading through three innovations: asymmetric priors, a margin-aware orthogonality and compactness loss, and a direction-aware ordinal loss, achieving state-of-the-art performance on multiple DR benchmark…
tags:
  - "NeurIPS 2025 Workshop (GenAI for Health)"
  - "Medical Imaging"
  - "diabetic retinopathy"
  - "ordinal classification"
  - "Wasserstein autoencoder"
  - "label distribution"
  - "imbalanced learning"
date: 2026-05-08
content_hash: 119ba9671505628d
---

# Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading

**Conference**: NeurIPS 2025 Workshop (GenAI for Health)  
**arXiv**: [2509.26146](https://arxiv.org/abs/2509.26146)  
**Code**: N/A  
**Area**: Medical Imaging / Ordinal Classification  
**Keywords**: diabetic retinopathy, ordinal classification, Wasserstein autoencoder, label distribution, imbalanced learning

## TL;DR
This paper proposes CAP-WAE (Constrained Asymmetric Prior Wasserstein Autoencoder), which addresses the challenges of long-tailed distribution and ordinal structure in diabetic retinopathy (DR) grading through three innovations: asymmetric priors, a margin-aware orthogonality and compactness loss, and a direction-aware ordinal loss, achieving state-of-the-art performance on multiple DR benchmarks.

## Background & Motivation
**Background**: DR grading is a canonical ordinal classification task (grades 0–4) with severely long-tailed data — severe DR (grades 3/4) samples are scarce yet clinically most critical.

**Limitations of Prior Work**:
   - Conventional methods employ isotropic Gaussian priors, which fail to model the heavy-tailed or skewed structure of minority classes.
   - Symmetric loss functions (e.g., cross-entropy) penalize under-estimation and over-estimation equally, which is inconsistent with clinical needs (under-estimation is more dangerous).
   - Latent spaces lack grade-level ordinality, with severe overlap between adjacent grades.

**Key Challenge**: Three intertwined challenges — long-tailed distribution, ordinal structure, and asymmetric clinical cost.

**Key Insight**: WAE is adopted over VAE to avoid posterior collapse; asymmetric priors are introduced to match the distributional characteristics of minority classes.

**Core Idea**: Asymmetric-prior WAE + margin-aware orthogonality and compactness loss + direction-aware ordinal soft labels, providing an end-to-end solution for long-tailed ordinal DR grading.

## Method

### Overall Architecture
Encoder → Latent space (constrained by MAOC loss) → Decoder (WAE reconstruction) + Ordinal grading head (direction-aware soft labels)

### Key Designs

1. **Wasserstein Autoencoder with Asymmetric Prior**

    - Function: Aligns the aggregated posterior $Q_Z$ with an asymmetric prior $P_Z$ via WAE.
    - Asymmetric Prior: Replaces the standard Gaussian with skewed distributions (e.g., skew-normal / log-normal).
    - Advantage: Preserves the heavy-tailed structure of minority classes, preventing the standard Gaussian prior from "squeezing" them.
    - WAE Objective: $\min_{E,D} \mathbb{E}[\|x - D(E(x))\|^2] + \lambda \cdot \text{MMD}(Q_Z, P_Z)$

2. **Margin-Aware Orthogonality and Compactness (MAOC) Loss**

    - Function: Enforces ordinal separability among grade-level representations in the latent space.
    - Orthogonality: Drives latent mean vectors of different grades to be mutually orthogonal: $\langle \mu_i, \mu_j \rangle \to 0$.
    - Compactness: Encourages within-grade clustering: $\text{Var}(z | y=k) \to \text{small}$.
    - Margin Awareness: Enforces a minimum separation between adjacent grades: $\|\mu_k - \mu_{k+1}\| \geq m$.
    - Formulation: $\mathcal{L}_{MAOC} = \alpha \sum_{i \neq j} |\langle \mu_i, \mu_j \rangle| + \beta \sum_k \text{tr}(\Sigma_k) + \gamma \sum_k \max(0, m - \|\mu_k - \mu_{k+1}\|)$

3. **Direction-Aware Ordinal Loss**

    - Function: Generates soft labels reflecting clinical priorities.
    - A lightweight head predicts asymmetric divergence parameters $(\sigma_L^k, \sigma_R^k)$.
    - Soft labels: $\tilde{y}_j^k = \exp(-\frac{(j-k)^2}{2\sigma_{L/R}^{k,2}})$ (with distinct left and right divergences).
    - Core mechanism: $\sigma_L < \sigma_R$ assigns heavier penalties to under-estimation errors.
    - KL divergence measures the discrepancy between the predicted distribution and the soft-label distribution.

### Loss & Training
- Adaptive Multi-Task Learning (MTL) Weighting balances WAE reconstruction, MAOC, and ordinal losses.
- End-to-end training with no separate pretraining required.
- Data augmentation: standard flipping/rotation + Mixup.

## Key Experimental Results

### Main Results — DR Grading Benchmarks

| Method | EyePACS QWK↑ | EyePACS Acc↑ | EyePACS F1↑ | APTOS QWK↑ |
|------|-------------|-------------|-------------|------------|
| ResNet-50 + CE | 0.812 | 78.5 | 52.3 | 0.845 |
| CORN (Ordinal Regression) | 0.836 | 80.1 | 55.7 | 0.862 |
| UniOrdinal | 0.849 | 81.3 | 57.2 | 0.871 |
| BalancedMix | 0.841 | 80.8 | 58.9 | 0.868 |
| VAE + Ordinal | 0.853 | 81.7 | 58.4 | 0.875 |
| **CAP-WAE (Ours)** | **0.878** | **83.6** | **63.1** | **0.894** |

### Ablation Study

| Configuration | EyePACS QWK↑ | Macro-F1↑ |
|------|-------------|-----------|
| Baseline (WAE + CE) | 0.838 | 54.8 |
| + Asymmetric Prior | 0.852 | 57.3 |
| + MAOC Loss | 0.861 | 59.6 |
| + Direction-Aware Ordinal | 0.871 | 61.8 |
| + Adaptive MTL Weighting | **0.878** | **63.1** |

### Per-Grade F1 (EyePACS)

| Grade | Baseline F1 | CAP-WAE F1 | Gain |
|------|-----------|-----------|------|
| Grade 0 (majority) | 89.2 | 90.1 | +0.9 |
| Grade 1 | 61.3 | 67.8 | +6.5 |
| Grade 2 | 52.7 | 61.3 | +8.6 |
| Grade 3 (minority) | 38.1 | 52.4 | +14.3 |
| Grade 4 (minority) | 32.8 | 43.9 | +11.1 |

### Key Findings
- CAP-WAE surpasses the Prev. SOTA by 2.5–4.2 percentage points on QWK.
- Minority classes (Grades 3/4) show the most significant F1 improvements (+11–14%), validating the effectiveness of the asymmetric design.
- The MAOC loss contributes most to latent space structuring.
- The direction-aware loss reduces under-estimation errors by approximately 23%.
- t-SNE visualizations reveal compact, well-ordered grade clusters in the latent space.

## Highlights & Insights
- **Clinically Motivated Design**: Asymmetric penalties (heavier for under-estimation) directly correspond to clinical requirements.
- **WAE Outperforms VAE**: Posterior collapse is avoided, yielding a more controllable latent space.
- **Three Complementary Innovations**: Asymmetric prior (distribution level) + MAOC (representation level) + ordinal loss (supervision level).

## Limitations & Future Work
- Validation is limited to DR grading; generalization to other ordinal tasks (e.g., age estimation, pain scoring) remains unexplored.
- As a NeurIPS Workshop paper, space constraints limit the depth of ablation studies.
- The choice of asymmetric prior family (skew-normal vs. log-normal) lacks a comparative analysis.
- Multi-center generalizability has not been verified.

## Related Work & Insights
- **CORAL (CVPR 2020)**: A classical method for ordinal regression.
- **Label Distribution Learning (Geng 2016)**: Foundational theory for LDL.
- **WAE (Tolstikhin et al. 2018)**: Wasserstein Autoencoder.
- Insight: The asymmetric prior concept is broadly applicable to other long-tailed ordinal tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Asymmetric priors combined with direction-aware ordinal loss are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation and per-grade analyses are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation–method–experiment logic is clear and well-structured.
- Value: ⭐⭐⭐⭐ Practical value for long-tailed ordinal classification in medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Divide, Conquer, and Aggregate: Asymmetric Experts for Class-Imbalanced Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/divide_conquer_and_aggregate_asymmetric_experts_for_class-imbalanced_semi-superv.md)
- [\[CVPR 2025\] Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data](../../CVPR2025/medical_imaging/domain_adaptive_diabetic_retinopathy_grading_with_model_absence_and_flowing_data.md)
- [\[CVPR 2026\] KLIP: localized distribution shift detection via KL-divergence with diffusion priors in Inverse Problems](../../CVPR2026/medical_imaging/klip_localized_distribution_shift_detection_via_kl-divergence_with_diffusion_pri.md)
- [\[ICLR 2026\] Frequency-Balanced Retinal Representation Learning with Mutual Information Regularization](../../ICLR2026/medical_imaging/frequency-balanced_retinal_representation_learning_with_mutual_information_regul.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](../../CVPR2025/medical_imaging/cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)

</div>

<!-- RELATED:END -->
