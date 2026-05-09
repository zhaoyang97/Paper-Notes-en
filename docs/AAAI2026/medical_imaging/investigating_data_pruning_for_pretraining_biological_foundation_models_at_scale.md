---
title: >-
  [Paper Note] Investigating Data Pruning for Pretraining Biological Foundation Models at Scale
description: >-
  [AAAI 2026][Medical Imaging][Data Pruning] This paper proposes a post-hoc data pruning framework based on influence functions, leveraging Subset-Based Self-Influence estimation and two selection strategies (Top-k Influence and Coverage-Centric Influence). Under an extreme pruning rate exceeding 99%, an RNA-FM pretrained on only 0.2M sequences matches or surpasses the full model trained on 23M sequences across multiple downstream tasks, revealing substantial redundancy in biological sequence datasets.
tags:
  - AAAI 2026
  - Medical Imaging
  - Data Pruning
  - Biological Foundation Models
  - Influence Functions
  - Coreset Selection
  - RNA-FM
  - ESM
  - Protein Language Models
date: 2026-05-08
content_hash: 9a5efaf09fed802c
---

# Investigating Data Pruning for Pretraining Biological Foundation Models at Scale

**Conference**: AAAI 2026
**arXiv**: [2512.12932](https://arxiv.org/abs/2512.12932)
**Code**: [github.com/victor-yifanwu/bio-coreset](https://github.com/victor-yifanwu/bio-coreset)
**Area**: Medical Imaging / Bioinformatics / Foundation Models
**Keywords**: Data Pruning, Biological Foundation Models, Influence Functions, Coreset Selection, RNA-FM, ESM, Protein Language Models

## TL;DR
This paper proposes a post-hoc data pruning framework based on influence functions, leveraging Subset-Based Self-Influence estimation and two selection strategies (Top-k Influence and Coverage-Centric Influence). Under an extreme pruning rate exceeding 99%, an RNA-FM pretrained on only 0.2M sequences matches or surpasses the full model trained on 23M sequences across multiple downstream tasks, revealing substantial redundancy in biological sequence datasets.

## Background & Motivation

**Background**: Biological foundation models (BioFMs) such as RNA-FM (23M RNA sequences) and ESM (2.78B protein sequences) have demonstrated strong performance on structure prediction and functional annotation tasks, yet their training costs are prohibitively high (RNA-FM: 8×A100 GPUs, 30 days), severely limiting reproducibility and accessibility for academic laboratories.

**Gap in Data Pruning for Biological Domains**:
- Extensive data pruning work exists in CV/NLP, but pruning for BioFM pretraining remains largely unexplored.
- Training-dynamics-based methods (EL2N, AUM) require the full training process, making them infeasible for BioFMs.
- Local-density-based methods require pairwise similarity computation, which does not scale to millions of sequences.
- Influence-function-based methods require inverting the full-training-set Hessian, which is infeasible for models with hundreds of millions of parameters.

**Core Problem**: Can a post-hoc method identify the most informative training subset without accessing the full training process?

**Key Insight**: Leverage influence function theory to approximate full-dataset curvature information on a small subset, enabling efficient estimation of per-sample importance.

## Method

### Overall Architecture

Three-stage pipeline:
1. **Influence Score Estimation**: Subset-based self-influence functions
2. **Coreset Selection**: Top I or CCI strategy
3. **Pretraining from Scratch**: Retrain BioFMs on the selected coreset

### Subset-Based Self-Influence

**Classical Influence Functions**: The influence of a training sample $z_{tr}$ on a validation sample $z_{val}$ is defined as:
$$\mathcal{I}(z_{tr}; z_{val}) = g_{z_{val}}^\top H_{\theta^*}^{-1} g_{z_{tr}}$$

where $g$ denotes gradients and $H$ denotes the Hessian. Computing $H^{-1}$ is infeasible for large models.

**Key Innovation — Subset Approximation**:

**Assumption 1**: A model $\tilde{\theta}$ trained on a randomly sampled subset $D_{sub}$ is approximately optimal on that subset.

Under this assumption, the full-training-set Hessian $H_{tr}$ is replaced by the subset Hessian $\tilde{H}_{sub}$:

$$\mathcal{I}(z_{tr}, D_{sub}) = \tilde{g}_{z_{tr}}^\top \tilde{H}_{sub}^{-1} \tilde{g}_{z_{tr}}$$

**Theoretical Support (Proposition 1)**: Under the flat loss landscape condition of large models (confirmed by recent studies), subset curvature sufficiently approximates full-dataset curvature.

**Further Acceleration — Diagonal Empirical Fisher Approximation**:

$$\tilde{H}_{sub}^{-1} \approx \text{diag}(\tilde{F}_{sub})^{-1}$$

where $\text{diag}(\tilde{F}_{sub}) = \frac{1}{M}\sum_{m=1}^M \tilde{g}_{z_m} \odot \tilde{g}_{z_m}$

Computational complexity is reduced from $O(M \cdot d^2 + d^3)$ to $O(M \cdot d)$, making the approach feasible for models with billions of parameters.

**Practical Note**: Lightweight fine-tuning for one epoch on the subset is sufficient to satisfy Assumption 1 at negligible cost.

### Two Coreset Selection Strategies

1. **Top-k Influence (Top I)**: Selects the $k$ samples with the highest influence scores.
    - Prioritizes samples with the greatest impact on model parameters.
    - Theoretically corresponds to the most informative data points.

2. **Coverage-Centric Influence (CCI)**: Applies stratified sampling over the influence score distribution.
    - Maintains a balanced distribution of "easy" and "hard" samples.
    - Inspired by Sorscher et al. 2022: retaining only the hardest samples under extreme pruning leads to overfitting.
    - Stratified sampling ensures distributional coverage.

### Experimental Setup

- **Extreme pruning rate**: Only 0.2M sequences retained (RNA: ~1% of 23M; protein: ~4.4% of 4.5M).
- Models trained from scratch for 10 epochs on the coreset.
- Evaluation on diverse downstream tasks.

## Key Experimental Results

### RNA-FM Experiments

#### Functional and Engineering Prediction Tasks

| Method | Data Size | TypeCls ACC(%) | TypeCls F1(%) | Modif AUC(%) | CRI-On SC(%) | CRI-On MSE↓ |
|--------|-----------|----------------|---------------|--------------|--------------|-------------|
| RNA-FM | 23M | 91.93 | 91.87 | 94.98 | 31.87 | .0118 |
| Random | 2M | 82.21 | 82.01 | 92.82 | 26.72 | .0158 |
| Random | 0.2M | 82.15 | 81.97 | 91.86 | 26.67 | .0161 |
| **Top I** | 0.2M | 82.51 | 82.53 | 93.20 | 27.08 | .0149 |
| **CCI** | 0.2M | **82.88** | **83.12** | **93.86** | **32.90** | **.0135** |

- CCI **surpasses the full RNA-FM** (23M sequences) on CRISPR On-Target prediction.
- The 0.2M coreset outperforms 2M random selection.

#### Structure and Interaction Prediction Tasks

| Method | Sec. Struct. F1(%) | Distance Map SC(%) | Contact Map Top-1.0L(%) | RBP Interaction ACC(%) |
|--------|-------------------|--------------------|-------------------------|------------------------|
| RNA-FM | 62.20 | 89.21 | 93.93 | 72.47 |
| Random 0.2M | 55.60 | 84.90 | 94.18 | 69.65 |
| **Top I** | **57.05** | **86.47** | **94.36** | **71.25** |
| CCI | 56.36 | 85.59 | 94.20 | 69.46 |

- **Top I outperforms CCI** on structure-related tasks, as high-influence samples encode richer structural information.
- Top I **surpasses the full RNA-FM** on contact map prediction.

### ESM-C Protein Experiments (Generalizability Validation)

| Method | Data Size | Localization ACC(%) | Sec. Struct. ACC(%) | PPI MAE↓ | PPI RMSE↓ |
|--------|-----------|--------------------|--------------------|----------|-----------|
| ESM-C | 2.78B | 91.63 | 86.10 | 1.92 | 2.44 |
| Random | 2M | 75.76 | 67.20 | 2.39 | 2.87 |
| Random | 0.2M | 73.64 | 66.18 | 2.51 | 3.01 |
| **Top I** | 0.2M | 77.13 | 69.34 | **2.06** | **2.64** |
| **CCI** | 0.2M | **79.25** | **71.48** | 2.14 | 2.69 |

- Both Top I and CCI at 0.2M outperform 2M random selection, further confirming substantial redundancy in protein data.
- CCI performs better in the protein setting.

### Ablation Study: Necessity of Fine-tuning

| Variant | Modif AUC(%) | Distance Map SC(%) |
|---------|--------------|-------------------|
| Top I (w/o ft) | 92.94 | 84.13 |
| CCI (w/o ft) | 93.31 | 84.95 |
| **Top I** | **93.20** | **86.47** |
| **CCI** | **93.86** | **85.59** |

Lightweight fine-tuning on the subset prior to influence score computation consistently improves results, validating the importance of Assumption 1.

## Highlights & Insights

1. **Reveals substantial redundancy in biological training data**: Less than 1% of the data suffices to approach or exceed the full model's performance.
2. **Post-hoc framework requires no training process**: Only pretrained model weights and lightweight subset fine-tuning are needed, applicable even to publicly released models without disclosed training details.
3. **Complementarity of Top I and CCI**:
    - CCI excels at functional/engineering prediction (benefits from distributional coverage).
    - Top I excels at structural/interaction prediction (benefits from information density).
4. **Complete theoretical derivation**: Each step — from classical influence functions to subset approximation to diagonal Fisher factorization — is theoretically grounded.
5. **High practical value**: Enables academic laboratories with limited computational resources to pretrain BioFMs at drastically reduced cost.

## Limitations & Future Work

1. RNA experiments are validated only on RNA-FM; larger RNA models (e.g., Evo 2) are not evaluated.
2. Protein experiments are limited to 4.5M sequences due to resource constraints (far smaller than ESM-C's 2.78B training set); coreset effectiveness at full scale remains unverified.
3. The error bound for Assumption 1 (approximate optimality on the subset) is not quantitatively analyzed.
4. The diagonal Fisher approximation may be inaccurate when the Hessian has strong off-diagonal structure.
5. Validation is restricted to self-supervised pretraining (MLM); data pruning effects in supervised fine-tuning settings are not discussed.
6. No direct comparison with density- or diversity-based methods (e.g., Facility Location) is provided.

## Related Work & Insights

- **Data Pruning**: EL2N (training dynamics), Sorscher et al. 2022 (data density), D2 Pruning (difficulty-diversity balance)
- **Influence Functions**: Koh & Liang 2017 (classical IF), DataInf, TRAK (efficient approximations)
- **Biological Foundation Models**: RNA-FM (Chen 2022, 23M sequences), ESM-C/ESM3 (Hayes 2025, 2.78B sequences), Evo 2

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — Subset influence function approximation offers a theoretical contribution; first systematic introduction of data pruning to BioFM pretraining.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Dual-modality validation on RNA and protein with comprehensive evaluation across diverse downstream tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and experimental presentation is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — Directly reduces BioFM pretraining cost, offering substantial practical value to computationally constrained research groups.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[AAAI 2026\] Personalization of Large Foundation Models for Health Interventions](personalization_of_large_foundation_models_for_health_interventions.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[AAAI 2026\] ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)

</div>

<!-- RELATED:END -->
