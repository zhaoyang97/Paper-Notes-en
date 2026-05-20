---
title: >-
  [Paper Note] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation
description: >-
  [AAAI 2026][Medical Imaging][Ambiguous medical image segmentation] This paper proposes the ATFM framework, which decouples prediction accuracy and diversity into distribution-level and sample-level optimization through a…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Ambiguous medical image segmentation"
  - "truncated diffusion model"
  - "Flow Matching"
  - "Gaussian truncation representation"
  - "semantic consistency"
date: 2026-05-08
content_hash: 957d8c51e92bbeed
---

# Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation

**Conference**: AAAI 2026
**arXiv**: [2511.06857v2](https://arxiv.org/abs/2511.06857v2)  
**Code**: [https://github.com/PerceptionComputingLab/ATFM](https://github.com/PerceptionComputingLab/ATFM)  
**Area**: Medical Imaging
**Keywords**: Ambiguous medical image segmentation, truncated diffusion model, Flow Matching, Gaussian truncation representation, semantic consistency

## TL;DR

This paper proposes the ATFM framework, which decouples prediction accuracy and diversity into distribution-level and sample-level optimization through a data-hierarchical inference paradigm. By integrating two modules — Gaussian Truncation Representation (GTR) and Segmentation Flow Matching (SFM) — ATFM simultaneously improves prediction accuracy, fidelity, and diversity in ambiguous medical image segmentation.

## Background & Motivation

Ambiguous Medical Image Segmentation (AMIS) aims to generate multiple plausible segmentation predictions for a single medical image, reflecting the inherent inter-annotator ambiguity. Clinically, high diversity captures the intrinsic ambiguity of the image, while high accuracy supports reliable diagnostic decisions — both are indispensable.

Existing methods face an inherent trade-off between accuracy and diversity:
- **Stochastic methods** (Prob. U-Net, PHiSeg): sacrifice accuracy for diversity, yielding low-confidence diagnoses
- **Multi-annotator-aware methods**: improve both by modeling annotator styles, but suppress low-frequency patterns, degrading segmentation quality
- **cVAE/diffusion methods**: inject randomness to enhance diversity, but single-stage inference couples accuracy and diversity optimization

Truncated Diffusion Probabilistic Models (TDPMs) show promise through an inference paradigm shift — truncating the diffusion process at $T_{\text{trunc}}$ and estimating the truncated distribution with an auxiliary network. However, directly applying TDPMs introduces three issues: (1) a unified inference objective cannot decouple accuracy from diversity; (2) sampling-based approximation of the truncated distribution leads to insufficient fidelity; (3) lack of semantic guidance after truncation impairs prediction plausibility.

## Core Problem

How to **simultaneously improve** prediction accuracy and diversity in ambiguous medical image segmentation, breaking the inherent trade-off observed in existing methods?

## Method

### Overall Architecture

ATFM consists of three core components, organized around a redefined inference paradigm:

1. **Data-Hierarchical Inference**: Redefines the AMIS-specific inference paradigm, improving accuracy at the data distribution level and enhancing diversity at the data sample level
2. **GTR (Gaussian Truncation Representation)**: Explicitly models a Gaussian distribution at the truncation point to improve prediction fidelity and the reliability of the truncated distribution
3. **SFM (Segmentation Flow Matching)**: Introduces semantics-aware flow transformation to enhance diversity while ensuring prediction plausibility

### Key Designs

#### Data-Hierarchical Inference

The core idea is to decouple accuracy and diversity by **marginalizing randomness in the diffusion process**:

- **Distribution level (before truncation)**: Marginalizes randomness across multiple annotator samples $\{s_1, s_2, \dots, s_n\}$; supervises the explicit distribution $P \sim \mathcal{N}(\mu, \Sigma)$ at the truncation point to approximate the true annotation distribution $Q \sim \mathcal{N}'(\mu', \Sigma')$, **optimizing overall accuracy**
- **Sample level (after truncation)**: Samples from the high-fidelity truncated distribution and generates diverse predictions $\{pred_i\}_{i=1}^n$ through the diffusion process, supervised per-sample against ground truth $\{gt_i\}_{i=1}^n$, **enhancing diversity**

This hierarchical design ensures that accuracy optimization does not compromise diversity, and diversity enhancement is built upon a globally aligned distribution.

#### Gaussian Truncation Representation (GTR)

GTR replaces the sampling-based approximation of conventional TDPMs with explicit Gaussian modeling:

- Extracts image semantic features $Z$ via a segmentation backbone $f_\theta$
- Estimates mean $\mu$ and covariance $\Sigma$ using independent convolutional layers $g_\phi$ and $h_\psi$, respectively (low-rank parameterization with rank $r=10$)
- Truncation-point distribution: $X_{T_{\text{trunc}}} \sim \mathcal{N}(\mu, \Sigma)$

Theoretical justification (Theorem 1 & 2):
- **Theorem 1**: The marginal distribution at any diffusion timestep can be parameterized as $\mathcal{N}(\mu, DD^\top + L)$
- **Theorem 2**: For any Gaussian distribution, there exists a timestep at which the diffusion process produces the same distribution

Therefore, the Gaussian distribution in GTR is a valid and optimal choice for the truncated distribution.

Network architecture: Standard encoder–decoder, 4 resolution levels, encoder filter sizes 32→64→128→192, decoder uses transposed convolution upsampling and skip connections.

#### Segmentation Flow Matching (SFM)

SFM replaces DDPM with Flow Matching after truncation, and introduces semantic consistency modeling:

- **Optimal Transport schedule**: Linear interpolation along the shortest path from source $X_{T_{\text{trunc}}}$ to target $X_1$
    - $X_t = t \times X_1 + (1-t) \times X_{T_{\text{trunc}}}$
- **ST-Net (Semantic-aware Transformation Network)**: Time-conditioned U-Net predicting the flow transformation direction $g_\theta(X_t)$
    - 4-level encoder–decoder, 15 residual blocks
    - Sinusoidal timestep embedding fused into each residual block via MLP
    - Linear attention in all layers; full self-attention in the bottleneck
- **Intermediate prediction**: $x_1^t = x_t + g_\theta(X_t) \times (1-t)$, projected in latent space via analytic geometry
- **Semantic consistency**: At each timestep $t$, computes Dice loss between the intermediate prediction $x_1^t$ and all ground-truth annotations, explicitly modeling semantic consistency among state, prediction, and ground truth

Advantage of FM over DDPM: avoids the interference of Gaussian constraints on fine-grained predictions.

### Loss & Training

**Two-stage training**:

1. **GTR stage**: Train GTR to convergence, then freeze parameters
    - $\mathcal{L}_{\text{Prior}} = -\log \int p(Y|X_{T_{\text{trunc}}}) p(X_{T_{\text{trunc}}}|X) dX_{T_{\text{trunc}}} \approx \frac{1}{M}\sum_{i=1}^M -\log p(Y|X_{T_{\text{trunc}}}^i)$
    - Monte Carlo sampling with $M=20$

2. **SFM stage**: Train SFM on top of the frozen GTR
    - $\mathcal{L}_{\text{FM}}$: standard Flow Matching loss
    - $\mathcal{L}_{\text{SF}}$: semantic consistency loss (Dice loss)
    - $\mathcal{L}_{\text{SFM}} = \mathcal{L}_{\text{FM}} + \alpha \cdot \mathcal{L}_{\text{SF}}$
    - $\alpha$: $10^{-3}$ for LIDC, $10^{-4}$ for ISIC3

**Training configuration**:
- Single RTX 3090 (24 GB)
- GTR: 1000 epochs (LIDC) / 400 epochs (ISIC3)
- SFM: 200 epochs (LIDC) / 120 epochs (ISIC3)
- Adam optimizer, learning rate $10^{-4}$
- $\lambda = 10^{-3}$ ($T=1000$), linear schedule

## Key Experimental Results

| Dataset | Metric | Ours (ATFM) | Prev. SOTA | Gain |
|---------|--------|-------------|------------|------|
| LIDC | GED₁₆↓ | Best | CCDM / AB | Consistently superior |
| LIDC | GED₁₀₀↓ | Best | Runner-up | **11.5%** |
| LIDC | HM-IoU₃₂↑ | Best | Runner-up | **≥7.3%** |
| LIDC | MDM₃₂↑ | Best | — | Leading |
| ISIC3 | GED↓ | Best | Runner-up | **12%** |
| ISIC3 | HM-IoU↑ | Best | — | Best |
| ISIC3 | MDM↑ | Best | — | Best |

**Inference efficiency** (generating 100 samples):

| Method | Steps | Time |
|--------|-------|------|
| CIMD | Multi-step | 420s |
| AB | Multi-step | 1050s |
| CCDM | Multi-step | 1100s |
| **ATFM** | **GTR + 25 steps** | **113s** |

ATFM is substantially faster than other diffusion-based methods, requiring only 25 diffusion steps plus a single GTR estimation.

### Ablation Study

Five variants compared on the LIDC dataset:

| Variant | Description | Conclusion |
|---------|-------------|------------|
| Act. GTR | GTR + activation layer only | Baseline |
| SFM w/o $\mathcal{L}_{\text{SF}}$ | SFM without semantic loss | Insufficient diversity |
| SFM | Full SFM | Strong standalone performance |
| ATFM w/o $\mathcal{L}_{\text{SF}}$ | Full framework without semantic loss | Reduced plausibility |
| **ATFM** | **Full framework** | **Best** |

- ATFM outperforms Act. GTR and SFM alone by **≥10%** and **≥6%**, respectively, validating data-hierarchical inference and the synergy of both modules
- The average gap between models with and without $\mathcal{L}_{\text{SF}}$ reaches **11%**, highlighting the critical role of semantic consistency modeling
- 25 inference steps achieves the best balance between performance and efficiency
- An excessively small $\alpha$ limits the effect of $\mathcal{L}_{\text{SF}}$; an excessively large $\alpha$ weakens $\mathcal{L}_{\text{FM}}$

## Highlights & Insights

1. **Inference paradigm innovation**: The first work to redefine the TDPMs inference paradigm as an AMIS-specific data-hierarchical inference, fundamentally decoupling accuracy and diversity
2. **Theoretical completeness**: Theorems 1 & 2 provide rigorous theoretical guarantees for GTR's Gaussian modeling
3. **Coherent three-module design**: Data-Hierarchical Inference + GTR + SFM each address a specific problem with clear design logic
4. **Significant efficiency advantage**: 113s vs. 420–1100s, representing a 3.7–9.7× speedup with superior performance
5. **FM as a replacement for DDPM**: Avoids the interference of Gaussian constraints on fine-grained segmentation, representing a well-motivated technical choice

## Limitations & Future Work

1. **Limited dataset scale**: Validation is restricted to LIDC (lung CT) and a subset of ISIC3 (skin lesions, only 300 images), without covering additional modalities and organs
2. **Fixed annotation count**: LIDC uses 4 annotations and ISIC3 uses 3; the effect of varying annotation counts on the method remains unexplored
3. **Two-stage training**: GTR must be trained to convergence before freezing, making the pipeline complex; end-to-end joint training may be preferable
4. **Missing table values**: Quantitative table values in the HTML paper were not fully displayed, limiting direct comparison
5. **3D extension**: Experiments are conducted only on 2D slices; applicability to 3D volumetric segmentation remains to be verified

## Related Work & Insights

| Method Category | Representative Work | Limitations | ATFM Advantage |
|----------------|--------------------|-----------|--------------  |
| Model ensembles / multi-head | SSN, MoSE | Do not alter the inference process; constrained by model selection | Redefines the inference paradigm |
| cVAE methods | Prob. U-Net, PHiSeg | Single-stage inference; accuracy and diversity are coupled | Hierarchically decouples the two objectives |
| Diffusion methods | CIMD, CCDM | Gaussian constraints interfere with fine-grained predictions; slow inference | FM avoids Gaussian constraints; efficient 25-step inference |
| Multi-annotator-aware | c-Prob. U-Net, c-SSN | Suppress low-frequency patterns | GTR explicitly models and preserves low-frequency patterns |
| Conventional TDPMs | TDPM | Inaccurate sampling approximation; lack of semantic guidance | GTR explicit modeling + SFM semantic supervision |

**Broader implications**:

1. **Redefining inference paradigms**: The idea of redefining the inference process without modifying the model architecture is transferable to other generative tasks requiring multi-objective balance
2. **Generality of truncated diffusion**: The "truncation + replacement of intermediate inference path" strategy in TDPMs reveals structural flexibility in diffusion model inference, applicable to other conditional generation tasks
3. **Flow Matching for segmentation**: Introducing FM into segmentation tasks is an emerging trend that avoids the mismatch between DDPM's Gaussian assumption and the one-hot label space
4. **Explicit vs. implicit distribution modeling**: GTR's explicit Gaussian modeling approach can be generalized to other scenarios requiring distribution modeling at intermediate representations

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The data-hierarchical inference paradigm is a meaningful contribution; the three-module design logic is clear
- **Technical Depth**: ⭐⭐⭐⭐ — Backed by theoretical guarantees (two theorems) with complete methodological derivation
- **Experimental Thoroughness**: ⭐⭐⭐ — Only two datasets, but ablation and hyperparameter analyses are comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured; the "Summarized Advantage" for each module aids understanding
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for AMIS; the application of FM in medical segmentation offers a valuable reference

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/medical_imaging/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/medical_imaging/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[AAAI 2026\] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)
- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)

</div>

<!-- RELATED:END -->
