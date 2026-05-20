---
title: >-
  [Paper Note] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration
description: >-
  [NeurIPS 2025][Image Restoration][Adverse Weather Restoration] This paper proposes the MODEM framework, which combines Morton-encoded spatial scanning with selective state space models (SSMs) to capture spatially heterog…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "Adverse Weather Restoration"
  - "State Space Model"
  - "Morton Order"
  - "Degradation Estimation"
  - "Mamba"
date: 2026-05-08
content_hash: ceaae38830129533
---

# MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration

**Conference**: NeurIPS 2025
**arXiv**: [2505.17581](https://arxiv.org/abs/2505.17581)  
**Code**: Coming soon  
**Area**: Image Restoration
**Keywords**: Adverse Weather Restoration, State Space Model, Morton Order, Degradation Estimation, Mamba

## TL;DR

This paper proposes the MODEM framework, which combines Morton-encoded spatial scanning with selective state space models (SSMs) to capture spatially heterogeneous weather degradation patterns. Equipped with a dual degradation estimation module that provides both global and local priors, MODEM achieves state-of-the-art unified adaptive restoration across multiple adverse weather degradation types.

## Background & Motivation

Image restoration under adverse weather conditions is a fundamental problem in computer vision. The central challenge lies in the **highly non-uniform and spatially heterogeneous nature of weather degradation**: haze manifests as smooth intensity attenuation, whereas rain streaks and snowflakes produce localized, sharp occlusions.

**Limitations of Prior Work**:
- **Task-specific methods**: Train separate models for each weather type (deraining, dehazing, desnowing), lacking scalability.
- **Unified frameworks** (TransWeather, Histoformer, etc.): Although capable of handling multiple weather types within a single model, they still lack **explicit degradation estimation mechanisms** to model fine-grained spatially varying degradation patterns.

**Core Idea**: Weather degradation is treated as the evolution of latent variables in a state space. The degradation characteristic at each pixel constitutes a "state," and these states evolve spatially under the joint influence of local (e.g., rain streaks) and non-local (e.g., drifting haze) patterns — a formulation that naturally aligns with the hidden-state recurrence of SSMs.

**Key Connection**: In the SSM formulation $y_k = CA h_{k-1} + CB x_k$, the term $CA h_{k-1}$ models long-range degradation context (e.g., global haze layer), while $CB x_k$ captures local degradation details (e.g., raindrops).

## Method

### Overall Architecture

MODEM employs a two-stage training strategy:
- **Stage 1**: DDEM receives both the degraded image $I_{LQ}$ and the ground truth $I_{GT}$ simultaneously to learn the degradation mapping; the backbone network receives only $I_{LQ}$ along with the degradation prior for restoration.
- **Stage 2**: DDEM receives only $I_{LQ}$ (supervised by the frozen DDEM from Stage 1), enabling pure degradation estimation at inference time.

The backbone network consists of $N$ stacked MDSL (MOS2D Degradation-Aware Layer) blocks.

### Key Designs

1. **Morton-Order 2D-Selective-Scan (MOS2D)**:
   Conventional SSM-based image processing relies on raster scanning (row-by-row), which disrupts spatial locality — pixels that are spatially adjacent may be far apart in the resulting 1D sequence. The authors propose scanning via Morton encoding (Z-order curve): by interleaving the binary bits of coordinates $(i,j)$ as $z = \text{interleave}(i,j)$, the 2D feature map is unfolded into a locality-preserving 1D sequence. This enables the SSM to simultaneously capture local and long-range dependencies through a structured traversal.

2. **Dual Degradation Estimation Module (DDEM)**:
   After extracting degradation features $F$ from the input, DDEM generates two complementary degradation priors:
    - **Global degradation descriptor** $Z_0 = \sigma(\text{Linear}(\text{MLP}(\text{AvgPool}(F)))) \in \mathbb{R}^{C_d}$: encodes weather type and severity.
    - **Spatially adaptive degradation kernel** $Z_1 = \text{Conv}(F) \times \text{Conv}(F)^T \in \mathbb{R}^{C_{d1} \times C_{d2}}$: encodes local degradation structure and variation.

3. **Dual Degradation Modulation**:

    - **DAFM (Degradation-Adaptive Feature Modulation)**: Uses $Z_0$ to generate channel-wise adaptive weights and biases, applying FiLM modulation as $F_{\text{DAFM}} = (Z_0^w \odot F_i) + Z_0^b$.
    - **DSAM (Degradation-Selective Attention Modulation)**: Uses $Z_1$ to construct an attention matrix that guides the generation of SSM parameters $B$, $C$, and $\Delta$: $F_{\text{DSAM}} = W_F F_{\text{DAFM}} \times \text{Softmax}(W_Z Z_1)$.

   SSM parameters are dynamically generated from $F_{\text{DSAM}}$, ensuring that state evolution and output are adaptive to the degradation characteristics.

### Loss & Training

- **Stage 1**: $\mathcal{L}_1 + \mathcal{L}_{cor}$ (Pearson correlation coefficient loss)
- **Stage 2**: $\mathcal{L}_1 + \mathcal{L}_{cor} + \mathcal{L}_{KL}$ (KL divergence to enforce consistency of the degradation representation $\tilde{Z}$)
- 4× RTX 3090, AdamW + Cosine Annealing Restart
- Trained on a joint all-weather dataset

## Key Experimental Results

### Main Results (Table 1, Unified Model Comparison)

| Dataset | Metric | MODEM | Histoformer | Prev. SOTA | Gain |
|--------|------|-------|-------------|-----------|------|
| Snow100K-S | PSNR | **38.08** | 37.41 | 36.92 | +0.67 |
| Snow100K-L | PSNR | **32.52** | 32.16 | 31.92 | +0.36 |
| Outdoor-Rain | PSNR | **33.10** | 32.08 | 31.39 | +1.02 |
| RainDrop | PSNR | 33.01 | **33.06** | 32.38 | -0.05 |
| **Average** | PSNR | **34.18** | 33.68 | 33.04 | **+0.50** |

### Ablation Study (Table 7)

| Configuration | Outdoor PSNR | RainDrop PSNR | Notes |
|------|-------------|---------------|------|
| Full model | **33.10** | **33.01** | Complete MODEM |
| w/o Morton | 32.89 | 32.69 | Morton scanning contributes to spatial modeling |
| w/o DDEM | 32.37 | 32.38 | Degradation estimation is a core component |
| w/o DAFM | 32.19 | 32.62 | Global modulation is important for mixed degradation |
| w/o DSAM | 32.77 | 32.72 | Local modulation is critical for localized degradation such as raindrops |

### Key Findings

- The PSNR gain of 1.02 dB on Outdoor-Rain is the most pronounced improvement, confirming the advantage of the proposed approach on mixed degradation (rain + haze).
- MODEM surpasses Histoformer on real-world snow scenes without additional fine-tuning, demonstrating strong generalization.
- T-SNE visualizations show that MODEM's features exhibit better cluster separation across different weather types.
- Perceptual metrics (LPIPS / Q-Align / MUSIQ) also achieve state-of-the-art performance.

## Highlights & Insights

- **Degradation as state-space evolution**: Modeling weather degradation as hidden-state recurrence in an SSM is an elegant analogy with clear theoretical grounding.
- **Locality preservation of Morton scanning**: Compared to raster, sequential, or local scanning, the Z-order curve better preserves 2D spatial proximity after 1D unfolding.
- **Complementarity of dual degradation priors**: The global prior encodes "what type of weather," while the local prior encodes "how degradation varies at this location."

## Limitations & Future Work

- The two-stage training increases overall training cost.
- Morton encoding requires image dimensions to be powers of two, necessitating padding in practice.
- MODEM slightly underperforms Histoformer on RainDrop (−0.05 dB), indicating room for improvement in purely localized degradation scenarios.
- No comprehensive comparison with recent diffusion-based restoration methods is provided.

## Related Work & Insights

- MODEM is complementary to the histogram statistical priors in Histoformer: the former explicitly estimates degradation, while the latter leverages global statistical information.
- SSM/Mamba-based image restoration methods (MambaIR, FourierMamba) provide a foundation; MODEM's contribution lies in degradation-guided modulation atop this basis.
- Morton encoding originates from computational geometry, and its application in image processing remains relatively novel.

## Rating

- Novelty: ⭐⭐⭐⭐ The degradation-as-state-space perspective is novel; the Morton scanning combined with dual degradation modulation constitutes a coherent design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset, multi-metric evaluation (PSNR/SSIM/LPIPS/Q-Align/MUSIQ), task-specific comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The connection between degradation and SSMs is explained intuitively, with rich illustrations.
- Value: ⭐⭐⭐⭐ Achieves state-of-the-art unified multi-weather restoration with direct applicability to real-world vision systems operating under adverse conditions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[ICCV 2025\] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)](../../ICCV2025/image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[NeurIPS 2025\] MAP Estimation with Denoisers: Convergence Rates and Guarantees](map_estimation_with_denoisers_convergence_rates_and_guarantees.md)
- [\[AAAI 2026\] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](../../AAAI2026/image_restoration/clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)

</div>

<!-- RELATED:END -->
