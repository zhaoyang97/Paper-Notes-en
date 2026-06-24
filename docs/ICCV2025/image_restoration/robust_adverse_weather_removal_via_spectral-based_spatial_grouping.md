---
title: >-
  [Paper Note] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)
description: >-
  [ICCV 2025][Image Restoration][All-in-One weather removal] SSGformer proposes an All-in-One adverse weather image restoration method based on spectral decomposition and grouping attention: it extracts high-frequency edge information via the Sobel operator and analyzes low-frequency degradation textures via SVD, fuses both to generate spatial grouping masks, and performs channel and spatial attention within groups to achieve robust removal of multiple weather degradations (rai…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "All-in-One weather removal"
  - "spectral decomposition"
  - "spatial grouping"
  - "Transformer"
  - "Sobel operator"
  - "SVD"
  - "attention mechanism"
date: 2026-05-08
content_hash: 17a0272b5d187827
---

# Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)

**Conference**: ICCV 2025
**arXiv**: [2507.22498](https://arxiv.org/abs/2507.22498)  
**Code**: [https://github.com/jeongyh98/SSGformer](https://github.com/jeongyh98/SSGformer)  
**Area**: Image Restoration / Adverse Weather Removal
**Keywords**: All-in-One weather removal, spectral decomposition, spatial grouping, Transformer, Sobel operator, SVD, attention mechanism

## TL;DR

SSGformer proposes an All-in-One adverse weather image restoration method based on spectral decomposition and grouping attention: it extracts high-frequency edge information via the Sobel operator and analyzes low-frequency degradation textures via SVD, fuses both to generate spatial grouping masks, and performs channel and spatial attention within groups to achieve robust removal of multiple weather degradations (rain, snow, haze, raindrops).

## Background & Motivation

**Background**: Adverse weather conditions (rain, snow, haze, raindrops) severely degrade image quality and affect downstream vision tasks. Early works built dedicated models for individual weather types; in recent years, All-in-One (AiO) methods have emerged to handle multiple weather degradations within a unified framework. Representative methods include AIRFormer, Fourmer, AdaIR, and WeatherDiff.

**Limitations of Prior Work**:
   - **Global operation in frequency-domain methods**: Existing frequency-domain restoration methods (Fourmer via Fourier transform, AIRFormer via wavelet transform) apply global filtering over the entire image's frequency content. This is effective for repetitive degradations (e.g., blur, uniform noise), but adverse weather degradations are **highly non-uniform and localized** (e.g., raindrops appear only at local positions), making global filtering insufficiently precise.
   - **Resolution loss in wavelet transforms**: Wavelet transforms halve the image resolution; restoring the original resolution after frequency-domain processing requires upsampling, which may introduce artifacts.
   - **Lack of fine spatial-context-aware grouping**: Different weather degradations are spatially non-uniform, requiring the identification and grouping of regions with similar degradation characteristics for targeted restoration.

**Key Challenge**: AiO models face a fundamental tension between "diverse degradation patterns" and "a unified processing framework." The high randomness and locality of weather degradations demand that the model both extract effective spectral priors and organize and process features in a spatially aware manner.

**Key Insight**: Extract spectral information while preserving spatial details — using the Sobel operator (no resolution reduction) for high-frequency edge features and SVD (no resolution reduction) for low-frequency degradation texture features, then generating spatial grouping masks from these features to perform within-group attention.

## Method

### Overall Architecture

SSGformer is a 4-stage Transformer Encoder-Decoder network comprising three core modules:

1. **Spectral-based Decomposition Prompt (SDP)**: Spectral analysis and feature fusion
2. **Mask Generator (MG)**: Generates spatial grouping masks from spectral features
3. **Spatial Grouping Transformer Block (SGTB)**: Mask-guided grouping attention

Data flow: degraded image $I_D$ → SDP extracts spectral feature $F_S$ → MG generates grouping mask $M_p$ → SGTB performs within-group attention → outputs clean image $I_C$

### Key Designs

1. **Spectral-based Decomposition Prompt (SDP)**:

    - **Sobel operator**: Detects high-frequency information by emphasizing grayscale intensity changes to extract edge features $F_{Sobel} \in \mathbb{R}^{H \times W \times 1}$, preserving spatial resolution.
    - **SVD filter**: Applies singular value decomposition to the degraded image, truncates high-frequency components (retaining the top $k$ singular values), and obtains low-frequency degradation texture features $F_{SVD} \in \mathbb{R}^{H \times W \times 1}$, preserving spatial resolution.
    - **Spectral Feature Fusion Module**: Refines $F_{Sobel}$ and $F_{SVD}$ separately (Sobel refinement block via convolution + feature reorganization; SVD refinement block via deformable convolution combined with $I_D$), then models the mutual information between high- and low-frequency features through multi-head linear attention, outputting the fused feature $F_S$.

2. **Mask Generator (MG)**:

    - **Function**: Generates spatial grouping masks $M_p$ from the fused spectral feature $F_S$.
    - **Mechanism**: The mask clusters image regions by spatial similarity of degradation characteristics and texture features. Regions within the same group share similar degradation patterns, facilitating within-group information exchange for restoration.
    - **Design Motivation**: Weather degradations are spatially non-uniform — raindrops cover local areas, snowflakes are scattered, and haze is denser at greater distances. Spectral-feature-guided grouping adaptively identifies these degraded regions.

3. **Spatial Grouping Transformer Block (SGTB)**:

    - Two variants: **SGTB-C** (grouped channel attention) and **SGTB-S** (grouped spatial attention).
    - **Function**: Groups features using the grouping mask and performs attention within each group.
    - **Mechanism**: Channel attention captures feature relationships across channels within the same group; spatial attention captures relationships across spatial positions within the same group. The dual attention cooperatively balances feature-level and spatial-level dependencies.
    - **Design Motivation**: Performing attention within regions of similar degradation characteristics is both more efficient and more precise than global attention, avoiding inappropriate information mixing between clean and degraded regions.

### Loss & Training

- Standard image restoration losses (L1 loss + perceptual loss)
- Multi-stage training: Encoder progressively downsamples across 4 stages; Decoder symmetrically upsamples with refinement blocks
- SGTB-C and SGTB-S alternate within each stage, with the count controlled by $L_p$ per stage $p$

## Key Experimental Results

### Main Results

Evaluated on standard weather removal benchmarks (covering deraining, desnowing, dehazing, and raindrop removal):

- SSGformer achieves **state-of-the-art performance** across multiple All-in-One benchmarks
- Performs particularly well in complex multi-weather mixed degradation scenarios
- Demonstrates robust consistency across diverse degradation types

### Comparison with Existing AiO Methods

- Vs. Fourmer (global Fourier filtering): SSGformer shows greater advantages in locally degraded scenarios
- Vs. AIRFormer (wavelet prior): SSGformer preserves spatial resolution without upsampling artifacts
- Vs. AdaIR (frequency-domain degradation recognition): SSGformer's spatial grouping mechanism more effectively handles non-uniform degradations
- Requires no external knowledge (LLM/VLM), relying purely on image-internal information

### Ablation Study

- **Effectiveness of Sobel + SVD combination**: Using Sobel or SVD alone underperforms their combination. Sobel captures high-frequency edges (degradation structure); SVD captures low-frequency components (degradation texture patterns); the two are complementary.
- **Necessity of the grouping mask**: Removing the MG module (replacing it with direct global attention) significantly degrades performance, confirming that the grouping mechanism is the key driver of improvement.
- **Grouped vs. global attention**: Superior to global attention in both efficiency and effectiveness, as grouping reduces interference from irrelevant regions.
- **Role of multi-head linear attention in SDP**: Better models the interaction between high- and low-frequency features compared to simple concatenation.

### Key Findings

- Spectral priors do not require Fourier/wavelet transforms: conventional Sobel edge detection and SVD low-rank approximation effectively extract degradation-relevant spectral information without sacrificing spatial resolution.
- Spatial grouping is critical for AiO weather removal: clustering spatially similar degraded regions and performing within-group attention is more precise than global attention.
- The dual channel + spatial attention design of SGTB balances multi-scale dependencies effectively.

## Highlights & Insights

- **"Spatially resolution-preserving spectral analysis" philosophy**: Conventional frequency-domain methods (FFT, DWT) alter the spatial dimension, whereas SSGformer selects the Sobel operator and SVD to extract spectral information without changing spatial resolution — a simple yet effective design choice.
- **Degradation-aware nature of grouping attention**: The mask is not a fixed grid or regular partition but is adaptively generated based on degradation characteristics, meaning the model automatically identifies "which regions suffer from similar degradations and should be processed together."
- **Independence from external knowledge**: Against the trend of incorporating LLM/VLM external knowledge in recent methods, SSGformer achieves robust restoration based solely on image-internal information (spectral decomposition + spatial clustering), demonstrating that carefully designed internal prior exploitation is sufficient to reach state-of-the-art performance.
- **Novel use of SVD for degradation texture analysis**: SVD is conventionally used for low-rank approximation or dimensionality reduction; here it is applied to analyze texture patterns of degraded images (truncated low-frequency information = global degradation pattern), representing a creative application extension.

## Limitations & Future Work

- **Computational efficiency**: Although grouped attention is more efficient than global attention, SVD decomposition itself carries non-trivial computational overhead, potentially limiting applicability to high-resolution images or real-time scenarios.
- **Grouping granularity**: The granularity of masks generated by MG may affect performance; both overly fine and overly coarse groupings can be problematic, yet the paper provides limited discussion on adaptive granularity control.
- **SVD truncation parameter $k$**: The number of retained singular values is a hyperparameter that may require tuning for different weather types.
- **Mixed weather degradations**: Although the method claims to handle multiple weather types, no dedicated evaluation is provided for images simultaneously suffering from multiple degradations (e.g., rain + haze).
- **Fixed Sobel kernel**: Using a fixed Sobel kernel for edge detection may be less flexible than a learnable edge detector.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](../../ECCV2024/image_restoration/learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)
- [\[ECCV 2024\] Restoring Images in Adverse Weather Conditions via Histogram Transformer](../../ECCV2024/image_restoration/restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->
