---
title: >-
  [Paper Note] CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration
description: >-
  [CVPR 2025][Autonomous Driving][Infrared-visible image fusion] CAWM-Mamba proposes the first end-to-end unified framework that simultaneously addresses infrared-visible image fusion and compound adverse weather restoration (e.g., fog + rain, rain + snow). By featuring weather-aware preprocessing, cross-modal feature interaction, and wavelet-domain frequency-SSM decoupling multi-frequency degradations, it comprehensively outperforms SOTA models on the AWMM-100K and standard fu…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Infrared-visible image fusion"
  - "compound adverse weather"
  - "wavelet-domain state space model"
  - "cross-modal interaction"
  - "Mamba"
date: 2026-05-08
content_hash: a4a94f65180b6bd3
---

# CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration

**Conference**: CVPR 2025  
**arXiv**: [2603.02560](https://arxiv.org/abs/2603.02560)  
**Code**: [GitHub](https://github.com/Feecuin/CAWM-Mamba)  
**Area**: Image Fusion / Adverse Weather Restoration  
**Keywords**: Infrared-visible image fusion, compound adverse weather, wavelet-domain state space model, cross-modal interaction, Mamba

## TL;DR
CAWM-Mamba proposes the first end-to-end unified framework that simultaneously addresses infrared-visible image fusion and compound adverse weather restoration (e.g., fog + rain, rain + snow). By featuring weather-aware preprocessing, cross-modal feature interaction, and wavelet-domain frequency-SSM decoupling multi-frequency degradations, it comprehensively outperforms SOTA models on the AWMM-100K and standard fusion datasets.

## Background & Motivation
**Background**: Infrared-visible image fusion (IVIF) generates highly informative fused images by integrating complementary information from two modalities (thermal radiation from infrared + texture details from visible light). It is widely applied in surveillance, autonomous driving, etc.

**Limitations of Prior Work**: Existing adverse weather fusion methods can only handle single degradations (only fog, only rain, or only snow), failing when encountering compound degradations common in the real world (e.g., fog + rain, rain + snow). The superposition of various degradation mechanisms makes the patterns highly complex.

**Key Challenge**: Different weather degradations affect different frequency bands—fog mainly affects low frequencies (causing global brightness/contrast degradation), while rain/snow primarily impact high frequencies (local texture interference). Moreover, high-frequency degradations are directional (vertical for rain, oblique for wind-driven snow), requiring joint spatial-frequency modeling.

**Goal**: How to simultaneously handle the fusion and restoration under multiple compound degradations using a single end-to-end model with unified weights?

**Key Insight**: Utilize wavelet transform to decouple features into low-frequency (global structure) and high-frequency (directional details), which are processed by SSMs with different scanning strategies respectively; extract global weather embeddings as priors for guidance.

**Core Idea**: Weather-aware preprocessing + cross-modal feature interaction + wavelet-domain frequency-SSM degradation decoupling + unified degradation representation.

## Method

### Overall Architecture
Three core modules are cascaded: (1) WAPM: enhances degraded visible features and extracts weather embeddings; (2) CFIM: achieves cross-modal alignment and complementary information exchange; (3) WSSB: performs wavelet-domain feature decoupling, utilizing Freq-SSM to handle anisotropic high-frequency degradation and CDSM to learn a unified degradation representation.

### Key Designs

1. **Weather-Aware Preprocess Module (WAPM)**

    - **Function**: Preprocesses degraded visible images and extracts global weather embeddings $F_\text{embedding} \in \mathbb{R}^{B \times 48}$.
    - **Mechanism**: CNN feature extraction $\rightarrow$ channel attention to enhance degradation-related features $\rightarrow$ adaptive average pooling + MLP to extract global weather embeddings.
    - **Design Motivation**: Weather embeddings provide explicit degradation type priors for subsequent modules, which is much more efficient than having the network learn them implicitly.

2. **Cross-modal Feature Interaction Module (CFIM)**

    - **Function**: Fuses complementary features of infrared and visible modalities and achieves cross-modal alignment.
    - **Mechanism**: Divides feature channels equally into two parts $\rightarrow$ uses MaxPooling for infrared features (extracting salient thermal targets) and AvgPooling for visible features (maintaining texture smoothness) $\rightarrow$ applies cross-attention weights to exchange information $\rightarrow$ global pooling + channel attention to assist cross-modal alignment.
    - **Design Motivation**: The thermal target saliency from infrared enhances the texture response in visible light, while spatial details in visible light optimize infrared target boundaries, achieving mutual benefit.

3. **Wavelet Space State Block (WSSB)**

    - **Function**: Decouples multi-frequency degradations using wavelet space decomposition and processes them with direction-aligned SSMs.
    - **Mechanism**: Haar DWT decomposes features into $\{I_{LL}, I_{LH}, I_{HL}, I_{HH}\}$ $\rightarrow$ low-frequency components are processed by standard 2D-SSM (global structure) $\rightarrow$ high-frequency components are processed via Freq-SSM with direction-aligned scanning + channel attention $\rightarrow$ IDWT reconstruction.
    - **Design Motivation**: Wavelet transform simultaneously provides both spatial-temporal localization (unlike Fourier transform, which lacks spatial localization) and directional decomposition (unlike Laplacian pyramids, which lack directionality), naturally fitting anisotropic weather degradations.

4. **Freq-SSM (Frequency-Selective State Space Model)**

    - **Function**: Performs bidirectional scanning along the primary directions of high-frequency subbands to accurately capture directional degradations.
    - **Mechanism**: LH subbands undergo vertical scanning only, HL subbands undergo horizontal scanning only, and HH subbands undergo diagonal scanning—eliminating redundant computations in orthogonal directions.
    - **Design Motivation**: Standard 2D-SSM scans all directions indiscriminately, which leads to redundancy and feature loss for directional weather degradations (such as vertical rain streaks).

5. **Common Degradation Space Mechanism (CDSM)**

    - **Function**: Learns a unified degradation representation to improve generalization across various degradation types.
    - **Mechanism**: Maps degradation features to a shared latent space using residual convolutional blocks + channel attention.
    - **Design Motivation**: Reduces dependency on specific degradation types, enhancing robustness in compound scenarios.

### Loss & Training
A combination of fusion loss and perceptual loss is used for end-to-end training. Specific loss function details are not elaborated in the abstract.

## Key Experimental Results

### Main Results (AWMM-100K, Compound Weather)

| Scenario | CAWM-Mamba $Q_{MI}$ | Second-Best Method | CAWM-Mamba SSIM | Second-Best SSIM |
|------|---------------------|---------|-----------------|---------|
| Fog+Rain | **0.3604** | AWFusion 0.2749 | **0.3805** | AWFusion 0.3371 |
| Rain+Snow | **Best** | — | **Best** | — |

CAWM-Mamba ranks first on 9 out of 10 metrics in the fog+rain scenario, with an Avg.Rank of 1.00 (compared to 3.00 for the second-best, AWFusion).

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full CAWM-Mamba | Best | Complete model |
| w/o WAPM | — | Lacks weather priors, leading to decreased degradation awareness |
| w/o CFIM | — | Insufficient cross-modal information exchange |
| w/o WSSB | — | Loss of frequency-domain degradation decoupling capability |
| Regular SSM instead of Freq-SSM | — | Direction-alignment advantage disappears |

### Key Findings
- Significant advantages in compound degradation scenarios: On the $Q_{CV}$ metric, it drops from 1200+ of the second-best method to 194.34 (a massive difference), demonstrating that wavelet-domain decoupling effectively suppresses cross-interference of compound degradations.
- Achieves SOTA performance in single degradation scenarios as well, proving that the model is not overfitted to compound degradations.
- The fusion results perform best in downstream tasks (semantic segmentation, object detection), confirming its practical application value.

## Highlights & Insights
- **First fusion framework for compound adverse weather**: Solves compound degradations like fog+rain and rain+snow with unified weights, filling a gap in the field.
- **Direction-aligned design of Freq-SSM**: Matches SSM scanning directions to the intrinsic directions of wavelet subbands, eliminating redundancy and enhancing directional degradation modeling—this design logic can be applied to other low-level vision tasks requiring directional awareness.
- **Weather embedding as an explicit prior**: Global weather embeddings provide degradation type information for downstream modules, which is more efficient than forcing the network to infer the degradation type implicitly.
- **Wavelet vs Fourier vs Laplacian Choice**: The paper clearly justifies the advantages of wavelets in terms of simultaneous space-frequency localization and directional decomposition.

## Limitations & Future Work
- Only Haar wavelets (the simplest wavelet basis) are utilized; more complex wavelets might yield better frequency resolution.
- Performance under extreme compound degradations (e.g., three overlapping factors like fog+rain+snow) remains untested.
- The weather embedding is a fixed 48-dimensional vector, which may lack representational capacity for more complex degradation combinations.
- Lacks real-time analysis—the inference speed of the SSM + wavelet transform is critical for autonomous driving deployment.

## Related Work & Insights
- **vs AWFusion**: AWFusion models the degradation process using the atmospheric scattering model but only handles single degradations; CAWM-Mamba does not rely on physical degradation models, making it more versatile.
- **vs Text-IF**: Text-IF leverages semantic text to guide degradation-aware fusion, but is also confined to single degradations.
- **vs CDDFuse**: CDDFuse performs cross-modal decomposition fusion under clean conditions, without considering weather degradations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First compound adverse weather fusion framework + Wavelet-SSM coupling design with direction-aligned Freq-SSM.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively validated across compound + single weather + downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Complete technical details, though the contribution list has repeated numbering as "1.".
- Value: ⭐⭐⭐⭐⭐ Addresses the highly demanded multi-modal fusion under compound weather scenarios for real-world autonomous driving/surveillance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)
- [\[CVPR 2026\] Hybrid Robust Collaborative Perception with LiDAR-4D Radar Fusion under Adverse Weather Conditions](../../CVPR2026/autonomous_driving/hybrid_robust_collaborative_perception_with_lidar-4d_radar_fusion_under_adverse_.md)
- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)

</div>

<!-- RELATED:END -->
