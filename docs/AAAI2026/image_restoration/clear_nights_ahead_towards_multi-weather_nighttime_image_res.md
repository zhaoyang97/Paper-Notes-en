---
title: >-
  [Paper Note] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration
description: >-
  [AAAI 2026][Image Restoration][nighttime image restoration] This paper is the first to define and explore the multi-weather nighttime image restoration task. It constructs the AllWeatherNight dataset (8K training + 1K sy…
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "nighttime image restoration"
  - "multi-weather"
  - "Retinex prior"
  - "dynamic MoE"
  - "AllWeatherNight dataset"
date: 2026-05-08
content_hash: 3d699e166ca39613
---

# Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration

**Conference**: AAAI 2026
**arXiv**: [2505.16479](https://arxiv.org/abs/2505.16479)  
**Code**: [https://henlyta.github.io/ClearNight/](https://henlyta.github.io/ClearNight/)  
**Area**: Image Restoration / Nighttime Adverse Weather Removal
**Keywords**: nighttime image restoration, multi-weather, Retinex prior, dynamic MoE, AllWeatherNight dataset

## TL;DR

This paper is the first to define and explore the multi-weather nighttime image restoration task. It constructs the AllWeatherNight dataset (8K training + 1K synthetic test + 1K real-world test) and proposes the ClearNight unified framework, which simultaneously removes compound degradations—haze, rain streaks, raindrops, snow, and flare—in a single stage via Retinex dual-prior guidance and weather-aware dynamic specificity–commonality collaboration. With only 2.84M parameters, ClearNight comprehensively surpasses state-of-the-art methods.

## Background & Motivation

In nighttime scenes, adverse weather degradations are tightly coupled with non-uniform illumination (flare effects, halos), severely impacting downstream tasks such as autonomous driving and video surveillance. Three major gaps exist in prior research:

| Gap | Specific Issue | Representative Works |
|-----|---------------|----------------------|
| Dataset absence | No dataset covering multi-weather + nighttime + flare simultaneously | UNREAL-NH (haze only), GTAV-NightRain (rain only) |
| Method limitation | Nighttime methods handle only a single degradation type | TKL (dehazing), FSDGN (deraining) |
| Neglected degradation coupling | Illumination and weather degradations are intertwined (e.g., rain streaks appear brighter under lights, haze is denser in dark regions) | Daytime methods WeatherDiff, AWRaCLe cannot handle this |

ClearNight must address two core challenges: (1) the lack of realistic multi-weather nighttime training samples; and (2) the inability of existing models to effectively handle coupled degradations. A cascade approach (dehazing followed by deraining) yields only 27.8 dB PSNR and is slow.

## Method

### Overall Architecture

ClearNight adopts a DehazeFormer-based encoder–decoder architecture integrating two core modules: **Retinex dual-prior guidance** (decoupling illumination from texture) and **weather-aware dynamic specificity–commonality collaboration** (WDS + TFB dual-branch processing of multi-weather degradations). The entire model contains only 2.84M parameters.

### Key Designs

1. **AllWeatherNight Dataset Construction**

    - 2,000 high-quality nighttime images are selected from BDD100K and ExDark as ground truth (triple filtering by brightness/gradient/variance + manual secondary screening).
    - **Illumination-aware degradation synthesis**: Flare is first simulated by convolving with the Atmospheric Point Spread Function (APSF) ($X^\text{flare} = \alpha X + \beta(L * K^\text{APSF})$, where $\beta$ is adaptively set according to the light source pixel ratio), followed by illumination-aware weather degradation synthesis ($X^d = X^\text{flare} + \sum_{e \in \mathcal{E}} \omega_e \cdot \mathcal{F}_e^G(X^\text{flare})$).
    - Key innovation: The weight map $\omega_e$ uses the illumination map from Retinex decomposition, naturally coupling weather degradation intensity with illumination (except raindrops, where $\omega_\text{RD}=1$, as they are governed by local background).
    - t-SNE validation: The illumination-aware synthetic data distribution is substantially closer to real-world data than conventional uniform synthesis.

2. **Retinex Dual-Prior Guidance**

    - The degraded image is decomposed via Retinex into a reflectance component $R^d$ and an illumination component $I^d$ ($X^d = R^d \cdot I^d$).
    - **Illumination prior** $i_n$: Injected into the first three Transformer Feature Blocks (TFBs) to guide the network toward non-uniform illumination regions.
    - **Reflectance prior** $r_n$: Injected into WDS blocks to enhance texture representation, aiding in distinguishing weather degradation types and recovering background details.
    - A shared-weight multi-scale prior extraction unit (MPE) extracts three-scale priors via dilated convolutions.

3. **Weather-Aware Dynamic Specificity–Commonality Collaboration**

    - **Commonality branch**: Sequential Transformer Feature Blocks (TFBs) capturing shared features across all weather degradations.
    - **Specificity branch (WDS)**: Dynamically selects the top-10 units from 25 candidates to adaptively construct a sub-network.
    - **Weather Guider**: Performs multi-label classification (BCE loss), learns weather-specific prototypes, and automatically associates different weather types with distinct candidate unit combinations.

### Loss & Training

$$L_\text{total} = L_1 + 0.1 \cdot L_\text{perceptual} + 0.001 \cdot L_\text{bce} + 0.01 \cdot L_\text{lb} + 0.02 \cdot L_\text{depth}$$

Adam optimizer, lr=2×10⁻⁴, cosine annealing, 100 epochs, patch size 256×256, 2.84M parameters, 0.32s inference time.

## Key Experimental Results

### Main Results: AllWeatherNight Synthetic Test Set (Scene-Level)

| Scene | Method | PSNR↑ | SSIM↑ |
|-------|--------|-------|-------|
| Rain Scene | TKL | 29.09 | 0.8769 |
| | AWRaCLe | 31.54 | 0.9210 |
| | **ClearNight** | **32.59** | **0.9223** |
| Snow Scene | RAMiT | 29.12 | 0.8889 |
| | AWRaCLe | 29.43 | 0.8738 |
| | **ClearNight** | **30.65** | **0.9041** |
| Haze | RAMiT | 36.44 | 0.9738 |
| | **ClearNight** | **36.47** | 0.9621 |
| Rain Streaks Only | DEA-Net | 32.76 | 0.9285 |
| | **ClearNight** | **33.62** | **0.9331** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Note |
|---------------|-------|-------|------|
| Baseline (DehazeFormer) | 28.80 | 0.8825 | No prior, no dynamic branch |
| + Illumination prior | 32.13 | 0.9148 | +3.33 dB, largest single contribution |
| + Reflectance prior | 32.39 | 0.9189 | Texture enhancement effective |
| + Dynamic routing | 32.49 | 0.9207 | WDS dynamic routing effective |
| + Weather Guider | **32.59** | **0.9223** | Full ClearNight |

### Key Findings

- The illumination prior contributes the most (+3.33 dB), validating the central importance of illumination decoupling in nighttime restoration.
- Real-world evaluation: rain streak scene NIQE 3.7653 (best), snow scene NIQE 3.2191 (best).
- Models trained on AllWeatherNight achieve significantly better NIQE on real-world data than models trained on combinations of existing nighttime datasets.
- ClearNight matches or surpasses dedicated single-weather methods, demonstrating the effectiveness of the unified framework.

## Highlights & Insights

- The first multi-weather nighttime image restoration framework and dataset, filling a critical research gap.
- The illumination-aware degradation synthesis strategy cleverly employs the Retinex illumination map as degradation weights, substantially improving synthesis realism.
- The WDS Weather Guider's dynamic unit assignment is interpretable, with different weather types automatically activating distinct candidate unit combinations.
- With only 2.84M parameters, ClearNight outperforms large models such as DEA-Net (15M+) and AWRaCLe across multi-weather nighttime scenarios.

## Limitations & Future Work

- Performance is limited under extreme dynamic illumination changes (e.g., rapidly flickering light sources).
- The dataset primarily covers driving/detection scenes; indoor nighttime scenes are underrepresented.
- Standalone flare removal performance is modest (PSNR 38.77 vs. RAMiT 43.01) due to insufficient dedicated flare training data.
- Only four predefined degradation types (haze/rain streaks/raindrops/snow) are supported; rare weather conditions such as dust storms are not covered.

## Related Work & Insights

| Direction | Representative Methods | Difference from Ours |
|-----------|----------------------|----------------------|
| Daytime multi-weather restoration | WeatherDiff, WGWS, AWRaCLe | Neglect nighttime illumination–weather degradation coupling |
| Nighttime single-weather restoration | TKL (dehazing), FSDGN (deraining) | Handle only single degradation; cannot address composite scenes |
| Nighttime datasets | UNREAL-NH, GTAV-NightRain, RVSD | Cover only a single weather type; no flare synthesis |
| Dynamic networks | MoE, dynamic filtering | ClearNight's WDS integrates weather classification for semantically guided dynamic routing |

## Rating

- Novelty: ⭐⭐⭐⭐ First to define the multi-weather nighttime restoration task; dataset and illumination-aware synthesis are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on synthetic + real-world data, ablation study, t-SNE analysis, and cascade comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure; motivation for illumination-aware degradation synthesis is well articulated.
- Value: ⭐⭐⭐⭐ Fills an important gap; both the dataset and method provide lasting contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)
- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](../../ICCV2025/image_restoration/mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [\[NeurIPS 2025\] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](../../NeurIPS2025/image_restoration/rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)

</div>

<!-- RELATED:END -->
