---
title: >-
  [Paper Note] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer
description: >-
  [AAAI 2026][weather data synthesis] This work introduces SynWeather, the first unified multi-region multi-variable weather observation synthesis dataset (covering 4 regions × 4 variables × 6 satellites)…
tags:
  - "AAAI 2026"
  - "weather data synthesis"
  - "diffusion Transformer"
  - "multi-region multi-variable"
  - "radar reflectivity"
  - "precipitation estimation"
date: 2026-05-08
content_hash: 1af2fc927a6eefcd
---

# SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer

**Conference**: AAAI 2026
**arXiv**: [2511.08291](https://arxiv.org/abs/2511.08291)  
**Code**: [https://github.com/Dtdtxuky/SynWeather](https://github.com/Dtdtxuky/SynWeather)  
**Area**: Meteorological Data Synthesis / Diffusion Models
**Keywords**: weather data synthesis, diffusion Transformer, multi-region multi-variable, radar reflectivity, precipitation estimation

## TL;DR

This work introduces SynWeather, the first unified multi-region multi-variable weather observation synthesis dataset (covering 4 regions × 4 variables × 6 satellites), and proposes SynWeatherDiff, a general probabilistic generative model based on a Diffusion Transformer. By leveraging text prompts to distinguish region–variable task combinations, SynWeatherDiff outperforms both task-specific models and existing general-purpose models across multiple synthesis tasks.

## Background & Motivation

**Background**: With advances in geostationary satellites (GOES, Himawari, Meteosat) and radar systems, large volumes of meteorological observation data are available for weather forecasting, disaster monitoring, and climate research. However, due to inherent instrument limitations—sparse radar coverage in topographically complex or economically underdeveloped regions, unavailability of visible-light satellites at night, and low temporal resolution of polar-orbiting satellites—raw data exhibit temporal and spatial gaps. Data synthesis techniques are therefore required to fill in missing information.

**Limitations of Prior Work**: (1) Existing datasets are limited to a single region or variable: HKO-7 covers only Hong Kong radar, SEVIR covers only the contiguous United States, and DigitalTyphoon uses only a single Himawari channel. (2) Existing methods employ task-specific networks customized for each variable synthesis task (e.g., SRViT for VIL synthesis, Deep-STEP for precipitation estimation), precluding unified modeling across variables and regions. (3) Deterministic models (e.g., UNets trained with MSE) produce overly smoothed outputs and fail to capture fine-grained structures in high-intensity regions such as convective precipitation cores.

**Key Challenge**: Meteorological variables across different regions share physical relationships (e.g., the Z–R relationship between precipitation and radar reflectivity, the spatial correspondence between visible-light and microwave brightness temperature), yet existing methods model each independently, ignoring complementary cross-variable and cross-region information. Furthermore, deterministic modeling is fundamentally at odds with the stochastic nature of meteorological events.

**Goal**: How can general probabilistic synthesis of weather observation data be performed across multiple regions and variables? Specific sub-problems include: (1) How to construct a unified multi-region multi-variable dataset? (2) How to design a single model capable of handling synthesis from different satellite sources and different variables? (3) How to generate probabilistic outputs with fine-grained high-value regions?

**Key Insight**: Drawing inspiration from general-purpose models in natural image generation, the paper employs text prompts to differentiate region–variable task combinations, and uses a Diffusion Transformer for probabilistic generation to address the over-smoothing problem of deterministic models.

**Core Idea**: Build a unified dataset combined with a text-prompt-driven Diffusion Transformer to achieve general-purpose modeling of multi-region multi-variable weather synthesis within a probabilistic framework.

## Method

### Overall Architecture

SynWeatherDiff consists of three components: (1) a **General AutoEncoder** that compresses different meteorological variables into a shared latent space; (2) a **ViT encoder** that extracts features from satellite infrared observations; and (3) a **text-guided Diffusion Transformer** that performs conditioned denoising generation in the latent space using text prompts and satellite features. The input is 10-channel infrared satellite observations (C07–C16), and the output is the target meteorological variable (radar reflectivity / precipitation / visible light / microwave brightness temperature). The text prompt format is: *"Synthesize the [variable] variable over the [region] region using corresponding satellite imagery."*

### Key Designs

1. **SynWeather Unified Dataset**:

    - **Function**: Provides the first standardized multi-region multi-variable benchmark for meteorological data synthesis.
    - **Mechanism**: Covers 4 regions (contiguous United States CONUS, Europe, East Asia, and tropical cyclone regions) and 4 meteorological variables (composite radar reflectivity CR, precipitation, visible light, and microwave brightness temperature MWBT), integrating all infrared channels (10 channels) from 6 geostationary satellites (GOES-16/17/18, Meteosat-11, Himawari-8/9) as input. Data are unified to 1-hour temporal resolution and 4 km spatial resolution, cropped into 256×256 patches (128-step sliding window), with invalid patches filtered by connected-component area thresholds. Precipitation undergoes log transformation (long-tail distribution); all other variables are min-max normalized.
    - **Design Motivation**: All 10 infrared channels are used rather than the conventional 3-channel approach, as different spectral ranges (SWIR, WV, LWIR, GAS) provide complementary contributions to different variable synthesis tasks—ablation experiments confirm that removing any channel group degrades performance.

2. **General AutoEncoder**:

    - **Function**: Compresses different meteorological variables into a unified low-dimensional latent space.
    - **Mechanism**: The target variable $Y_{r,b} \in \mathbb{R}^{1 \times H \times W}$ is encoded into a latent representation $z_{r,b} \in \mathbb{R}^{C_z \times H_z \times W_z}$. Training uses pixel reconstruction loss + KL divergence + adversarial loss. All meteorological variables share a single autoencoder.
    - **Design Motivation**: Meteorological images contain substantial redundancy (simultaneous occurrence of precipitation or typhoons over hundreds of kilometers is rare), and the compressed latent space removes redundancy while preserving physical information. Different variables share physical similarities (e.g., the Z–R relationship between precipitation and CR), and a shared latent space facilitates cross-variable knowledge transfer.

3. **Text-Guided Diffusion Transformer (Text-Guided DiT)**:

    - **Function**: Performs probabilistic conditioned denoising generation in the latent space, conditioned on satellite inputs and text prompts.
    - **Mechanism**: Satellite observations $X_r$ are encoded via a ViT encoder; text prompts $P_{r,b}$ are encoded via the CLIP text encoder (with only the last Transformer block fine-tuned). An **early fusion** strategy is adopted: the noisy latent $z^t_{r,b}$ is concatenated with satellite encoder features and patchified, then concatenated with text tokens, and conditioned denoising is performed via the self-attention layers of the DiT. The training objective is the standard noise prediction loss $\mathcal{L} = \mathbb{E}_{z,\epsilon,t}[\|\epsilon_\theta(z^t_{r,b}, t, X_{r,b}, P_{r,b}) - \epsilon\|^2_2]$.
    - **Design Motivation**: Compared to the cross-attention fusion in SD3, the early fusion strategy allows satellite features to participate more directly in the denoising process. Text prompts provide a flexible task control interface—the model can be extended to new region–variable combinations without modifying its architecture. The probabilistic framework of the DiT inherently avoids the over-smoothing problem of deterministic models.

### Loss & Training

The autoencoder is trained with pixel reconstruction loss + KL divergence + adversarial loss. The DiT is trained with standard noise prediction MSE loss, using the AdamW optimizer with cosine learning rate decay (5e-4 → 1e-5) for 600K steps, batch size 16, on 4× A100 GPUs. The six standard tasks are sampled uniformly (1/6 each) for training the general model; ablation experiments explore different task sampling ratios.

## Key Experimental Results

### Main Results

| Task | Metric | SynWeatherDiff (General) | WeatherGFM (General) | UNet (Specific) | ViT (Specific) |
|------|--------|--------------------------|----------------------|-----------------|----------------|
| CONUS CR | RMSE↓ | **2.820** | 3.124 | 3.395 | 3.487 |
| CONUS CR | CSI/25↑ | **0.382** | 0.366 | 0.299 | 0.309 |
| CONUS Precipitation | CSI/2↑ | **0.312** | 0.288 | 0.231 | 0.250 |
| CONUS Precipitation | CSI/15↑ | **0.113** | 0.090 | 0.059 | 0.038 |
| Europe Precipitation | CSI/5↑ | **0.079** | 0.013 | 0.016 | 0.044 |
| East Asia Visible | SSIM↑ | 0.868 | 0.822 | **0.917** | 0.870 |
| MWBT | LPIPS↓ | **0.254** | 0.325 | 0.329 | 0.324 |

### Ablation Study

| Sampling Strategy | CONUS Precip. CSI/2 | CONUS CR CSI/25 | Europe Visible SSIM | MWBT SSIM |
|-------------------|---------------------|-----------------|---------------------|-----------|
| Uniform (1/6) | 0.312 | 0.382 | 0.864 | 0.837 |
| CR-dominant (1/2) | 0.320 | **0.403** | 0.857 | 0.843 |
| Precip.-dominant (1/2) | 0.292 | 0.343 | 0.855 | 0.841 |
| Visible-dominant (1/2) | 0.298 | 0.377 | **0.877** | 0.842 |
| MWBT-dominant (1/2) | 0.295 | 0.374 | 0.879 | 0.842 |

### Key Findings

- **General model outperforms task-specific models**: SynWeatherDiff comprehensively surpasses all task-specific models on precipitation synthesis (the most challenging task), achieving CSI/15 of 0.113 vs. 0.059 for UNet—a 91% improvement.
- **Visible-light synthesis is an exception**: UNet operates directly in pixel space; since visible-light images contain abundant high-frequency detail, encoding through the autoencoder results in information loss. This points toward a direction for improving the autoencoder.
- **Complementarity and conflict among tasks**: A higher CR sampling ratio benefits precipitation (Z–R physical relationship) and MWBT (similar physical domain); visible light and MWBT mutually benefit each other (spatial correspondence in strong convective systems); however, CR and visible light exhibit conflicting sampling preferences.
- **OOD generalization capability**: On the never-seen "East Asia precipitation" task, the general model (text prompt: "East Asia + precipitation") outperforms a task-specific model trained solely on CONUS precipitation, demonstrating the effectiveness of cross-region knowledge transfer.
- **Input channel ablation**: Using all 10 channels outperforms using only 3. Water vapor and longwave infrared channels are critical for precipitation/CR synthesis, while shortwave infrared is more important for visible-light synthesis. Removing any channel group leads to performance degradation.
- **Clear advantage in fine-grained generation**: Visualizations show that SynWeatherDiff recovers the number and spatial distribution of scattered small weather cells, whereas UNet/ViT/WeatherGFM frequently merge multiple small cells into a single large blob or lose intensity centers.

## Highlights & Insights

- **The dataset contribution may prove more enduring than the model**: SynWeather is the first unified meteorological synthesis dataset spanning 4 regions × 4 variables × 6 satellite sources, defining 7 standard tasks (including 1 OOD task) and a complete evaluation protocol. This infrastructure-level contribution is of substantial value for advancing the field.
- **Text prompts enable flexible task control**: Unlike training a separate model per task, text prompts allow the same model to switch tasks by simply changing the input text. This "one model, multiple tasks" paradigm represents a forward-looking approach to meteorological AI.
- **Advantage of probabilistic models in high-value regions**: Deterministic models tend to regress toward the mean, smoothing out the intensity cores of extreme weather events. SynWeatherDiff's probabilistic generation recovers high-threshold CSI performance (e.g., CSI/15), which is directly relevant to hazardous weather warning applications.

## Limitations & Future Work

- Visible-light synthesis underperforms UNet; the autoencoder presents a bottleneck in reconstructing high-frequency detail, motivating improvements to the autoencoder or the introduction of skip connections.
- Inference is constrained to region–variable combinations seen during training—while the text-prompt framework is flexible, it cannot generalize to entirely unseen region–variable pairs.
- Data imbalance: visible-light samples are most abundant (~500K), whereas precipitation and MWBT samples are relatively scarce (~10–20K), potentially biasing training.
- Only 4 km spatial resolution is explored; synthesis at higher resolutions (e.g., 1 km) has not been investigated.
- Integration with numerical weather prediction systems (e.g., data assimilation) has not been validated.

## Related Work & Insights

- **vs. WeatherGFM (zhao2024weathergfm)**: WeatherGFM is also a general-purpose meteorological model but relies on deterministic prediction. SynWeatherDiff's probabilistic framework outperforms WeatherGFM on most tasks, with particularly significant CSI gains in high-value precipitation regions.
- **vs. SRViT (stock2024srvit)**: SRViT is a task-specific model for VIL/CR synthesis. As a general-purpose model, SynWeatherDiff substantially surpasses it on CONUS CR with an RMSE of 2.820 vs. 3.561.
- **vs. Deep-STEP & TomoPE**: Both are task-specific precipitation estimation models. SynWeatherDiff's CSI/15 (0.113) far exceeds Deep-STEP (0.007) and TomoPE (0.036), confirming the advantage of probabilistic models in extreme-value regions.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of the first multi-region multi-variable meteorological synthesis dataset with a general Diffusion Transformer model is novel; text-prompt-driven task control is a creative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six standard tasks + 1 OOD task, 7 baseline comparisons, and complete ablations over sampling ratios and input channels.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction is described in detail, experimental analysis is thorough, and the paper is well-structured.
- Value: ⭐⭐⭐⭐⭐ The dataset-plus-baseline combination provides an important contribution to the meteorological AI community; probabilistic generation has direct application value for extreme weather warning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer](../../ICCV2025/others/layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)
- [\[NeurIPS 2025\] UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](../../NeurIPS2025/others/uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)
- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[AAAI 2026\] The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)
- [\[AAAI 2026\] Regular Games – an Automata-Based General Game Playing Language](regular_games_--_an_automata-based_general_game_playing_language.md)

</div>

<!-- RELATED:END -->
