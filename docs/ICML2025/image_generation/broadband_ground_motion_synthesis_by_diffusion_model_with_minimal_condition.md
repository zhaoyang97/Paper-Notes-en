---
title: >-
  [Paper Note] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition
description: >-
  [ICML2025][Image Generation][diffusion model] Proposed HEGGS (High-fidelity Earthquake Groundmotion Generation System), which leverages the naturally paired characteristics of waveforms in seismic datasets, combined with a conditional latent diffusion model and an Amplitude Correction Module (ACM), to generate high-fidelity three-component seismic waveforms end-to-end with minimal conditional information (latitude, longitude, focal depth, magnitude).
tags:
  - "ICML2025"
  - "Image Generation"
  - "diffusion model"
  - "seismic waveform"
  - "ground motion synthesis"
  - "latent diffusion"
  - "paired data"
date: 2026-05-08
content_hash: fd0986c4690abc45
---

# Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition

**Conference**: ICML2025  
**arXiv**: [2412.17333](https://arxiv.org/abs/2412.17333)  
**Code**: Not open-sourced  
**Area**: Diffusion Models / Seismic Waveform Synthesis  
**Keywords**: diffusion model, seismic waveform, ground motion synthesis, latent diffusion, paired data

## TL;DR

Proposed HEGGS (High-fidelity Earthquake Groundmotion Generation System), which leverages the naturally paired characteristics of waveforms in seismic datasets, combined with a conditional latent diffusion model and an Amplitude Correction Module (ACM), to generate high-fidelity three-component seismic waveforms end-to-end with minimal conditional information (latitude, longitude, focal depth, magnitude).

## Background & Motivation

- **Data Scarcity**: Major earthquake events are inherently rare, leading to insufficient training data. Existing methods (primarily GAN-based) generate waveforms that lack seismological realism (e.g., inaccurate P/S wave arrival times, amplitude distortion).
- **Expensive Conditional Information**: Previous methods typically require additional geological information such as focal mechanisms and VS30, which are costly to obtain and not always available.
- **Insufficient Generation Quality**: GAN-based methods such as SeismoGen, ConSeisGen, and BBGAN perform poorly on seismological metrics like phase arrival times and envelope correlation.
- **Core Motivation**: To exploit the naturally paired characteristics of "the same seismic event observed by multiple stations" in regional seismic datasets, allowing the diffusion model to implicitly learn source features from paired waveforms without explicitly requiring expensive geological conditions.

## Method

### Minimal Condition Set

Requires only 6 conditional parameters:
- Station latitude and longitude $(s_{lat}, s_{lon})$
- Epicenter latitude and longitude $(e_{lat}, e_{lon})$
- Focal depth $e_{dep}$
- Earthquake magnitude $M_L$

### Pair-Exploiting Diffusion Model

For the same seismic event, a pair of waveforms $(W^{src}, W^{tgt})$ is sampled from the dataset and converted into spectrograms $(X^{src}, X^{tgt})$. Assuming that sufficient seismic features can be extracted from $X^{src}$ and the target condition vector $\vec{c}_{tgt}$ to generate $X^{tgt}$, a latent transformation mapping $\eta$ is defined:

$$\eta(x_t^{src}, \vec{c}_{tgt}, t) \sim q(x_t^{tgt} | X^{tgt})$$

The diffusion loss in the sample space is:

$$\mathcal{L}_{DM}' = \mathbb{E}_{(X^{src}, X^{tgt}, \vec{c}_{tgt}), \epsilon, t} \| X^{tgt} - \mathbf{m}_\theta(x_t^{src}, \vec{c}_{tgt}, t) \|^2$$

where $\mathbf{m}_\theta(x, \vec{c}, t) = \mathbf{x}_\theta(\eta(x, \vec{c}, t), \vec{c}, t)$, which is a composition of the latent transformation and the denoising model.

### End-to-End Training

An autoencoder (encoder $\mathcal{E}_{AE}$ + decoder $\mathcal{D}_{AE}$) is introduced to perform diffusion in the latent space. However, end-to-end loss is used to directly supervise the waveform reconstruction:

$$\mathcal{L}_{ours} = \mathbb{E}_{(X^{src}, X^{tgt}, \vec{c}_{tgt}), \epsilon, t} \| X^{tgt} - \mathcal{D}_{AE}(\mathbf{m}_\theta(z_t^{src}, \vec{c}_{tgt}, t)) \|^2$$

where $z_t^{src} = \sqrt{\bar\alpha_t} \mathcal{E}_{AE}(X^{src}) + \sqrt{1 - \bar\alpha_t} \epsilon$.

### Amplitude Correction Module (ACM)

An ACM is appended after the decoder to perform amplitude correction, addressing the issue that the pretrained VAE cannot capture amplitude information, thereby enabling the model to directly generate unnormalized, real-amplitude waveforms.

### Inference Process

- **With Reference Waveforms**: Uses $W^{src}$ encoded and corrupted with noise as the starting point, performing reverse denoising combined with $\vec{c}_{tgt}$.
- **Without Reference Waveforms**: Initiates from pure Gaussian noise, generating solely based on the condition vector (applicable to virtual seismic simulation).

## Key Experimental Results

### Datasets

Three seismic databases from three continents: SCEDC (North America), KMA (East Asia), and INSTANCE (Europe).

### Quantitative Comparison (SCEDC Dataset, Partial Results)

| Model | Input | P_MAE(s)↓ | S_MAE(s)↓ | env.corr↑ | MSE↓ |
|------|------|-----------|-----------|-----------|------|
| SeismoGen | w/o $W^{src}$ | 1.956 | 3.625 | 0.490 | 1.412 |
| ConSeisGen | w/o $W^{src}$ | 3.972 | 6.899 | 0.325 | 0.746 |
| BBGAN | w/o $W^{src}$ | 6.421 | 10.42 | 0.195 | 1.615 |
| LDM | w/ $W^{src}$ | 0.563 | 0.781 | 0.773 | 0.243 |
| **HEGGS** | **w/o** $W^{src}$ | **0.503** | **0.800** | **0.796** | **0.153** |
| **HEGGS** | **w/** $W^{src}$ | **0.476** | **0.548** | **0.819** | **0.151** |

- HEGGS achieves state-of-the-art performance across all three datasets, reducing the P/S wave arrival time error by an order of magnitude compared to GAN-based methods.
- Even without reference waveforms (pure noise generation), HEGGS outperforms all baseline models that use reference waveforms.

### Ablation Study (SCEDC)

| Configuration | P_MAE(s) | S_MAE(s) | env.corr |
|------|----------|----------|----------|
| LDM (Normalized Waveform) | 1.114 | 1.729 | 0.693 |
| + Paired Data | 0.563 | 0.781 | 0.773 |
| + End-to-End Training | 0.801 | 1.537 | 0.624 |
| + ACM (=HEGGS) | **0.476** | **0.548** | **0.819** |

Paired training doubles the precision of phase arrival times; ACM enables the model to process real amplitudes and further improves the generation quality.

## Highlights & Insights

1. **Ingenious Exploitation of Data Characteristics**: Leveraging the naturally paired property where the same seismic event is simultaneously recorded by multiple stations allows the model to implicitly learn source features during training, eliminating the need for explicitly provided expensive conditions.
2. **End-to-End Training Bypassing Pretraining Bottlenecks**: Due to the lack of pretrained VAEs in the seismic domain, the end-to-end framework jointly optimizes the encoder, diffusion module, and decoder.
3. **Dual-Mode Inference**: Supports both conditional generation utilizing existing seismic waveforms (yielding higher precision) and generation starting from pure noise to synthesize virtual seismographs (offering greater application flexibility).
4. **Magnitude Control Capability**: Modifying $M_L$ in the conditioning vector generates waveforms of different magnitudes. Spectral analysis shows that the trends of corner frequency and seismic moment $M_0$ align well with theoretical expectations.
5. **Single-GPU Trainable**: Training can be completed on a single GPU, demonstrating high computational efficiency.

## Limitations & Future Work

- **Regional Limitations**: The model is trained independently on each regional dataset. A globally unified model has not yet been implemented, and cross-region generalization capability remains unverified.
- **Limited Conditions**: Although the minimal condition set lowers the barrier to practical use, it may limit the generation quality in extreme scenarios (such as extrapolation to super-large magnitudes).
- **Spectral Resolution**: The resolution of synthesized spectrograms is slightly lower than that of real data, lacking fine-grained detail in high-frequency feature reconstruction.
- **Lack of Downstream Verification**: Although claimed to be applicable to early warning systems and seismic design, its effectiveness on actual downstream tasks has not been demonstrated.
- **No Code Availability**: Reproducibility is limited as the source code is not open-sourced.

## Related Work & Insights

- **SeismoGen / ConSeisGen / BBGAN**: GAN-based seismic waveform generation methods, all of which are outperformed as baselines.
- **Latent Diffusion Model (LDM)**: Rombach et al., 2022. HEGGS introduces paired training and end-to-end learning on top of LDM.
- **Inspiration from Music Generation**: The methodology design is inspired by the conditional music generation method in Ghosal et al., 2023 (first generating spectrograms, then converting to waveforms).
- **EQTransformer**: Used to evaluate the P/S wave arrival quality of the generated waveforms.
- **Cross-Domain Insight**: The migration potential of diffusion models to "signal generation" tasks—showing that the same LDM framework can be successfully transferred from images to seismic waveforms.

## Rating

- Novelty: ⭐⭐⭐⭐ — The paired training strategy and end-to-end framework are novel in the seismological domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation covering datasets from three continents, multiple seismological metrics, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Structurally clear, mathematically rigorous, and provides adequate seismological background.
- Value: ⭐⭐⭐⭐ — High practical application potential for earthquake engineering and early warning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Model Immunization from a Condition Number Perspective](model_immunization_from_a_condition_number_perspective.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](../../ICCV2025/image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](../../ECCV2024/image_generation/smoodi_stylized_motion_diffusion_model.md)
- [\[CVPR 2026\] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](../../CVPR2026/image_generation/d2c_diffusion_dataset_condensation.md)

</div>

<!-- RELATED:END -->
