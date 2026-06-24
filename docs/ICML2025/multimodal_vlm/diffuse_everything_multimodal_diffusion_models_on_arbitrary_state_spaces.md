---
title: >-
  [Paper Note] Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces
description: >-
  [ICML 2025][Multimodal VLM][Multimodal Diffusion Models] This work proposes a unified framework for constructing multimodal diffusion models on arbitrary state spaces. By introducing independent decoupled noise schedules for each modality, it simultaneously achieves both unconditional and modality-conditional generation within a single model without requiring external tokenizers or VAE preprocessing.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Multimodal Diffusion Models"
  - "Mixed State Spaces"
  - "Decoupled Noise Schedule"
  - "Joint Text-Image Generation"
  - "Tabular Data Synthesis"
date: 2026-05-08
content_hash: 9757a0143e1591bc
---

# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces

**Conference**: ICML 2025  
**arXiv**: [2506.07903](https://arxiv.org/abs/2506.07903)  
**Code**: [https://github.com/kevinjrojas/DiffuseEverything](https://github.com/kevinjrojas/DiffuseEverything)  
**Area**: Diffusion Models / Multimodal Generation  
**Keywords**: Multimodal Diffusion Models, Mixed State Spaces, Decoupled Noise Schedule, Joint Text-Image Generation, Tabular Data Synthesis

## TL;DR
This work proposes a unified framework for constructing multimodal diffusion models on arbitrary state spaces. By introducing independent decoupled noise schedules for each modality, it simultaneously achieves both unconditional and modality-conditional generation within a single model without requiring external tokenizers or VAE preprocessing.

## Background & Motivation

**Background**: Diffusion models have achieved immense success in single-modality data (image, video, text) generation, where methods like DDPM, Score-based models, and Flow Matching have become mainstream.

**Limitations of Prior Work**: Existing multimodal diffusion approaches rely heavily on external preprocessing protocols. For instance, to jointly generate text and images, they typically require first discretizing text via a tokenizer and encoding images into a continuous latent space using a VAE, before performing diffusion in a unified format. This pipeline is highly sensitive to the precision of the encoder/decoder.

**Key Challenge**: Text is categorical (discrete) while images are Gaussian (continuous); their state spaces are inherently different. Traditional methods attempt to force all modalities into the same state space, which introduces additional information loss, and the quality of the encoder is difficult to guarantee when data is limited.

**Goal**: How to directly construct a joint diffusion model on the raw state spaces of different modalities without relying on external encoders/decoders?

**Key Insight**: Starting from the mathematical framework of diffusion processes, continuous diffusion (Gaussian) and discrete diffusion (categorical) are unified under a general Markov process framework. The diffusion rates of different modalities are coordinated by designing independent noise schedules for each modality.

**Core Idea**: Introduce a **decoupled noise schedule** that allows each modality to add and remove noise at its own pace, thereby naturally handling mixed state space data within a single model.

## Method

### Overall Architecture

The input consists of coupled multimodal data (such as text-image pairs or mixed-type tabular data), where each modality defines its own forward diffusion process in its native state space. The model jointly denoises in the reverse process, outputting the generation results of each modality.

Mechanism:
- Continuous modalities (e.g., image pixels/latent space) use a **Gaussian diffusion process**.
- Discrete modalities (e.g., text tokens) use a **categorical diffusion process** (uniform-state or absorbing-state).
- Both share the denoising network but utilize independent noise schedules.

### Key Designs

1. **Unified Multimodal Diffusion Framework**:

    - Unifies continuous and discrete diffusion as a generalized Markov noise process.
    - For continuous modalities $x_c$, the forward process is defined as $q_t(x_c|x_0) = \mathcal{N}(\alpha_t^c x_0, \sigma_t^c \mathbf{I})$.
    - For discrete modalities $x_d$, the forward process is defined as $q_t(x_d|x_0) = \text{Cat}(\alpha_t^d x_0 + (1-\alpha_t^d)\pi)$.
    - The key is that $\alpha_t^c$ and $\alpha_t^d$ can be configured independently.
    - **Design Motivation**: Since the information density and complexity of different modalities vary drastically, a unified noise schedule would cause one modality to finish diffusing prematurely while another has not yet begun effective denoising.

2. **Decoupled Noise Schedule**:

    - Defines independent signal-to-noise ratio (SNR) curves for each modality.
    - Continuous modalities use $\text{SNR}_c(t) = \alpha_t^{c2} / \sigma_t^{c2}$.
    - Discrete modalities use the corresponding retention probability $\alpha_t^d$.
    - By independently adjusting the schedule parameters of each modality, the denoising difficulty and generation quality of each modality can be controlled.
    - **Design Motivation**: The discrete space of text is much smaller than the continuous space of images. If the same noise schedule were used, text would be completely destroyed at very early timesteps, preventing the model from learning effective text denoising.

3. **Unifying Conditional and Unconditional Generation**:

    - Utilizing the decoupled schedule, the noise level of one modality can be set to 0 (i.e., keeping it as clean data) while only performing diffusion on the other modality.
    - This naturally enables conditional generation: text-to-image generation given text, or image-to-text generation given an image.
    - Unconditional generation consists of all modalities simultaneously starting from noise.
    - **Design Motivation**: To avoid training multiple independent models, as a single model can switch generation modes simply by adjusting the noise schedules.

### Loss & Training

- The overall loss is a weighted sum of the denoising loss of each modality:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \lambda_c(t) \|\hat{x}_c - x_c\|^2 + \lambda_d(t) \text{CE}(\hat{x}_d, x_d) \right]$$

- Continuous modalities employ the MSE loss (standard denoising objective).
- Discrete modalities employ the cross-entropy (CE) loss.
- The weights $\lambda_c(t)$ and $\lambda_d(t)$ are adaptively adjusted based on each modality's SNR.
- During training, timesteps $t$ are randomly sampled, and each modality receives different noise levels according to its respective schedule.

## Key Experimental Results

### Main Results: Joint Text-Image Generation

| Method | Image FID ↓ | Text PPL ↓ | Joint Generation Quality |
|------|-----------|-----------|-------------|
| Ours (Decoupled Schedule) | **Competitive** | **Competitive** | Best |
| Unified Schedule Baseline | Poor | Poor | Modalities interfere with each other |
| Independent Models | Good | Good | Cannot generate jointly |
| Tokenizer + Diffusion | Limited by tokenizer quality | Limited by decoding quality | Depends on external modules |

### Ablation Study: Impact of Decoupled Schedules

| Configuration | Image Quality | Text Quality | Description |
|------|---------|---------|------|
| Fully Decoupled Schedule | ✓ Best | ✓ Best | Each modality independently optimizes noise pacing |
| Unified Schedule (Synchronized) | ✗ Poor | ✗ Poor | Text is destroyed prematurely; image denoising is sufficient |
| Image Only Decoupled | ✓ Good | △ Moderate | Text remains limited |
| Text Only Decoupled | △ Moderate | ✓ Good | Image remains limited |

### Key Findings
- Decoupled noise scheduling is key to achieving high-quality joint multimodal generation; unified scheduling leads to severe inter-modality interference.
- The framework is also applicable to mixed-type tabular data (containing continuous and categorical columns), achieving competitive performance on tabular data synthesis tasks.
- A single model can realize various modes like unconditional generation and single-modality conditional generation without requiring separate training for each task.

## Highlights & Insights
- **Theoretical Elegance**: Unified continuous and discrete diffusion under the same framework, where decoupled scheduling acts as a simple yet effective design.
- **No External Dependencies**: Eliminates reliance on tokenizers/VAEs, operating directly on raw native state spaces.
- **High Flexibility**: The framework can be extended to any number and type of modality combinations.
- **Tabular Data Application**: Demonstrates the applicability of diffusion models to non-visual multimodal tasks.

## Limitations & Future Work
- Currently verified primarily on relatively small-scale datasets, and has not yet been tested on large-scale text-to-image tasks (e.g., LAION scale).
- Hyperparameter selection for decoupled schedules requires tuning for specific combinations of modalities.
- Image generation quality still lags behind specialized single-modality diffusion models (e.g., SDXL, DALL-E 3).
- Joint generation of more modalities (e.g., audio, video) has not been explored.

## Related Work & Insights
- **Discrete Diffusion**: Discrete diffusion models such as D3PM and MDLM provide the theoretical foundation for text modalities.
- **Multimodal Generation**: Works like UniDiffuser have also explored joint multimodal diffusion, but rely on external modules like VAEs.
- **Insights**: The concept of decoupled noise scheduling can be extended to further mixed state space scenarios, such as molecule generation (continuous coordinates + discrete atom types).

## Rating
- Novelty: ⭐⭐⭐⭐ Decoupled noise scheduling is a clean and elegant innovation, and the unified framework holds theoretical value.
- Experimental Thoroughness: ⭐⭐⭐ Validated on text-image and tabular data scenarios, but at a relatively small scale.
- Writing Quality: ⭐⭐⭐⭐ The mathematical derivations are clear, and the framework description is comprehensive.
- Value: ⭐⭐⭐⭐ Provides fresh avenues for native multimodal diffusion, serving as an inspiration for mixed state space modeling.

## Related Papers

- [\[CVPR 2026\] Grounding Everything in Tokens for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/grounding_everything_in_tokens_for_multimodal_large_language_models.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](../../CVPR2025/multimodal_vlm/molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[CVPR 2026\] Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models](../../CVPR2026/multimodal_vlm/sparse-lavida_sparse_multimodal_discrete_diffusion_language_models.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Grounding Everything in Tokens for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/grounding_everything_in_tokens_for_multimodal_large_language_models.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](../../CVPR2025/multimodal_vlm/molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[CVPR 2026\] Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models](../../CVPR2026/multimodal_vlm/sparse-lavida_sparse_multimodal_discrete_diffusion_language_models.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)

</div>

<!-- RELATED:END -->
