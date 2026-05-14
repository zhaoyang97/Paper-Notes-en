---
title: >-
  [Paper Note] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space
description: >-
  [Image Generation] This paper proposes MotionStreamer, which integrates a continuous causal latent space with a diffusion head into an autoregressive framework for text-conditioned streaming human motion generation…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 2a909cfec6ccac1b
---

# MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space

> **Conference**: ICCV 2025
> **arXiv**: [2503.15451](https://arxiv.org/abs/2503.15451)
> **Code**: [Project Page](https://zju3dv.github.io/MotionStreamer/)
> **Area**: Motion Generation · Diffusion Models · Autoregressive
> **Keywords**: streaming motion generation, causal latent space, diffusion head, autoregressive, text-to-motion

## TL;DR

This paper proposes MotionStreamer, which integrates a continuous causal latent space with a diffusion head into an autoregressive framework for text-conditioned streaming human motion generation, supporting online multi-turn generation and dynamic motion composition.

## Background & Motivation

Streaming motion generation requires a model to incrementally generate coherent human motion sequences in response to incremental text inputs, which is critical for real-time applications in gaming, animation, and robotics. Existing methods face the following core challenges:

**Fixed-length limitation of diffusion models**: Conventional diffusion-based motion generation models (e.g., MDM, MLD) require predefined motion lengths and cannot dynamically respond to online text inputs, lacking incremental generation capability.

**Latency and error accumulation in GPT-based methods**: Autoregressive methods based on discrete VQ (e.g., T2M-GPT, MotionGPT) use non-causal tokenizers, which prevent online decoding of partial tokens and introduce high latency; quantization errors from discretization also accumulate progressively in long-sequence autoregressive generation.

**Limitations of fixed-window methods**: Real-time methods such as DART rely on fixed-window local motion primitives and cannot model variable-length historical context.

The core mechanism of MotionStreamer is to **integrate a diffusion head into an autoregressive framework to predict continuous motion latent variables, and to introduce a causal motion compressor for online decoding**, thereby achieving streaming generation, online responsiveness, and long-term consistency simultaneously.

## Method

### Overall Architecture

MotionStreamer consists of three core components (as shown in Fig. 2):

- **Pre-trained text encoder**: Uses T5-XXL to extract text features $T_i \in \mathbb{R}^{1 \times d_t}$
- **Causal Temporal Autoencoder (Causal TAE)**: Encodes raw motion sequences into continuous causal latent variable sequences
- **Diffusion autoregressive model**: Based on a Transformer with a diffusion head, autoregressively predicts the next motion latent variable in the causal latent space

### Key Design 1: Causal Temporal Autoencoder (Causal TAE)

The Causal TAE employs 1D causal convolutions to construct encoder $\mathcal{E}$ and decoder $\mathcal{D}$, ensuring causality through a dedicated temporal padding scheme: for a convolutional layer with kernel size $k_t$, stride $s_t$, and dilation rate $d_t$, $(k_t - 1) \times d_t + (1 - s_t)$ frames are padded at the beginning of the sequence.

Given a motion sequence $X = \{x_1, \ldots, x_N\}$ ($x_t \in \mathbb{R}^{272}$), the Causal TAE outputs continuous latent variables $Z = \{z_1, \ldots, z_{N/l}\}$ ($z_i \in \mathbb{R}^{d_c}$), where $l=4$ is the temporal downsampling rate and $d_c=16$ is the latent dimension.

The motion representation adopts 272-dimensional SMPL 6D rotation vectors: $x = \{\dot{r}^x, \dot{r}^z, \dot{r}^a, j^p, j^v, j^r\}$, which directly drive the SMPL character without post-processing.

### Key Design 2: Diffusion Autoregressive Generator

Each training sample is represented as $S_i = (T_i, C_i, Z_i)$, where $C_i$ is the historical motion latent and $Z_i$ is the current motion latent. These are concatenated along the temporal axis and fed into a Transformer with causal masking to obtain intermediate latent variables $\{c_i^1, \ldots, c_i^n\}$, which are then passed to a diffusion head (a lightweight MLP) to predict the motion latents.

The training loss follows the standard diffusion objective:

$$\mathcal{L} = \mathbb{E}_{\epsilon, t}\left[\|\epsilon - \epsilon_\theta(Z_t | t, C_i, T_i)\|^2\right]$$

### Key Design 3: Two-Forward Training Strategy

To mitigate exposure bias in autoregressive training, a two-forward strategy is proposed: the first forward pass uses ground-truth latents; the second replaces a portion of the ground-truth latents with predictions from the first pass for a mixed forward pass, with gradients backpropagated only through the second pass. The replacement ratio is controlled by a cosine scheduler.

### Continuous Stop Condition

An "impossible pose" (an all-zero vector) is encoded as a reference terminal latent variable. Generation terminates when the distance between the generated latent and this reference falls below a threshold, enabling automatic determination of generation length.

### Loss & Training

Causal TAE training uses the $\sigma$-VAE loss augmented with a root joint loss:

$$\mathcal{L} = \mathcal{L}_{recon} + D_{KL}(q(z|x) \| p(z)) + \lambda \mathcal{L}_{root}$$

## Key Experimental Results

### Main Results: Text-to-Motion Generation (Tab. 1)

| Method | FID ↓ | R@3 ↑ | MM-Dist ↓ | Div → |
|--------|-------|-------|-----------|-------|
| Real motion | 0.002 | 0.914 | 15.151 | 27.492 |
| MDM | 23.454 | 0.764 | 17.423 | 26.325 |
| T2M-GPT | 12.475 | 0.838 | 16.812 | 27.275 |
| MoMask | 12.232 | 0.846 | 16.138 | 27.127 |
| **Ours** | **11.790** | **0.859** | **16.081** | 27.284 |

MotionStreamer outperforms all baseline methods on FID, R@3, and MM-Dist.

### Long-Sequence Generation (Tab. 2, BABEL Dataset)

| Method | Sub-seq FID ↓ | Trans. FID ↓ | PJ → | AUJ ↓ |
|--------|--------------|-------------|------|-------|
| DoubleTake | 23.937 | 51.232 | 0.48 | 1.83 |
| FlowMDM | 18.736 | 34.721 | 0.06 | 0.51 |
| VQ-LLaMA | 24.342 | 36.293 | 0.08 | 1.20 |
| **Ours** | **15.743** | **32.888** | **0.04** | 0.90 |

For long-horizon generation, MotionStreamer achieves substantially lower sub-sequence FID and transition FID compared to all baselines.

### Ablation Study (Tab. 3)

| Compressor | Recon. FID ↓ | MPJPE ↓ | Gen. FID ↓ |
|-----------|-------------|---------|-----------|
| VQ-VAE | 5.173 | 63.9 mm | 13.226 |
| AE | 0.001 | 1.7 mm | 43.828 |
| VAE (non-causal) | 2.092 | 26.2 mm | 19.902 |
| **Causal TAE** | 0.661 | 22.9 mm | **11.790** |

Key finding: the AE achieves the best reconstruction quality but the worst generation performance (due to lack of latent space regularization); the Causal TAE achieves the best overall balance between reconstruction and generation.

### Key Findings

- The continuous latent space avoids information loss from VQ discretization, effectively reducing error accumulation.
- The causal structure of the latent space naturally aligns with the causal masking used in autoregressive generation.
- First-frame latency experiments demonstrate that the Causal TAE achieves the lowest first-frame latency, which does not grow with sequence length.

## Highlights & Insights

1. The **continuous and causal latent space design** is the key innovation, simultaneously addressing both VQ error accumulation and online decoding.
2. The **Two-Forward strategy** effectively mitigates exposure bias in autoregressive training while preserving parallel training efficiency.
3. The **continuous stop condition** is more elegant than binary classifier-based stopping, avoiding class imbalance issues.
4. The framework supports a rich set of applications including multi-turn generation, long-sequence generation, and dynamic motion composition.

## Limitations & Future Work

- The reliance on the SMPL skeleton representation makes it difficult to generalize directly to non-humanoid characters.
- The historical context length is bounded by the Transformer sequence length.
- Semantic drift remains present to some degree in long-sequence generation.

## Related Work & Insights

- Diffusion-based motion generation: MDM, MLD, MotionDiffuse
- Autoregressive motion generation: T2M-GPT, MotionGPT, MoMask
- Real-time control: CAMDM, AMDM, DART, CLoSD

## Rating

- **Novelty**: ★★★★☆ — The combination of a causal latent space and a diffusion head is a novel design.
- **Technical Depth**: ★★★★☆ — The Two-Forward strategy and continuous stop condition are elegantly designed.
- **Experimental Thoroughness**: ★★★★☆ — Comprehensive validation across multiple benchmarks with detailed ablations.
- **Writing Quality**: ★★★★☆ — Well-structured with expressive figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)

</div>

<!-- RELATED:END -->
