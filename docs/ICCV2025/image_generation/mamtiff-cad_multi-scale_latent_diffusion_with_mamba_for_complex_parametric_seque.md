---
title: >-
  [Paper Note] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence
description: >-
  [Image Generation] This paper proposes MamTiff-CAD, a framework that combines a Mamba+-based encoder with a Transformer decoder in an autoencoder to learn latent representations of CAD command sequences, followed by a multi-scale Transformer diffusion model for generation. It is the first method to generate complex CAD models with sequence lengths of 60–256 commands.
tags:
  - Image Generation
date: 2026-05-08
content_hash: d427521e369eaabe
---

# MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2511.17647](https://arxiv.org/abs/2511.17647)
- **Code**: Not released
- **Area**: Diffusion Models · CAD Generation
- **Keywords**: Parametric CAD, Mamba, Long Sequence Modeling, Multi-Scale Transformer, Diffusion Models

## TL;DR
This paper proposes MamTiff-CAD, a framework that combines a Mamba+-based encoder with a Transformer decoder in an autoencoder to learn latent representations of CAD command sequences, followed by a multi-scale Transformer diffusion model for generation. It is the first method to generate complex CAD models with sequence lengths of 60–256 commands.

## Background & Motivation

Parametric CAD constructs 3D models through command sequences (sketches, extrusions, Boolean operations, etc.) and is central to industrial design. Existing deep learning methods (e.g., DeepCAD) are constrained by the quadratic complexity of Transformers, limiting them to short sequences (<60 commands) and preventing the generation of industrially complex CAD models.

**Core Challenges**:
1. Long-sequence modeling bottleneck: The $O(n^2)$ complexity of Transformers restricts sequence scaling.
2. Local–global constraint balance: CAD models simultaneously encode fine-grained local geometry and global topological constraints.
3. Dataset gap: Existing datasets (DeepCAD averages 15 commands) contain no complex long-sequence CAD models.

## Method

### Overall Architecture (Two-Stage)

**Stage 1**: An autoencoder with a Mamba+ encoder and a Transformer decoder encodes CAD sequences into a latent representation $Z$.

**Stage 2**: A multi-scale Transformer diffusion model learns the distribution in latent space to generate new CAD models.

### CAD Parametric Representation

Each command is represented as $m_i = (C_i, p_i)$, where $C_i$ is one of six command types and $p_i \in \mathbb{R}^{16}$ contains coordinates, angles, extrusion parameters, etc. Continuous parameters are normalized to a $2 \times 2 \times 2$ cube and quantized into 256 discrete levels. Sequences are padded with EOS tokens to a fixed length of 256.

### Key Designs

#### Mamba+ Encoder (with Forget Gate)

The core innovation is a **dual-branch design with a forget gate**:

$$G_f = 1 - G_{b2}$$

$$x'' = G_f \cdot x'$$

$$h_{\text{out}} = x'' + h_{\text{SSM}}$$

- Branch b1: 1D convolution + SSM block for sequential feature extraction.
- Branch b2: SiLU activation to produce a control signal.
- The forget gate $G_f$ modulates retention of historical information, preventing loss of critical long-range dependencies.

Four stacked Mamba+ blocks enable efficient encoding of CAD sequences from length 60 to 256.

#### Transformer Decoder (Non-Autoregressive)

Four Transformer blocks take the latent vector $Z$ and learnable positional embeddings as input. All 256 command positions are decoded in parallel:

$$p(\hat{M} | z, \Theta) = \prod_{i=1}^{N_c} p(\hat{C}_i, \hat{p}_i | z, \Theta)$$

### Loss & Training

#### Autoencoder Training Loss

$$L = \sum_{i=1}^{N_c} \ell(p_i(t_i)) + \beta \sum_{i=1}^{N_c} \sum_{j=1}^{N_p} \ell(q_{i,j}(a_{i,j}))$$

$\beta=2$ balances parameter reconstruction loss and command type loss. Padding commands and unused parameters are excluded.

#### Multi-Scale Transformer Diffusion Generator (MST-D)

Three parallel attention branches capture dependencies at different scales:
- Window 64: local geometric constraints
- Window 128: medium-range topological dependencies
- Window 256: global semantic consistency

Adaptive fusion:
$$\mathbf{H} = \text{MLP}(\sigma(\mathbf{W}_g [\mathbf{H}_l \| \mathbf{H}_m \| \mathbf{H}_g]) \odot [\mathbf{H}_l \| \mathbf{H}_m \| \mathbf{H}_g])$$

Standard DDPM noise prediction loss:
$$L_{\text{diff}} = \mathbb{E}_{t,Z_0,\epsilon}[\|\epsilon - \epsilon_\theta(Z_t, t)\|_2^2]$$

## Key Experimental Results

### Main Results

#### Autoencoder Reconstruction (ABC-256 Dataset)

| Method | Cmd Acc.↑ | Param Acc.↑ | MCD↓ | Invalid Rate↓ | STEP Rate↑ |
|--------|-----------|-------------|------|---------------|------------|
| DeepCAD | 92.24% | 75.93% | 41.02 | 33.11% | 70.46% |
| MT-CAD | 89.72% | 66.87% | 121.35 | 39.89% | 63.97% |
| **MamTiff-CAD** | **99.99%** | **99.93%** | **0.75** | **8.50%** | **93.93%** |

Command and parameter accuracy approach 100%; MCD drops from 41.02 to 0.75 (−98%), and the invalid rate decreases from 33.11% to 8.50%.

#### Unconditional Generation

| Method | MMD↓ | JSD↓ | COV↑ | Unique↑ | Novel↑ | STEP Rate↑ |
|--------|------|------|------|---------|--------|------------|
| DeepCAD | 2.66 | 6.49 | 56.66% | 75.8 | 88.0 | 23.96% |
| SkexGen | 2.31 | 4.53 | 57.76% | 80.5 | 96.9 | 75.26% |
| **MamTiff-CAD** | **1.43** | **3.19** | **64.16%** | **90.8** | **95.6** | **85.38%** |

JSD (distribution divergence) of 3.19 is best-in-class; STEP conversion success rate of 85.38% substantially surpasses DeepCAD's 23.96%.

#### ABC-256 Dataset Contribution

The paper introduces a dataset of 13,705 CAD models with an average sequence length of 99 (6.6× that of DeepCAD), spanning lengths of 60–256. It is split into 10,964 training, 1,370 validation, and 1,371 test samples.

## Highlights & Insights

1. **Long-sequence breakthrough**: First method to achieve industrial-grade CAD generation with up to 256 commands.
2. **Forget gate in Mamba+**: Effectively mitigates information loss in long-range dependencies.
3. **Multi-scale diffusion**: Three-level local–medium–global attention simultaneously enforces geometric and topological constraints.
4. **Dataset contribution**: ABC-256 fills the gap in long-sequence CAD datasets.

## Limitations & Future Work

- The fixed length of 256 prevents modeling of sequences exceeding this limit.
- Only unconditional generation is supported; text- or image-guided generation is absent.
- Evaluation focuses solely on geometric quality without assessing engineering usability.
- Computational overhead: two-stage training requires 300K + 200K epochs in total.

## Related Work & Insights

- **CAD Generation**: DeepCAD, SkexGen, HNC-CAD
- **Long-Sequence Models**: Mamba, sparse attention
- **3D Diffusion Generation**: 3DShape2VecSet, DiT-3D, DiffCAD

## Rating
- **Novelty**: ★★★★☆ — The combination of Mamba+ and multi-scale diffusion is well-targeted.
- **Technical Depth**: ★★★★☆ — Architecture design is sound and experimental validation is thorough.
- **Practicality**: ★★★★☆ — Addresses real demands in industrial-grade CAD generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)

</div>

<!-- RELATED:END -->
