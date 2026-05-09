---
title: >-
  [Paper Note] I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting
description: >-
  [ICCV 2025][Time Series][4D occupancy forecasting] This paper proposes I²-World, which decouples 3D scene tokenization into two complementary processes — intra-scene multi-scale residual quantization and inter-scene temporal quantization — thereby retaining the high compression ratio of 3D tokenizers while incorporating the temporal modeling capability of 4D tokenizers, enabling efficient and high-quality 4D occupancy forecasting.
tags:
  - ICCV 2025
  - Time Series
  - 4D occupancy forecasting
  - scene tokenization
  - world model
  - autoregressive generation
  - autonomous driving
date: 2026-05-08
content_hash: 38d096414c6cadbc
---

# I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting

**Conference**: ICCV 2025
**arXiv**: [2507.09144](https://arxiv.org/abs/2507.09144)
**Code**: [GitHub](https://github.com/lzzzzzm/II-World)
**Area**: Time Series
**Keywords**: 4D occupancy forecasting, scene tokenization, world model, autoregressive generation, autonomous driving

## TL;DR

This paper proposes I²-World, which decouples 3D scene tokenization into two complementary processes — intra-scene multi-scale residual quantization and inter-scene temporal quantization — thereby retaining the high compression ratio of 3D tokenizers while incorporating the temporal modeling capability of 4D tokenizers, enabling efficient and high-quality 4D occupancy forecasting.

## Background & Motivation

Occupancy-based world models are used in autonomous driving to predict the evolution of future 3D scenes, and are critical for handling corner cases. Existing approaches face a fundamental tension:

**3D scene tokenizers** (e.g., OccWorld, OccLLaMA): compress single-frame scenes into compact tokens with high reconstruction fidelity, but cannot model temporal dynamics, limiting predictive capability.

**4D scene tokenizers** (e.g., OccSora, DOME): process spatiotemporal token sequences with strong dynamic modeling ability, but suffer from an explosive growth in token dimensionality, leading to prohibitive computational overhead that is incompatible with real-time requirements.

Furthermore, most existing methods adopt GPT-style decoder-only autoregressive architectures or diffusion models, both of which are resource-intensive. The core problem addressed in this paper is: **how to simultaneously encode spatial detail and temporal dynamics within a compact token representation?**

## Method

### Overall Architecture

I²-World consists of two core components trained in two stages:

- **Stage 1**: Train the I²-Scene Tokenizer (VAE pipeline) to learn scene compression and reconstruction.
- **Stage 2**: Freeze the tokenizer and train I²-Former to learn dynamic transitions.

Given the current occupancy $O_t \in \mathbb{R}^{H \times W \times Z}$ and $G$ frames of history, the tokenizer outputs compact continuous tokens $\hat{B}_t \in \mathbb{R}^{h \times w \times C}$ (where $h, w \ll H, W$). I²-Former autoregressively predicts future $K$-frame tokens, which are then decoded into occupancy.

### Key Designs

1. **Intra-Scene Tokenizer (intra-frame multi-scale residual quantization)**: Inspired by RQ-VAE and VAR, the encoder output feature $B_t$ is iteratively quantized into $S$ multi-scale token maps $\{b_t^s\}_{s=1}^S$. Quantization proceeds from low to high resolution; at each scale, features are matched against a shared codebook $\mathcal{C} \in \mathbb{R}^{N \times C}$ and the residual is passed to the next level. The core formulation is:

    $b_t^s = f_{intp}(B_t, h_s, w_s), \quad \hat{b}_t^{s_{i,j}} = \mathcal{Q}(b_t^{s_{i,j}}, \mathcal{C})$
    $B_t = B_t - f_{intp}(\hat{b}_t^s, h, w), \quad \hat{B}_t = \hat{B}_t + \phi_s(\hat{b}_t^s)$

   where $\phi_s$ is a learnable convolutional layer that mitigates information loss from resolution scaling. **Design Motivation**: 3D occupancy naturally supports multi-scale representation, with coarse-to-fine levels progressively supplying spatial detail.

2. **Inter-Scene Tokenizer (inter-frame temporal quantization)**: A historical feature queue $\{B_{t-g}\}_{g=1}^G$ is maintained, with each frame aligned to the current coordinate system via ego-pose transformation matrix $T_{t-g}^t$. The residual from the Intra-Scene stage is then added to aligned historical features and subjected to $G$ rounds of temporal quantization:

    $b_t^{S+g} = B_t + B_{t-g}', \quad \hat{B}_t = \hat{B}_t + \psi_g(\hat{b}_t^{S+g})$

   Crucially, the same codebook $\mathcal{C}$ is shared with the Intra-Scene Tokenizer; only $G+S$ lightweight convolutional layers are added to jointly encode spatial and temporal information.

3. **I²-Former (encoder–decoder architecture)**: Unlike GPT-style decoder-only designs, I²-Former adopts an Intra-Encoder + Inter-Decoder structure:

    - **Intra-Encoder**: Employs Spatial Self-Attention (SSA) and multi-head cross-attention to fuse plan embeddings with scene tokens, using a 3-layer multi-scale design (2× downsampling per layer) with FPN aggregation. A key output is the regressed **transformation matrix** $T_{t+k}^{t+k+1} \in \mathbb{R}^{4 \times 4}$, which encodes inter-frame spatiotemporal transitions.
    - **Inter-Decoder**: Conditioned on the transformation matrix, the decoder maintains a historical token queue and autoregressively generates the next-frame token via SSA and lightweight MLP-based temporal fusion (channel-wise concatenation followed by a single-layer MLP).

   **Design Motivation**: The transformation matrix provides finer-grained spatial constraints than trajectory-based guidance, enabling controllable generation.

### Loss & Training

**Tokenizer stage**:
$$\mathcal{L}_{token} = \mathcal{L}_{focal}(O_t, \hat{O}_t) + \mathcal{L}_{lov}(O_t, \hat{O}_t) + \mathcal{L}_{vq}$$

where $\mathcal{L}_{vq}$ includes a codebook alignment term and commitment loss ($\beta=1$), applied only to the Intra-Scene component to ensure training stability.

**Generation stage**:
$$\mathcal{L}_{gen} = \sum_{k=1}^K w_k \mathcal{L}_{mse}(\hat{B}_{t+k}', \hat{B}_{t+k})$$

The transformation matrix is decomposed into translation (L2 loss) and rotation (quaternion cosine loss), each supervised separately.

## Key Experimental Results

### Main Results (Occ3D-nuScenes)

| Method | Input | mIoU Avg (%) | IoU Avg (%) | FPS |
|--------|-------|-------------|-------------|-----|
| OccWorld-O | Occ | 17.14 | 26.63 | 18.0 |
| DOME | Occ | 27.10 | 36.36 | 6.54 |
| UniScene | Occ | 31.76 | 34.84 | - |
| **I²-World-O** | **Occ** | **39.73** | **49.80** | **37.04** |
| DOME-STC | Camera | 14.53 | 23.33 | 2.75 |
| **I²-World-STC** | **Camera** | **18.97** | **28.77** | **4.21** |

- Under GT occupancy input, mIoU improves by **25.1%** (39.73 vs. 31.76) and IoU by **36.9%**.
- Training requires only **2.9 GB** of GPU memory; inference runs at **37 FPS** (RTX 4090), far surpassing existing methods.

### Ablation Study

| Component | mIoU (%) | IoU (%) | Note |
|-----------|---------|---------|------|
| Baseline (single-scale) | 66.52 | 61.07 | No Inter-Scene |
| + Inter-Scene (no alignment) | 70.37 | 62.18 | Temporal modeling +5.7% mIoU |
| + Alignment (Intra single-scale) | 77.12 | 64.20 | Alignment +15.9% mIoU |
| + Multi-scale Intra + Alignment | **81.22** | **68.30** | Best reconstruction quality |
| Trans condition only | 28.74 | 36.44 | Translation contributes most |
| Trans + Rot + Encoder + MS | **39.73** | **49.80** | Components are complementary |

### Key Findings

- Inter-frame ego-pose alignment is the key to effective temporal modeling, contributing a 15.9% mIoU gain.
- Translation conditioning is the primary driver of generation quality (+67.8% mIoU vs. baseline); rotation contributes less individually but is complementary to translation.
- Zero-shot transfer to Occ3D-Waymo yields strong results: mIoU of 43.73 at 10 Hz vs. 28.34 for copy-paste baseline.

## Highlights & Insights

- **Elegant decoupled design**: intra-scene captures spatial detail while inter-scene encodes temporal dynamics; the shared codebook ensures spatiotemporal consistency.
- **Extreme efficiency**: 2.9 GB memory and 37 FPS make occupancy-based world models viable for real-time deployment for the first time.
- **Controllable generation**: the transformation matrix enables direct manipulation of scene evolution at meter/radian precision, supporting corner case simulation.

## Limitations & Future Work

- The transformation matrix may produce unrealistic outputs for motion patterns not covered in the training set (e.g., reversing).
- It remains unclear whether continuous tokens (as opposed to discrete VQ tokens) limit integration with discrete generative paradigms such as LLMs.
- Future work may explore longer-horizon forecasting (>3s) and strategies to mitigate cumulative errors.

## Related Work & Insights

- The multi-scale residual quantization paradigm from **RQ-VAE / VAR** is worth exploring in a broader range of 3D tasks.
- Using transformation matrices as conditioning signals outperforms conventional trajectory guidance, offering inspiration for other scene generation tasks.
- The dual encoder–decoder design shares conceptual similarity with iVideoGPT, where decoupling redundancy is key to efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ — The intra/inter decoupled tokenization design is original.
- Technical Depth: ⭐⭐⭐⭐ — Multi-scale residual quantization + temporal quantization + transformation-matrix-conditioned generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of main results, ablations, generalization, and visualization.
- Value: ⭐⭐⭐⭐⭐ — Exceptional efficiency; the first real-time 4D occupancy world model.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](../../CVPR2026/time_series/a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](../../NeurIPS2025/time_series/diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)

<!-- RELATED:END -->
