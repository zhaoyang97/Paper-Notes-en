---
title: >-
  [Paper Note] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention
description: >-
  [ICLR 2026][Audio & Speech][audio-visual speech separation] This paper proposes Dolphin, a model that maps lip movements to discrete semantic tokens via a dual-path lightweight video encoder (DP-LipCoder), and introduces a Global-Local Attention (GLA) separator. Dolphin surpasses state-of-the-art methods on three benchmarks while reducing parameters by 50%+, MACs by 2.4×, and GPU inference latency by 6×.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - audio-visual speech separation
  - discrete lip semantics
  - vector quantization
  - global-local attention
  - lightweight
date: 2026-05-08
content_hash: 9c985b36eb17aa49
---

# Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention

**Conference**: ICLR 2026  
**arXiv**: [2509.23610](https://arxiv.org/abs/2509.23610)  
**Code**: Available ([https://cslikai.cn/Dolphin](https://cslikai.cn/Dolphin))  
**Area**: Audio & Speech  
**Keywords**: audio-visual speech separation, discrete lip semantics, vector quantization, global-local attention, lightweight

## TL;DR

This paper proposes Dolphin, a model that maps lip movements to discrete semantic tokens via a dual-path lightweight video encoder (DP-LipCoder), and introduces a Global-Local Attention (GLA) separator. Dolphin surpasses state-of-the-art methods on three benchmarks while reducing parameters by 50%+, MACs by 2.4×, and GPU inference latency by 6×.

## Background & Motivation

Audio-Visual Speech Separation (AVSS) leverages visual cues (lip movements) to extract target speaker speech from noisy mixed audio. Existing methods face two fundamental tensions:

**Path dependency in visual encoders**: Large-scale pretrained video backbones (e.g., 3D ResNet-18) offer strong semantic alignment but at prohibitive computational cost; direct compression severely degrades semantic representation; lightweight encoders designed from scratch can only extract shallow pixel-level features.

**Efficiency–quality trade-off in separators**: High-performance methods (e.g., AV-Mossformer2) have enormous parameter counts unsuitable for deployment; lightweight alternatives (RTFSNet, AVLiT) rely on multiple iterations, resulting in still-high inference latency.

## Method

### Overall Architecture

Dolphin consists of five core components:
- **Pretrained video encoder DP-LipCoder**: Maps lip video to reconstruction features $\mathbf{V}_r$ and semantic features $\mathbf{V}_s$
- **Audio encoder**: A 1D convolutional layer encoding mixed audio into $\mathbf{X} \in \mathbb{R}^{N_a \times T_a}$
- **Audio-Visual Fusion (AVF) module**: Fuses visual and audio features
- **Separator**: An encoder-decoder architecture based on TDANet, with GLA blocks embedded at each layer
- **Audio decoder**: A 1D transposed convolution outputting the time-domain separated signal

### Key Designs

#### 1. DP-LipCoder: Dual-Path Lightweight Video Encoder

A dual-path autoencoder is designed based on the MagVIT video generation architecture:

- **Reconstruction path**: Extracts compressed visual features $\mathbf{V}_r$, preserving auxiliary cues such as speaker identity and facial expressions. The encoder consists of cascaded 3D residual blocks, spatial attention blocks, and alternating spatial downsampling.
- **Semantic path**: An encoder with the same structure but non-shared parameters, augmented at the end with a **Vector Quantization (VQ) module**. Knowledge distillation from AV-HuBERT maps continuous video to audio-aligned discrete semantic tokens $\mathbf{V}_s$.
- **Decoder**: The outputs of both paths are summed and fused for video reconstruction.

Three training losses are jointly optimized:
$$\mathcal{L} = \mathcal{L}_{\text{commit}} + \mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{recon}}$$

| Loss | Role |
|------|------|
| $\mathcal{L}_{\text{recon}}$ | Reconstruction loss; drives the reconstruction path to capture speaker visual cues |
| $\mathcal{L}_{\text{distill}}$ | AV-HuBERT teacher distillation; guides the semantic path to extract audio-aligned features |
| $\mathcal{L}_{\text{commit}}$ | VQ commitment loss; constrains consistency between encoder outputs and the codebook |

During AVSS inference, **only the encoder and VQ module are executed**, with no need for the decoder. Compared to 3D ResNet-18: parameters are reduced by 93% (0.78M vs. 11.19M), MACs reduced by 70%, and SI-SNRi drops by only 0.2 dB.

#### 2. GLA Block: Global-Local Attention

**GA Block (Global Attention)**:
- Coarse-grained self-attention (CSA): downsamples to length $T_a/2^Q$, performs MHSA, then upsamples back to original length
- Computational complexity reduced to $1/2^{2Q}$ of the original
- Followed by an FFN (with DWConv1D, kernel=3)

**LA Block (Local Attention)**:
- **Heat Diffusion Attention (HDA)**: Learnable multi-scale filtering designed with physics-based priors from the heat diffusion equation
- Maps to the frequency domain via DCT and applies exponential decay filtering:
  $$\tilde{\mathbf{A}}(p) = \mathbf{A}(p) \cdot \exp(-\mathbf{k}_c (p\pi/T_a)^2)$$
- $\mathbf{k}_c \in \mathbb{R}^{N_a}$ is a learnable channel-adaptive diffusion coefficient
- Maps back to time domain via IDCT, followed by a gating mechanism: $\breve{\mathbf{F}}_0 = \mathcal{P}(\hat{\mathbf{x}} \odot \text{SiLU}(\mathbf{z}))$
- Advantage: Not limited by finite convolutional receptive fields; fewer parameters than large-kernel Conv1D with finer modeling

#### 3. Encoder-Decoder Separator

- **Encoder**: $Q=4$ layers, each with 2 GLA blocks + downsampling, extracting multi-scale features
- Features from all scales are downsampled to the lowest resolution and summed to obtain a global representation $\mathcal{G}$, enhanced by a top-level GA block
- **Decoder**: $Q=4$ layers, each with a TDA block (upsampling) + 3 GLA blocks
- **Directly outputs target speaker features** without mask multiplication, avoiding distortion introduced by conventional masking

#### 4. Audio-Visual Fusion Module

Two fusion mechanisms from RTFSNet are extended to the time domain: video-guided gating fusion $\mathcal{F}_1$ and attention-based cross-feature-space fusion $\mathcal{F}_2$, with visual features upsampled along the temporal dimension only.

### Loss & Training

- Separator optimization objective: SI-SNR
- Adam optimizer, lr=1e-3; learning rate halved after 15 epochs of validation loss plateau, early stopping after 30 epochs of stagnation
- L2 gradient clipping threshold of 5, batch size=48, 8× RTX 5090 GPUs
- DP-LipCoder parameters are frozen; only the separation network is trained

## Key Experimental Results

### Main Results

**Table 1: Pretrained Video Encoder Comparison (LRS2)**

| Method | SI-SNRi(dB)↑ | SDRi(dB)↑ | PESQ↑ | Params(MB)↓ | MACs(G/s)↓ |
|--------|-------------|-----------|-------|-------------|------------|
| 3D ResNet-18 | 17.0 | 17.1 | 3.30 | 11.19 | 7.95 |
| AE | 15.2 | 15.4 | 3.15 | 0.05 | 0.17 |
| LipCoder | 16.3 | 16.4 | 3.24 | 0.65 | 5.33 |
| **DP-LipCoder** | **16.8** | **16.9** | **3.29** | **0.78** | **2.38** |

**Table 2: AVSS SOTA Comparison (Three Datasets)**

| Method | LRS2 SI-SNRi | LRS3 SI-SNRi | VoxCeleb2 SI-SNRi |
|--------|-------------|-------------|-------------------|
| IIANet | 16.0 | 18.3 | 13.6 |
| AV-Mossformer2 | 15.1 | 17.7 | 14.0 |
| **Dolphin** | **16.8** | **18.8** | **14.6** |

**Table 3: Efficiency Comparison (Including Video Encoder)**

| Method | Params(M)↓ | MACs(G)↓ | GPU Latency(ms)↓ |
|--------|-----------|---------|-----------------|
| IIANet | 15.01 | 26.51 | 142.30 |
| AV-Mossformer2 | 68.52 | 124.46 | 62.30 |
| **Dolphin** | **7.00** | **10.89** | **33.24** |

### Ablation Study

**GLA Component Ablation (LRS2)**:

| GA | LA | SI-SNRi↑ | Params(MB)↓ |
|----|---|---------|------------|
| ✗ | ✗ | 10.4 | 2.04 |
| ✓ | ✗ | 15.9 | 5.23 |
| ✗ | ✓ | 15.6 | 3.81 |
| ✓ | ✓ | **16.8** | 7.00 |

HDA layer vs. Conv1D: HDA achieves 16.9 dB SI-SNRi, outperforming Conv1D at 16.5 dB, with fewer parameters (7.00M vs. 7.57M).

### Key Findings

1. VQ discrete encoding improves SI-SNRi by at least 1.0 dB over continuous autoencoders; the VQ module alone contributes approximately 0.5 dB.
2. DP-LipCoder generalizes to other AVSS models: replacing the video encoder reduces parameters by 10M+ with only marginal performance degradation.
3. A single-iteration architecture with GLA outperforms multi-iteration approaches.
4. Compared to SOTA IIANet: parameters reduced by 53%, MACs by 59%, and GPU inference is 4.3× faster.

## Highlights & Insights

- **Superiority of discrete representations**: Mapping the video stream to a "visual vocabulary" yields more compact and discriminative representations than continuous alternatives — offering broad inspiration for multimodal system design.
- **Heat diffusion physics prior**: Incorporating the heat equation into local attention enables fine-grained local feature modeling by learning only scaling and gating parameters, reducing overfitting risk.
- **Dual-path complementarity philosophy**: The reconstruction path retains auxiliary identity and expression information, while the semantic path extracts audio-aligned information.

## Limitations & Future Work

1. The method relies on clean, synchronized lip video and lacks robustness to large head pose variations, occlusion, and extreme lighting conditions.
2. Deployment on extremely resource-constrained devices remains challenging; quantization and pruning are worth exploring.
3. Discrete tokens may lose fine-grained articulatory cues; hierarchical codebooks or hybrid discrete-continuous representations could be explored.
4. Validation is conducted only on English datasets; cross-lingual generalization remains to be investigated.

## Related Work & Insights

- **TDANet** provides the foundational separation architecture; this work adds GLA blocks and removes iterative processing.
- **AV-HuBERT** serves as the teacher model for semantic distillation.
- **MagVIT**'s video generation architecture is creatively repurposed as a video encoder.
- Insight: Physics-based priors (heat diffusion) can be injected into attention mechanisms as inductive biases.

## Rating

- Novelty: ⭐⭐⭐⭐ — Dual-path discrete encoding + heat diffusion local attention
- Technical Depth: ⭐⭐⭐⭐ — Multi-module design with comprehensive ablation studies
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, multi-dimensional efficiency comparisons, and ablations
- Value: ⭐⭐⭐⭐⭐ — Significant efficiency gains with clear deployment scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)

</div>

<!-- RELATED:END -->
