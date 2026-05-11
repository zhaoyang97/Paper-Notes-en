---
title: >-
  [Paper Note] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation
description: >-
  [CVPR 2026][Model Compression][Visual autoregressive generation] This work reformulates the visual autoregressive model (VAR) from a full-context-dependent next-scale prediction paradigm into a Markovian scale prediction process. By introducing a sliding-window history compensation mechanism for non-full-context modeling, the method achieves a 10.5% FID reduction and 83.8% peak memory reduction on ImageNet.
tags:
  - CVPR 2026
  - Model Compression
  - Visual autoregressive generation
  - Markov process
  - multi-scale prediction
  - memory efficiency
  - image generation
date: 2026-05-08
content_hash: c1bee5e0115ec0bc
---

# Markovian Scale Prediction: A New Era of Visual Autoregressive Generation

**Conference**: CVPR 2026
**arXiv**: [2511.23334](https://arxiv.org/abs/2511.23334)
**Code**: [Available](https://luokairo.github.io/markov-var-page/)
**Area**: Model Compression
**Keywords**: Visual autoregressive generation, Markov process, multi-scale prediction, memory efficiency, image generation

## TL;DR

This work reformulates the visual autoregressive model (VAR) from a full-context-dependent next-scale prediction paradigm into a Markovian scale prediction process. By introducing a sliding-window history compensation mechanism for non-full-context modeling, the method achieves a 10.5% FID reduction and 83.8% peak memory reduction on ImageNet.

## Background & Motivation

Visual autoregressive modeling (VAR) replaces next-token prediction with next-scale prediction, generating images in a coarse-to-fine manner and achieving breakthroughs in visual generation. However, VAR's **full-context dependency**—where predicting the current scale requires attending to all previous scales—introduces three major problems:

**Prohibitive computational cost**: Token counts grow quadratically with scale, and cross-scale cumulative modeling causes super-linear computation growth. At 1024×1024 resolution, a depth-24 VAR model reaches a peak memory of 117.9 GB.

**Persistent error accumulation**: The unidirectional causal chain of autoregressive generation cannot correct early prediction errors. Experiments show that perturbations injected at early scales have far greater impact on FID than those at later scales (perturbation at the first scale causes the largest FID degradation), and full-context dependency repeatedly exploits erroneous information, exacerbating accumulation.

**Cross-scale interference**: Full-context attention causes gradients from different scales to compete and conflict in the shared feature space. The authors compute RFA (Residual-Feature Alignment) scores—cosine similarities between the residual features of the current scale output and the input features of each prior scale—finding that **early scales generally exert a negative influence on current-scale representation learning**.

The core motivation derives from the information-theoretic concept of **sufficient statistics**: in a sequential chain, each node inherently maintains representative historical information, so effective prediction can be achieved through appropriate distillation without requiring the full history.

## Method

### Overall Architecture

Markov-VAR reformulates VAR as a non-full-context Markovian process:

- **Original VAR modeling**: $p(R_1, \ldots, R_T) = \prod_{t=1}^{T} p(R_t | \langle\text{sos}\rangle, R_{<t})$, where each scale depends on all previous scales.
- **Markov-VAR modeling**: $p(R_1, \ldots, R_T) = \prod_{t=1}^{T} p(R_t | M_{t-1})$, where each scale depends only on the current Markov state.

Here $M_t = f_\phi(R_t, M_{t-1})$ is a representative dynamic state, with $M_0 = \langle\text{sos}\rangle$.

### Key Designs

#### 1. Markov State Definition

- **Function**: Treats the features of each scale directly as the Markov state.
- **Mechanism**: Information theory establishes that the mutual information between the full history $c_{<t}$ and the current timestep $c_t$ is highly redundant; a sufficient statistic $c_{t-1}$ exists such that $I(c_{t-1}; c_t) = I(c_{<t}; c_t)$.
- **Design Motivation**: The sequential unidirectional autoregressive structure causes each scale to already encode representative historical information, making it a natural Markov state. This assumption eliminates full-context dependency and fundamentally avoids KV cache computation.

#### 2. Sliding-Window History Compensation Mechanism

- **Function**: Compresses recent scale information via a sliding window to compensate for information loss caused by non-full-context modeling.
- **Mechanism**: Given a sliding window of size $N$, $\mathcal{W}_t = \{E_{t-1}, E_{t-2}, \ldots, E_{t-N}\}$, the token sequences within the window are concatenated into $\hat{X}_t$, and aggregated into a fixed-dimensional history vector via cross-attention:

$$h_{t-1} = \text{Attn}(q, \hat{X}_t, \hat{X}_t)$$

where $q$ is a learnable global state query. The history vector is broadcast and concatenated with the current scale features to form the representative dynamic state:

$$M_{t-1} = \text{Concat}(E_{t-1}, H_{t-1})$$

- **Design Motivation**: Window size $N=3$ is verified as optimal through ablation, consistent with RFA analysis—the most recent 3 scales contribute positively to current-scale learning, while earlier scales introduce interference.

#### 3. Markovian Attention

- **Function**: Redesigns the attention mask to restrict each scale to attend only to its current dynamic state $M_{t-1}$.
- **Mechanism**: Unlike VAR's full causal attention, Markovian attention strictly confines the attention scope of each scale to within its dynamic state.
- **Design Motivation**: Eliminating cross-scale interference allows each scale to learn distinctive representations; removing the need for KV cache fundamentally reduces computational cost.

### Loss & Training

- **Loss function**: Cross-entropy $\mathcal{L} = \sum_{t=1}^{T} CE(\hat{R}_t, R_t)$
- **Training scheme**: Teacher-forcing with Markovian attention mask
- **Optimizer**: AdamW, lr=$8 \times 10^{-5}$, $\beta_1=0.9$, $\beta_2=0.95$
- **Scale**: Batch size 768–1536, epochs 200–400, 8×H200 GPUs
- **Tokenizer**: Multi-scale VQ-VAE tokenizer pretrained by VAR
- **Positional encoding**: Rotary Positional Embedding (RoPE)
- **Network architecture**: LLaMA-style attention and MLP blocks, width $w=64d$, attention heads $h=d$

## Key Experimental Results

### Main Results (ImageNet 256×256 Class-Conditional)

| Model | Params | FID↓ | IS↑ | Precision↑ | Recall↑ |
|-------|--------|------|-----|------------|---------|
| VAR-d16 | 310M | 3.61 | 225.6 | 0.81 | 0.52 |
| **Markov-VAR-d16** | 329M | **3.23** | **256.2** | **0.84** | 0.52 |
| VAR-d20 | 600M | 2.67 | 254.4 | 0.81 | 0.57 |
| **Markov-VAR-d20** | 623M | **2.44** | **286.1** | 0.83 | 0.56 |
| VAR-d24 | 1.0B | 2.17 | 271.9 | 0.81 | 0.59 |
| **Markov-VAR-d24** | 1.02B | **2.15** | **310.9** | 0.83 | 0.59 |
| DiT-XL/2 (Diffusion) | 675M | 2.27 | 278.2 | 0.83 | 0.57 |

Efficiency comparison (batch=25, single H200):

| Model | Resolution | Inference Time (s)↓ | Peak Memory (GB)↓ | Memory Reduction |
|-------|------------|---------------------|-------------------|-----------------|
| VAR-d24 | 256 | 0.711 | 12.4 | — |
| Markov-VAR-d24 | 256 | 0.608 | 4.7 | **-62.1%** |
| VAR-d24 | 512 | 1.335 | 31.4 | — |
| Markov-VAR-d24 | 512 | 1.261 | 8.1 | **-74.2%** |
| VAR-d24 | 1024 | 5.891 | 117.9 | — |
| Markov-VAR-d24 | 1024 | 5.322 | 19.1 | **-83.8%** |

### Ablation Study

History compensation mechanism (depth-16):

| Method | Params | FID↓ | IS↑ |
|--------|--------|------|-----|
| No history compensation | 300M | 3.64 | 247.7 |
| Global history (full-context compensation) | 324M | 3.41 | 245.2 |
| Mixed history | 359M | 3.45 | 257.4 |
| **Sliding window (Ours)** | 329M | **3.23** | **256.2** |

Sliding window size:

| Window Size | FID(d16)↓ | IS(d16)↑ | FID(d20)↓ | IS(d20)↑ |
|-------------|-----------|----------|-----------|----------|
| 1 | 3.53 | 237.8 | 2.50 | 267.9 |
| 2 | 3.39 | 248.6 | 2.47 | 281.4 |
| **3** | **3.23** | **256.2** | **2.44** | **286.1** |
| 4 | 3.33 | 252.3 | 2.56 | 278.2 |

### Key Findings

1. The d16 model achieves FID improvement from 3.61→3.23 (10.5% gain) and IS improvement from 225.6→256.2 (13.6% gain).
2. At 1024 resolution, peak memory is reduced from 117.9 GB to 19.1 GB (83.8% reduction) without any KV cache.
3. Window size $N=3$ is optimal across all model depths, showing strong consistency between theoretical analysis and empirical results.
4. Favorable scaling laws: both loss and error rate follow power-law decay as model size increases, with $R^2 > 0.99$.
5. Markov-VAR-d20 achieves competitive performance using only ~70% of the parameters of M-VAR-d20.

## Highlights & Insights

1. **Elegant unification of theory and experiment**: The Markov assumption is motivated from the information-theoretic concept of sufficient statistics, with direct empirical support from RFA analysis and perturbation experiments.
2. **Deep validation of "less is more"**: Reducing context dependency actually improves generation quality, as full-context modeling introduces cross-scale interference.
3. **Architecture-level efficiency gains**: The elimination of KV cache is a fundamental advantage whose benefit continues to grow with increasing resolution.
4. **Minimalist design**: The history compensation mechanism requires only a single cross-attention module and one learnable query, adding minimal parameters while yielding significant gains.

## Limitations & Future Work

1. Validation is limited to ImageNet class-conditional generation; performance on more complex tasks such as text-to-image generation remains to be explored.
2. The method relies on the VQ-VAE tokenizer pretrained by VAR; a stronger tokenizer may yield further improvements.
3. A single learnable query may limit the expressiveness of historical information; multi-query or adaptive query designs are worth exploring.
4. Integration with acceleration techniques such as quantization and distillation has not been investigated.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The Markov assumption challenges full-context dependency with a theoretically counterintuitive yet compelling motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage of performance, efficiency, ablation, and scaling laws, with multi-resolution validation and publicly released full model weights.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation analysis is rigorous (RFA and perturbation experiments), figures are well-crafted, and the narrative is logically coherent.
- **Value**: ⭐⭐⭐⭐⭐ — Simultaneously improves both quality and efficiency; the 83.8% memory reduction is of substantial practical significance for high-resolution generation deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](../../ICLR2026/model_compression/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[CVPR 2026\] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyp.md)

</div>

<!-- RELATED:END -->
