---
title: >-
  [Paper Note] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective
description: >-
  [ICLR 2026][Video Generation][Autoregressive] This paper proposes Lumos-1, a unified video generation model built on a standard LLM architecture. It addresses visual spatiotemporal encoding via MM-RoPE (distributed multi…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Autoregressive"
  - "Discrete Diffusion"
  - "RoPE"
  - "Unified Model"
date: 2026-05-08
content_hash: 59df2a2692ab107e
---

# Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective

**Conference**: ICLR 2026
**arXiv**: [2507.08801](https://arxiv.org/abs/2507.08801)  
**Code**: [https://github.com/alibaba-damo-academy/Lumos](https://github.com/alibaba-damo-academy/Lumos)  
**Area**: Diffusion Models / Video Generation
**Keywords**: Autoregressive, Discrete Diffusion, RoPE, Unified Model, Video Generation

## TL;DR

This paper proposes Lumos-1, a unified video generation model built on a standard LLM architecture. It addresses visual spatiotemporal encoding via MM-RoPE (distributed multi-modal RoPE) and inter-frame loss imbalance via AR-DF (autoregressive discrete diffusion forcing). Trained with only 48 GPUs, Lumos-1 achieves competitive performance on GenEval, VBench-I2V, and VBench-T2V.

## Background & Motivation

**Rise of autoregressive video generation**: The remarkable success of LLMs on language tasks has motivated exploration of autoregressive video generation.

**Limitations of existing AR video models**:
- Architectures deviate from standard LLMs (e.g., NOVA, Phenaki)
- Reliance on external text encoders (e.g., LlamaGen, Fluid)
- Extremely low decoding efficiency due to token-by-token generation (e.g., Loong)

**Inadequacy of 1D RoPE for video**: The one-dimensional positional encoding of LLMs cannot model the three-dimensional spatiotemporal correlations in video.

**Spectral imbalance in 3D RoPE**: In naive 3D RoPE, the temporal dimension occupies high-frequency bands while spatial dimensions are compressed toward near-zero frequencies, undermining spatial modeling capacity.

**Loss imbalance in random mask prediction**: Spatial redundancy across frames causes the mask prediction loss on subsequent frames to be far lower than on the first frame, biasing the model toward easier tasks.

**Training efficiency goal**: The work aims to achieve competitive performance under limited resources (48 GPUs) and limited data.

## Method

### Overall Architecture

Lumos-1 is built on the standard Llama architecture. It uses the Cosmos discrete tokenizer ($8 \times 8 \times 4$ compression) to convert video into discrete tokens, with text and visual tokens processed jointly in a unified model. Two core innovations are introduced: **MM-RoPE** for injecting spatiotemporal priors and **AR-DF** for efficient discrete diffusion training and inference.

### Key Design 1: MM-RoPE (Multi-Modal RoPE)

**Problem analysis**: Naive 3D RoPE allocates $d/2$ dimensions to time/height/width in a $2:3:3$ ratio, resulting in:

- Temporal channels occupying high-frequency bands; spatial channels near zero frequency
- Height and width being symmetrically important yet asymmetrically allocated in the spectrum
- Low-index channels rotating too fast causing aliasing; high-index channels rotating too slowly lacking resolution

**MM-RoPE solution**:

- **Distributed frequency allocation**: Channels are divided into multiple meta MM-RoPE components (16 dimensions each), with each component encoding spatiotemporal information internally using the $2:3:3$ ratio
- Each component covers a different frequency range, ensuring both temporal and spatial dimensions are modeled across the full spectrum
- **Text RoPE unchanged**: Visual tokens use MM-RoPE, while text tokens retain the original 1D RoPE of the LLM
- **Scaled 3D positions**: 3D positions in latent space are multiplied by the compression ratio to map back to RGB space, balancing the positional range between text and visual tokens

| RoPE Type | Text Compatible | 3D Structure | Full Spectrum | Scaled Strategy |
|-----------|----------------|--------------|---------------|-----------------|
| M-RoPE | ✔ | ✔ | ✗ | ✗ |
| VideoRoPE | ✔ | ✔ | ✗ | ✔ |
| **MM-RoPE** | **✔** | **✔** | **✔** | **✔** |

### Key Design 2: AR-DF (Autoregressive Discrete Diffusion Forcing)

**Training — temporal tube masking**:

- A mask pattern is sampled for the first frame: $\bm{M} \sim \text{Bernoulli}(1-\rho)$
- The same mask pattern is replicated across all frames: $\widetilde{\bm{X}}_v^{(t)} = \bm{M} \odot \bm{X}_v^{(t)} + (1 - \bm{M}) \odot [\text{MASK}]$
- This prevents subsequent frames from taking a "copy shortcut" by observing unmasked positions in prior frames

**Inference — partial observation caching**:

- Frames are generated autoregressively; after each frame is generated, a fraction $\rho_{\text{inf}}$ of tokens are randomly replaced with [MASK]
- Partially observed frames are stored in the KV cache as conditions for subsequent frames
- This aligns with the training condition where subsequent frames only observe partial prior frames, avoiding inference–training discrepancy and quality degradation

### Loss & Training

Standard cross-entropy loss computed only on masked tokens: $\mathcal{L}(\widehat{\bm{X}}, \bm{X}, \bm{M})$

## Key Experimental Results

### Main Results

#### GenEval (Text-to-Image)

| Model | Params | Training Data | Representation | Overall ↑ |
|-------|--------|---------------|----------------|-----------|
| EMU3 | 8B | - | Discrete | 0.66 |
| Show-o2 | 7B | 66M | Continuous | 0.76 |
| Fluid | 10.5B+4.7B | 680M | Continuous | 0.69 |
| **Lumos-1 (3.6B, 512×512)** | **3.6B** | **60M** | **Discrete** | **0.791** |

#### VBench-I2V

| Model | Params | Video Data | Total ↑ | I2V Score |
|-------|--------|------------|---------|-----------|
| COSMOS | 5B+11B | 100M | 84.16 | 92.51 |
| CogVideoX | 5.6B+4.8B | - | 86.70 | 94.79 |
| VideoMAR | 1.4B+1.5B | 0.5M | 84.82 | 94.02 |
| **Lumos-1 (3.6B)** | **3.6B** | **10M** | **84.72** | **93.34** |

**Key Findings**: Trained with only 48 GPUs, 60M images, and 10M videos, Lumos-1 achieves performance comparable to or better than EMU3 (8B), demonstrating the effectiveness of MM-RoPE and AR-DF.

## Highlights & Insights

1. **Spectral analysis of 3D RoPE**: The first systematic analysis of frequency allocation imbalance in 3D RoPE for video generation, with an elegant distributed solution
2. **Attribution of loss imbalance**: The inter-frame loss imbalance is clearly attributed to spatial information redundancy rather than simple data imbalance
3. **Inference–training consistency**: AR-DF introduces partial observation masking at inference to align with training conditions, offering a clean and effective design
4. **Extremely low training cost**: Competitive performance achieved with only 48 GPUs, validating design efficiency
5. **Pure LLM architecture**: No external text encoder required; a standard Llama architecture directly processes multimodal tokens

## Limitations & Future Work

1. The use of a discrete tokenizer (Cosmos) yields lower reconstruction quality than continuous tokenizers, limiting generation fidelity
2. Training data scale (60M images + 10M videos) is far smaller than commercial models; part of the performance gap is attributable to data
3. The position scaling strategy (multiplying by compression ratio) is heuristic; the authors acknowledge it may not be optimal
4. Video resolution and duration are limited (672×384×25); scalability to long videos or higher resolutions is not validated
5. Inference speed, while faster than next-token decoding, lacks comprehensive latency reporting

## Related Work & Insights

- **Chameleon / EMU3**: Pioneers of unified multimodal LLMs; Lumos-1 builds upon this line with targeted optimization for video generation
- **M-RoPE (Qwen2-VL)**: An early work extending RoPE to 3D; MM-RoPE improves its frequency allocation
- **Diffusion Forcing (Chen et al.)**: The original formulation with bidirectional dependencies does not suffer from loss imbalance; AR-DF targets causal dependency scenarios
- Insight: Frequency allocation is a critical design consideration for any RoPE-based multimodal model; tube masking is generalizable to other AR + mask prediction paradigms

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Both MM-RoPE and AR-DF are original contributions supported by rigorous analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across T2I/I2V/T2V tasks, though absolute performance is constrained by compute budget
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem analysis is thorough, figures are well-designed, and formulations are clear
- Value: ⭐⭐⭐⭐⭐ — Open-source and practically useful; provides an important technical blueprint for unified LLM-based video generation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](../../CVPR2026/video_generation/cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[ICLR 2026\] QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](quantsparse_comprehensively_compressing_video_diffusion_transformer_with_model_q.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](streaming_autoregressive_video_generation_via_diagonal_distillation.md)
- [\[NeurIPS 2025\] Photography Perspective Composition: Towards Aesthetic Perspective Recommendation](../../NeurIPS2025/video_generation/photography_perspective_composition_towards_aesthetic_perspective_recommendation.md)

</div>

<!-- RELATED:END -->
