---
title: >-
  [Paper Note] CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models
description: >-
  [ICML 2026][Multimodal VLM][3D Generation] CG-MLLM proposes a Mixture-of-Transformer-based multi-modal large language model. By utilizing a dual Transformer architecture consisting of TokenAR (per-token autoregressive) a…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "3D Generation"
  - "Multi-modal Large Language Model"
  - "Mixture-of-Transformer"
  - "Spatial Intelligence"
  - "3D Understanding"
date: 2026-05-08
content_hash: 66ddafc564e7e027
---

# CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2601.21798](https://arxiv.org/abs/2601.21798)  
**Code**: To be confirmed  
**Area**: Multi-modal VLM / 3D Vision  
**Keywords**: 3D Generation, Multi-modal Large Language Model, Mixture-of-Transformer, Spatial Intelligence, 3D Understanding  

## TL;DR
CG-MLLM proposes a Mixture-of-Transformer-based multi-modal large language model. By utilizing a dual Transformer architecture consisting of TokenAR (per-token autoregressive) and BlockAR (block-level parallel) branches, combined with a pre-trained VLM backbone and a 3D VAE latent space, it achieves end-to-end high-resolution 3D content generation and 3D captioning within a single MLLM framework for the first time, reaching SOTA performance among MLLM-based 3D generation methods.

## Background & Motivation

**Background**: Large language models have made breakthrough progress in modalities such as text, image, and video, with many MLLMs performing exceptionally well in 2D vision-language understanding and generation tasks. However, progress in the field of 3D content generation has been slow, showing a significant gap compared to 2D multi-modal generation.

**Limitations of Prior Work**: Current MLLMs for 3D generation mainly follow two routes: (1) generating meshes as text/discrete tokens, where the token budget limits the complexity and resolution of the mesh; (2) using low-resolution voxel VAEs or Lego-like structures to generate coarse 3D proxy shapes, which still require additional 3D diffusion models for refined geometry. Neither can generate high-resolution 3D objects end-to-end within the LLM stage.

**Key Challenge**: 3D geometry inherently forms long-range, highly interdependent sequences. Pure token-level autoregressive modeling leads to severe efficiency issues, while existing MoT methods bind Transformers by task (understanding vs. generation), which lacks flexibility.

**Goal**: To build a unified language-image-3D multi-modal large language model that simultaneously achieves precise spatial understanding and high-fidelity spatial content generation within a single model.

**Key Insight**: The authors observe that token-level serial modeling and block-level parallel modeling can be decoupled into different Transformer branches, bound by generation mode (serial vs. parallel) rather than by task, allowing for flexible integration of various pre-trained encoders.

**Core Idea**: Use a dual-Transformer MoT architecture (TokenAR + BlockAR) to integrate a pre-trained Qwen3-VL backbone with the Hunyuan3D-2.1 VAE latent space, enabling native high-resolution 3D generation within an MLLM.

## Method

### Overall Architecture
CG-MLLM adopts a decoder-only architecture consisting of three stages: (1) **Multi-modal Encoding**—text uses a BBPE tokenizer, images are compressed via a SigLIP-2 encoder and a 2-layer MLP, and 3D assets are encoded into latent representations via a frozen Hunyuan3D-2.1 Spatial-VAE; (2) **MoT Modeling**—the TokenAR Transformer handles token-level sequence modeling, and the BlockAR Transformer handles block-level parallel modeling, with both sharing attention mechanisms; (3) **Multi-modal Decoding**—text tokens are decoded via the tokenizer, while 3D tokens are restored to meshes via the VAE decoder and enhanced for visual quality through a material generator.

### Key Designs

1. **Dual Transformer MoT Architecture (TokenAR + BlockAR)**:

    - **Function**: Decouples serial modeling from parallel modeling, enabling the model to possess both token-level language/vision understanding and block-level 3D spatial generation capabilities.
    - **Mechanism**: Both TokenAR and BlockAR are initialized from pre-trained Qwen3-VL weights. TokenAR maintains original token-level autoregressive capabilities; BlockAR performs block-level parallel prediction on 3D latent tokens, sharing position indices within each block to maintain permutation invariance of point features. A hybrid masking mechanism is employed—causal masks for sequential tokens and parallel masks for tokens within the same block, combined adaptively in the attention layers.
    - **Design Motivation**: Unlike task-based binding (understanding vs. generation), mode-based binding allows flexible integration with any encoder and achieves approximately $3\times$ speedup at a 4096-token resolution via block-level parallelism.

2. **3D Spatial-VAE Integration and Positional Encoding Strategy**:

    - **Function**: Encodes 3D objects into a high-dimensional latent space and aligns them with the VLM semantic space.
    - **Mechanism**: Utilizes Hunyuan3D-2.1's Spatial-VAE (downsampling factor of 20, latent dimension of 64) to extract point clouds from 3D object surfaces and encode them into latent representations, aligned with LLM hidden dimensions through a Connector layer. Positional embeddings within blocks are intentionally omitted for 3D tokens, assigning only block-level position indices to preserve the permutation invariance of point features while maintaining global spatial structure. The VAE is frozen throughout training.
    - **Design Motivation**: Reusing mature 3D VAE geometric priors avoids the high cost of training a 3D encoder from scratch; the positional encoding strategy ensures that the unordered nature of point cloud features is not disrupted by positional information.

3. **Progressive Resolution Training Strategy**:

    - **Function**: Gradually increases 3D generation resolution in two stages to stabilize the training process.
    - **Mechanism**: In the first stage (alignment stage), 90% of conditional inputs are dropped to train unconditional generation and initial understanding at a 512-token resolution. In the second stage (progressive resolution stage), the resolution is gradually increased from 512 to 4096 tokens, while the dropout probability is reduced from 90% to 10%. An AdamW optimizer is used, with the learning rate adjusted from $1 \times 10^{-4}$ to $5 \times 10^{-5}$.
    - **Design Motivation**: Directly training high-resolution 3D tokens (4096) places excessive pressure on LLM sequence length and memory; a progressive strategy allows the model to master coarse-grained structures before refining geometric details.

### Loss & Training
Classifier-Free Guidance (CFG) is used, with a CFG scale of 7.5 and 50 sampling steps during inference. A logit-normal sampler is employed for timesteps. Training is conducted on 16 NVIDIA H20 GPUs, with the maximum sequence length increased from 36,864 to 51,200.

## Key Experimental Results

### Main Results: 3D Generation Quality Comparison

| Method | Type | p-FID↓ | p-KID↓ | CLIP-IQA+↑ | MUSIQ↑ | CLIP↑ | User Study↑ |
|------|------|--------|--------|------------|--------|-------|------------|
| Michelangelo | Non-MLLM | 17.96 | 0.56 | 0.45 | 71.42 | 84.08 | 2.60 |
| CraftsMan | Non-MLLM | 14.09 | 0.40 | 0.45 | 71.09 | 84.86 | 3.15 |
| TRELLIS | Non-MLLM | 7.36 | 0.12 | 0.44 | 66.97 | 84.13 | 3.28 |
| SAR3D | MLLM | 30.07 | 1.00 | 0.42 | 66.01 | 82.86 | 2.93 |
| ShapeLLM-Omni | MLLM | 13.11 | 0.29 | 0.37 | 55.71 | 84.18 | 2.30 |
| **Ours (CG-MLLM)** | MLLM | **12.55** | **0.27** | **0.45** | **71.65** | **84.47** | **3.32** |

CG-MLLM leads across all metrics among MLLM-based methods, with p-FID reduced by 58% and p-KID by 73% compared to SAR3D.

### Ablation Study

| HY2.1-VAE | MoT | LLM Backbone | #Tokens | p-FID↓ | p-KID↓ |
|-----------|-----|--------------|---------|--------|--------|
| ✗ | ✗ | Qwen2.5-0.5B | 512 | 53.66 | 1.76 |
| ✓ | ✗ | Qwen2.5-0.5B | 512 | 44.91 | 1.42 |
| ✓ | ✓ | Qwen2.5-0.5B | 512 | 30.60 | 0.77 |
| ✓ | ✓ | Qwen3VL-2B | 512 | 15.61 | 0.43 |
| ✓ | ✓ | Qwen2.5-0.5B | 4096 | 16.57 | 0.53 |
| ✓ | ✓ | Qwen3VL-2B | 4096 | **12.55** | **0.27** |

The HY2.1-VAE, MoT architecture, larger token budgets, and stronger VLM backbones all yield consistent gains, aligning with scaling law trends.

### Main Results: 3D Captioning Comparison

| Model | Input | BLEU-1↑ | ROUGE-L↑ | METEOR↑ |
|------|------|---------|----------|---------|
| 3D-LLM | 3D Latent | 16.91 | 19.48 | 19.73 |
| ShapeLLM-Omni-7B | 3D Latent | 18.51 | 21.37 | 19.89 |
| Qwen3-VL-2B | Image | 3.13 | 7.21 | 11.92 |
| **CG-MLLM-2B (Ours)** | **Image** | **13.51** | **19.13** | **14.28** |

Under image-only input conditions, CG-MLLM's captioning ability significantly outperforms the same-sized Qwen3-VL (BLEU-1 increased by $4.3\times$), proving that 3D generation training can feed back into perceptual capabilities.

## Highlights & Insights
- **Generation Feeds Understanding**: Joint 3D generation training not only grants the model generative capabilities but also significantly enhances 3D structural reasoning based on 2D images, validating the hypothesis that "learning to generate helps in understanding."
- **Mode-based vs. Task-based Binding**: Binding Transformers by generation mode (serial/parallel) rather than task (understanding/generation) is a simple yet crucial design choice that maintains architectural scalability.
- **Failure of AdaLN in MLLM**: The authors found that AdaLN introduces extra scaling factors in shared causal-parallel attention mechanisms, which disrupts training stability. This provides a useful reference for future MLLM+Diffusion work.

## Limitations & Future Work
- The overall quality has not yet surpassed top non-MLLM methods (e.g., TRELLIS); narrowing this gap remains an open problem.
- The quality of 3D caption datasets is limited (usually < 20 words), restricting 3D understanding capabilities.
- The watertight preprocessing of Hunyuan3D-2.1 VAE causes loss in data precision, and the token count is only 4K (high-quality methods can reach 40K+).
- Hallucinations may occur during input ambiguity or semantic confusion (e.g., generating a rabbit when the input is a sheep).

## Related Work & Insights
- **SAR3D / ShapeLLM-Omni**: Previous MLLM 3D generation methods using tokens and voxel VAEs, respectively; CG-MLLM outperforms them on all metrics.
- **TRELLIS**: A non-MLLM 3D generation SOTA; its p-FID of 7.36 is still lower than CG-MLLM's, indicating that the pure LLM paradigm still has a gap in 3D precision.
- **Mixture-of-Transformers**: The MoT concept is reinterpreted as mode-based rather than task-based binding.

## Rating
- Novelty: ★★★★☆ — The design of dual Transformers bound by generation mode is novel, and the exploration of 3D MLLM is valuable.
- Experimental Thoroughness: ★★★★☆ — Comprehensive ablation (5 groups), though a gap remains with non-MLLM SOTAs.
- Writing Quality: ★★★☆☆ — Methodology is clearly described, but some paragraphs are lengthy.
- Value: ★★★★☆ — The first end-to-end high-resolution 3D generation MLLM, opening a new direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/multimodal_vlm/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)

</div>

<!-- RELATED:END -->
