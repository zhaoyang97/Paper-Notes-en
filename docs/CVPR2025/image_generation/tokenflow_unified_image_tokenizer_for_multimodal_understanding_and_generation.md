---
title: >-
  [Paper Note] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
description: >-
  [CVPR 2025][Image Generation][Unified image tokenizer] Proposes TokenFlow, a unified image tokenizer that decouples semantic and pixel-level feature learning through a dual-codebook + shared-mapping architecture, achieving discrete visual inputs outperforming LLaVA-1.5 13B (+7.2%) for the first time while reaching SOTA GenEval of 0.55 in autoregressive generation.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Unified image tokenizer"
  - "dual codebook"
  - "VQ encoding"
  - "multimodal understanding"
  - "autoregressive generation"
date: 2026-05-08
content_hash: e3219bfb5c92df52
---

# TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.03069](https://arxiv.org/abs/2412.03069)  
**Code**: [GitHub](https://github.com/ByteVisionLab/TokenFlow)  
**Area**: Image Generation  
**Keywords**: Unified image tokenizer, dual codebook, VQ encoding, multimodal understanding, autoregressive generation

## TL;DR

Proposes TokenFlow, a unified image tokenizer that decouples semantic and pixel-level feature learning through a dual-codebook + shared-mapping architecture, achieving discrete visual inputs outperforming LLaVA-1.5 13B (+7.2%) for the first time while reaching SOTA GenEval of 0.55 in autoregressive generation.

## Background & Motivation

Multimodal unification (understanding + generation) faces a representation dilemma:
- **Semantic vs. Pixel**: Understanding tasks require high-level semantic representations (e.g., CLIP features), while generation tasks need fine-grained pixel-level information (e.g., VQGAN tokens).
- **Limitations of Prior Work**:
    - Reconstruction-oriented VQ encoders (VQGAN): Perform well in generation but suffer a collapse in understanding (MME-P of only 756 vs. 1461 for CLIP).
    - Semantic-oriented VQ encoders (VQKD/CLIP distillation): Perform decently in understanding but result in severely blurry reconstructions.
    - Janus: Dual encoders increase complexity but do not fundamentally resolve the representation conflict.

Key Insight: If the tokenizer can map patches that are both "semantically similar and pixel-wise similar" to the same index, then the quantized features can be used simultaneously for both understanding and generation.

## Method

### Overall Architecture

TokenFlow adopts a dual-encoder + dual-codebook + shared-index architecture:
1. Semantic encoder $\mathcal{E}_{sem}$ (initialized with CLIP) extracts high-level features.
2. Pixel encoder $\mathcal{E}_{pix}$ extracts low-level features.
3. Dual-codebook shared index mapping: Index selection is done by minimizing the weighted sum of distances.
4. Dual decoders are used for semantic alignment and image reconstruction respectively.
5. Quantized features are concatenated for downstream understanding and generation.

### Key Designs

#### Key Design 1: Dual-Codebook Quantization with Shared Mapping

- **Function**: Encoding both semantic and pixel information in a single index space.
- **Mechanism**: Maintains two codebooks $\mathbf{Z}_{sem} \in \mathbb{R}^{K \times d_{sem}}$ and $\mathbf{Z}_{pix} \in \mathbb{R}^{K \times d_{pix}}$, sharing the codebook size $K$. The quantized index is selected through joint distance minimization: $i^* = \arg\min_i (d_{sem,i} + w_{dis} \cdot d_{pix,i})$
- **Design Motivation**: Single-codebook methods (like VQGAN or VQKD) lose either semantic or pixel details. Dual-codebooks sharing a single index space mean the same index can retrieve both semantic embeddings (for understanding) and pixel embeddings (for reconstruction), achieving "one index, two uses". Visualization confirms that the clustering in TokenFlow reflects both semantic and visual similarities.

#### Key Design 2: Multi-Scale VQ + High Utilization of Large Codebook

- **Function**: Providing rich codebook representation capabilities and high utilization.
- **Mechanism**: Adopts an MSVQ (Multi-Scale VQ) structure for next-scale prediction. Keeps 95%+ utilization even when scaling the codebook size to 131K entries.
- **Design Motivation**: Large codebooks offer more possibilities for semantic-pixel combinations, but traditional VQ methods suffer a rapid drop in utilization under large codebooks. The shared mapping strategy naturally promotes full utilization of codebook entries, as an entry is selected only if it is close in both distances, reducing redundancy and dead codes.

#### Key Design 3: Multi-Step Sampling Inference Strategy

- **Function**: Solving image collapse and repetitive textures caused by single top-k sampling in the next-scale paradigm.
- **Mechanism**: Two-step sampling: First, sample with large $(k_1, p_1)$ to gain diversity, then refine consistency on the same scale using small $(k_2, p_2)$.
- **Design Motivation**: The cross-entropy training objective mainly establishes attention relations with top-1 prediction, where independent top-k sampling might yield unrelated tokens. Progressively narrowing the sampling space maintains creative diversity while strengthening consistency.

### Loss & Training

$\mathcal{L}_{total} = \mathcal{L}_{sem} + \mathcal{L}_{VQ} + \mathcal{L}_{pix}$, where the pixel loss includes $\ell_2$ reconstruction, LPIPS perceptual loss, and GAN adversarial loss.

## Key Experimental Results

### Main Results 1: Multimodal Understanding (LLaVA-1.5 Framework)

| Method | Type | MME-P↑ | SEED-B↑ | TextVQA↑ |
|------|------|--------|---------|----------|
| CLIP ViT-B/14 (Cont.) | Sem. | 1460.9 | 64.1 | 53.4 |
| VQGAN | Pix. | 756.1 | 38.2 | 46.8 |
| VQKD | Sem. | 1252.4 | 57.8 | 48.2 |
| **TokenFlow** | **Unified** | **Outperforms LLaVA-1.5 13B** | **+7.2% Avg.** | - |

Demonstrates for the first time that discrete visual inputs can outperform continuous CLIP in understanding tasks.

### Main Results 2: Image Reconstruction and Generation

| Metric | Value |
|------|------|
| rFID@384×384 | **0.63** |
| GenEval@256×256 | **0.55** (Autoregressive SOTA) |
| Codebook Utilization | **95%+** (131K entries) |

### Key Findings

- Understanding only requires features of the final scale (residuals/all scales tend to introduce noise instead).
- Scaling up the codebook size consistently improves understanding and generation performance (unique to TokenFlow).
- Multi-step sampling is significantly superior to single-step top-k sampling.
- Training requires less than 24 hours on 8×A100 GPUs (for the understanding fine-tuning part).

## Highlights & Insights

1. **One index, two uses**: Shared mapping serves as an elegant bridge connecting understanding and generation.
2. **Feasibility of large codebooks**: 95%+ utilization at a scale of 131K is unprecedented, demonstrating that joint distance selection naturally prevents codebook collapse.
3. **Discrete outperforming continuous for the first time**: Shatters the conventional belief that "discrete tokens are inherently weaker than continuous features in understanding".

## Limitations & Future Work

- Generating resolution is limited to 256×256; high-resolution generation capabilities remain to be validated.
- The parameter size and complexity of having a dual encoder, dual codebook, and dual decoder are relatively high.
- The semantic encoder relies on CLIP initialization, which may limit generalization in domains poorly covered by CLIP.
- Unification of video understanding and generation has not yet been explored.

## Related Work & Insights

- **Chameleon / EMU3**: Unified approaches using a single VQ tokenizer, resulting in limited understanding.
- **Janus**: Dual encoders but without shared mapping; high complexity and does not fundamentally resolve representation conflicts.
- **LlamaGen**: Baseline for autoregressive image generation, which TokenFlow outperforms on GenEval.

## Rating

⭐⭐⭐⭐⭐ — The dual-codebook shared mapping is an ingenious design, and achieving discrete visual performance that outperforms LLaVA-1.5 13B for the first time is a milestone. It features clear theoretical motivation, comprehensive experiments, and dual SOTA performance in both generation and understanding. The resolution limitation is the primary area for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](../../NeurIPS2025/image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2025\] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)

</div>

<!-- RELATED:END -->
