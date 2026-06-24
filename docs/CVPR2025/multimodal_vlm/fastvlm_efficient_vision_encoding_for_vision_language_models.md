---
title: >-
  [Paper Note] FastVLM: Efficient Vision Encoding for Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Vision Encoder] Proposes FastViTHD, a hybrid convolution-transformer vision encoder that achieves $32\times$ spatial downsampling through a 5-stage architecture. Under comparable accuracy, it generates $16\times$ fewer vision tokens and achieves a $3.7\times$ faster encoding speed compared to ViT-L/14, reducing TTFT by up to $85\times$.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Vision Encoder"
  - "Hybrid Convolution-Transformer"
  - "Efficient VLM"
  - "Token Compression"
  - "TTFT Optimization"
date: 2026-05-08
content_hash: bab0fd3296f346ac
---

# FastVLM: Efficient Vision Encoding for Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.13303](https://arxiv.org/abs/2412.13303)  
**Code**: [https://github.com/apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)  
**Area**: Multimodal VLM  
**Keywords**: Vision Encoder, Hybrid Convolution-Transformer, Efficient VLM, Token Compression, TTFT Optimization

## TL;DR
Proposes FastViTHD, a hybrid convolution-transformer vision encoder that achieves $32\times$ spatial downsampling through a 5-stage architecture. Under comparable accuracy, it generates $16\times$ fewer vision tokens and achieves a $3.7\times$ faster encoding speed compared to ViT-L/14, reducing TTFT by up to $85\times$.

## Background & Motivation

**Background**: Current mainstream VLMs (e.g., LLaVA) employ ViT as the vision encoder, improving performance on text-intensive tasks by scaling up input resolution. ViT performs $14\times$ or $16\times$ spatial downsampling on images to produce patch tokens, resulting in massive numbers of vision tokens at high resolutions.

**Limitations of Prior Work**: The number of tokens in ViT scales quadratically with resolution—a 336px resolution produces 576 tokens, whereas a 1024px resolution yields thousands. This leads to two critical bottlenecks: (1) high vision encoder latency, and (2) prolonged TTFT caused by the LLM prefilling a vast number of vision tokens. Existing remedies (such as token pruning or dynamic resolution like AnyRes) are after-the-fact mitigations that introduce extra overhead or disrupt semantic continuity.

**Key Challenge**: High resolution is essential for VLM performance (especially for OCR and document understanding), but the ViT architecture is inherently inefficient at processing high-resolution inputs—the accuracy gains from resolution scaling are offset by the exponential increase in latency.

**Goal**: To design an architecturally efficient vision encoder that produces an extremely small number of tokens at high resolutions, fundamentally resolving the trade-off between resolution, latency, and accuracy.

**Key Insight**: Hybrid convolution-transformer architectures possess inherent hierarchical downsampling capabilities, reducing spatial resolution at each stage. By introducing an additional fifth stage, $32\times$ downsampling (instead of the conventional $16\times$) is achieved, allowing self-attention to run on feature maps of extremely low resolution.

**Core Idea**: Replace ViT with a 5-stage hybrid convolution-transformer (FastViTHD) as the VLM vision encoder. This architecture-level token reduction achieves a vastly superior accuracy-efficiency trade-off compared to token pruning.

## Method

### Overall Architecture
Standard LLaVA architecture: FastViTHD Vision Encoder $\rightarrow$ Vision-Language Projector $\rightarrow$ LLM Decoder (Vicuna/Qwen2). Input images are encoded by FastViTHD into a small number of vision tokens (e.g., only 256 tokens for a 1024px image), which are then projected and concatenated with text tokens before being fed into the LLM.

### Key Designs

1. **FastViTHD 5-Stage Architecture**:

    - **Function**: Extracts visual features from images while drastically reducing spatial resolution.
    - **Mechanism**: Processed sequentially through five stages. The first three stages utilize lightweight RepMixer convolutional blocks for local feature extraction and spatial downsampling, while stages 4–5 employ multi-head self-attention for global feature modeling. The stage depths are set to [2, 12, 24, 4, 2], with the embedding dimensions doubling at each stage [96, 192, 384, 768, 1536]. The key innovation lies in Stage 5: it introduces an additional downsampling layer so that self-attention operates on the $32\times$ downsampled feature map (compared to the $16\times$ downsampling in conventional hybrid architectures like ViTamin), directly reducing the token count by $4\times$.
    - **Design Motivation**: A 4-stage design still requires running self-attention on relatively large feature maps at high resolutions, resulting in latencies that can exceed ConvNeXt-L. The extra downsampling in Stage 5, achieved with a minimal parameter increase (totalling 125M, which is still $2.4\times$ smaller than the 304M of ViT-L/14), delivers a fundamental advantage in token count and latency.

2. **Multi-Scale Feature Aggregation**:

    - **Function**: Replenishes low-level, local detailed information lost in high-level features.
    - **Mechanism**: Features from early convolutional stages are pooled to the same spatial dimensions as the final feature map via 2D depthwise separable convolutions (DWConv), and then concatenated with the output of Stage 5. DWConv slightly outperforms AvgPool because it preserves more local structural information.
    - **Design Motivation**: $32\times$ downsampling inevitably discards a large amount of fine-grained information (e.g., character boundaries, small objects). Multi-scale aggregation compensates for this loss.

3. **Static Resolution Outperforms AnyRes Dynamic Resolution**:

    - **Function**: Determines the optimal input resolution processing strategy.
    - **Mechanism**: Directly resizes the input image to target resolutions (e.g., 512/768/1024) and feeds it into the encoder, rather than splitting high-resolution images into tiles, encoding them separately, and then splicing them (as done in LLaVA-NeXT). Experiments demonstrate that static scaling achieves a superior accuracy-latency trade-off at almost all resolutions. AnyRes only shows a marginal advantage at extremely high resolutions ($1536\times 1536+$) with a few tiles ($2\times 2$).
    - **Design Motivation**: Tiling disrupts semantic continuity across tiles and introduces a massive number of extra tokens. FastViTHD inherently supports arbitrary input resolutions without requiring tiling.

### Loss & Training
A 3-stage training pipeline is adopted: Stage 1 freezes both the encoder and LLM, training only the projector (558K alignment data) $\rightarrow$ Stage 1.5 unfreezes the encoder and LLM to perform resolution adaptation on 15M dense captioning data $\rightarrow$ Stage 2 performs full-parameter fine-tuning on 1.1M–23.1M visual instruction tuning data $\rightarrow$ Optional Stage 3 further boosts performance using MammothVL 10.6M CoT reasoning data. CLIP pre-training of FastViTHD utilizes the DataCompDR-1B dataset.

## Key Experimental Results

### Main Results

| Method | Encoder | Resolution | #Token | TTFT(ms) | GQA | TextVQA | DocVQA | Avg |
|------|--------|--------|--------|----------|-----|---------|--------|-----|
| LLaVA-1.5 | ViT-L/14 | 336 | 576 | 127ms+ | 62.0 | 58.2 | 28.1 | 60.1 |
| FastVLM | FastViT | 768 | 576 | 34.5ms | 62.7 | 62.3 | 34.4 | 62.6 |
| FastVLM | FastViTHD | 1024 | 256 | 236ms | 63.1 | 64.4 | 35.6 | 63.9 |
| LLaVA-OV (0.5B) | SigLIP-SO400M | 1152 | 7290 | 14124ms | - | - | 70.0 | - |
| FastVLM (0.5B) | FastViTHD | 1024 | 256 | 166ms | 63.1 | 62.9 | 70.4 | - |
| FastVLM (7B) | FastViTHD+Qwen2 | 1024 | 256 | 641ms | 65.2 | 73.4 | 82.7 | - |

### Ablation Study

| Configuration | GQA | TextVQA | DocVQA | Avg-5 | Description |
|------|-----|---------|--------|-------|------|
| Without Multi-Scale Features | 62.7 | 62.3 | 34.4 | 62.6 | baseline |
| +Multi-Scale (AvgPool) | 63.0 | 62.2 | 35.1 | 62.7 | Slight improvement |
| +Multi-Scale (DWConv) | 63.0 | 62.5 | 34.7 | 62.9 | DWConv is superior |
| ViT+MQT pruning 16 tokens | 57.6 | - | - | - | Poor token pruning performance |
| FastViTHD 256px 16 tokens | 60.6 | 53.1 | - | - | Architectural token reduction is vastly superior to pruning |

### Key Findings
- **Architecture-level token reduction $\gg$ token pruning**: FastViTHD@256 using only 16 tokens out-performs ViT-L/14 + MQT pruning (16 tokens) by 3 percentage points on GQA, proving that designing from the encoder architecture is far more effective than post-hoc pruning.
- **Stage 5 is the key innovation**: A 4-stage design ($16\times$ downsampling) suffers from latencies exceeding ConvNeXt-L at high resolutions; the 5-stage design ($32\times$ downsampling) achieves the Pareto frontier.
- **Diminishing returns of resolution scaling on small LLMs**: Matching high resolution with a 0.5B LLM is inferior to pairing medium resolution with a 7B LLM. Small LLMs cannot effectively exploit large token counts, and their TTFT is dominated by encoder latency.
- **Data scaling continuously improves performance**: Scaling instruction-tuning data from 1.1M to 23.1M continues to yield gains, showing that FastViTHD is not a performance bottleneck.

## Highlights & Insights
- **Resolving token redundancy via architecture is more elegant than token pruning**: While pruning methods on ViTs require complex strategies to select which tokens to keep, FastViTHD naturally produces a small set of high-quality tokens via hierarchical downsampling, avoiding information loss. This approach can be extended to video VLMs to address the temporal token explosion.
- **The finding that static resolution outperforms AnyRes is highly practical**: It reveals that the currently popular AnyRes/tiling strategies are not necessarily optimal; simple resizing works better when paired with an efficient encoder. This challenges the dominant paradigm that high-resolution inputs must be processed via tiling.
- **An $85\times$ TTFT speedup** (vs. LLaVA-OneVision) is of great practical value for on-device VLM deployment.

## Limitations & Future Work
- FastViTHD's CLIP pre-training still relies on traditional contrastive learning paradigms, without exploring newer pre-training methodologies (e.g., SigLIP, EVA-CLIP).
- $32\times$ downsampling inevitably discards extremely fine-grained information, potentially imposing a performance ceiling on tasks requiring pixel-level understanding (e.g., grounding, fine-grained OCR).
- Lack of comparison under identical settings against recent SOTA models that utilize dynamic resolution, such as InternVL and Qwen-VL.
- Whether the hyperparameters of the 5-stage design (depth and dimension of each stage) are optimal remains unexplored; the search space is vast, but the paper only reports a single configuration.

## Related Work & Insights
- **vs. LLaVA-NeXT AnyRes**: AnyRes handles high resolutions by tiling, which introduces massive token counts and high latency. FastVLM demonstrates that an efficient encoder combined with static resolution achieves a much better accuracy-latency trade-off.
- **vs. VisionZip / DynamicLLaVA (token pruning)**: These methods perform token selection/merging after the ViT output, which is less ideal because information is already dispersed across a vast number of tokens during the ViT encoding phase. FastViTHD controls token count from the encoder design stage onward.
- **vs. ConvNeXt-XXL**: ConvNeXt is also an efficient, convolution-only encoder, but FastViTHD@1024 achieves comparable accuracy with $6.8\times$ fewer parameters and $1.7\times$ faster speed. This is due to the hybrid architecture combining the local efficiency of convolutions with the global modeling power of self-attention.

## Rating
- Novelty: ⭐⭐⭐⭐ The 5-stage hybrid architecture design is simple and effective, though the overall framework remains a direct drop-in replacement for LLaVA.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive comparative experiments, covering comparisons with ViT/ConvNeXt/token pruning, varying LLM scales, static vs. dynamic resolution, and data scale ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-organized experiments and highly convincing Pareto analysis curves, though the methodology section is relatively brief.
- Value: ⭐⭐⭐⭐⭐ Directly useful for mobile and real-time VLM deployments; coming from Apple, it carries high engineering credibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[CVPR 2025\] NVILA: Efficient Frontier Visual Language Models](nvila_efficient_frontier_visual_language_models.md)
- [\[ICLR 2026\] Revisiting Multimodal Positional Encoding in Vision-Language Models](../../ICLR2026/multimodal_vlm/revisiting_multimodal_positional_encoding_in_visionlanguage_models.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](../../ECCV2024/multimodal_vlm/brave_broadening_the_visual_encoding_of_vision-language_models.md)
- [\[CVPR 2025\] What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)

</div>

<!-- RELATED:END -->
