---
title: >-
  [Paper Note] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models
description: >-
  [CVPR 2025][Video Understanding][Encoder-free video-language models] Video-Panda proposes the first encoder-free video-language model, which directly processes video inputs via a Spatio-Temporal Alignment Block (STAB) with only 45M parameters. It achieves performance comparable to methods using 300M-1.4B parameter encoders on open-ended video QA tasks, while delivering a 3-4x speedup in inference.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Encoder-free video-language models"
  - "spatio-temporal alignment"
  - "parameter-efficient"
  - "video question answering"
  - "lightweight"
date: 2026-05-08
content_hash: 76916f9539cccafd
---

# Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.18609](https://arxiv.org/abs/2412.18609)  
**Code**: [https://jh-yi.github.io/Video-Panda](https://jh-yi.github.io/Video-Panda)  
**Area**: Video Understanding  
**Keywords**: Encoder-free video-language models, spatio-temporal alignment, parameter-efficient, video question answering, lightweight

## TL;DR
Video-Panda proposes the first encoder-free video-language model, which directly processes video inputs via a Spatio-Temporal Alignment Block (STAB) with only 45M parameters. It achieves performance comparable to methods using 300M-1.4B parameter encoders on open-ended video QA tasks, while delivering a 3-4x speedup in inference.

## Background & Motivation

**Background**: Current video-language models (e.g., Video-ChatGPT, Video-LLaVA) typically rely on pre-trained large visual encoders (300M-1.4B parameters) to extract frame-level features, then map these visual features to the LLM's embedding space through an alignment module. Some methods even utilize both image and video encoders.

**Limitations of Prior Work**: Heavyweight encoders incur massive computational overhead. Processing multi-frame videos requires repeatedly passing frames through the encoder, leading to significant latency. Furthermore, simply adapting image-language model architectures for video understanding results in performance degradation, as they fail to capture video-specific spatio-temporal relationships.

**Key Challenge**: Video understanding requires rich spatio-temporal modeling capabilities but also faces heavy computational pressure from multi-frame processing. Existing methods address the former by stacking larger encoders, which in turn exacerbates the latter.

**Goal**: Design an extremely lightweight encoder-free architecture that directly processes raw video pixels, performing visual processing with fewer than 50M parameters while maintaining competitive video QA performance against encoder-based methods.

**Key Insight**: The authors observe that encoder-free methods in the image domain (e.g., Fuyu-8B, EVE) have demonstrated the feasibility of bypassing pre-trained encoders. However, simply extending these methods to video fails due to a lack of spatio-temporal modeling. Therefore, an alignment module specifically designed for videos is required.

**Core Idea**: Replace the traditional visual encoder with a dedicated Spatio-Temporal Alignment Block (STAB). By explicitly utilizing local/global spatio-temporal modeling and frame-level spatial modeling, video-language alignment is achieved with an extraordinarily small parameter footprint.

## Method

### Overall Architecture
The input video is uniformly sampled into 8 frames. Each frame is divided into patches and directly processed by the STAB module without passing through any pre-trained encoder. STAB internally consists of four sub-modules: Local Spatio-Temporal Encoding (LSTE), Local Spatial Downsampling (LSD), Frame-level Spatial relation Aggregation (FSRA), and Global Spatio-Temporal relation Aggregation (GSTRA). The processed visual tokens are mapped to the embedding space of the LLM (Vicuna-7B) via an MLP, and combined with text tokens to perform video QA. The training process consists of three stages: initial alignment, joint vision-language training, and instruction tuning.

### Key Designs

1. **Local Spatio-Temporal Encoding (LSTE) + Dynamic Position Encoding**:

    - **Function**: Extract fine-grained features and encode positional information within local spatio-temporal windows.
    - **Mechanism**: Use three cascaded 3D convolutions to process patch embeddings. Conv3D$_1$ and Conv3D$_3$ use $1\times1\times1$ kernels for channel compression/restoration, while Conv3D$_2$ uses a $3\times1\times1$ kernel to model temporal contexts, completed with residual connections. Then, dynamic position encoding (DPE) is implemented via a $3\times3\times3$ depthwise 3D convolution, computed as $L_{st} = L'_{st} + \text{DPE}(L'_{st})$.
    - **Design Motivation**: Videos exhibit rich temporal dynamics in local neighborhoods. 3D convolutions can efficiently capture these patterns, and DPE provides structure-aware positional information that is more flexible than fixed positional encodings.

2. **Spatial Downsampling + Dual-Path Frame-level/Global Spatio-Temporal Aggregation**:

    - **Function**: Compress spatial resolution and model frame-level content and video-level context respectively.
    - **Mechanism**: LSD downsamples features using learnable queries within a $2\times2$ window to halve the spatial dimensions. Then, FSRA uses frame-specific learnable queries to perform global spatial aggregation on each frame, yielding a frame summary $F_{s,t}$. Concurrently, GSTRA uses global queries to perform cross-spatiotemporal aggregation on all frame tokens, producing video-level context $G_{st}$. Finally, they are fused via a learnable weight $\alpha$: $F_{r,t} = f_{\text{proj}}(\alpha F_{s,t} + (1-\alpha)G_{st})$.
    - **Design Motivation**: Spatial content varies greatly across different frames, and FSRA preserves frame specificity. Meanwhile, video understanding requires global context (e.g., overall event comprehension), which GSTRA provides. Ablation studies show they are complementary; removing GSTRA drops performance on ActivityNet-QA by 1.8 points.

3. **Three-Stage Progressive Training Strategy**:

    - **Function**: Efficiently optimize and enable knowledge transfer.
    - **Mechanism**: Stage 1 (351K data) freezes the LLM and only trains the STAB, using a distillation loss with LanguageBind as the teacher: $\mathcal{L}_{\text{distill}} = -\frac{1}{M}\sum_i \frac{v_{\text{pred},i}^\top v_{\text{teacher},i}}{\|v_{\text{pred},i}\|\|v_{\text{teacher},i}\|}$ plus text cross-entropy. Stage 2 (702K data) performs end-to-end fine-tuning. Stage 3 (100K instruction data) enhances instruction-following capability.
    - **Design Motivation**: Direct end-to-end training can lead to collapse. Aligning visual representations first before gradually unlocking LLM parameters is more stable.

### Loss & Training
The training objective combines visual distillation loss (cosine similarity) and standard cross-entropy loss for text generation. The teacher model is LanguageBind (a contrastive-learning model pre-trained on ViT-L/14), with the input resized to 224x224.

## Key Experimental Results

### Main Results

| Method | Visual Params | MSVD-QA Acc/Score | MSRVTT-QA Acc/Score | TGIF-QA Acc/Score | ActivityNet-QA Acc/Score |
|------|-----------|-------------------|---------------------|-------------------|------------------------|
| Video-ChatGPT | 307M | 64.9/3.3 | 49.3/2.8 | 40.7/3.1 | 35.2/2.8 |
| Video-LLaVA | 425M | 64.8/- | 58.3/- | 41.7/- | 40.7/- |
| **Video-Panda** | **45M** | **64.7/3.8** | **54.8/3.4** | **42.9/3.2** | **40.0/3.3** |

### Ablation Study

| Configuration | MSVD-QA Acc/Score | ActivityNet-QA Acc/Score |
|------|-------------------|------------------------|
| w/o LSTE | 63.6/3.7 | 39.4/3.3 |
| w/o GSTRA | 63.0/3.7 | 38.2/3.2 |
| w/o GSTRA & LSTE | 62.2/3.7 | 38.1/3.2 |
| w/o LSD (avg pool) | 58.0/3.6 | 38.1/3.2 |
| **Video-Panda** | **64.7/3.8** | **40.0/3.3** |

### Key Findings
- LSD (attention-based downsampling) is the most critical component. Replacing it with average pooling causes MSVD-QA accuracy to plummet from 64.7 to 58.0, indicating that learnable spatial aggregation is far superior to simple pooling.
- Removing GSTRA has the largest impact on ActivityNet-QA (long videos) (-1.8), as long videos rely heavily on global context.
- Computational efficiency: Visual part FLOPs is only 105.5G (1/77 of Video-ChatGPT), and inference latency is 41ms vs 171ms (a 4.2x speedup).
- In fine-grained evaluation, Video-Panda outperforms Video-ChatGPT in both correctness (2.74 vs 2.40) and temporal understanding (2.26 vs 1.98).

## Highlights & Insights
- **Extreme parameter efficiency**: Achieving performance comparable to 307M-425M model sets using only 45M parameters proves that video understanding does not strictly require massive pre-trained encoders; the key lies in the design of spatio-temporal modeling.
- **Complementarity of dual-path aggregation**: Separating and then fusing frame-level (spatial) and video-level (spatio-temporal) representations is an elegant design. The learnable fusion weight automatically balances local details and global semantics.
- **Distillation-driven encoder-free training**: Using a pre-trained encoder as a teacher for distillation is an effective path for encoder-free methods to achieve robust visual representations. This paradigm can be transferred to other modalities.

## Limitations & Future Work
- The model still relies on LanguageBind as a distillation teacher, failing to completely bypass dependency on pre-trained visual models.
- Sampling only 8 frames may be insufficient for long videos or tasks that require fine-grained temporal reasoning.
- Resolution is limited to 448x448, which may limit performance on tasks requiring high-resolution visual details.
- Future work can explore self-distillation or teacher-free training paradigms to further reduce reliance on pre-trained models.

## Related Work & Insights
- **vs Video-ChatGPT**: The core difference lies in removing the ViT encoder and replacing it with STAB. It reduces parameters by 6.8x while achieving superior performance.
- **vs EVE**: EVE is a pioneer of encoder-free models in the image domain, but simply scaling it to video yields poor results (MSVD-QA 60.5 vs 64.7), highlighting the necessity of dedicated spatio-temporal modeling for video.
- This encoder-free + distillation paradigm can be extended to other multi-modal scenarios, such as audio-language or point-cloud-language models.

## Rating
- Novelty: ⭐⭐⭐⭐ The first encoder-free video-language model, with a well-designed STAB.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, fine-grained evaluations, detailed ablation study, and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Opens up a new direction for lightweight video-language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[ICCV 2025\] Breaking the Encoder Barrier for Seamless Video-Language Understanding](../../ICCV2025/video_understanding/breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)

</div>

<!-- RELATED:END -->
