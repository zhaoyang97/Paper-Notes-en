---
title: >-
  [Paper Note] Breaking the Encoder Barrier for Seamless Video-Language Understanding
description: >-
  [ICCV 2025][Video Understanding][encoder-free] This paper proposes ELVA, the first encoder-free Video Large Language Model (Video-LLM), which achieves performance comparable to encoder-based architectures through hierarchical token merging, video guidance supervision, and hybrid resolution inference, using only 7M publicly available video-text pairs while reducing FLOPs by 95% and inference latency by 92%.
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "encoder-free"
  - "Video-LLM"
  - "token merging"
  - "video guidance"
  - "hybrid resolution"
date: 2026-05-08
content_hash: 5830b201c2a805ba
---

# Breaking the Encoder Barrier for Seamless Video-Language Understanding

**Conference**: ICCV 2025
**arXiv**: [2503.18422](https://arxiv.org/abs/2503.18422)  
**Code**: None  
**Area**: Video Understanding / Video Large Language Models
**Keywords**: encoder-free, Video-LLM, token merging, video guidance, hybrid resolution

## TL;DR

This paper proposes ELVA, the first encoder-free Video Large Language Model (Video-LLM), which achieves performance comparable to encoder-based architectures through hierarchical token merging, video guidance supervision, and hybrid resolution inference, using only 7M publicly available video-text pairs while reducing FLOPs by 95% and inference latency by 92%.

## Background & Motivation

Existing Video-LLMs almost universally adopt an "encoder + decoder" framework (e.g., CLIP encoder + LLM), which suffers from three fundamental limitations:

**Accumulated computational overhead**: Videos require per-frame feature extraction through the visual encoder, with costs scaling linearly with frame count; large encoders (e.g., InternViT-6B) further exacerbate this issue.

**Spatiotemporal resolution constraints**: Encoders impose resolution biases on fixed-size visual representations, preventing dynamic adaptation to content.

**Multimodal interaction bottleneck**: Reliance on pre-extracted features limits low-level interaction between video pixels and text tokens, as well as inter-frame dependency modeling.

Encoder-free approaches have been explored in the image domain (Fuyu, EVE), but the high dimensionality and temporal dependencies of video data introduce additional challenges. ELVA aims to demonstrate that encoder-free Video-LLMs can achieve competitive performance.

## Method

### Overall Architecture

ELVA is built on the Qwen2 LLM backbone and directly processes raw video pixels. Key techniques include: a Native Video Tokenizer that preserves original resolution and aspect ratio, a lightweight video patch embedding layer for spatiotemporal pre-modeling, hierarchical token merging for progressive compression of redundant information, and video guidance supervision for learning spatiotemporal representations.

### Key Designs

1. **Native Video Tokenization**:

    - Video frames are directly partitioned into patches at their original resolution without preprocessing.
    - Special tokens are introduced: `<FRAME>` marks the start of each frame, and `<LINE>` marks the end of each patch row (in raster scan order).
    - Advantage: supports video input at arbitrary resolution and frame length.

2. **Video Patch Embedding Layer**:

    - A lightweight spatiotemporal pre-modeling module with only 9M parameters.
    - Learnable `<LINE>` tokens are appended to each row of patches, and learnable `<FRAME>` tokens are appended to each frame.
    - Long-range spatiotemporal relationships are established via cross-attention: `<FRAME>` tokens query intra-frame embeddings, and `<LINE>` tokens query intra-row embeddings.
    - Compared to naive patch embedding, this yields an average improvement of 2.53% on long-video tasks.

3. **Hierarchical Token Merging**:

    - Redundant tokens along the temporal dimension are progressively merged across different LLM layers.
    - An index matrix $\bm{M} \in \{0,1\}^{T \times (H \cdot W / P^2)}$ is maintained to compute cosine similarity between corresponding token positions in adjacent frames: $s_{ij} = \langle f^l_{ij}, f^l_{(i+1)j} \rangle$.
    - Tokens whose similarity exceeds threshold $\tau=0.6$ are merged by averaging.
    - In shallow layers, tokens are merged immediately upon exceeding the threshold; in deeper layers, merging continues until a target compression ratio (50%) is reached.
    - Compared to direct pooling, this approach preserves critical spatiotemporal information, with far less performance degradation on long-video tasks.

4. **Video Guidance Supervisor**:

    - A pretrained SigLIP video model serves as the teacher.
    - **Tube-wise alignment loss**: The visual features $\mathbf{f}_{\text{vis}}$ from the LLM's final layer are temporally mean-pooled and aligned with teacher features $\mathbf{f}_{\text{target}}$ via MSE: $\mathcal{L}_{\text{MSE}} = \text{MSE}(\frac{\mathbf{f}_{\text{vis}}}{\|\mathbf{f}_{\text{vis}}\|_2}, \frac{\mathbf{f}_{\text{target}}}{\|\mathbf{f}_{\text{target}}\|_2})$
    - **Frame-wise contrastive loss**: `<FRAME>` tokens are retained and frame-level mean-pooled features are used to compute InfoNCE contrastive loss $\mathcal{L}_{\text{Con}}$ across GPUs.
    - Total training loss: $L = L_{\text{Gen}} + L_{\text{MSE}} + L_{\text{Con}}$

### Loss & Training

Three-stage progressive training:

- **Stage 1 — Spatial Pre-training**: Images are treated as single-frame videos; training uses ELVA-Image (4M samples) to learn basic visual information.
- **Stage 2 — Spatiotemporal Pre-training**: ELVA-Video (3M samples) is added; all three loss functions are applied to learn spatiotemporal representations.
- **Stage 3 — Supervised Fine-Tuning (SFT)**: Only the text generation loss is used, with 665K image + 178K video SFT data.

High-quality dense captions re-annotated by Qwen2-VL are extensively used in the training data, yielding substantial improvements over original annotations.

## Key Experimental Results

### Main Results

| Model | Type | LLM | MSVD | ActivityNet | VideoMME | MLVU | CinePile |
|------|------|-----|------|-------------|----------|------|----------|
| Video-LLaVA | encoder | 7B | 70.7 | 45.3 | 39.9 | 47.3 | 22.5 |
| VideoLLaMA2 | encoder | 7B | 70.9 | 50.2 | 46.6 | 48.5 | 44.6 |
| Fuyu | encoder-free | 8B | 56.8 | 28.8 | 28.7 | 31.1 | 26.0 |
| EVE | encoder-free | 7B | 61.4 | 41.8 | 29.3 | 36.8 | 26.4 |
| **ELVA** | **encoder-free** | **7B** | **65.2** | **48.7** | **47.1** | **51.8** | **46.1** |

### Inference Efficiency Comparison (32 frames)

| Model | MEM (G) | FLOPs (T) | TTFT (s) |
|------|---------|-----------|----------|
| Encoder-based | 20.7 | 260 | 2.59 |
| ELVA (no merging) | 20.0 (-3%) | 75 (-71%) | 0.51 (-80%) |
| ELVA + Merge | 16.4 (-21%) | 25 (-90%) | 0.26 (-90%) |
| ELVA + Merge + HR | **15.5 (-25%)** | **14 (-95%)** | **0.22 (-92%)** |

The gains are even more pronounced at 128 frames: FLOPs are reduced by 96%, and TTFT is only 0.56s (vs. 15.18s for encoder-based models).

### Ablation Study

| Pre-training Objective | GQA | SEED_I | MSVD | VideoMME |
|-----------|-----|--------|------|----------|
| $\mathcal{L}_{\text{Gen}}$ only | 42.2 | 40.0 | 45.8 | 37.9 |
| + $\mathcal{L}_{\text{MSE}}$ | 43.6 | 42.6 | 47.1 | 38.1 |
| + $\mathcal{L}_{\text{Con}}$ | 42.4 | 41.0 | 47.4 | 38.5 |
| + Both | **44.4** | **44.8** | **48.0** | **38.5** |

| Data Quality | GQA | MSVD | VideoMME |
|---------|-----|------|----------|
| Original captions | 42.1 | 46.0 | 34.2 |
| Recaptioned Image+Video | **46.1** | **49.4** | **38.5** |

### Key Findings

- For the first time, an encoder-free Video-LLM is demonstrated to achieve performance comparable to encoder-based models.
- A video-pretrained teacher model (video-pretrained SigLIP) provides better guidance than an image-only pretrained encoder (approximately 1-point gain per task).
- High-quality re-annotated captions are critical: they yield 3–4% improvements over original captions across tasks.
- Short-video QA primarily benefits from spatial modeling (Stage 1 gains are rapid), while long-video tasks require spatiotemporal modeling (Stage 2 gains are more pronounced).
- Hybrid resolution inference: maintaining the number of high-resolution frames while adding low-resolution frames substantially improves long-video performance (VideoMME +5.9%) with negligible token overhead.
- Hierarchical merging at a 50% compression ratio incurs almost no accuracy loss while significantly reducing inference cost.

## Highlights & Insights

- **First validation of encoder-free Video-LLMs**: This work challenges the assumption that visual encoders are necessary, demonstrating that the LLM itself can directly learn video representations from pixels.
- **Substantial efficiency gains**: A 95% reduction in FLOPs and 92% reduction in latency make real-time video understanding practically feasible.
- **Elegant hybrid resolution strategy**: The flexibility of the encoder-free architecture is fully exploited by mixing high- and low-resolution frames within the same video.
- **Data quality over data scale**: High-quality re-annotated captions yield greater improvements than increasing data volume.

## Limitations & Future Work

- Training on only 7M samples represents a large data scale gap compared to encoder-based models trained on billions of samples.
- Performance on short-video benchmarks still slightly lags behind the strongest encoder-based models.
- The threshold and compression ratio for hierarchical merging require manual tuning; adaptive strategies warrant further exploration.
- The video guidance teacher model remains an external visual encoder, so the system is not entirely self-contained in principle.

## Related Work & Insights

- Compared to Fuyu (linear projection only) and EVE (image-only pre-training), ELVA's spatiotemporal pre-modeling and video guidance loss are the key differentiating factors.
- Hierarchical token merging shares conceptual similarities with methods such as ToME, but operates across layers inside the LLM, making it better suited for autoregressive generation.
- The hybrid resolution inference strategy is generalizable to other multimodal models that require processing long sequences.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First effective encoder-free Video-LLM with multiple key technical innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 video benchmarks, detailed ablations, and efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Thorough problem analysis with precise identification of three fundamental limitations.
- **Value**: ⭐⭐⭐⭐⭐ — A 95% FLOPs reduction establishes a new efficiency paradigm for video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](../../CVPR2025/video_understanding/video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4d_bench_benchmarking_multimodal_llms_for_4d_object_understanding.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)

</div>

<!-- RELATED:END -->
