---
title: >-
  [Paper Note] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM
description: >-
  [ICCV 2025][Multimodal VLM][VideoLLM] This paper proposes Dynamic-VLM, which employs a dynamic visual token compressor to flexibly adjust the number of tokens per frame according to video length. Combined with a 2-millio…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "VideoLLM"
  - "Visual Token Compression"
  - "Dynamic Compression"
  - "Synthetic Data"
  - "Multimodal"
date: 2026-05-08
content_hash: 0a89453cd93ce0c4
---

# Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM

**Conference**: ICCV 2025
**arXiv**: [2412.09530](https://arxiv.org/abs/2412.09530)
**Code**: None
**Area**: Multimodal Large Models / Video Understanding
**Keywords**: VideoLLM, Visual Token Compression, Dynamic Compression, Synthetic Data, Multimodal

## TL;DR

This paper proposes Dynamic-VLM, which employs a dynamic visual token compressor to flexibly adjust the number of tokens per frame according to video length. Combined with a 2-million-scale high-quality synthetic video QA dataset, the method achieves a 2.7% improvement over LLaVA-OneVision on VideoMME and a 10.7% improvement on MuirBench.

## Background & Motivation

Video large language models face two core challenges:

**Data Gap**: While the image domain benefits from abundant high-quality synthetic data (LLaVA, ShareGPT4V, etc.), fine-tuning data for the video domain still relies heavily on low-quality legacy datasets with a narrow range of task types (e.g., only activity recognition and object counting).

**Architectural Bottleneck**: Existing VideoLLM approaches either:
   - Compress video features into external memory modules → losing frame-level details
   - Naively extend the LLM context window → causing computational explosion and performance degradation
   - Apply fixed compression rates → resulting in information loss for short videos and excessive tokens for long videos

Core problem: **How to strike a balance between the context window and the number of processed frames?** Short videos should preserve more detail (less compression), while long videos should accommodate more frames (higher compression).

## Method

### Overall Architecture

Dynamic-VLM = ViT Visual Encoder + Dynamic Token Compressor + LLM.
- Visual Encoder: CLIP-ViT-Large@336p (default)
- LLM: Qwen-2.5 series (7B/14B)
- Training proceeds in three stages: pre-training → image instruction tuning → video instruction tuning

Given an input video $\mathcal{V} = \{X_0, ..., X_{N-1}\}$, each frame independently passes through the ViT to obtain visual features $F_i \in \mathbb{R}^{(H \times W) \times C}$, which are then compressed from 576 tokens to a target count $M$ via the dynamic compressor.

### Key Designs

1. **Dynamic Visual Token Compressor**: Three candidate schemes are explored:

    - **Dynamic Spatial Pooling**: $\hat{\mathcal{F}} = \text{AdaptiveAvgPool2d}(\mathcal{F}, [H, W])$, with $H=W \in [4, 28]$ and token count $M = H \times W$. **This scheme is ultimately adopted.**
    - **Dynamic Token Merging**: Bipartite soft matching based on ToMe, merging tokens by cosine similarity.
    - **Token Pruning**: Scoring via MLP + Gumbel Softmax, retaining Top-K tokens.

   Dynamic inference strategy: tokens per frame $= \max(N_{max}/T, 576)$, where $N_{max}$ is the maximum visual token budget (7B: 12K; 14B: 10K) and $T$ is the number of video frames. Short videos (fewer frames) → more tokens per frame; long videos (more frames) → fewer tokens per frame.

2. **2-Million-Scale Synthetic Video QA Dataset**: Collected from closed-source models GPT-4V/GPT-4o, with source videos drawn from three datasets:

    - **WebVid-10M** (349k videos): Deduplicated via captions with low-frequency noun downsampling.
    - **InternVid-10M** (547k videos): Videos only (caption quality is low).
    - **HDVILA-100M** (3.3M raw videos): Includes hour-long videos.

   Task design covers five major categories:
    - **Perception Tasks**: Entity recognition, attributes, spatial location, motion description.
    - **General Tasks**: Re-captioning, sentiment analysis, story writing.
    - **Temporal Tasks**: Dense video description, timestamped QA, general temporal QA.
    - **Reasoning Tasks**: Visual reasoning to improve fine-grained understanding.
    - **Formatting Tasks**: Multiple-choice QA format guidance.

3. **Training Pipeline**:

    - Pre-training: On llava-558K, the backbone is first frozen to train only the compressor, followed by end-to-end training on caption data.
    - Image Instruction Tuning: Large-scale public data (General VQA + OCR categories).
    - Video Instruction Tuning: 2M synthetic data + PerceptionTest + NextQA, with a 16K context window.

### Loss & Training

- Standard autoregressive language modeling loss (next token prediction).
- Token count randomly sampled during training: images $M \in [16, 576]$; videos $M \in [16, \min(N_{max}/T, 576)]$.
- System prompt: "You are a helpful visual assistant."
- Learning rate: LLM + compressor $2 \times 10^{-5}$; ViT $4 \times 10^{-6}$ (1/5 ratio).
- Video frames are fed in natural timestamp format: "1s: \<image\>; 2s: \<image\>; ..."

## Key Experimental Results

### Main Results (Tables)

**Multi-choice VideoQA:**

| Method | LLM Size | VideoMME (w/o/w sub) | MLVU | TempCompass | EgoSchema | PerceptionTest |
|--------|---------|---------------------|------|-------------|-----------|----------------|
| GPT-4o | N/A | 71.9/77.2 | 64.6 | 71.0 | - | - |
| LLaVA-OneVision | 7B | 58.2/61.5 | 64.7 | 64.8 | 60.1 | 57.1 |
| LLaVA-OneVision | 72B | 66.2/69.5 | 68.0 | - | 62.0 | 66.9 |
| **Dynamic-VLM** | **7B** | **60.9/64.0** | **65.0** | **62.2** | **68.6** | **68.8** |
| **Dynamic-VLM** | **14B** | **64.6/68.8** | **70.1** | **66.2** | **75.2** | **72.1** |

Dynamic-VLM-7B improves over LLaVA-OneVision-7B by **2.7%** on VideoMME and **8.5%** on EgoSchema. Dynamic-VLM-14B approaches the VideoMME performance of GPT-4o mini.

### Ablation Study (Tables)

**Compressor Architecture Comparison:**

| Compressor | VideoMME (w/o sub) | MSVD-QA Acc/Score |
|------------|-------------------|-------------------|
| Token Merging | 51.6% | 61.9/3.6 |
| Token Pruning | 47.6% | 59.5/3.5 |
| **Pooling** | **52.0%** | **62.0/3.6** |

**Tokens/Frame vs. Max Frames Trade-off (12K token budget):**

| Tokens/Frame | Max Frames | VideoMME |
|-------------|-----------|---------|
| 36 | 333 | 58.7% |
| 64 | 187 | 59.4% |
| **100** | **120** | **60.9%** |
| 144 | 83 | 59.7% |
| 256 | 46 | 59.3% |

The optimal point is 100 tokens/frame, indicating that a moderate compression rate achieves the best balance between information retention and frame coverage.

### Key Findings

- **Pooling is simplest yet most effective**: Adaptive pooling outperforms the other two compressors while being the easiest to implement.
- **100 tokens/frame is the sweet spot**: Under a 12K budget, this corresponds to up to 120 frames, balancing detail and temporal coverage.
- **Remarkable zero-shot multi-image understanding**: Without training on multi-image data, the model surpasses LLaVA-OneVision by 10.7% on MuirBench, reaching 50.7%.
- **Data quality > data quantity**: The carefully prompt-engineered 2M synthetic dataset yields substantial performance gains.

## Highlights & Insights

- **Elegant and concise dynamic strategy**: A single parameter (token budget) automatically determines the compression rate, eliminating the need for per-video manual tuning.
- **Comprehensive synthetic data engineering**: Covers five task categories — perception, general, temporal, reasoning, and formatting — with explicit quality filtering.
- **Smaller model outperforms larger**: Dynamic-VLM-7B surpasses LLaVA-OneVision-72B on multiple tasks.
- **Multi-image generalization**: The video-trained model naturally acquires multi-image understanding ability, demonstrating that video understanding capabilities transfer broadly.
- **Timestamp format**: Temporal information is injected as text in the format "1s: \<image\>", providing explicit temporal cues.

## Limitations & Future Work

- Reliance on closed-source models for training data generation limits reproducibility.
- Adaptive pooling is a hand-crafted compression scheme that does not adapt to scene content.
- A maximum of 256 frames per video limits support for very long videos (e.g., feature-length films).
- Audio modality fusion is not explored.
- The compression rate is fixed at inference time, and content-aware dynamic compression has not been investigated.

## Related Work & Insights

- Shares the same training paradigm as LLaVA-OneVision but extends it more deeply in the video direction.
- The dynamic token concept can be transferred to domains requiring large token counts, such as 3D point clouds and medical imaging.
- The synthetic data engineering methodology offers broadly applicable guidance: deduplication → filtering → multi-task prompt design.

## Rating

- **Novelty**: ⭐⭐⭐ The architectural design is relatively straightforward; the primary contributions lie in the dataset and engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers open-ended, multiple-choice, and zero-shot multi-image tasks with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed comparative analysis.
- **Value**: ⭐⭐⭐⭐ Highly practical, providing a complete training recipe for VideoLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph](dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[CVPR 2026\] DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](../../CVPR2026/multimodal_vlm/dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)
- [\[NeurIPS 2025\] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](../../NeurIPS2025/multimodal_vlm/dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)

</div>

<!-- RELATED:END -->
