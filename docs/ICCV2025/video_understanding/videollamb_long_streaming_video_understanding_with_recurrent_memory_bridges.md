---
title: >-
  [Paper Note] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges
description: >-
  [ICCV 2025][Video Understanding][Long video understanding] VideoLLaMB is proposed, achieving long streaming video understanding with linear GPU memory scaling via SceneTiling semantic segmentation…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Long video understanding"
  - "recurrent memory"
  - "streaming video"
  - "video-language models"
  - "frame retrieval"
date: 2026-05-08
content_hash: 2a52f214aa749f76
---

# VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges

**Conference**: ICCV 2025
**arXiv**: [2409.01071](https://arxiv.org/abs/2409.01071)
**Code**: [https://github.com/bigai-nlco/VideoLLaMB](https://github.com/bigai-nlco/VideoLLaMB)
**Area**: Video Understanding
**Keywords**: Long video understanding, recurrent memory, streaming video, video-language models, frame retrieval

## TL;DR

VideoLLaMB is proposed, achieving long streaming video understanding with linear GPU memory scaling via SceneTiling semantic segmentation, recurrent memory bridge layers, and a memory cache retrieval mechanism, yielding an average improvement of 4.2 points across 4 VideoQA benchmarks.

## Background & Motivation

Large-scale video-language models (e.g., GPT-4o) have demonstrated strong potential for streaming video understanding, yet face the following challenges:

**Computational bottleneck**: The high-dimensional data of long videos is computationally unaffordable for academic researchers.

**Information loss from compression**: Sampling, aggregation, and semantic merging strategies discard critical visual cues.

**Semantic discontinuity from segmentation**: Splitting videos into short clips disrupts semantic flow and impairs holistic understanding.

**Evaluation bias**: Existing benchmarks suffer from static and language biases, failing to comprehensively assess long-video capabilities.

Core motivation: To design an efficient framework that encodes the entire video sequence via a recurrent memory mechanism while preserving semantic continuity, without discarding visual information.

## Method

### Overall Architecture

VideoLLaMB consists of three core modules: (1) a SceneTiling semantic segmenter, (2) recurrent memory bridge layers, and (3) a memory cache retriever. After ViT encoding, the video is segmented by SceneTiling; the recurrent memory layers encode across semantic segments recursively; the memory cache maintains long-range dependencies via retrieval; and the enhanced representations are finally fed into the LLM.

### Key Designs

1. **SceneTiling Semantic Segmentation Algorithm**: A model-free scene segmentation algorithm inspired by TextTiling. It computes cosine similarity between adjacent frame [CLS] tokens, $c_i = S_C(\text{ViT}(v_i), \text{ViT}(v_{i+1}))$, followed by a depth score $d_i = (cl_i + cr_i - 2c_i)/2$. Segmentation is performed using $\mu + \alpha \cdot \sigma$ as the threshold. This algorithm ensures intra-segment semantic consistency and adapts to streaming video captioning without any training.

2. **Recurrent Memory Bridge Layers**: A recurrent memory token is introduced into a single-layer Transformer (Bridge Layer). For each semantic segment $s_i$, memory tokens are prepended as $[m_i; s_i]$, and self-attention yields $[m_{i+1}; o_i] = \text{BridgeLayer}([m_i; s_i])$. Memory tokens are updated by iterating over all semantic segments. This compresses historical video content into memory while preserving per-frame detail via projection.

3. **Memory Cache with Retrieval**: All historical memory tokens $M_i = [m_1, ..., m_i]$ are stored at each timestep $i$. The current memory is updated via a cross-attention self-retrieval mechanism: $m_{i+1} = \text{Softmax}(W_i^Q m_i (W_i^K M_i)^\top / \sqrt{d_k}) W_i^V M_i$, alleviating gradient vanishing and maintaining long-range dependencies.

### Loss & Training

- Training follows the same video data protocol as PLLaVA.
- The LLM backbone is Vicuna-7B-v1.5; the visual backbone is ViT-L/14.
- Both training and evaluation use 16 frames across 4 semantic segments.
- Time complexity is $\mathcal{O}(K^2)$; space complexity is $\mathcal{O}(K)$ (where $K$ is the number of segments), enabling linear GPU memory scaling.

## Key Experimental Results

### Main Results

EgoSchema zero-shot accuracy:

| Model | LLM | Frames | Accuracy |
|-------|-----|--------|----------|
| GPT-4o | OpenAI API | 16 | 72.2 |
| Video-LLaVA | Vicuna-7B | 8 | 40.2 |
| PLLaVA | Vicuna-7B | 16 | 45.6 |
| PLLaVA | Vicuna-7B | 32 | 43.8 |
| **VideoLLaMB** | Vicuna-7B | 32 (trained on 8) | **53.8** |

NExT-QA accuracy comparison:

| Model | Temporal | Causal | Description | All |
|-------|----------|--------|-------------|-----|
| PLLaVA* | 62.2 | 68.5 | 79.7 | 68.2 |
| **VideoLLaMB*** | **66.8** | **71.6** | 78.4 | **71.1** |

### Ablation Study

| Configuration | Key Improvement | Notes |
|---------------|----------------|-------|
| Base linear projection | — | Strong detail retention, weak memory |
| + Resampler | Semantic compression | Strong compression, detail loss |
| + Recurrent memory bridge layers | +4.2 avg | Balances compression and detail |
| + Memory cache retrieval | +Robustness on long videos | Mitigates gradient vanishing |
| + SceneTiling | +Semantic coherence | Training-free streaming captioning |

### Key Findings

- Robust performance is maintained when video length is scaled up to 8× the training length.
- Target frames are accurately retrieved across videos of 1–320 seconds on the NIAVH (Needle in a Video Haystack) benchmark.
- A single A100 GPU can process 320 frames (trained on only 16 frames).
- On the EgoPlan task, VideoLLaMB achieves the best performance among all 7B models, outperforming PLLaVA by 2.06 points.

## Highlights & Insights

- SceneTiling elegantly transfers the TextTiling concept to video segmentation, maintaining semantic consistency without any training.
- The recurrent memory bridge layers are implemented without modifying the visual encoder or LLM architecture, enabling a plug-and-play design.
- Linear memory scaling makes long-video understanding feasible for academic research.
- The NIAVH benchmark fills the gap in frame-level retrieval evaluation.

## Limitations & Future Work

- Based on a 7B model, a notable gap remains compared to large models such as GPT-4o.
- Segmentation quality depends on the representational capability of the ViT [CLS] token.
- The growing memory cache demands more efficient eviction and compression strategies for very long videos.
- Training on a limited number of frames (16) leaves the generalization to ultra-long videos to be further validated.

## Related Work & Insights

- The combination of recurrent memory and retrieval can be generalized to other multimodal tasks requiring long-range dependency modeling.
- The training-free streaming paradigm of SceneTiling has practical value for real-time video understanding.
- The bridge layer concept—balancing projection and compression—is worth adopting in other video-language models.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](../../NeurIPS2025/video_understanding/videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](../../NeurIPS2025/video_understanding/infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](../../ACL2026/video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)

</div>

<!-- RELATED:END -->
