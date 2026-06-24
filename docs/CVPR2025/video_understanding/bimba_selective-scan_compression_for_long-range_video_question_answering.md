---
title: >-
  [Paper Note] BIMBA: Selective-Scan Compression for Long-Range Video Question Answering
description: >-
  [CVPR 2025][Video Understanding][State Space Models] This paper proposes BIMBA, a spatiotemporal token selector based on Mamba selective scan. It compresses long video sequences of over 100K tokens by 16 times down to 6,400 key tokens, achieving state-of-the-art (SOTA) performance across 7 long-video VQA benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "State Space Models"
  - "Mamba"
  - "Video Compression"
  - "Long-Video QA"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: b0a6b3cd28af83af
---

# BIMBA: Selective-Scan Compression for Long-Range Video Question Answering

**Conference**: CVPR 2025  
**arXiv**: [2503.09590](https://arxiv.org/abs/2503.09590)  
**Code**: [https://sites.google.com/view/bimba-mllm](https://sites.google.com/view/bimba-mllm)  
**Area**: Video Understanding / Long-range Video Question Answering  
**Keywords**: State Space Models, Mamba, Video Compression, Long-Video QA, Multimodal Large Language Models

## TL;DR
This paper proposes BIMBA, a spatiotemporal token selector based on Mamba selective scan. It compresses long video sequences of over 100K tokens by 16 times down to 6,400 key tokens, achieving state-of-the-art (SOTA) performance across 7 long-video VQA benchmarks.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have made significant progress in video understanding, but handling long videos (minutes to hours) remains a core challenge. For instance, with LLaMA-3.2, the image encoder outputs 1,600 to 6,400 tokens per frame, meaning 128 frames will yield 200K to 800K tokens, far exceeding the context window and compute capacity of LLMs.

**Limitations of Prior Work**: (1) Spatial/temporal pooling discards crucial spatiotemporal information; (2) Convolution-based compression lacks the capability to model long-range dependencies; (3) The computational cost of self-attention grows quadratically with sequence length, making it infeasible for long sequences; (4) Perceiver/Q-former-based compression is efficient but lacks cross-frame analysis.

**Key Challenge**: Massive amounts of redundancy exist across long videos, yet pivotal events can occur in a split second. A mechanism is required to selectively retain critical information while substantially compressing the sequence length—demanding both long-range modeling capability and computational efficiency.

**Goal**: To design an efficient long-video token compression module that condenses 100K-level token sequences to thousand-level length while preserving crucial spatiotemporal dependencies.

**Key Insight**: The selective scan mechanism in Mamba (S6) perfectly satisfies these requirements, offering linear computational complexity, input-dependent selective retention, and natural long-range modeling capabilities.

**Core Idea**: Utilizing Mamba's selective scan as a spatiotemporal token selector, combined with interleaved query distribution and bidirectional scanning to match the spatiotemporal structure of videos, achieving 16x compression with even improved accuracy.

## Method

### Overall Architecture
Video frames $\rightarrow$ Pre-trained image encoder $\rightarrow$ Spatiotemporal token sequence (64 frames $\times$ 40 $\times$ 40 = 102,400 tokens) $\rightarrow$ BIMBA spatiotemporal token selector $\rightarrow$ Compressed tokens (16 $\times$ 20 $\times$ 20 = 6,400) $\rightarrow$ LLM decoder to generate the answer.

### Key Designs

1. **Spatiotemporal Token Selector**:

    - **Function**: Compresses a massive number of redundant video tokens into a small, information-dense set of query tokens.
    - **Mechanism**: (a) Initializes a small number of visual queries $Q$ using 3D adaptive average pooling (reducing from $L$ input tokens to $N$ queries, where $N \dots N \ll L$); (b) Concatenates queries with the spatiotemporal tokens to form the sequence $Z' = [Z; Q]$; (c) Applies Mamba selective scan layers, allowing queries to "absorb" key info from the massive token sequence via the selective State Space Model; (d) Extracts the updated queries $Q'$ and feeds them into the LLM.
    - **Design Motivation**: Unlike pooling, selective scan dynamically decides what to preserve or discard based on the input content, which is highly effective for highly redundant video data. Unlike self-attention, its computational complexity scales linearly.

2. **Interleaved Queries**:

    - **Function**: Prevents query positional bias.
    - **Mechanism**: Traditional approaches place queries at the end of the sequence, causing a bias where queries favor tokens from later parts of the video. This work uniformly interleaves queries with the spatiotemporal tokens, allowing queries to interact evenly with all parts of the video.
    - **Design Motivation**: Effectively eliminates positional bias, ensuring a more balanced transfer of spatiotemporal information to the query tokens.

3. **Bidirectional Selective Scan**:

    - **Function**: Enhances the capability to capture 2D/3D spatiotemporal structures.
    - **Mechanism**: The original Mamba is designed for 1D NLP sequences and is less sensitive to the spatial structure in vision tasks. This work adopts bidirectional (forward + backward) scanning, allowing the model to model spatiotemporal dependencies from both directions.
    - **Design Motivation**: Bidirectional scanning has been proven effective in Vision Mamba, enabling better capture of spatial and temporal structures.

### Loss & Training
Standard language modeling autoregressive loss. The image encoder is frozen, while the token selector and the LLM (via LoRA) are trained. Optional: Question-conditioned token selection (prepending question tokens to allow the selector to refer to the context of the question).

## Key Experimental Results

### Main Results
Achieved SOTA across 7 long-video VQA benchmarks:

| Benchmark | BIMBA-LLaMA | Prev. SOTA |
|------|-------------|----------|
| PerceptionTest | SOTA | - |
| NExT-QA | 76.61 | Lower than Ours |
| EgoSchema | 62.20+ | Lower than Ours |
| VNBench | SOTA | - |
| LongVideoBench | SOTA | - |
| Video-MME | SOTA | - |
| MLVU | SOTA | - |

### Ablation Study
Ablation study on the NExT-QA dataset (using LLaMA variants):

| Configuration | Accuracy |
|------|--------|
| Average Pooling Initialization + LN + Bidirectional Scan + Interleaved Queries (Full) | 75.57 |
| Learned Initialization (without pooling) | 68.91 (-6.66) |
| Remove Bidirectional $\rightarrow$ Unidirectional | 71.16 (-4.41) |
| End-concatenated Queries (no interleaving) | 73.23 (-2.34) |
| Remove LayerNorm | 70.56 (-5.01) |

### Key Findings
- BIMBA shows monotonic improvement across all sequence lengths, whereas pooling saturates or even degrades after 16 frames.
- Self-attention triggers OOM (Out Of Memory) beyond 9K–13K tokens, whereas BIMBA can handle up to 102K tokens.
- Its computational cost is close to that of pooling (the lowest), yet its accuracy is significantly higher.
- Question-conditioned selection yields an extra 1–2% gain, indicating that selecting tokens based on question context is highly effective.
- Interleaved queries (+2.34%) and bidirectional scan (+4.41%) contribute the most.

## Highlights & Insights
- **Mamba for Video Token Compression**: The content-aware selectivity of the selective scan is perfectly suited for highly redundant video features—offering a much more elegant approach than pooling or fixed compression strategies.
- **Interleaved Queries**: A simple yet effective design that eliminates sequence positional bias, offering generalizable value to long-range sequence modeling.
- **Optimal Accuracy-Efficiency Trade-off**: Achieving computational costs close to pooling and performance close to (or even better than) self-attention, making it an ideal compression module for long-video MLLMs.

## Limitations & Future Work
- The compression ratio is fixed (16x); adaptive compression ratios might be more optimal.
- Compression is currently applied after frame-level independent encoding, lacking early cross-frame interactions.
- Scalability on ultra-long (hour-level) videos remains to be validated.

## Related Work & Insights
- **vs. Pooling Methods (e.g., Video-ChatGPT)**: Pooling suffers from accuracy saturation on long sequences, whereas BIMBA shows continuous improvement.
- **vs. Perceiver/Q-former**: Fixed cross-attention lacks long-range selectivity; BIMBA yields superior accuracy.
- **vs. VideoMamba**: Utilizes Mamba to replace self-attention without token compression; this work focuses specifically on compression.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Mamba for video compression, interleaved queries, and bidirectional scanning is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 benchmarks + detailed ablations + computational cost analysis + multiple variants.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and well-designed experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a major boost to the advancement of long video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2026\] MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering](../../CVPR2026/video_understanding/mukv_multi-grained_kv_cache_compression_for_long_streaming_video_question-answer.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)

</div>

<!-- RELATED:END -->
