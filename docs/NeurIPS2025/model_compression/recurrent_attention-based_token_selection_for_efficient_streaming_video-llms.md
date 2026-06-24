---
title: >-
  [Paper Note] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs
description: >-
  [NeurIPS 2025][Model Compression][Streaming video understanding] This paper proposes rLiVS (Recurrent LLM-informed Visual Selection), a training-free and model-agnostic method for streaming video understanding. It achieves state-of-the-art performance on streaming video benchmarks through three complementary designs: LLM attention-guided visual token selection (retaining only ~6% of tokens), recurrent reuse of historical tokens, and caption-based retrieval for question answer…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Streaming video understanding"
  - "visual token compression"
  - "attention-based selection"
  - "long video"
  - "Video-LLM"
date: 2026-05-08
content_hash: e5c3188ff0ab4c2a
---

# Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.17364](https://arxiv.org/abs/2510.17364)  
**Code**: Unavailable  
**Area**: Model Compression
**Keywords**: Streaming video understanding, visual token compression, attention-based selection, long video, Video-LLM

## TL;DR

This paper proposes rLiVS (Recurrent LLM-informed Visual Selection), a training-free and model-agnostic method for streaming video understanding. It achieves state-of-the-art performance on streaming video benchmarks through three complementary designs: LLM attention-guided visual token selection (retaining only ~6% of tokens), recurrent reuse of historical tokens, and caption-based retrieval for question answering.

## Background & Motivation

- **Background**: Video-LLMs achieve strong performance on short video understanding, but face severe challenges in streaming scenarios involving hour-long online video processing with real-time question answering. The number of visual tokens grows linearly with frame count, making brute-force full-frame processing computationally infeasible for long videos and often exceeding context length limits.
- **Limitations of Prior Work**: Existing long-video understanding approaches each have notable drawbacks:
    - **Training-based methods** (e.g., VideoStreaming, Flash-VStream): require additional training, suffer from extrapolation issues on arbitrary-length videos, and incur high training costs.
    - **KV-cache methods** (e.g., ReKV): store complete decoder KV-caches, leading to large memory consumption (18.8 GB/hour) with significant redundancy.
    - **Caption-only methods** (e.g., Goldfish): process short clips independently, lacking temporal continuity and making entity tracking difficult.
- **Key Insight**: Inspired by cognitive neuroscience—where selective attention governs encoding under limited memory capacity and past experience shapes current attention—the authors propose combining LLM self-attention for visual token selection, recurrent propagation of historical context, and text-based retrieval for question answering.

## Method

### Overall Architecture

Long videos are segmented into short clips (e.g., 16 frames) and processed in a streaming fashion. For each clip: (1) historical selected tokens are concatenated with the current clip's tokens and fed into the LLM to generate a caption; (2) a small number of key tokens are selected from the current clip based on attention weights and appended to a FIFO history queue; (3) the generated caption is stored in long-term textual memory. At inference time, the most relevant captions are retrieved from textual memory and fed to the LLM to generate answers.

### Key Designs

1. **Attention-based Visual Token Selection**

   After caption generation, the already-computed attention matrices are used to measure the importance of each visual token. The attention coefficients from caption tokens to visual tokens are extracted from layer $l$, head $h$:

   $\mathbf{A}^{l,h}_V = \mathbf{A}^{l,h}[TN_V+N_I : TN_V+N_I+N_C, \; 0:TN_V]$

   A global importance score for each visual token $j$ is computed by averaging across all caption tokens, attention heads, and layers:

   $a_j = \frac{1}{L}\sum_{l=1}^{L}\frac{1}{H}\sum_{h=1}^{H}\left(\frac{1}{N_C}\sum_{i=1}^{N_C}\mathbf{A}^{l,h}_{V_{ij}}\right)$

   The top-$N_S$ tokens by score are retained ($N_S \ll N_V$); in practice, only 6.25% are kept (196 out of 3,136 tokens). Uniformly sampling 4 out of $L$ layers is sufficient for robust results.

   **Design Motivation**: Attention scores are signals already computed during caption generation, introducing no additional overhead, and naturally reflect which visual tokens are most relevant to the current linguistic understanding.

2. **Recurrent Long-Video Processing**

   A FIFO queue maintains historically selected tokens $[\mathbf{S}^{(0)}, \mathbf{S}^{(1)}, \ldots, \mathbf{S}^{(t)}]$, which are prepended as a context prefix when processing the next clip. Tokens from the earliest clip are discarded when the context window limit $W$ is exceeded.

   The recurrent design serves a dual purpose: (1) it enhances visual continuity and coherence across clips; (2) it guides LLM attention toward content consistent with historical context, reinforcing the selection effect.

3. **Caption-based Retrieval for Question Answering**

   The embeddings of captions from all clips $\{\mathbf{X}_C^{(t)}\}$ are stored. Given a question $q$, the average cosine similarity between question tokens $\mathbf{X}_q$ and caption tokens is computed. Maximal Marginal Relevance (MMR) is applied to balance relevance and diversity, and top-$K$ captions are retrieved. Only the retrieved captions (not visual tokens) are fed to the LLM for answer generation.

   **Design Motivation**: Experiments show that cosine similarities between visual tokens and questions are concentrated in $[-0.02, 0.06]$, offering nearly no discriminative power, whereas caption similarities span $[0.4, 0.9]$ with strong discriminability. Moreover, since LLMs are well-suited for long-form text reasoning, converting video QA into textual QA proves more effective.

### Loss & Training

The method is entirely training-free; it operates directly on pretrained Video-LLMs without any architectural modification, making it applicable to any short-video-pretrained Video-LLM.

## Key Experimental Results

### Main Results

Streaming benchmarks (RVS-Ego / RVS-Movie):

| Method | Backbone | RVS-Ego Acc | RVS-Movie Acc | Latency | VRAM |
|--------|----------|:-----------:|:-------------:|:-------:|:----:|
| Flash-VStream-7B | Dedicated | 57.3 | 53.1 | 2.1s | 19GB |
| ReKV | LLaVA-OV 7B | 63.7 | 54.4 | 2.7s | 36GB |
| **rLiVS** | LLaVA-OV 7B | **65.3** | **57.7** | **1.9s** | **25GB** |
| **rLiVS** | Qwen2.5-VL 7B | **68.1** | 56.1 | 2.7s | 19GB |
| ReKV | LLaVA-OV 0.5B | 54.7 | 44.6 | 1.6s | 19GB |
| rLiVS | LLaVA-OV 0.5B | 57.6 | 51.3 | 1.5s | 11GB |

Offline benchmarks:

| Method | VS-Ego Acc | VS-Movie Acc | MovieChat Acc | CG-Bench Acc |
|--------|:----------:|:------------:|:-------------:|:------------:|
| Flash-VStream-7B | 59.0 | 56.1 | - | - |
| Goldfish | - | - | 67.6 | - |
| **rLiVS** | **61.0** | **59.3** | **78.0** | **33.1** |

### Ablation Study

Token selection method comparison (NextQA, 6% retention):

| Selection Method | Accuracy |
|-----------------|:--------:|
| Full model (100%) | 78.6 |
| Uniform sampling (6%) | 75.5 |
| Mean pooling (6%) | 70.7 |
| K-Means (6%) | 76.8 |
| **Attention selection (6%)** | **77.0** |
| Attention selection (12%) | 78.4 |

Design choice ablation (streaming benchmarks):

| Configuration | RVS-Ego Acc | RVS-Movie Acc | Note |
|---------------|:-----------:|:-------------:|------|
| rLiVS (full) | 65.3 | 57.7 | Recurrent + attention selection + caption QA |
| w/o recurrence | 62.5 | 53.7 | Recurrence contributes 3–4% gain |
| Visual token retrieval for QA | 58.2 | 48.4 | Captions far outperform visual tokens |
| Uniform sampling instead of attention | 64.2 | 56.0 | Attention selection gains 1–2% |

### Key Findings

- Retaining only 6% of visual tokens results in a performance drop of merely 1.6% on NextQA; at 12%, performance is nearly lossless.
- Recurrently propagating historical tokens improves long-video understanding by 3–4 percentage points.
- Captions substantially outperform visual tokens as the information carrier for both retrieval and question answering.
- A 0.5B model equipped with rLiVS surpasses most competing methods that require 7B parameters.
- A context length of 10K tokens represents the optimal trade-off between efficiency and effectiveness.

## Highlights & Insights

- The method is remarkably simple and elegant: it repurposes the LLM's already-computed attention for token selection at zero additional cost.
- The model-agnostic design enables plug-and-play integration with arbitrary Video-LLMs such as LLaVA-OV and Qwen2.5-VL.
- Zero KV-cache storage: unlike ReKV, no full KV-cache needs to be stored (saving 18.8 GB/hour).
- The design is inspired by cognitive science: attention → selective memory → recurrent processing mirrors human visual information processing.

## Limitations & Future Work

- Focusing exclusively on selected tokens may cause the method to miss fine-grained details.
- The FIFO memory buffer operates on temporal rather than semantic priority, potentially discarding critical early-stage information.
- Recurrent caption generation may introduce cross-segment redundancy.
- The method fully inherits the capabilities and limitations of the pretrained backbone.
- Future work could explore adaptive compression rates that dynamically adjust the retention ratio according to scene complexity.

## Related Work & Insights

- ReKV stores complete KV-caches for streaming understanding → rLiVS achieves better results with far fewer tokens.
- Goldfish processes short clips independently → rLiVS enhances temporal continuity through recurrence.
- Using attention as a token importance indicator → this principle can be generalized to other multimodal long-context scenarios.
- The insight of "converting video QA into textual QA" is worth adopting in other long-video systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Integrates known concepts (attention selection, recurrent processing, caption QA) into an elegant unified framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers both streaming and offline benchmarks with comprehensive ablations and efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear exposition with complete algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, efficient, and practical — establishes a strong baseline for streaming video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[NeurIPS 2025\] Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)
- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](../../ICML2025/model_compression/orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)

</div>

<!-- RELATED:END -->
