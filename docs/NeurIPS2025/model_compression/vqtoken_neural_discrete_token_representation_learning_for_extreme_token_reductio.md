---
title: >-
  [Paper Note] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models
description: >-
  [NeurIPS 2025][Model Compression][token reduction] VQToken introduces the first vector-quantization-based framework for extreme video token compression. By adaptively discretizing continuous ViT embeddings into a compact codebook and preserving spatiotemporal positional information via a token hash function, it achieves only 0.66% accuracy loss on NextQA-MC using merely 0.07% of the original tokens (approximately 13 tokens).
tags:
  - NeurIPS 2025
  - Model Compression
  - token reduction
  - vector quantization
  - video LLM
  - discrete representation
  - token information density
date: 2026-05-08
content_hash: b29980a692c983e2
---

# VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2503.16980](https://arxiv.org/abs/2503.16980)
**Code**: [https://github.com/](https://github.com/) (available, includes Homepage, GitHub, HuggingFace)
**Area**: Model Compression / Video LLM Efficiency
**Keywords**: token reduction, vector quantization, video LLM, discrete representation, token information density

## TL;DR

VQToken introduces the first vector-quantization-based framework for extreme video token compression. By adaptively discretizing continuous ViT embeddings into a compact codebook and preserving spatiotemporal positional information via a token hash function, it achieves only 0.66% accuracy loss on NextQA-MC using merely 0.07% of the original tokens (approximately 13 tokens).

## Background & Motivation

Video large language models (vLLMs) face severe computational efficiency challenges: video inputs require tokenizing each frame's pixels and concatenating them into extremely long sequences, and the Transformer attention mechanism exhibits $O(n^2 DL)$ complexity with respect to sequence length $n$. The critical bottleneck lies in the disproportionate impact of token sequence length $n$ compared to model parameters and layer count.

Existing token compression methods suffer from three major problems:

**Token Pruning** directly discarding tokens loses critical information and disrupts positional encoding.

**Token Merging** (e.g., ToMe, VidToMe) applies fixed compression ratios, lacks flexibility, and still produces excessively long sequences.

3. Compressed tokens remain highly similar in the continuous domain, resulting in low information density that resists further compression.

The root causes are: (1) fixed-count/ratio compression strategies either compress insufficiently or incur excessive loss; (2) the absence of adaptive, context-aware mechanisms for selecting the most informative tokens; (3) no utilization of vector quantization to cluster tokens into discrete categories for improved information density.

The paper's core mechanism: **applying vector quantization (VQ) to cluster continuous ViT embeddings into a minimal set of discrete tokens, coupled with a token hash function to preserve spatiotemporal positional relationships, achieving 99.9%+ token compression with negligible performance degradation.**

## Method

### Overall Architecture

The VQToken pipeline proceeds as follows:
1. Input video is tokenized by a ViT into a continuous visual token sequence.
2. An Adaptive Discrete Process clusters and quantizes tokens into a compact codebook.
3. A token hash function records the original spatiotemporal position of each token and maps it to the nearest codebook entry.
4. A VQ-Attention module integrates the codebook with positional indices to produce a compressed token sequence retaining positional information.
5. The compressed tokens and tokenized query are fed into the LLM for zero-shot inference.

### Key Designs

1. **Adaptive Discrete Process**:

    - Applies cosine-similarity-based vector quantization clustering to continuous ViT token embeddings.
    - Fixed-length compression: employs standard K-Means.
    - Adaptive-length compression: employs an adaptive K-Means variant that dynamically determines the number of clusters $K$ based on video content complexity.
    - Output: $K$ cluster centroids serving as discrete codebook entries, along with the cluster assignment of each token.

2. **Concise Token Codebook**:

    - Each codebook entry $b_k$ is the centroid of all token embeddings within the corresponding cluster: $b_k = \frac{1}{|s_k|} \sum_{i \in s_k} t_i$
    - The codebook $B \in \mathbb{R}^{K \times D}$ captures representative visual patterns with minimal redundancy.

3. **Token Hash Function Mapping**:

    - Constructs a 3D index map $M \in \{1,...,K\}^{T \times H \times W}$ (where $T$ denotes frame count and $H/W$ denote ViT grid dimensions).
    - $M_{f,h,w} = c_i$ records the cluster index assigned to each grid position.
    - Preserves spatiotemporal positional encoding, serving as a lightweight substitute for expensive motion tracking methods such as optical flow.

4. **VQ-Attention Module**:

    - Flattens the index map and projects it via MLP: $\widetilde{M} = \text{MLP}(\text{Flatten}(M)) \in \mathbb{R}^{K \times D}$
    - Multi-head attention fuses the codebook and positional information: $B' = \text{MultiHeadAttn}(Q=BW_Q, K=BW_K, V=\widetilde{M}W_V)$
    - The output $B' \in \mathbb{R}^{K \times D}$ constitutes the final compressed tokens, carrying motion context.

### Loss & Training

- Based on LLaVA-OneVision 0.5B (Qwen2 backbone).
- Training data: LLaVA-Video-178K dataset, 1.3M instruction-following samples.
- 4× A100 GPUs, 85K iterations, AdamW with cosine decay.
- VQ-Attention learning rate: $1\times10^{-5}$; ViT backbone learning rate: $2\times10^{-6}$.
- ZeRO-2 optimization, batch size 8, gradient accumulation over 2 steps.

### Token Information Density Metric (TokDense)

- Definition: $\text{TokDense} = \frac{\text{Accuracy}}{\text{Token Count}}$
- Measures each retained token's contribution to task performance.
- The paper additionally defines module complexity (overhead of the token compression module itself) and LLM complexity (downstream inference cost after compression).

## Key Experimental Results

### Main Results: Comparison with vLLM Baselines

| Model | Params | Zero-Shot | Accuracy (%) | Token Count % |
|-------|--------|-----------|--------------|---------------|
| LLaVA-OneVision | 0.5B | ✓ | 57.2 | 100% |
| LLaVA-OV-SI | 0.5B | ✓ | 53.6 | 27% |
| VQToken (Ours) | 0.5B | ✓ | **57.5** | **0.14%** |
| Mistral | 7B | ✓ | 51.1 | 100% |
| LLoVi | 7B | ✓ | 54.3 | 100% |
| MVU | 7B | ✓ | 55.2 | 100% |

### Extreme Token Compression Tasks

**Fixed-Length Subtask**:

| Method | Token=12 | Token=32 | Token=64 |
|--------|----------|----------|----------|
| Token Pruning | 29.12 | 34.50 | 31.31 |
| ToMe | 35.72 | 38.50 | 40.10 |
| VidToMe | 39.64 | 45.10 | 46.20 |
| VQToken (Ours) | **57.03** | **57.46** | **57.10** |

**Adaptive-Length Subtask**:

| Method | Avg. Token Count | Accuracy | TokDense |
|--------|-----------------|----------|----------|
| Interpolating | 3136 | 57.20 | 0.018 |
| DyCoke | 1662.12 | 57.70 | 0.035 |
| Ours-Fixed (m=32) | 32 | 57.46 | 1.796 |
| Ours-Dynamic | **13.08** | **57.72** | **4.413** |

### Ablation Study

| VLM | Codebook | Hash Fn. | VQ-Attn | Accuracy | Tokens | TokDense |
|-----|----------|----------|---------|----------|--------|----------|
| ✓ | — | — | — | 57.2 | 23328 | 0.002 |
| ✓ | ✓ | — | — | 35.2 | 32 | 1.100 |
| ✓ | ✓ | ✓ | ✓ | **57.5** | **32** | **1.797** |
| ✓ | rand | ✓ | ✓ | 37.7 | 32 | 1.178 |
| ✓ | ✓ | rand | ✓ | 46.9 | 32 | 1.466 |

### Key Findings

- VQToken matches and marginally surpasses the full-token baseline using only 0.14% (32) of the original tokens.
- In adaptive mode, an average of only 13.08 tokens are required, achieving a TokDense of 4.413—245× that of the interpolation method.
- VQToken with 0.5B parameters outperforms multiple 7B vLLMs in the zero-shot setting.
- All three components (codebook, hash function, VQ-Attention) are indispensable: omitting positional information while retaining the codebook alone results in a 22% accuracy drop, and VQ-Attention is the key to performance recovery.
- Performance is balanced across all 20 subtasks of MVBench, with particular strength in action recognition and object interaction tasks.

## Highlights & Insights

- **Viability of extreme compression**: This work provides the first demonstration that video tokens can be compressed to 0.07% of the original (in dynamic mode) with negligible performance loss, fundamentally revising the understanding of video token redundancy.
- **Discretization vs. continuous representation**: Unlike existing token pruning/merging approaches that operate entirely in continuous space, VQToken transforms video representations into a discrete codebook, substantially improving information density.
- **Elegant token hash function design**: A concise 3D index map replaces expensive motion tracking (e.g., optical flow), preserving spatiotemporal relationships with virtually no additional computational overhead.
- **TokDense evaluation metric**: The proposed "accuracy per token" metric provides a more principled basis for comparing methods in extreme compression scenarios.

## Limitations & Future Work

- Validation is limited to LLaVA-OneVision at the 0.5B scale; performance on larger LLMs remains unexplored.
- K-Means clustering requires processing the full token sequence at once, making streaming inference difficult.
- The robustness and generalizability of the adaptive K-Means strategy for selecting $K$ warrants further validation across additional datasets.
- Performance on tasks requiring fine-grained spatial localization (e.g., video grounding) has not been evaluated.
- Approximately 4–6% accuracy degradation relative to the full-token baseline is observed on ActNet-QA and LongVideoBench, indicating residual information loss under extreme compression for certain long-video understanding tasks.

## Related Work & Insights

- The paper stands in direct contrast to ToMe/VidToMe (token merging): VQToken achieves orders-of-magnitude higher compression through discretization.
- VQToken is complementary to DyCoke (dynamic KV cache compression): DyCoke compresses the KV cache at inference time, while VQToken compresses tokens at the input stage.
- VQ has been extensively validated in visual generation (VQ-VAE, VQ-GAN); this paper is the first to apply it to token compression in vLLMs.
- Insight: the discretization-plus-positional-index paradigm may generalize to other ultra-long sequence tasks, such as long-document understanding and audio processing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ICLR 2026\] Parallel Token Prediction for Language Models](../../ICLR2026/model_compression/parallel_token_prediction_for_language_models.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)

</div>

<!-- RELATED:END -->
