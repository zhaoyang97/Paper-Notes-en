---
title: >-
  [Paper Note] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference
description: >-
  [CVPR 2026][Video Understanding][sparse attention] This paper identifies a strong "vertical vector" sparsity pattern in the attention maps of video models and proposes VecAttention, a fine-grained vector-wise sparse attention framework. Through TilingSelect and minS filtering, the method efficiently selects important KV vectors, achieving accuracy on par with full attention at over 78% sparsity while delivering a 2.65× speedup in attention computation.
tags:
  - CVPR 2026
  - Video Understanding
  - sparse attention
  - vector-wise sparsity
  - long-context acceleration
  - video generation
date: 2026-05-08
content_hash: 988709f7e1c01806
---

# VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference

**Conference**: CVPR 2026
**arXiv**: [2603.29494](https://arxiv.org/abs/2603.29494)
**Code**: [https://github.com/anminliu/VecAttention](https://github.com/anminliu/VecAttention)
**Area**: Video Understanding
**Keywords**: sparse attention, vector-wise sparsity, long-context acceleration, video understanding, video generation

## TL;DR

This paper identifies a strong "vertical vector" sparsity pattern in the attention maps of video models and proposes VecAttention, a fine-grained vector-wise sparse attention framework. Through TilingSelect and minS filtering, the method efficiently selects important KV vectors, achieving accuracy on par with full attention at over 78% sparsity while delivering a 2.65× speedup in attention computation.

## Background & Motivation

1. **Background**: Video understanding and generation models operate on extremely long token sequences (17K–119K), making attention computation the primary inference bottleneck. Sparse attention methods (e.g., FlexPrefill, XAttention) accelerate inference by skipping unimportant attention computations.
2. **Limitations of Prior Work**: Existing methods rely on coarse-grained sparsity patterns (e.g., block-level, row-level), which reduce computational complexity at the cost of accuracy—since important and unimportant tokens may coexist within a single block or row, coarse-grained skipping discards critical information.
3. **Key Challenge**: Finer granularity preserves more important information but incurs greater selection overhead—the communication and computation costs of token-level selection may negate the speedup gains.
4. **Goal**: Identify the accuracy-efficiency optimal sparsity granularity and design efficient selection and computation kernels to match.
5. **Key Insight**: A systematic analysis of sparsity structures in video attention maps reveals a "vertical vector" pattern—important KV tokens tend to be important across all query heads, appearing as consistently bright columns. This structural property enables efficient selection via query pooling.
6. **Core Idea**: Vector-level granularity ($P_q=64$) + minS filtering (more efficient than topK) + TilingSelect (fusing selection into GEMM to reduce HBM access).

## Method

### Overall Architecture

Full-sequence Q/K/V → Stage 1: Query pooling + minS filtering to select important KV vectors (TilingSelect kernel) → Stage 2: FlashAttention-2-style sparse attention computed only over selected KV vectors → Output.

### Key Designs

1. **minS Filtering Strategy**

   - **Function**: Efficiently identifies the set of KV vectors each query group should attend to.
   - **Mechanism**: Query pooling yields $Q_p$; similarity scores $s_i$ with all keys are computed; a mask $M_i = (s_i \geq (m_i^s - \alpha))$ is applied, where $m_i^s = \text{rowmax}(s_i)$ and $\alpha$ is the filter ratio. The intuition is to retain all KV tokens within $\alpha$ of each row's maximum score.
   - **Design Motivation**: More efficient than topK—topK requires full-row sorting ($O(N \log N)$), whereas minS requires only one rowmax operation plus a threshold comparison ($O(N)$). Ablations show minS is 3.77× faster than topP.

2. **TilingSelect Kernel**

   - **Function**: Fuses important vector selection into the GEMM operation to reduce memory access.
   - **Mechanism**: During tiled GEMM computation of $Q_p \cdot K^T$, minS filtering and cross-tile rowmax accumulation are performed simultaneously, avoiding the allocation of an intermediate tensor of size $N^2$. Memory footprint is reduced from $\Theta(N^2 P_q^{-1})$ to $\Theta(N^2 P_q^{-1}(1-\rho))$.
   - **Design Motivation**: At $N=64$K, standard selection requires 18.3 GB of intermediate storage; TilingSelect reduces this to 1.8 GB (10.2× savings).

3. **Dynamic Per-Head Filter Ratio**

   - **Function**: Adaptively adjusts the sparsity level for each attention head.
   - **Mechanism**: Dynamic programming is used to predict the optimal $\alpha$ value for each head based on its attention distribution characteristics, allowing different heads to operate at different sparsity ratios.
   - **Design Motivation**: Attention distributions vary substantially across heads—some heads are naturally sparse (permitting aggressive filtering), while others exhibit flat distributions (requiring more KV tokens to be retained).

### Loss & Training

No training is required; VecAttention is a purely inference-time method. Hyperparameters include vector size $P_q=64$, K tile size $B_k=16$, and group size $G_k$ (16 for understanding tasks, 8192 for generation tasks).

## Key Experimental Results

### Main Results

| Method | Sparsity | VideoMME↑ | LongVideoBench↑ | VCRBench↑ | Avg.↑ |
|--------|----------|-----------|-----------------|-----------|-------|
| Full Attention | 0% | 65.7 | 59.4 | 32.9 | 52.7 |
| FlexPrefill | 76.5% | 52.3 | 59.0 | 30.0 | 47.1 |
| XAttention | 78.1% | 56.0 | 59.9 | 32.5 | 49.5 |
| AnchorAttention | 78.6% | 57.4 | 59.4 | 31.3 | 49.4 |
| **VecAttention** | **78.6%** | **60.6** | **59.0** | **33.8** | **51.1** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Vector size $P_q=32$ | Higher accuracy but increased overhead | Granularity too fine |
| Vector size $P_q=64$ | Optimal balance | Default |
| Vector size $P_q=128$ | Accuracy drops | Granularity too coarse |
| minS vs. topP | 3.77× speedup | minS is more efficient |
| TilingSelect | 10.2× memory savings | 18.3 GB → 1.8 GB |

### Key Findings

- At 78.6% sparsity, VecAttention achieves an average accuracy of 51.1%, only 1.6% below full attention (52.7%), substantially outperforming competing methods at the same sparsity level (47.1–49.5%).
- The maximum applicable sparsity reaches 93%, far exceeding competing methods' limits of 85–88%.
- The method is equally effective for video generation: on Wan2.1-T2V, it matches full-attention PSNR/SSIM at 52.3% sparsity.
- Attention computation is accelerated by 2.65×, with end-to-end TTFT speedup of 1.17×.

## Highlights & Insights

- **Discovery of the Vertical Vector Sparsity Pattern**: This empirical observation provides the theoretical basis for fine-grained sparsity—the importance of KV tokens is highly consistent across queries, making query-pooling-based selection nearly lossless.
- **Efficiency Gap Between minS and topK**: Reducing selection complexity from $O(N \log N)$ to $O(N)$ is a modest algorithmic change that yields a 3.77× practical speedup.
- **Unified Applicability to Video Understanding and Generation**: The same framework is effective on both VLMs and DiTs, suggesting that the vertical vector pattern is a general property of video attention.

## Limitations & Future Work

- Whether the vertical vector pattern generalizes to other modalities (e.g., plain text, audio) remains unverified.
- The additional overhead of fine-grained selection may not be cost-effective for shorter sequences.
- Evaluation is limited to video understanding and generation; complex reasoning tasks (e.g., agent workflows, RAG) are not tested.
- Future work could explore the benefits of other fine-grained patterns (horizontal, diagonal) for specific tasks.

## Related Work & Insights

- **vs. FlexPrefill**: This block-level sparse method achieves only 52.3% on VideoMME at comparable sparsity, versus VecAttention's 60.6%—the accuracy gap is attributable to granularity differences.
- **vs. XAttention**: Also a sparse attention method for video, but it underperforms VecAttention on understanding tasks and is limited in maximum achievable sparsity.
- **vs. FlashAttention-2**: VecAttention's computation kernel builds directly on FlashAttention-2's tiling strategy and can be viewed as its sparse extension.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The vector-level sparsity granularity and minS selection strategy are genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Validated on both video understanding and generation across multiple models and benchmarks, with detailed micro-benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ — Systematic and well-structured, with a clear logical flow from observation to design to implementation.
- **Value**: ⭐⭐⭐⭐⭐ — Long-video inference acceleration addresses a pressing practical need; the 2.65× speedup has direct industrial relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2026\] Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining](cluster-wise_spatio-temporal_masking_for_efficient_video-language_pretraining.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)

</div>

<!-- RELATED:END -->
