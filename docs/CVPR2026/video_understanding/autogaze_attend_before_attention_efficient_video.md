---
title: >-
  [Paper Note] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing
description: >-
  [CVPR 2026][Video Understanding][Autoregressive gazing] This paper proposes AutoGaze, a lightweight 3M-parameter module that autoregressively selects the minimal multi-scale patch set minimizing reconstruction loss prior…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Autoregressive gazing"
  - "token compression"
  - "multi-scale patch selection"
  - "video MLLM"
  - "high-resolution long video"
date: 2026-05-08
content_hash: f8152f292a2e8606
---

# AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing

**Conference**: CVPR 2026
**arXiv**: [2603.12254](https://arxiv.org/abs/2603.12254)  
**Code**: [autogaze.github.io](https://autogaze.github.io/)  
**Area**: Video Understanding / Efficient Inference
**Keywords**: Autoregressive gazing, token compression, multi-scale patch selection, video MLLM, high-resolution long video

## TL;DR

This paper proposes AutoGaze, a lightweight 3M-parameter module that autoregressively selects the minimal multi-scale patch set minimizing reconstruction loss prior to ViT processing, removing redundant information from video. It achieves 4×–100× token compression and up to 19× ViT speedup, enabling MLLMs to scale to 1K-frame 4K-resolution video and reach 67.0% on VideoMME.

## Background & Motivation

**Background**: MLLMs such as Qwen2.5-VL and NVILA have achieved general-purpose video understanding, yet are constrained by computational cost to short, low-resolution video. The human visual system efficiently processes high-FPS high-resolution video streams by selectively attending to informative regions via saccades.

**Limitations of Prior Work**: (1) Existing MLLMs process every pixel and every frame with equal weight, ignoring the substantial spatiotemporal redundancy in video (e.g., a static background needs to be observed only once); (2) existing token compression methods prune tokens only inside the LLM or between ViT and LLM—the ViT still processes all pixels and remains the scaling bottleneck; (3) heuristic approaches (attention-score-based) underperform learned methods, while search- and reasoning-based approaches introduce additional overhead.

**Key Challenge**: A fundamental tension exists between the need for long-duration, high-resolution video understanding and the computational bottleneck of processing all pixels through the ViT.

**Goal**: Efficiently remove redundant patches from video before the ViT while retaining sufficient information for downstream understanding.

**Key Insight**: Patch selection is formulated as an autoregressive sequence generation problem, where which patches to select and when to stop are jointly determined by minimizing reconstruction loss.

**Core Idea**: Train a lightweight model to learn "where to look first," mimicking human saccadic attention, and remove redundancy before ViT processing.

## Method

### Overall Architecture

AutoGaze operates at the very front of the video processing pipeline: (1) a convolutional encoder extracts frame-level features for each frame; (2) a Transformer decoder autoregressively outputs a sequence of patch indices while simultaneously predicting the current reconstruction loss; (3) gazing for the current frame stops automatically when the predicted reconstruction loss falls below a user-specified threshold $\epsilon$; (4) only the selected multi-scale patches are forwarded to the downstream ViT and MLLM.

### Key Designs

1. **Autoregressive Gazing**:
    - Function: Autoregressively selects the most informative patches frame by frame and patch by patch, leveraging history from previous frames and already-selected patches to avoid redundant selections.
    - Mechanism: Patch selection is formulated as a sequence-to-sequence problem with a vocabulary of patch indices $\{1, \dots, V\}$. The decoder produces a probability distribution over the next patch to select, conditioned on current and historical frame features and previously selected patches.
    - Optimization objective: $\min_{p_1^1, \dots, p_{N^T}^T} L(\mathbf{X}^{1:T}, \text{Recon}(\mathbf{X}^1[p_1^1], \dots, \mathbf{X}^T[p_{N^T}^T]))$
    - Automatic stopping: The decoder additionally predicts the reconstruction loss at each step; gazing for the current frame halts when the predicted loss $< \epsilon$.
    - Design Motivation: The autoregressive formulation allows the model to exploit the gazing history of previous frames to avoid redundant selections (e.g., a static background already attended to in prior frames), which heuristic methods cannot achieve.

2. **Multi-Scale Gazing**:
    - Function: Allocates patches of different resolutions to regions of varying detail, using coarser patches to cover low-detail areas.
    - Mechanism: The decoder vocabulary includes patches at multiple scales, enabling the model to adaptively select the optimal scale for each region.
    - Validation: Experiments confirm that AutoGaze tends to select finer scales in high-detail regions (high Laplacian variance) and coarser scales in low-detail regions (Spearman $\rho = 0.12, p < 0.001$).
    - Design Motivation: Uniform regions can be represented losslessly at low resolution, obviating the need for full-resolution patches.

3. **Two-Stage Training: NTP Pre-training + RL Post-training**:
    - Function: Supervised learning first establishes foundational gazing capability; RL subsequently overcomes the suboptimality of greedy search.
    - NTP Pre-training: Approximately optimal gazing sequences are collected via greedy search on 250K videos and used to train with cross-entropy loss $L_{NTP} = -\sum_t \sum_k \log \pi_\theta(\tilde{p}_k^t | \mathbf{X}^{1:t}, \tilde{p}_{1:k-1}^t)$.
    - RL Post-training: A simplified GRPO algorithm is employed, using the negative reconstruction loss as reward, with advantage $G_k^t = \sum_{\tau=t}^T \gamma^{N^t-k+\sum_{s=t+1}^\tau N^s} \cdot (-l_{N^\tau}^\tau)$.
    - Multi-token prediction: Multiple heads simultaneously output multiple patch indices; 10-token prediction accelerates the gazing process by approximately 5× with negligible performance degradation.
    - Design Motivation: Greedy search data represents only a suboptimal solution; RL can discover superior gazing sequences.

4. **Arbitrary-Resolution/Duration Inference**:
    - Function: Trained on 16-frame 224×224 video; generalizes to arbitrary resolution and duration at inference.
    - Mechanism: Video is partitioned into $16 \times 224 \times 224$ spatiotemporal tiles; AutoGaze is applied to each tile independently, and gaze locations are subsequently merged.
    - ViT Adaptation: Position encodings for frames and locations at different scales are interpolated separately; patch embeddings at each scale are computed independently and concatenated before being fed into the ViT.

### Loss & Training

- Reconstruction model: Custom VideoMAE with block-causal attention.
- Reconstruction loss: Weighted sum of pixel reconstruction loss and perceptual loss.
- Training data: 800K videos (first-person, third-person, natural scenes, text-rich videos), of which 250K include greedy gazing sequences.
- Video sampling: 16 frames at 224 resolution.
- Optimal reconstruction loss threshold: 0.7 (downstream MLLM performance degradation < 0.5%).

## Key Experimental Results

### Main Results (Comparison with State-of-the-Art MLLMs)

| Model | Max #F | Max Res | VideoMME (w/o) | VideoMME (w/) | MVBench | HLVid |
|-------|--------|---------|----------------|---------------|---------|-------|
| GPT-4o | - | - | 71.9 | 77.2 | 64.6 | 49.3 |
| Qwen2.5-VL-7B | 48 | 896 | 65.1 | 71.6 | 69.6 | 48.1 |
| VideoChat-Flash | 10000 | 448 | 65.3 | 69.7 | 74.0 | 46.6 |
| NVILA-8B-Video | 256 | 448 | 64.2 | 70.0 | 68.1 | 42.5 |
| **+ AutoGaze** | **1024** | **3584** | **67.0** | **71.8** | 69.7 | **52.6** |
| Gain (vs NVILA) | ×4 | ×8 | +2.8 | +1.8 | +1.6 | **+10.1** |

### Ablation Study

| Training Strategy | Recon. Loss | Gaze Ratio |
|-------------------|-------------|------------|
| No training | 0.7 | 0.263 |
| NTP pre-training only | 0.7 | 0.102 |
| RL post-training only | 0.7 | 0.209 |
| **NTP + RL** | 0.7 | **0.094** |

| Multi-token Pred. Count | Multi-Scale | Gaze Ratio | Inference Latency |
|-------------------------|-------------|------------|-------------------|
| 1 | ✓ | 0.074 | 0.949s |
| 5 | ✓ | 0.078 | 0.246s |
| 10 | ✓ | 0.094 | 0.193s |
| 10 | ✗ | 0.220 | 0.467s |

### Key Findings

- 30FPS 4K-resolution video requires only approximately 1% of patches to reach the 0.7 reconstruction loss threshold.
- ViT speedup reaches up to 19×; MLLM speedup up to 10×, making 4K-resolution video processing feasible.
- A 10.1% improvement on the HLVid high-resolution long-video benchmark (42.5→52.6%) surpasses GPT-4o (49.3%).
- AutoGaze exhibits strong OOD generalization: consistent gazing patterns are maintained on unseen semantics (CCTV, robotics, object-replacement video) and style-transferred video.
- Compared to existing token compression methods, AutoGaze is the only approach that reduces both ViT and LLM latency simultaneously (ViT: 2.20s→0.55s).

## Highlights & Insights

- **Addressing the Bottleneck at Its Source**: AutoGaze is the first method to apply token compression before the ViT, breaking away from the prior convention of compressing only after ViT processing.
- **Biomimetic Design**: The autoregressive gazing mechanism directly emulates the saccadic behavior of the human visual system, exploiting temporal information to avoid redundant fixations.
- **Extreme Lightweight Design**: Powerful gazing capability is achieved with only 3M parameters, with negligible computational overhead from the gazing module itself.
- **HLVid: First High-Resolution Long-Video Benchmark**: 268 questions requiring 1K–2K resolution for answering, filling a gap in evaluation methodology.
- **Two-Stage Training Paradigm**: The NTP+RL combination outperforms either component alone; RL transcends the performance ceiling imposed by greedy search data.

## Limitations & Future Work

- On certain benchmarks that do not require high resolution, scaling resolution excessively can be harmful, necessitating adaptive strategies.
- The approach relies on VideoMAE as the reconstruction model; reconstruction quality may itself limit the quality of gazing decisions.
- The HLVid benchmark is limited in scale (268 QAs), and its representativeness warrants further validation.
- Multi-scale patch input requires modifications to the ViT (separate patch embeddings per scale), raising compatibility concerns for already-deployed models.

## Related Work & Insights

- **NVILA** (Lin et al., 2024): Serves as the base MLLM; AutoGaze enables it to achieve 1K-frame 4K-resolution processing.
- **GRPO** (Shao et al.): Used for RL post-training of the gazing policy, validating the effectiveness of RL for visual token selection.
- **ToMe / FastV / LongVU**: Prior token compression methods operate only within the ViT or LLM; AutoGaze is the first to move compression upstream.
- **Insights**: The paradigm of autoregressive sequence generation with reconstruction loss as reward is extensible to token selection in other modalities such as images and 3D point clouds.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐⭐** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2026\] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)

</div>

<!-- RELATED:END -->
