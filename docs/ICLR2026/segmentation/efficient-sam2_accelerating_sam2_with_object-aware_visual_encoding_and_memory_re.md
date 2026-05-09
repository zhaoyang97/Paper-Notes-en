---
title: >-
  [Paper Note] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval
description: >-
  [ICLR 2026][Segmentation][SAM2] This paper identifies sparse perception patterns in SAM2 analogous to biological vision — the decoder focuses on foreground while the encoder computes broadly, and only a small subset of tokens in memory frames are active with temporally consistent saliency. Based on these observations, Efficient-SAM2 is proposed, which eliminates redundant computation via object-aware Sparse Window Routing (SWR) and Sparse Memory Retrieval (SMR), achieving 1.68× end-to-end speedup on SAM2.1-L with only 1% accuracy loss.
tags:
  - ICLR 2026
  - Segmentation
  - SAM2
  - video object segmentation
  - post-training acceleration
  - sparse window routing
  - memory compression
date: 2026-05-08
content_hash: 5b55e09642c27078
---

# Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval

**Conference**: ICLR 2026
**arXiv**: [2602.08224](https://arxiv.org/abs/2602.08224)
**Code**: [GitHub](https://github.com/jingjing0419/Efficient-SAM2)
**Area**: Video Segmentation / Model Acceleration
**Keywords**: SAM2, video object segmentation, post-training acceleration, sparse window routing, memory compression

## TL;DR
This paper identifies sparse perception patterns in SAM2 analogous to biological vision — the decoder focuses on foreground while the encoder computes broadly, and only a small subset of tokens in memory frames are active with temporally consistent saliency. Based on these observations, Efficient-SAM2 is proposed, which eliminates redundant computation via object-aware Sparse Window Routing (SWR) and Sparse Memory Retrieval (SMR), achieving 1.68× end-to-end speedup on SAM2.1-L with only 1% accuracy loss.

## Background & Motivation

**Background**: SAM2 achieves outstanding performance in video object segmentation (VOS) through its streaming memory mechanism, but the high computational cost of its large-scale visual backbone and per-frame memory interaction precludes real-time video processing.

**Limitations of Prior Work**: EdgeTAM achieves edge-level efficiency via knowledge distillation into a lightweight model combined with a spatial perceiver for memory compression, but incurs high training cost and notable performance degradation. General token merging methods such as ToMe are incompatible with SAM2's hierarchical window attention architecture and suffer severe accuracy degradation on segmentation tasks.

**Core Observation — Sparse Perception Patterns**: Through attention matrix visualization, the authors identify two key sources of redundancy:
   - **Encoder–Decoder Attention Mismatch**: The prompt-to-image attention in the mask decoder is highly concentrated on foreground objects and potential distractors, whereas the upstream image encoder, unaware of prompt intent, distributes attention broadly, resulting in substantial wasted computation on background regions.
   - **Temporal Consistency in Memory Frame Sparsity**: During memory attention, when the same memory frame is repeatedly queried by different query frames, attention consistently concentrates on the same small subset of tokens (cosine similarity approaching 1), making full-token recomputation entirely unnecessary.

**Key Insight**: Accelerate SAM2 in a post-training manner without modifying its architecture, exploiting its inherent sparsity to maximize compatibility and deployment convenience.

## Method

### Overall Architecture
Efficient-SAM2 is an object-aware computation scheduling and optimization framework that targets the two primary latency bottlenecks of SAM2 — the image encoder and memory attention — with corresponding sparse acceleration strategies.

### Key Design 1: Sparse Window Routing (SWR)

1. **Object-Aware Router**: Exploits temporal consistency and perceptual saliency from the previous frame to predict object-relevant windows in the current frame: $\mathcal{W}_{obj} = \mathcal{W}_{pred} \oplus \mathcal{W}_{salient}$
    - $\mathcal{W}_{pred}$: Windows covered by the OR of three candidate prediction masks from the previous frame, with dilation applied to prevent boundary escape.
    - $\mathcal{W}_{salient}$: When the tracking confidence $s_{obj}$ falls below a threshold, windows whose cumulative saliency — derived from the previous frame's decoder cross-attention weights — exceeds the threshold are selected, ensuring re-tracking and robustness to distractors.
2. **Lightweight Shortcut Branch**: Background windows bypass the full Transformer block and are routed through a shortcut consisting of only two linear layers (parameter count $d^2+2d$, far fewer than the full block's $12d^2+13d$).
3. **Training**: A reconstruction loss $\mathcal{L} = \|F_M^s - F_M^t\|_2^2$ aligns shortcut outputs with the original memory-conditioned features. Only 30 unlabeled samples from the SA-V training set are required, with training completing in approximately 1 hour on an A6000 GPU.

### Key Design 2: Sparse Memory Retrieval (SMR)

1. **Saliency Pattern Identification at First Recall**: When a memory frame $M_{t-1}$ first participates in memory attention, the average attention weights are computed per layer and the top-$(1-s)K$ tokens are selected as the saliency pattern $S_{t-1}^l$, which is cached in a FIFO retrieval queue.
2. **Saliency Pattern Reuse**: For the subsequent $m+1$ time steps, the cached pattern is directly reused, allowing only the salient tokens to participate in attention computation. This reduces per-layer complexity from $O((m+1)NKd)$ to $O(2NKd + (m-1)Nkd)$, where $k = (1-s)K \ll K$.
3. **Sparsity Configuration**: $s=0.95$ (retaining only 5% of tokens per frame); prompt frames and the most recent frame are kept intact, yielding an overall sparsity ratio of approximately 0.68.

### Loss & Training
The SWR shortcut branch is trained via a simple reconstruction pipeline. Memory-conditioned features are chosen as the reconstruction target — rather than raw encoded features or decoder features — as they contain moderate background information useful for branch learning, without being suppressed by the decoder's strong background inhibition. SMR requires no training.

## Key Experimental Results

### Main Results (SAM2.1-B+, $\Delta t=1$)

| Method | Accelerated Module | SA-V test J&F | Speedup |
|--------|--------------------|---------------|---------|
| SAM2.1-B+ (original) | - | 77.7 | 1.00× |
| ToMe | Encoder | 55.3 | 1.36× |
| ALGM | Encoder | 71.9 | 1.05× |
| **SWR (ours)** | Encoder | **75.0** | **1.69×** |
| MemPool | Memory | 72.3 | 2.14× |
| SMR-random | Memory | 76.7 | 1.73× |
| **SMR (ours)** | Memory | **77.8** | **1.82×** |
| EdgeTAM (distillation) | Both | 72.1 | 1.63× |
| **Efficient-SAM2** | Both | **75.5** | **1.74×** |

### SAM2.1-L Results

| Method | SA-V test J&F | DAVIS 2017 | End-to-End Speedup |
|--------|---------------|------------|--------------------|
| Original | 79.2 | 89.9 | 1.00× |
| SWR | 77.5 | 89.7 | 1.83× |
| SMR | 79.3 | 89.9 | 1.78× |
| **SWR+SMR** | **78.2** | **89.5** | **1.68×** |

### Key Findings
- SWR substantially outperforms all token merging methods for encoder acceleration (ToMe: 56.4 vs. SWR: 75.0), as window-level routing naturally aligns with SAM2's window attention architecture.
- SMR incurs virtually no performance loss at 95% sparsity (77.8 vs. 77.7), validating the temporal consistency hypothesis of memory frame saliency.
- On DAVIS 2017, Efficient-SAM2 exhibits negligible performance degradation (89.3 vs. 89.7), though 1–3 point drops are observed on the more challenging SA-V and MOSE benchmarks.
- Compared to EdgeTAM, Efficient-SAM2 requires no large-scale retraining and achieves higher performance (75.5 vs. 72.1) with slightly greater speedup (1.74× vs. 1.63×).

## Highlights & Insights
- **Post-Training Acceleration Paradigm**: No end-to-end retraining is required; only 30 samples and approximately 1 hour of training are needed for the shortcut branch, substantially lowering the barrier to deployment.
- The design philosophy of working from sparse perception patterns is elegant — rather than forcibly pruning the model, it eliminates redundancy by following the model's own behavioral tendencies.
- SMR's first-recall caching and reuse strategy ingeniously exploits temporal consistency, achieving a remarkably clean design with near-zero overhead.
- SWR leverages attention saliency feedback from the decoder to guide computation allocation in the encoder — a strong exemplar of cross-module information reuse.

## Limitations & Future Work
- SWR relies on the quality of previous-frame predictions to estimate target windows; tracking failures or rapid object motion may lead to cascading degradation.
- The sparsity ratio $s=0.95$ and confidence threshold $\theta_{obj}=5$ are manually set; adaptive adjustment could yield further improvements.
- Validation is limited to the semi-supervised VOS setting; applicability to interactive segmentation and multi-object tracking scenarios remains unexplored.
- The shortcut branch is extremely lightweight and may discard important information in scenes with complex dynamic backgrounds.

## Related Work & Insights
- **vs. EdgeTAM**: Distillation-based approaches require full retraining, whereas the post-training acceleration paradigm proposed here is more flexible and achieves superior performance.
- **vs. ToMe/ALGM**: General ViT acceleration methods fail on SAM2's window attention architecture; window-level routing operates at a more appropriate granularity.
- **vs. MemPool**: Simple pooling-based memory compression loses fine-grained information (72.3), whereas SMR's selective retention strategy is considerably more precise (77.8).

## Rating
- Novelty: ⭐⭐⭐⭐ Targeted acceleration design derived from SAM2's sparse perception patterns, offering a distinctive perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four VOS benchmarks, two model scales, comprehensive ablations, and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between observations and proposed solutions, with intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly practical; post-training acceleration for SAM2 addresses broad industrial demand.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](../../AAAI2026/segmentation/eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[NeurIPS 2025\] Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2](../../NeurIPS2025/segmentation/vanish_into_thin_air_cross-prompt_universal_adversarial_attacks_for_sam2.md)

<!-- RELATED:END -->
