---
title: >-
  [Paper Note] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection
description: >-
  [AAAI 2026][Segmentation][SAM2] This paper proposes SAM-DAQ, which adapts SAM2 to RGB-D video salient object detection (VSOD) via a Depth-guided Parallel Adapter (DPA) and a Query-driven Temporal Memory (QTM) module, addressing three key challenges: manual prompt dependency, excessive GPU memory consumption, and computational overhead.
tags:
  - AAAI 2026
  - Segmentation
  - SAM2
  - RGB-D Salient Object Detection
  - Video Understanding
  - Depth Guidance
  - Query-driven Memory
date: 2026-05-08
content_hash: 5ad8738875c6df58
---

# SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection

**Conference**: AAAI 2026
**arXiv**: [2511.09870](https://arxiv.org/abs/2511.09870)
**Code**: [https://github.com/LinJ0866/SAM-DAQ](https://github.com/LinJ0866/SAM-DAQ)
**Area**: Segmentation
**Keywords**: SAM2, RGB-D Salient Object Detection, Video Understanding, Depth Guidance, Query-driven Memory

## TL;DR

This paper proposes SAM-DAQ, which adapts SAM2 to RGB-D video salient object detection (VSOD) via a Depth-guided Parallel Adapter (DPA) and a Query-driven Temporal Memory (QTM) module, addressing three key challenges: manual prompt dependency, excessive GPU memory consumption, and computational overhead.

## Background & Motivation

Video salient object detection (VSOD) aims to identify the most visually attractive objects in a video. RGB-D VSOD incorporates depth information to leverage spatial structure, effectively mitigating challenges such as cluttered backgrounds, occlusion, and low illumination. However, directly applying SAM2 to RGB-D VSOD faces **three critical issues**:

**Manual prompt dependency**: SAM2 requires user-provided prompts (points, boxes, or masks) to guide segmentation, which are unavailable during RGB-D VSOD inference. Existing prompt-free strategies—such as generating pseudo-prompts or using only the encoder for feature extraction—either yield limited performance or fail to fully exploit SAM2's architectural advantages.

**High GPU memory from serial adapters**: Existing parameter-efficient fine-tuning methods (e.g., inserting serial adapters or LoRA between Transformer blocks) incur extremely high training memory (up to 91–95 GB) due to backpropagation gradients traversing the entire encoder.

**Computational overhead of memory attention**: SAM2's memory mechanism captures inter-frame dependencies via a memory bank, but computing associations between the current frame features and a large memory bank is computationally expensive.

The core approach addresses these via: (1) replacing serial adapters with parallel skip-connection adapters to substantially reduce memory usage; and (2) replacing the memory bank and prompt embeddings with learnable queries to unify temporal modeling and prompt generation.

## Method

### Overall Architecture

SAM-DAQ builds on SAM2-Large and consists of three core components:

1. **Parallel Adapter Multimodal Image Encoder (PAMIE)**: Integrates depth-guided parallel adapters via skip connections into a frozen encoder, enabling prompt-free fine-tuning and RGB-D feature fusion.
2. **Query-driven Temporal Memory module (QTM)**: Replaces the memory bank and prompt embeddings with frame-level and video-level learnable queries that selectively extract temporally consistent features.
3. **Mask Decoder**: Retains the original SAM2 mask decoder.

### Key Designs

#### 1. **Parallel Adapter Multimodal Image Encoder (PAMIE)**

**Depth Adapter**: A skip-connection adapter is inserted between the input and output of each Hiera block:

$$\tilde{\mathbf{F}}_D^{i-1} = \text{Adapter}(\mathbf{F}_D^{i-1})$$
$$\mathbf{F}_D^i = \text{Hiera}^i(\mathbf{F}_D^{i-1}) + \text{DS}(\tilde{\mathbf{F}}_D^{i-1})$$

where the Adapter consists of a down-projection linear layer, an activation function, and an up-projection linear layer, and DS denotes bilinear downsampling.

**Depth-guided Parallel Adapter (DPA)**: RGB and depth features are concatenated before being fed into the adapter:

$$\tilde{\mathbf{F}}_{RGB}^{i-1} = \text{Adapter}(\text{Cat}(\mathbf{F}_{RGB}^{i-1}, \mathbf{F}_D^{i-1}))$$
$$\mathbf{F}_{RGB}^i = \text{Hiera}^i(\mathbf{F}_{RGB}^{i-1}) + \text{DS}(\tilde{\mathbf{F}}_{RGB}^{i-1})$$

**Design Motivation**: The parallel skip connection allows gradients to bypass the heavy Transformer computation during backpropagation, reducing GPU memory from 91.9 GB (serial adapter) to 21.0 GB. RGB-D feature fusion is achieved through the concatenation operation within the adapter.

After an FPN, three-level image embeddings $\mathbf{E}_I = \{\mathbf{E}_I^i\}_{i=2}^{4}$ are produced. A **self-inference scheme** is also introduced: lightweight convolutions followed by sigmoid generate intermediate predictions at each level, with supervision applied only at the highest level.

#### 2. **Query-driven Temporal Memory Module (QTM)**

Two sets of learnable queries are introduced:

- **Frame-level queries** $\mathbf{Q}_f \in \mathbb{R}^{N_f \times c}$ ($N_f = 30$): Static queries that interact with the highest-level image embedding of each frame to extract saliency-relevant frame features.
- **Video-level queries** $\mathbf{Q}_v \in \mathbb{R}^{N_v \times c}$ ($N_v = 8$): Dynamic queries that are iteratively updated across frames to capture temporal dependencies.

The interaction process is formulated as:

$$\mathbf{E}_f = \text{Linear}(\mathbf{Q}_f' \cdot \mathbf{E}_I^4)$$
$$\tilde{\mathbf{Q}}_v = \text{CA}(\mathbf{Q}_v', \mathbf{E}_f) + \mathbf{Q}_v'$$

The video-level queries $\tilde{\mathbf{Q}}_v$ are element-wise multiplied with $\mathbf{E}_I^4$ to produce learnable embeddings $\mathbf{E}_L$, which replace SAM's sparse prompt embeddings.

**Temporal Update Mechanism**: SAM2's memory encoder processes the current frame's image embedding and prediction:

$$\mathbf{F}_m = \text{Linear}(\text{ME}(\mathbf{E}_{I,t}, \mathbf{P}_t))$$
$$\mathbf{Q}_{v,t+1} = \mathbf{Q}_{v,t} + \text{FFN}(\text{SA}(\text{CA}(\mathbf{Q}_{v,t}, \mathbf{F}_m)))$$

**Design Motivation**: (1) Frame-level queries selectively attend to visually salient regions via token-level attention rather than dense pixel-level feature matching. (2) Video-level queries establish temporal dependencies through iterative updates, replacing the high computational cost of a large memory bank. (3) Using sparse embeddings alone (rather than dense embeddings or a combination) yields the best performance, as the token-level interaction of the queries is structurally consistent with the sparse prompt paradigm used during SAM pretraining.

#### 3. **Mask Decoder**

The original SAM2 mask decoder is retained, receiving learnable embeddings $\mathbf{E}_L$ and multi-level image embeddings $\mathbf{E}_I$ to produce the final segmentation output.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \alpha \cdot \mathcal{L}_{inter}$$

- $\mathcal{L}_{pred}$: BCE loss on the final prediction.
- $\mathcal{L}_{inter}$: BCE loss on the intermediate predictions.
- $\alpha$: Weight for the intermediate loss.

Training settings: The SAM encoder is fully frozen; only the adapters and QTM modules are trained (19.2M trainable parameters). Input resolution is 1024×1024; 10 frames are randomly sampled per video. AdamW optimizer (lr=0.0001, weight decay=0.05), batch size=1, 2000 training iterations. **Training completes in only 3 hours on a single RTX-3090 (24 GB).**

## Key Experimental Results

### Main Results

**Quantitative comparison on three datasets**:

| Method | Source | RDVS $E_\xi$↑ | RDVS $S_\alpha$↑ | ViDSOD-100 $F_\beta$↑ | ViDSOD-100 MAE↓ | DViSal $F_\beta$↑ | DViSal MAE↓ |
|--------|--------|-----------|-----------|------------|---------|-----------|---------|
| DCTNet+ | TIP'24 | 0.909 | 0.876 | 0.809 | 0.030 | 0.689 | 0.095 |
| MDSAM | MM'24 | 0.813 | 0.791 | 0.815 | 0.026 | 0.715 | 0.071 |
| SAM2-UNet | arXiv'24 | 0.888 | 0.843 | 0.829 | 0.025 | 0.747 | 0.064 |
| KAN-SAM | ICME'25 | 0.888 | 0.854 | 0.846 | 0.025 | 0.783 | 0.052 |
| **SAM-DAQ** | **Ours** | **0.913** | **0.879** | **0.868** | **0.020** | **0.818** | **0.046** |

Compared to KAN-SAM, SAM-DAQ achieves average improvements of 1.5%, 1.0%, and 2.4% in E-measure, S-measure, and F-measure, respectively.

### Ablation Study

**PAMIE ablation (RDVS dataset)**:

| Configuration | Trainable/Total Params (M) | Memory (GB) | $E_\xi$ | $S_\alpha$ | $F_\beta$ | MAE |
|---------------|---------------------------|-------------|---------|---------|---------|-----|
| w/o depth projector | - | 20.3 | 0.899 | 0.870 | 0.808 | 0.023 |
| Serial adapter | 17.4/236.0 | **91.9** | 0.860 | 0.830 | 0.778 | 0.028 |
| LoRA | 56.0/274.6 | **95.0** | 0.889 | 0.877 | 0.824 | 0.027 |
| w/o multimodal fusion | - | 17.9 | 0.876 | 0.853 | 0.782 | 0.029 |
| **DPA (Ours)** | **19.2/237.9** | **21.0** | **0.913** | **0.879** | **0.827** | **0.026** |

**QTM embedding strategy ablation**:

| Strategy | $E_\xi$ | $S_\alpha$ | $F_\beta$ | MAE | Note |
|----------|---------|---------|---------|-----|------|
| Sparse embedding (Ours) | **0.913** | **0.879** | **0.827** | **0.026** | Best |
| Dense embedding | 0.875 | 0.856 | 0.783 | 0.032 | Incompatible |
| Both combined | 0.862 | 0.839 | 0.763 | 0.033 | Conflict degrades performance |

**Update mechanism ablation**:

| Strategy | $E_\xi$ | $F_\beta$ | Note |
|----------|---------|---------|------|
| No update | 0.883 | 0.788 | Temporal information absent |
| SAM2 memory bank | 0.853 | 0.796 | Conventional approach |
| Multiplicative update | 0.895 | 0.804 | Suboptimal |
| **Additive update (Ours)** | **0.913** | **0.827** | Best |

### Key Findings

- **Parallel vs. serial adapter**: The parallel skip connection reduces memory from 91.9 GB to 21.0 GB (a 77% reduction) while achieving superior performance ($E_\xi$: 0.913 vs. 0.860).
- **Sparse-only embedding is optimal**: Using sparse embeddings alone outperforms dense or combined strategies, as QTM's token-level interaction is structurally aligned with SAM's pretraining sparse prompt paradigm.
- **Query count sensitivity**: 8 video-level queries and 30 frame-level queries constitute the optimal configuration; reducing video-level queries to 5 or increasing to 10 both degrade performance.
- **Hidden dimension of 64 is optimal**: Both larger and smaller values lead to performance degradation.
- **Intermediate supervision is effective only at the highest level**: Adding supervision at lower levels degrades overall performance.
- **Additive update outperforms multiplicative update and the original SAM2 memory bank.**

## Highlights & Insights

1. The **parallel skip-connection adapter** is an elegant solution to the memory bottleneck in SAM fine-tuning: gradients bypass the frozen Transformer, reducing memory by more than 4×.
2. The **query-driven design unifies prompt generation and temporal modeling** — two seemingly distinct problems — within a single framework: learnable queries serve simultaneously as SAM prompt embeddings and cross-frame memory representations.
3. **Exceptional training efficiency**: only 3 hours on a single RTX-3090, which is highly practical for real-world deployment.
4. The finding that **sparse-only embeddings outperform dense embeddings** is insightful: the sparse prompt paradigm used during SAM pretraining dictates that structural consistency should be maintained during fine-tuning.

## Limitations & Future Work

- Validation is limited to RGB-D VSOD; extension to multi-object video segmentation or other multimodal tasks remains unexplored.
- The number of video-level queries is fixed and cannot adapt to video length or scene complexity.
- The effect of depth map quality variation on performance is not thoroughly analyzed.
- Optical flow and other motion cues are not incorporated, which may limit performance in fast-motion scenarios.
- Frame-level queries are static; dynamic frame-level queries represent a promising direction for future exploration.

## Related Work & Insights

- The memory efficiency of parallel adapters can be generalized to all SAM/SAM2-based fine-tuning scenarios.
- Query-driven memory can inspire temporal modeling designs in other video foundation models.
- The Cat-Adapter design for depth-guided fusion is concise yet effective, and can serve as a lightweight scheme for multimodal fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of parallel adapters and query-driven memory is novel and addresses practical engineering bottlenecks.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three datasets, comprehensive ablations, parameter sensitivity analysis, and memory usage comparisons.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with well-defined problem formulation.)
- Value: ⭐⭐⭐⭐ (The memory efficiency gains are of significant practical value, though the task scope is relatively narrow.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[CVPR 2026\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](../../CVPR2026/segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[AAAI 2026\] Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries](vista_scene-aware_optimization_for_streaming_video_question_answering_under_post.md)

</div>

<!-- RELATED:END -->
