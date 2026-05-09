---
title: >-
  [Paper Note] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation
description: >-
  [CVPR 2026][Model Compression][Multi-view geometry estimation] This paper proposes DAGE, a dual-stream Transformer architecture that decouples global consistency modeling (low-resolution stream) from fine-grained detail preservation (high-resolution stream), fusing them via a lightweight Cross-Attention Adapter. DAGE achieves high-quality depth/point map estimation and camera pose prediction at 2K resolution and over 1000-frame sequences, running 2–28× faster than Pi3 and establishing a new state of the art on video geometry estimation.
tags:
  - CVPR 2026
  - Model Compression
  - Multi-view geometry estimation
  - dual-stream Transformer
  - depth estimation
  - knowledge distillation
  - high-resolution inference
date: 2026-05-08
content_hash: f42f453354441e07
---

# DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation

**Conference**: CVPR 2026
**arXiv**: [2603.03744](https://arxiv.org/abs/2603.03744)
**Code**: [https://github.com/dage-site](https://github.com/dage-site)
**Area**: Model Compression
**Keywords**: Multi-view geometry estimation, dual-stream Transformer, depth estimation, knowledge distillation, high-resolution inference

## TL;DR

This paper proposes DAGE, a dual-stream Transformer architecture that decouples global consistency modeling (low-resolution stream) from fine-grained detail preservation (high-resolution stream), fusing them via a lightweight Cross-Attention Adapter. DAGE achieves high-quality depth/point map estimation and camera pose prediction at 2K resolution and over 1000-frame sequences, running 2–28× faster than Pi3 and establishing a new state of the art on video geometry estimation.

## Background & Motivation

Estimating 3D geometry and camera poses from multi-view images is a fundamental problem in computer vision. Three challenges must be addressed simultaneously: (1) global cross-view consistency, (2) high-resolution fine-grained detail preservation, and (3) computationally efficient scaling to long sequences.

- **Feed-forward multi-view methods** (VGGT, Pi3) achieve cross-view consistency via global attention, but their $O(N^2)$ complexity limits resolution and frame count, resulting in blurred details.
- **Single-view methods** (DepthPro, MoGe2) handle high resolutions but lack multi-view consistency.
- **Video diffusion models** (GeoCrafter) are computationally expensive and generally cannot estimate camera poses.

**Root Cause**: the quadratic complexity of global attention with respect to resolution versus the demand for high-resolution detail preservation. DAGE's **Starting Point**: **decoupling resolution from sequence length**.

## Method

### Overall Architecture

Given $N$ uncalibrated RGB images, DAGE predicts per-frame 3D point maps, camera poses, and global metric scale. The architecture consists of three components: an LR Stream, an HR Stream, and a lightweight Adapter.

### Key Designs

1. **Low-Resolution Stream (LR Stream)**:

    - **Function**: Processes all frames at 252 px to extract globally consistent features and estimate camera poses.
    - **Mechanism**: DINOv2 tokenizer + alternating Frame/Global Attention. Feature distillation from a Pi3 teacher model compensates for information loss at low resolution.
    - **Design Motivation**: Global attention is tractable at low resolution; pose estimation does not require high-frequency detail.

2. **High-Resolution Stream (HR Stream)**:

    - **Function**: Processes each frame independently at native resolution (up to 2K).
    - **Mechanism**: Freezes MoGe2's 24-layer ViT encoder for per-frame independent encoding. Computational cost scales linearly with resolution.
    - **Design Motivation**: Frozen weights preserve zero-shot generalization and prevent overfitting on small datasets.

3. **Lightweight Adapter**:

    - **Function**: Injects globally consistent information from the LR stream into the HR stream.
    - **Mechanism**: Cross-Attention (HR as Q, LR as K/V) followed by Self-Attention to restore intra-frame spatial coherence, stacked in 5 blocks.
    - **Design Motivation**: Cross-Attention naturally supports arbitrary token-count ratios between streams.

4. **RoPE Positional Encoding Strategy**:

    - Self-Attention: interpolated RoPE to stabilize the positional spectrum at high resolutions.
    - Cross-Attention: snap-to-grid mapping of HR tokens to the nearest LR grid cell.
    - **Design Motivation**: Standard RoPE degrades severely beyond training resolution.

### Loss & Training

- Point map $\ell_1$ loss (globally aligned, without confidence weighting)
- Camera pose loss (rotation geodesic distance + translation $\ell_1$)
- Gradient loss (multi-scale Scharr/Laplace filter supervision on inverse-depth gradients, replacing multi-scale alignment)
- Normal loss and distillation loss
- HR ViT is frozen; LR stream is initialized from Pi3; training uses 18 datasets.

## Key Experimental Results

### Main Results: Video Point Map Estimation (Average Rank across 8 Datasets)

| Method | Multi-view | High-Res | Pose | Avg. Rank |
|--------|------------|----------|------|-----------|
| VGGT | Yes | No | Yes | 3.4 |
| Pi3 | Yes | No | Yes | 3.3 |
| GeoCrafter | Yes | Partial | No | 3.9 |
| **DAGE** | **Yes** | **Yes** | **Yes** | **1.6** |

### Ablation Study

| Configuration | Key Change | Observation |
|---------------|------------|-------------|
| Adapter injected at intermediate layers | Consistency degrades | Full global processing is necessary |
| Concatenation instead of CrossAttn | Quality degrades | Fixed scale ratio is insufficient |
| Without gradient loss | Sharpness degrades | Gradient supervision is critical for fine detail |
| MoGe2 multi-scale alignment | Consistency degrades | Per-patch independent alignment breaks cross-view consistency |

### Runtime Efficiency (A100, 100-frame video)

| Method | 540p FPS | 2K FPS | 540p VRAM |
|--------|----------|--------|-----------|
| Pi3 | 32.7 | OOM | 37.3 GB |
| VGGT | 13.5 | OOM | 71.3 GB |
| **DAGE** | **65.4** | **5.6** | **12.4 GB** |

### Key Findings

- Average rank of 1.6 significantly outperforms Pi3 (3.3) and VGGT (3.4).
- Clear advantage in high-resolution scenarios: UrbanSyn Rel error is 47% lower than Pi3.
- 540p speed is 2× that of Pi3; Pi3/VGGT run out of memory at 2K while DAGE achieves 5.6 FPS.
- Pose accuracy at 252 px matches Pi3/VGGT at 518 px.

## Highlights & Insights

- **"Decoupling resolution from sequence length" is the central insight**: global consistency does not require high resolution; detail preservation does not require cross-view attention.
- **Frozen HR ViT + lightweight adapter**: an efficient transfer learning paradigm.
- **Snap-to-grid RoPE**: an elegant solution for cross-scale attention.
- **Gradient loss replacing multi-scale alignment**: maintaining a single global alignment is more important in multi-view settings.

## Limitations & Future Work

- The LR stream is fixed at 252 px, which may be insufficient for certain scenes.
- The method depends on pretrained weights from MoGe2 and Pi3.
- Dynamic scenes (moving objects) have not been evaluated.
- The 5-layer Adapter still incurs memory pressure on extremely long sequences.

## Related Work & Insights

- The alternating attention in Pi3/VGGT forms the foundation of the LR stream; DAGE's contribution lies in constraining it to low resolution.
- MoGe2's coarse-to-fine loss is abandoned (as it disrupts multi-view consistency), reflecting a principled design conflict.
- Knowledge distillation is reframed from "model compression" to "resolution compensation."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-stream decoupling design and snap-to-grid RoPE are insightful and original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 datasets, 4 tasks, detailed ablations, and runtime comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly argued; architecture is systematically described.
- **Value**: ⭐⭐⭐⭐⭐ Addresses practical bottlenecks in high-resolution multi-view geometry estimation with state-of-the-art results and practical efficiency.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](memory_efficient_transfer_learning_with_fading_side_networks.md)
- [\[CVPR 2026\] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[CVPR 2026\] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)
- [\[NeurIPS 2025\] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method](../../NeurIPS2025/model_compression/deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)

<!-- RELATED:END -->
