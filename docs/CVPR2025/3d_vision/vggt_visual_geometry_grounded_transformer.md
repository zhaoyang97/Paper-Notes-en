---
title: >-
  [Paper Note] VGGT: Visual Geometry Grounded Transformer
description: >-
  [CVPR 2025][3D Vision][3D Reconstruction] VGGT is a large feed-forward Transformer that directly predicts camera parameters, depth maps, point clouds, and 3D point trajectories from one to hundreds of images in less than a second, outperforming existing methods without post-processing optimization.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Reconstruction"
  - "Feed-Forward Transformer"
  - "Multi-Task Learning"
  - "Depth Estimation"
  - "Point Cloud Tracking"
date: 2026-05-08
content_hash: 3c8e56238603760e
---

# VGGT: Visual Geometry Grounded Transformer

**Conference**: CVPR 2025  
**arXiv**: [2503.11651](https://arxiv.org/abs/2503.11651)  
**Code**: [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)  
**Area**: 3D Vision  
**Keywords**: 3D Reconstruction, Feed-Forward Transformer, Multi-Task Learning, Depth Estimation, Point Cloud Tracking

## TL;DR

VGGT is a large feed-forward Transformer that directly predicts camera parameters, depth maps, point clouds, and 3D point trajectories from one to hundreds of images in less than a second, outperforming existing methods without post-processing optimization.

## Background & Motivation

Traditional 3D reconstruction relies on visual geometry methods and iterative optimization (such as Bundle Adjustment), which are computationally expensive and involve complex pipelines. Although recent methods like DUSt3R/MASt3R have made progress, they can only handle two images, and still require post-processing to fuse pairwise reconstruction results when handling multiple images.

The Core Problem proposed in this paper is: **Can a pure feed-forward neural network directly complete all tasks of 3D reconstruction?** The authors believe that with the enhanced capability of networks and the accumulation of large-scale 3D annotated data, end-to-end methods without geometric post-processing have become feasible.

Limitations of Prior Work:
- DUSt3R/MASt3R require 10+ seconds of global alignment post-processing.
- Existing multi-view 3D methods usually focus only on a single task (such as monocular depth estimation or novel view synthesis).
- There is no unified model that can simultaneously predict all key 3D properties with high quality.

The Goal of VGGT is to build a **general 3D foundation model** similar to GPT in NLP and DINO in computer vision, which can serve as a backbone to enhance downstream tasks.

## Method

### Overall Architecture

VGGT adopts a standard large Transformer architecture (about 1.2 billion parameters). It uses DINO as the image encoder to transform input images into tokens, which are then processed by an **Alternating-Attention (AA)** mechanism. Finally, it outputs camera parameters $\mathbf{g}_i$, depth maps $D_i$, point clouds $P_i$, and tracking features $T_i$ through different prediction heads. The first image is chosen as the world reference coordinate system.

### Key Designs

**1. Alternating-Attention (AA) Mechanism**

- **Function**: Alternates between intra-frame self-attention and global self-attention to balance intra-frame feature normalization and cross-frame information fusion.
- **Mechanism**: Intra-frame attention processes tokens $t_k^I$ of each frame individually, while global attention jointly processes tokens $t^I$ from all frames. An alternating structure of $L=24$ layers is used.
- **Design Motivation**: Pure global self-attention suffers from performance degradation due to the lack of intra-frame normalization. Although cross-attention provides strong information fusion, its performance is inferior to self-attention variants. Ablation studies show that AA reduces the Overall metric on ETH3D from 1.061/0.827 to 0.709.

**2. Over-complete Predictions**

- **Function**: Simultaneously trains and predicts camera parameters, depth maps, point clouds, and point trajectories, even though mathematical redundancy exists among them.
- **Mechanism**: Joint training is performed using a multi-task loss $\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda\mathcal{L}_{\text{track}}$. During inference, point clouds unprojected from depth and camera parameters are more accurate than those directly output by the point cloud head.
- **Design Motivation**: Although point clouds can be derived from depth maps and camera parameters, joint training on these correlated tasks mutually enhances accuracy. Ablation studies confirm that removing any task degrades point cloud accuracy.

**3. Learnable Camera Tokens and Register Tokens**

- **Function**: Allows the Transformer to distinguish the first frame (reference coordinate system) from other frames and provides additional information capacity.
- **Mechanism**: One camera token $t_i^{\mathbf{g}}$ and four register tokens $t_i^R$ are appended to each frame. The first frame is initialized with a different learnable token than other frames. The output camera token predicts camera parameters through 4 additional self-attention layers.
- **Design Motivation**: Distinct initialization tokens allow the network to perceive the coordinate system convention (first frame is identity), and intra-frame self-attention aligns the camera token with the corresponding frame's image tokens.

### Loss & Training

The total loss is the sum of four terms: $\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + 0.05 \cdot \mathcal{L}_{\text{track}}$

- **Camera Loss**: Huber loss $\sum_{i=1}^{N} \|\hat{\mathbf{g}}_i - \mathbf{g}_i\|_\epsilon$
- **Depth Loss**: Uncertainty-weighted reconstruction loss + gradient loss + uncertainty regularization term
- **Point Cloud Map Loss**: Same structure as the depth loss but utilizing point cloud uncertainty
- **Tracking Loss**: L1 correspondence position error + visibility binary cross-entropy

## Key Experimental Results

### Main Results: Camera Pose Estimation (RealEstate10K & CO3Dv2, 10 frames)

| Method | Re10K AUC@30↑ | CO3Dv2 AUC@30↑ | Time |
|------|-------------|---------------|------|
| DUSt3R | 67.7 | 76.7 | ~7s |
| MASt3R | 76.4 | 81.8 | ~9s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| Fast3R | 72.7 | 82.5 | ~0.2s |
| **VGGT (FF)** | **85.3** | **88.2** | **~0.2s** |
| VGGT + BA | 93.5 | 91.8 | ~1.8s |

### Ablation Study: Impact of Architectural Designs on Point Cloud Estimation (ETH3D)

| Architectural Variant | Acc.↓ | Comp.↓ | Overall↓ |
|---------|-------|--------|----------|
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attn Only | 1.032 | 0.621 | 0.827 |
| **Alternating-Attn** | **0.901** | **0.518** | **0.709** |

### Key Findings

- **Without post-processing**, VGGT outperforms all baselines that require optimization by more than 6 AUC points in camera pose estimation.
- Depth estimation on DTU sees a significant drop in Overall error from DUSt3R's 1.741 to 0.382, matching methods that use ground-truth camera parameters.
- Outperforms specialized method Roma on ScanNet two-view matching (AUC@5: 33.9 vs 31.8).
- Joint training on multiple tasks benefits each sub-task; removing any single task degrades overall performance.
- Demonstrates strong capabilities in dynamic point tracking and novel view synthesis after being fine-tuned as a feature backbone.

## Highlights & Insights

1. **Unified Foundation Model Paradigm**: Analogs the 3D reconstruction problem to GPT in NLP and DINO in computer vision, proving the feasibility of a general 3D foundation model. The "scaling law" strategy with 1.2 billion parameters is implemented with comprehensive success in the 3D domain for the first time.
2. **Minimalist Design Philosophy**: Relies purely on standard Transformer + alternating attention, without any 3D inductive biases (no cross-attention, no geometric constraint modules), letting data-driven learning achieve 3D understanding.
3. **Decomposition at Inference Outperforms Direct Prediction**: Although the point cloud map is jointly supervised during training, unprojecting depth and camera parameters during inference yields more accurate results than directly using the point cloud head—a counter-intuitive but practical finding.

## Limitations & Future Work

- The model requires a massive amount of 3D annotated data and computational resources (training for 9 days on 64 A100 GPUs).
- Currently processes only static scenes, lacking explicit modeling for dynamic objects (although partially resolvable through fine-tuning).
- Robustness in extreme cases with no overlap or severe occlusion remains to be verified.
- Future directions: more efficient training strategies, integration with language models for 3D understanding and interaction, and extension to video and dynamic scenes.

## Related Work & Insights

- **Relationship with DUSt3R/MASt3R**: VGGT inherits the idea of point cloud map representation, but completely eliminates the need for post-processing through multi-frame processing and alternating attention.
- **Relationship with VGGSfM**: Work from the same research group (Oxford VGG). VGGSfM uses end-to-end differentiable BA, whereas VGGT replaces it with a pure feed-forward mechanism.
- **Insights for 3D Foundation Models**: Proves that a sufficiently large Transformer with enough 3D data can learn "implicit multi-view triangulation," without requiring explicit geometric reasoning modules.

## Rating

⭐⭐⭐⭐⭐

A blockbuster joint work by Meta AI and Oxford VGG, achieving for the first time a true "one feed-forward model to solve all 3D tasks". It achieves comprehensive SOTA on multiple benchmarks with open-sourced code, exerting a profound impact on subsequent 3D foundation model research. The only limitation is the extremely high barrier of training resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](../../ICLR2026/3d_vision/quantized_visual_geometry_grounded_transformer.md)
- [\[CVPR 2025\] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [\[CVPR 2026\] OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer](../../CVPR2026/3d_vision/omnivggt_omni-modality_driven_visual_geometry_grounded_transformer.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)

</div>

<!-- RELATED:END -->
