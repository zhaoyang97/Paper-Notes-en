---
title: >-
  [Paper Note] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] SelfSplat proposes an generalizable 3D Gaussian Splatting framework that is pose-free and 3D prior-free. By unifying self-supervised depth/pose estimation with the 3D-GS representation, coupled with a match-aware pose network and a pose-aware depth refinement module, it significantly outperforms existing pose-free methods on the RealEstate10K, ACID, and DL3DV datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Self-Supervised Learning"
  - "Pose-Free Reconstruction"
  - "Depth Estimation"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: f8f27855a3f9f04c
---

# SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2411.17190](https://arxiv.org/abs/2411.17190)  
**Code**: [https://gynjn.github.io/selfsplat/](https://gynjn.github.io/selfsplat/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Self-Supervised Learning, Pose-Free Reconstruction, Depth Estimation, Novel View Synthesis

## TL;DR

SelfSplat proposes an generalizable 3D Gaussian Splatting framework that is pose-free and 3D prior-free. By unifying self-supervised depth/pose estimation with the 3D-GS representation, coupled with a match-aware pose network and a pose-aware depth refinement module, it significantly outperforms existing pose-free methods on the RealEstate10K, ACID, and DL3DV datasets.

## Background & Motivation

**Background**: NeRF and 3D Gaussian Splatting (3D-GS) have achieved great success in 3D reconstruction and novel view synthesis. To avoid scene-specific optimization, feed-forward generalizable 3D reconstruction models (such as pixelSplat and MVSplat) can predict 3D geometry in a single forward pass. However, these methods still require accurate camera poses as input, which severely limits their application to "in-the-wild" data.

**Limitations of Prior Work**: Recent pose-free methods suffer from various limitations. FlowCAM relies on error-prone pretrained optical flow models; DBARF requires expensive scene-specific fine-tuning; CoPoNeRF is pose-free during inference but still requires GT pose supervision during training; and NeRF-based methods (e.g., FlowCAM, DBARF) incur high inference costs. Furthermore, 3D-GS, as an explicit representation, is highly sensitive to minor errors in 3D positions—even slight deviations can destroy multi-view consistency, making the simultaneous prediction of Gaussian attributes and camera poses an extremely challenging task.

**Key Challenge**: (1) The pose-free setting is inherently ill-posed—it demands accurate 3D reconstruction without GT data and learned geometric information; (2) The explicit representation of 3D-GS is extremely sensitive to pose errors, whereas accurate pose estimation in turn relies on a solid 3D representation, leading to a chicken-and-egg dilemma.

**Goal**: Design a fully self-supervised framework to learn generalizable 3D reconstruction from pose-free monocular videos, without requiring pretrained 3D prior models or scene-specific fine-tuning.

**Key Insight**: The authors observe that self-supervised depth/pose estimation and 3D-GS can mutually benefit each other. The geometric consistency constraints of self-supervised learning help guide the localization of 3D Gaussians, while the high-quality view synthesis capability of 3D-GS in turn improves pose estimation accuracy. The key lies in how to effectively unify both frameworks and resolve the issues arising from their combination.

**Core Idea**: Unify self-supervised depth/pose estimation with pixel-aligned 3D-GS into a single framework, and resolve the issues of inaccurate poses and inconsistent depths in pure self-supervised methods via a match-aware pose network and a pose-aware depth refinement module.

## Method

### Overall Architecture

SelfSplat takes three pose-free images $(I_{c_1}, I_t, I_{c_2})$ as input and consists of four core components: (1) multi-view and monocular dual encoders to extract features; (2) fusion and dense prediction modules to generate depth maps and Gaussian attributes; (3) a match-aware pose network to estimate inter-frame camera poses; and (4) a pose-aware depth refinement module that utilizes estimated poses to optimize depth consistency. Finally, the estimated poses are used to transform the Gaussians from each view into a unified coordinate system, followed by rasterized rendering and joint training with a combination of reprojection and rendering losses.

### Key Designs

1. **Multi-View and Monocular Dual-Encoder Architecture**:

    - **Function**: Combine cross-view matching information with robust monocular depth priors.
    - **Mechanism**: The multi-view branch uses a weight-shared ResNet to extract 4x downsampled features, which are then processed by 6 Swin Transformer blocks for cross-view self/cross-attention, yielding cross-view aware features $F^{\text{mv}}$. The monocular branch uses a ViT initialized with CroCo v2 weights to process each image independently, obtaining robust monocular features $F^{\text{mono}}$. The two features are fused at multiple scales via a DPT module—first downsampling the multi-view features to match the resolution, then using a CNN pyramid + reassemble/fusion blocks to generate dense predictions. Crucially, Gaussian attributes are not generated for the target view $I_t$, forcing the network to learn novel-view generalization.
    - **Design Motivation**: Multi-view matching performs poorly in occluded, textureless, and reflective areas, where monocular features provide complementary, robust estimations. Using CroCo v2 (instead of DepthAnything) avoids introducing 3D priors, preserving a fully self-supervised paradigm.

2. **Match-Aware Pose Estimation Network**:

    - **Function**: Leverage cross-view context to improve pose estimation accuracy.
    - **Mechanism**: A 2D U-Net with cross-attention blocks is used to process the three images and extract match-aware features $F^{\text{ma}}_k \in \mathbb{R}^{H \times W \times 3}$. The match-aware features, raw images, and camera intrinsic ray embeddings $E^{\text{int}} = K^{-1}p(x,y)$ are concatenated and fed into PoseNet to estimate the relative transformation $T_{c_1 \to t} \in SE(3)$ for each image pair. Unlike traditional CNN pose networks, cross-attention allows the network to exploit correspondence information across multiple views.
    - **Design Motivation**: Pure CNN pose networks lack cross-view interaction, making it difficult to establish precise geometric correspondences. Incorporating a match-aware module provides additional cross-view knowledge.

3. **Pose-Aware Depth Refinement Module**:

    - **Function**: Use estimated pose information to optimize cross-view depth consistency.
    - **Mechanism**: The initial depth estimations $\tilde{D}_{c_1}, \tilde{D}_{c_2}$ may be inconsistent across different views, leading to Gaussian overlaps and poor reconstruction quality. The depth refinement module is a lightweight 2D U-Net with cross-attention, which takes the current depth, original image, and the estimated pose encoded by Plücker ray embedding $E^{\text{ext}}(T_{c_1 \to t}) \in \mathbb{R}^{H \times W \times 6}$ as inputs, and outputs a residual depth $\Delta D_k$. The final depth is $D_k = \tilde{D}_k + \Delta D_k$. The pose information provides spatial relationships of surrounding views for depth refinement, making depth estimations across different views more consistent.
    - **Design Motivation**: Independent depth estimation across views lacks consistency; introducing pose as additional context addresses this problem, and residual learning ensures more stable refinement.

### Loss & Training

The total loss is formulated as $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{proj}} + \lambda_2 \mathcal{L}_{\text{ren}}$, where the reprojection loss $\mathcal{L}_{\text{proj}}$ calculates the error between the projected image and the target image using a combination of SSIM and L1. The rendering loss $\mathcal{L}_{\text{ren}}$ calculates the SSIM + L2 error between the 3D-GS rendered image and the input image. For $I_t$, the rendered depth (and not the estimated depth) is used to compute the reprojection loss to maintain scale consistency. Camera intrinsics are assumed to be known (obtained from sensor metadata).

## Key Experimental Results

### Main Results

**RealEstate10K Novel View Synthesis (Average)**:

| Method | Pose-Free | No 3D Prior | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|---------|-------|-------|--------|
| DBARF | ✓ | ✗ | 12.57 | 0.494 | 0.474 |
| FlowCAM | ✓ | ✗ | 22.29 | 0.711 | 0.313 |
| CoPoNeRF | ✗ (requires GT for training) | ✓ | 21.03 | 0.693 | 0.256 |
| **SelfSplat** | **✓** | **✓** | **24.22** | **0.813** | **0.188** |

**ACID Novel View Synthesis (Average)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| FlowCAM | 25.59 | 0.721 | 0.294 |
| **SelfSplat** | **26.71** | **0.801** | **0.196** |

### Ablation Study

Detailed ablation studies were conducted to verify the contributions of individual components (match-aware pose network, depth refinement module, monocular encoder, etc.). The specific data are available in the ablation tables of the complete paper.

### Key Findings

- SelfSplat comprehensively outperforms all baselines in the most demanding setting (pose-free + no 3D prior + no fine-tuning).
- On RE10K, the PSNR is 1.93 dB higher than FlowCAM (the strongest pose-free method), and LPIPS is reduced by 40%.
- It demonstrates strong cross-dataset generalization: when trained on RE10K, it still performs well when evaluated on DL3DV.
- Its advantages are even more pronounced in large-baseline (large view changes) scenes.

## Highlights & Insights

- The "self-supervised + 3D-GS mutual benefit" design concept is the core highlight: geometric consistency constraints in self-supervised learning help locate Gaussians, while high-quality rendering of 3D-GS helps refine pose estimation, establishing a positive feedback loop.
- The depth refinement module utilizing pose information is a clever design—feeding the pose estimation outputs back into the depth estimation establishes a bidirectional information flow between components.
- The fully self-supervised design (independent of pretrained 3D models like DepthAnything) makes the method more academically significant and generalizable.
- Based on 3D-GS, it achieves fast inference without requiring expensive volume rendering like NeRF-based methods.

## Limitations & Future Work

- Assuming known camera intrinsics limits its application under fully unknown settings.
- Scale ambiguity in self-supervised depth estimation remains an issue.
- The three-frame input setup limits the scale of reconstructed scenes.
- There is still room for improvement in scenarios with extreme viewpoint changes or heavy occlusions.

## Related Work & Insights

- Combining the self-supervised paradigm in monocular depth estimation (SfMLearner, Monodepth2) with 3D-GS is a natural yet effective direction.
- CroCo's cross-view completion pretraining provides effective monocular features while avoiding the introduction of 3D priors.
- The concept of the match-aware pose network can be applied to other tasks requiring precise frame-to-frame alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified framework of self-supervision and 3D-GS is novel, and the bidirectional information flow between match-aware pose and pose-aware depth has depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensively validated on three large-scale real-world datasets, including cross-dataset generalization and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear methodology description, sufficient experimental analysis, and a fair, comprehensive comparison with existing methods.
- Value: ⭐⭐⭐⭐ — Addresses an important limitation of 3D-GS (pose dependency) and promotes the development of pose-free 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[CVPR 2025\] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)

</div>

<!-- RELATED:END -->
