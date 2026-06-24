---
title: >-
  [Paper Note] DepthSplat: Connecting Gaussian Splatting and Depth
description: >-
  [CVPR 2025][3D Vision][Gaussian Splatting] Unifies Gaussian Splatting (3DGS) and depth estimation, two tasks typically studied independently: uses pre-trained monocular depth features to enhance the multi-view depth model, improving 3DGS reconstruction quality, while concurrently leveraging the photometric rendering loss of 3DGS as an unsupervised pre-training target to learn a robust depth model. Both tasks achieve state-of-the-art (SOTA) performance across multiple datasets…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gaussian Splatting"
  - "depth estimation"
  - "multi-view reconstruction"
  - "feed-forward 3D reconstruction"
  - "monocular depth prior"
date: 2026-05-08
content_hash: 1ac765a72fa9f2c6
---

# DepthSplat: Connecting Gaussian Splatting and Depth

**Conference**: CVPR 2025  
**arXiv**: [2410.13862](https://arxiv.org/abs/2410.13862)  
**Code**: [https://github.com/cvg/depthsplat](https://github.com/cvg/depthsplat)  
**Area**: 3D Vision  
**Keywords**: Gaussian Splatting, depth estimation, multi-view reconstruction, feed-forward 3D reconstruction, monocular depth prior

## TL;DR

Unifies Gaussian Splatting (3DGS) and depth estimation, two tasks typically studied independently: uses pre-trained monocular depth features to enhance the multi-view depth model, improving 3DGS reconstruction quality, while concurrently leveraging the photometric rendering loss of 3DGS as an unsupervised pre-training target to learn a robust depth model. Both tasks achieve state-of-the-art (SOTA) performance across multiple datasets.

## Background & Motivation

Feed-forward 3DGS methods (such as MVSplat) rely on multi-view feature matching to locate 3D Gaussian centers, but they underperform in challenging scenarios such as occlusions, textureless regions, and reflective surfaces. On the other hand, while monocular depth models (such as Depth Anything V2) exhibit robust predictions across diverse scenes, they lack cross-view scale consistency, which limits their application in 3D reconstruction. The key insight of this paper is that **the geometric consistency of multi-view depth and the robust prior of monocular depth are complementary**, and integrating them can simultaneously enhance both tasks.

## Method

### Overall Architecture

DepthSplat employs a shared architecture to connect depth estimation and 3DGS. The model consists of two branches: a multi-view feature matching branch (for cost volume construction) and a monocular depth feature extraction branch (from Depth Anything V2). The outputs of both branches are concatenated and passed through a 2D U-Net to regress depth, which is then back-projected into 3D to serve as Gaussian centers. A lightweight head is appended to predict other Gaussian parameters.

### Key Designs

1. **Multi-View Feature Matching Branch (Cost Volume Construction)**:
    - Function: Models multi-view geometric consistency information.
    - Mechanism: A lightweight ResNet extracts features, which exchange information via multi-view Swin Transformer cross-attention. A cost volume $\bm{C}_i \in \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times D}$ is then constructed using plane-sweep stereo, where $D$ represents the number of depth candidates.
    - Design Motivation: The cost volume encodes multi-view photometric consistency, which is core to multi-view depth estimation, but fails to handle textureless and reflective regions.

2. **Monocular Depth Feature Fusion**:
    - Function: Provides robust depth priors for challenging regions.
    - Mechanism: Direct usage of ViT features from Depth Anything V2 (rather than explicit depth maps). These features are bilinearly interpolated to the same resolution as the cost volume and concatenated along the channel dimension before being fed into a U-Net for depth regression.
    - Design Motivation: Compared to complex fusion strategies (such as attention-based fusion or explicit scale alignment), simple concatenation performs surprisingly well (Tab. 3) and avoids error propagation. The key is utilizing features instead of raw depth predictions, allowing the network to adaptively learn the fusion details.

3. **Hierarchical Matching and Unsupervised Pre-training**:
    - Function: Improves resolution and depth accuracy; utilizes 3DGS as an unsupervised target to pre-train the depth model.
    - Mechanism: Employs a 2-scale hierarchical architecture (1/8 + 1/4 resolution) to progressively refine depth from coarse to fine. The 3DGS rendering loss allows the entire depth model to be trained end-to-end using only photometric supervision, enabling unsupervised pre-training on large-scale multi-view datasets.
    - Design Motivation: Hierarchical matching improves efficiency and accuracy. Unsupervised pre-training breaks the bottleneck of depth annotations; fine-tuning after pre-training significantly outperforms training from scratch.

### Loss & Training

- **Depth Estimation**: $L_{\text{depth}} = \alpha |D_{\text{pred}} - D_{\text{gt}}| + \beta (|\partial_x D| + |\partial_y D|)$, with $\alpha=\beta=20$ (in disparity space).
- **3DGS Rendering**: $L_{\text{gs}} = \text{MSE} + 0.05 \cdot \text{LPIPS}$.
- The learning rate for the monocular ViT backbone is set to $2\times10^{-6}$ (low learning rate to preserve pre-trained knowledge), and $2\times10^{-4}$ for other layers.

## Key Experimental Results

### Main Results

| Dataset | Metric | DepthSplat | Prev. SOTA | Gain |
|--------|------|-----------|----------|------|
| ScanNet (2-view depth) | Abs Rel | 3.8 | 4.7 (NeuralRecon) | -19% |
| RealEstate10K (2-view GS) | PSNR | 27.47 | 26.39 (MVSplat) | +1.08dB |
| DL3DV (2-view GS) | PSNR | 21.28 | 19.28 (MVSplat) | +2.00dB |

### Ablation Study

| Configuration | Abs Rel↓ | PSNR↑ | Description |
|------|---------|-------|------|
| Full (ViT-S, 1-scale) | 8.46 | 26.84 | Baseline |
| w/o monocular features | 12.25 | 26.04 | Depth performance drops severely, proving the monocular prior is critical |
| w/o cost volume | 11.34 | 23.24 | PSNR drops dramatically, proving multi-view consistency is indispensable |
| ViT-L, 2-scale | **5.57** | **27.47** | Best performance with larger backbone + hierarchical matching |
| No pre-training + fine-tuning | 10.86 | - | Training from scratch |
| GS pre-training + fine-tuning | **10.20** | - | Unsupervised pre-training improves depth |

### Key Findings

- Removing the monocular features hurts depth prediction performance (+3.79 Abs Rel) more than removing the cost volume (+2.88), but the cost volume is more critical for 3DGS (a PSNR drop of 3.6dB).
- The unsupervised GS pre-training $\rightarrow$ fine-tuning strategy consistently outperforms training from random initialization on TartanAir, ScanNet, and KITTI.
- Supports feed-forward reconstruction using up to 12 input views (at 512×960 resolution) in under 0.6 seconds.

## Highlights & Insights

- **Simple and Effective Fusion Design**: Simply concatenating pre-trained monocular features with the cost volume outperforms various complex fusion schemes, embodying a "less is more" design philosophy.
- **Mutual Benefits**: Bidirectional mutual reinforcement (depth $\rightarrow$ 3DGS and 3DGS $\rightarrow$ depth) creates a virtuous cycle.
- Achieving SOTA requires only 2 days of training on 4 GPUs, far lower than methods like GS-LRM (which requires 64 A100 GPUs for 2 days).

## Limitations & Future Work

- The benefits of unsupervised pre-training are limited on "easy" datasets (e.g., ScanNet) and are primarily significant in challenging scenes.
- Currently, only the top-2 neighboring views are used for cross-attention and cost volume construction; more efficient utilization of additional views is worth exploring.
- The monocular backbone freezes most parameters; joint training might yield further improvements.

## Related Work & Insights

- Competing with MVSplat and pixelSplat, the proposed method achieves significant performance gains through monocular prior integration.
- A key difference from TranSplat lies in early fusion versus late refinement, preventing error propagation.
- The unsupervised depth pre-training approach can be extended to more geometric learning tasks (such as normal estimation, optical flow, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of bidirectionally connecting depth and 3DGS is clear and powerful, though individual components are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive and rigorous experimental design featuring multiple datasets, multiple tasks, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear and well-illustrated, with well-explained methodological motivations.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful and efficient unified framework with open-source code and high reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
