---
title: >-
  [Paper Note] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][Visual Foundation Models] This paper proposes Feat2GS, a unified framework that decodes 2D features of Visual Foundation Models (VFMs) into 3D Gaussian attributes via a lightweight MLP. It probes the geometric and texture awareness of VFMs individually on the novel view synthesis task, comprehensively evaluating the 3D awareness of over 10 VFMs on large-scale diverse datasets without requiring 3D ground-truth data.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Visual Foundation Models"
  - "3D Perception"
  - "Gaussian Splatting"
  - "Novel View Synthesis"
  - "Probing Evaluation"
date: 2026-05-08
content_hash: 34a5c0435e122619
---

# Feat2GS: Probing Visual Foundation Models with Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2412.09606](https://arxiv.org/abs/2412.09606)  
**Code**: [https://github.com/fanegg/Feat2GS](https://github.com/fanegg/Feat2GS)  
**Area**: 3D Vision  
**Keywords**: Visual Foundation Models, 3D Perception, Gaussian Splatting, Novel View Synthesis, Probing Evaluation

## TL;DR
This paper proposes Feat2GS, a unified framework that decodes 2D features of Visual Foundation Models (VFMs) into 3D Gaussian attributes via a lightweight MLP. It probes the geometric and texture awareness of VFMs individually on the novel view synthesis task, comprehensively evaluating the 3D awareness of over 10 VFMs on large-scale diverse datasets without requiring 3D ground-truth data.

## Background & Motivation

**Background**: Visual Foundation Models (such as DINOv2, CLIP, DUSt3R, SAM, RADIO, etc.) are trained on massive 2D datasets and widely used for feature extraction in 3D-related tasks. These models exhibit significant differences in architecture (ViT, UNet), training strategies (contrastive learning, self-distillation, pointmap regression, denoising), and training data (2D vs. 3D).

**Limitations of Prior Work**: Existing 3D probing methods mainly fall into two categories: (1) single-view 2.5D estimation (depth/normal), and (2) two-view sparse 2D correspondence (matching/tracking). Both have severe limitations: they neglect the evaluation of texture awareness and require 3D ground-truth annotations, which heavily limits the scale and diversity of evaluation data. More critically, there is a lack of a unified and fair comparison framework across different VFMs.

**Key Challenge**: Comprehensively evaluating the 3D awareness of VFMs requires covering both geometry and texture, using dense rather than sparse evaluation, and employing large-scale, diverse datasets—yet existing methods cannot simultaneously satisfy these three requirements.

**Goal**: To design a unified 3D perception probing framework that can: (1) individually evaluate geometric and texture awareness, (2) perform dense, pixel-level evaluation, (3) require only 2D multi-view images, and (4) support sparse-view images captured in-the-wild.

**Key Insight**: The parameters of 3D Gaussian Splatting (3DGS) are naturally divided into geometric attributes (position $\mathbf{x}$, opacity $\alpha$, covariance $\Sigma$) and texture attributes (spherical harmonics coefficients $\mathbf{c}$). This decoupled characteristic can be directly utilized to separately probe the geometric and texture perception of VFMs.

**Core Idea**: Decoding 3DGS attributes from VFM features using a lightweight MLP, using novel view synthesis (NVS) quality (PSNR/SSIM/LPIPS) as proxy metrics for 3D awareness, to comprehensively evaluate 10 VFMs across 7 diverse datasets.

## Method

### Overall Architecture
The pipeline of Feat2GS consists of: (1) inputting uncalibrated, sparse multi-view images; (2) extracting feature maps for each image using pre-trained VFMs, performing PCA to reduce the channels to 256, and bilinearly upsampling to $512$ resolution; (3) initializing camera poses and point clouds using DUSt3R; (4) employing a 2-layer MLP readout layer to regress 3DGS attributes from VFM features; (5) executing differentiable rasterization rendering, optimizing the readout layer and camera parameters using photometric losses; and (6) computing NVS quality metrics on unseen test viewpoints.

### Key Designs

1. **GTA Three-Mode Probing (Geometry/Texture/All Probing)**:

    - **Function**: Individually evaluate the geometric and texture awareness of VFMs.
    - **Mechanism**: Three modes are introduced—**Geometry Mode**: The MLP decodes geometric parameters $\{x_i, \alpha_i, \Sigma_i\} = g_\Theta^{(G)}(f_i)$ from VFM features, leaving texture parameters freely optimized; **Texture Mode**: The MLP decodes texture parameters $\{c_i\} = g_\Theta^{(T)}(f_i)$ with geometric parameters freely optimized; **All Mode**: The MLP simultaneously decodes all parameters. By comparing the NVS quality across the three modes, the strengths and weaknesses of different VFMs in terms of geometry and texture can be precisely pinpointed.
    - **Design Motivation**: This is the core contribution of this work—leveraging the natural decoupling of 3DGS parameters to run independent evaluations for both kinds of 3D awareness.

2. **Lightweight Readout Layer Design**:

    - **Function**: Decoding 3DGS attributes from VFM features while preventing overfitting.
    - **Mechanism**: Utilizing only a 2-layer MLP (with a 256-dimensional hidden layer and ReLU activation). The network capacity is intentionally constrained to ensure that the 3DGS parameters are truly "read out" from the VFM features, rather than being learned by the network itself. This stands in sharp contrast to InstantSplat's free optimization, which is prone to overfitting under sparse views due to its millions of optimized parameters.
    - **Design Motivation**: To guarantee the fairness of probing—ensuring that different VFMs use identical readout architectures and training configurations.

3. **Warm Start Strategy**:

    - **Function**: Avoiding local optima when directly decoding 3D structures from 2D features.
    - **Mechanism**: Pre-training the readout layer for 1K steps with $\min_\Theta \|g_\Theta(f) - G_{init}\|$ target, using point clouds reconstructed by DUSt3R as the target, before switching to photometric loss optimization for another 7K steps. This provides matching initialization conditions for all VFMs.
    - **Design Motivation**: Direct optimization under sparse views is highly prone to failure; the warm start ensures that features from different VFMs can converge to reasonable solutions.

### Loss & Training
The optimization target is the photometric loss $\min_{\Theta,T} \|\mathcal{R}(g_\Theta(f), T) - \mathcal{I}\|$, which simultaneously optimizes MLP parameters, 3DGS parameters, and camera parameters. The Adam optimizer is used, with the MLP learning rate decayed from $10^{-2}$ to $10^{-4}$. Adaptive density control is omitted. All experiments are conducted on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results (Average NVS Metrics across 7 Datasets, Geometry Mode)

| VFM | Training Strategy | Training Data | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ |
|-----|---------|---------|-----------|-----------|------------|
| RADIO | Multi-teacher distillation | DataComp-1B | **Highest** | **Highest** | **Lowest** |
| MASt3R | Pointmap regression | 3D Hybrid | 2nd | 2nd | 2nd |
| DUSt3R | Pointmap regression | 3D Hybrid | 3rd | 3rd | 3rd |
| DINO | Self-distillation | ImageNet-1K | 4th | 4th | Medium |
| DINOv2 | Self-distillation | LVD-142M | Medium | Medium | Medium |
| CLIP | Contrastive VLM | WIT-400M | Medium | Medium | Medium |
| SD | Denoising VLM | LAION | Lowest | Lowest | Highest |

### Ablation Study (NVS Baseline Comparisons, Averaged over All Datasets)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| InstantSplat (SOTA) | 18.87 | 0.6044 | 0.3039 |
| Feat2GS w/ RADIO | 19.73 | 0.6513 | 0.3143 |
| Feat2GS w/ concat all | 19.80 | 0.6545 | 0.3105 |
| Feat2GS w/ DUSt3R | 19.66 | 0.6469 | 0.3247 |
| Feat2GS w/ DUSt3R* (Fine-tuned) | **19.75** | **0.6561** | **0.2928** |

### Key Findings
- **Geometric Awareness Top 4**: RADIO > MASt3R > DUSt3R > DINO. Training on 3D data (pointmap regression) is critical for geometric awareness, and depth regression (MiDaS) is far inferior to pointmap regression (DUSt3R).
- **Texture Awareness Top 3**: MAE > SAM > MASt3R. Masked image reconstruction pre-training helps preserve texture information, whereas RADIO, despite having the strongest geometry, has the poorest texture—reflecting the distilled texture invariance from DINO/CLIP.
- **All Mode dragged down by texture**: LPIPS deteriorates by +0.05 on average, and the blurry rendering in the All Mode stems from insufficient VFM texture perception.
- **Feature concatenation is effective**: Simply concatenating DINOv2+CLIP+SAM yields geometric results comparable to RADIO (which distills them); concatenating the best geometric feature with the best texture feature (RADIO+MAE+IUVRGB) can outperform the best single VFM.
- **2D metrics correlate strongly with 3D metrics**: On the DTU dataset, NVS quality strongly correlates with point cloud accuracy/completeness, validating the rationality of NVS as a proxy for 3D evaluation.

## Highlights & Insights
- The GTA three-mode probing is an exceptionally clever design—by leveraging the natural decoupling of 3DGS parameters, it independently evaluates geometric and textural 3D perception without extra overhead. This mindset of "employing intrinsic properties of existing tools to address research problems" is worth learning.
- The finding that "VFM texture awareness is generally poor" is crucial. Many VFMs sacrifice texture information for semantic understanding or geometric robustness, which limits their application in photometric consistency tasks.
- Simple feature concatenation outperforming the carefully distilled RADIO indicates that features from different VFMs are complementary in 3D tasks.
- This hints at a future design direction for 3D VFMs: predicting 3D Gaussians in a canonical space + training with photometric losses.

## Limitations & Future Work
- Limitations acknowledged by the authors: (1) Dependence on DUSt3R for pose and point cloud initialization; if initialization fails, Feat2GS also fails; (2) Assumption of static scenes captured within a short period, unable to process internet photo collections with large time spans; (3) Restriction to static scenes, without supporting dynamic videos.
- Self-identified limitations: Low resolution of VFM features (typically 1/16) limits the recovery of high-frequency details. Using DUSt3R point clouds for warm starting may introduce bias, favoring DUSt3R-family features.
- Improvement ideas: Exploring VFM feature upsamplers, extending to 4D Gaussian Splatting for dynamic scenes, and utilizing VFM features to estimate initial camera poses to eliminate the reliance on DUSt3R.

## Related Work & Insights
- **vs Probe3D**: Probe3D probes VFMs using 2.5D estimation (depth/normal), but ignores the texture dimension and requires 3D ground truth. Feat2GS covers both geometry and texture through NVS without requiring 3D annotations.
- **vs InstantSplat**: Both use DUSt3R initialization for sparse-view NVS, but InstantSplat's free optimization of 3DGS is prone to overfitting. Feat2GS effectively prevents overfitting through VFM feature-constrained readout.
- **vs 3D Feature Fields (LERF, N3F, etc.)**: These works assume that VFM features possess 3D consistency, while Feat2GS questions and validates this assumption.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Leveraging 3DGS parameter decoupling to probe VFM 3D awareness is a highly original problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive, involving 10 VFMs, 7 datasets, 3 probing modes, and 3D metric validation on DTU.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Highly logical, insightful discoveries, and beautiful visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Strongly guiding for both the VFM and 3D vision communities, with findings carrying broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [\[CVPR 2025\] Deformable Radial Kernel Splatting](deformable_radial_kernel_splatting.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)

</div>

<!-- RELATED:END -->
