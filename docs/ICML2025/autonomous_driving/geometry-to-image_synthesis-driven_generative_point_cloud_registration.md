---
title: >-
  [Paper Note] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration
description: >-
  [ICML 2025][Autonomous Driving][Point Cloud Registration] Proposes a new paradigm of Generative Point Cloud Registration, designing two registration-tailored controllable 2D generative models: DepthMatch-ControlNet and LiDARMatch-ControlNet, to generate cross-view consistent RGB image pairs from pure geometric point cloud pairs. It plug-and-play improves existing 3D registration methods through geometry-color feature fusion, validated on 3DMatch/ScanNet/Dur360BEV.
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "Point Cloud Registration"
  - "Generative Registration"
  - "ControlNet"
  - "Cross-view Consistency"
  - "Geometry-Color Fusion"
date: 2026-05-08
content_hash: 44259410c2e19057
---

# Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration

**Conference**: ICML 2025  
**arXiv**: [2512.09407](https://arxiv.org/abs/2512.09407)  
**Code**: None (Not released)  
**Area**: Autonomous Driving  
**Keywords**: Point Cloud Registration, Generative Registration, ControlNet, Cross-view Consistency, Geometry-Color Fusion

## TL;DR

Proposes a new paradigm of Generative Point Cloud Registration, designing two registration-tailored controllable 2D generative models: DepthMatch-ControlNet and LiDARMatch-ControlNet, to generate cross-view consistent RGB image pairs from pure geometric point cloud pairs. It plug-and-play improves existing 3D registration methods through geometry-color feature fusion, validated on 3DMatch/ScanNet/Dur360BEV.

## Background & Motivation

**Limitations of Prior Work**: Existing point cloud registration methods (e.g., ICP, FPFH, GeoTransformer, Predator) exhibit limited robustness in scenarios with low overlap, repetitive textures, and noise. RGB-D registration research has demonstrated that color/semantic information can significantly enhance the discriminative power of geometric descriptors, but corresponding RGB images are unavailable in pure geometric point cloud scenarios.

**Key Challenge**: Is it possible to "generate" useful color information to assist registration in the absence of real RGB images? The critical difficulty lies not in single-image generation, but in **paired generation**—the generated image pair must satisfy: (1) 2D-3D geometric consistency (generated images are spatially aligned with point clouds); (2) cross-view texture consistency (identical scene regions must share the same textures, otherwise noise matchings will be introduced).

**Goal**: Utilizing the depth-conditional generation capability of ControlNet to ensure geometric consistency, and achieving cross-view texture consistency through an innovative **coupled conditional denoising** mechanism, which works in both zero-shot and few-shot settings. **Core Idea**: **Vertically concatenate the source/target depth maps as a single conditional input, utilizing the self-attention mechanism of the UNet to naturally enable cross-view feature interaction without modifying the model architecture or weights**.

## Method

### Overall Architecture

Given a source/target point cloud pair $\mathcal{P}, \mathcal{Q}$, the pipeline consists of three phases: (1) **Geometric Representation Conversion**: Transforming point clouds into depth maps (perspective scenes) or equirectangular range images (LiDAR scenes); (2) **Registration-Tailored Image Generation**: Generating cross-view consistent RGB image pairs using DepthMatch-ControlNet or LiDARMatch-ControlNet; (3) **Geometry-Color Feature Fusion**: Leveraging pre-trained foundation vision models (DINOv2/Stable Diffusion) to extract zero-shot features of generated images, which are then weighted and concatenated with geometric descriptors for correspondence estimation and pose estimation.

### Key Designs

1. **Coupled Conditional Denoising**:
    - **Function**: Merging the denoising diffusion process of source/target images into one joint denoising process.
    - **Mechanism**: Vertically concatenating two noisy latent representations $\mathbf{x}_t^{\mathcal{P}} \in \mathbb{R}^{H' \times W' \times d}$ into $\mathbf{x}_t^{\mathcal{PQ}} \in \mathbb{R}^{2H' \times W' \times d}$, and concatenating the depth conditional maps into $\mathbf{d}_{\mathcal{PQ}} \in \mathbb{R}^{2H' \times W' \times d}$ accordingly. The original ControlNet denoiser can directly process the concatenated input: $\tilde{\epsilon}_\theta(\mathbf{x}_t^{\mathcal{PQ}}; t, \mathbf{c}, \mathbf{d}_{\mathcal{PQ}}) \rightarrow \mathbf{x}_{t-1}^{\mathcal{PQ}}$. The self-attention $\text{softmax}(\frac{QK^\top}{\sqrt{d}})V$ in UNet naturally covers all feature elements starging from both source and target views, achieving cross-view long-range dependency modeling.
    - **Design Motivation**: Independent denoising leaves the two images unaware of each other's colors, leading to texture inconsistency. The coupled approach requires no architectural modifications or parameter fine-tuning (zero-shot); it merely restructures the input layout to naturally leverage the existing self-attention mechanism for cross-view interactions.

2. **Coupled Prompt Guidance**:
    - **Function**: Designing specific text prompts to guide the denoiser to generate consistent, vertically stacked image pairs.
    - **Mechanism**: Utilizing a carefully crafted coupled prompt: "Generate two vertically stacked images that are captured from different viewpoints in a same scene. The images should feature the same environment... with very subtle differences between them. Overall, the layout and key elements remain the same."
    - **Design Motivation**: Even with the coupled denoising mechanism, the denoiser remains unaware of "what the user expects." By informing the model through prompts that it needs to generate spatially consistent image pairs, ControlNet can naturally recover consistent textures using its pre-trained semantic knowledge. This is the first work to discover and utilize this zero-shot capability of pre-trained ControlNet.

3. **LiDARMatch-ControlNet (LiDAR Panorama Extension)**:
    - **Function**: Extending the framework to 360° LiDAR point clouds to generate panoramic RGB image pairs.
    - **Mechanism**: Projecting the LiDAR point cloud into an equirectangular range image $\mathbf{D}^{\text{equi}} \in \mathbb{R}^{H \times W \times 1}$ to serve as the conditional input for ControlNet to generate panoramic RGB images. Applying few-shot fine-tuning using the Dur360BEV dataset (the only dataset providing complete 360°×180° spherical camera images).
    - **Design Motivation**: Achieving LiDAR point cloud to panoramic RGB image generation for the first time. Since there is no off-the-shelf ControlNet conditioned on range images, few-shot fine-tuning (requiring only ~10K panoramic pairs) is necessary.

### Zero-Shot Geometry-Color Feature Fusion

Leveraging the intermediate layers of pre-trained DINOv2 and Stable Diffusion to extract semantic and textual features of the generated images, respectively, which are then fused with geometric descriptors via weighted concatenation: $f_{\text{final}} = [f_{\text{geo}}; w_1 f_{\text{DINOv2}}; w_2 f_{\text{SD}}]$. It operates in a plug-and-play manner without extra training.

## Key Experimental Results

### Main Results: ScanNet Depth Camera Registration

| Method | Rot Acc@5° ↑ | Rot Acc@10° ↑ | Trans Acc@5cm ↑ | Trans Acc@10cm ↑ | Chamfer Acc@1mm ↑ |
|------|-------------|--------------|----------------|-----------------|-------------------|
| FCGF | 78.9 | 84.2 | 55.3 | 70.7 | 67.3 |
| Generative FCGF(DINOv2) | 81.0 | 86.2 | 57.3 | 72.6 | 68.9 |
| Generative FCGF(SD) | **82.9** | **90.0** | 56.4 | **73.0** | 67.7 |
| **Gain** | **+4.0** | **+5.8** | **+2.0** | **+2.3** | **+1.6** |

| Method | Rot Error Mean ↓ | Trans Error Mean ↓ | Chamfer Error Mean ↓ |
|------|-----------------|-------------------|---------------------|
| FCGF | 19.4 | 37.8 | 100.7 |
| Generative FCGF(SD) | **8.4** | **21.7** | **66.0** |
| **Gain** | **-11.0** | **-16.1** | **-34.7** |

The rotation error decreases from 19.4° to 8.4° (a 57% reduction), and the translation error decreases from 37.8cm to 21.7cm (a 43% reduction).

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Independent vs. Coupled Denoising | Coupled is significantly superior | Narrows texture inconsistency |
| DINOv2 vs. SD Features | SD is superior in rotation accuracy | DINOv2 is slightly better in translation |
| DINOv2 + SD Fusion | Comprehensively best | Complementary semantic and texture information |
| Zero-shot vs. Few-shot fine-tuning | Few-shot shows further improvement | Effective with only ~3K samples |
| Integration with different baselines | Consistent improvement | FPFH/Predator/FCGF/GeoTransformer |

### Key Findings

- Generative enhancement is consistently effective across all evaluated baseline methods, validating its plug-and-play universality.
- The improvement in rotation accuracy is most prominent (Rot Acc@10° from 84.2% $\rightarrow$ 90.0%), suggesting that color information is highly valuable for orientation-sensitive matching.
- The improvement in error metrics is more significant than that in accuracy metrics (rotation error reduced by 57%), showing that color features effectively eliminate large-error matchings.
- Panoramic image generation in LiDAR scenarios is achieved for the first time, with validated effectiveness on Dur360BEV.
- Few-shot fine-tuning (using only ~3K samples) can significantly improve upon zero-shot results.

## Highlights & Insights

- **New Paradigm of Generative Registration**: Shifting from "finding correspondences" to "generating color $\rightarrow$ enhancing correspondences," a novel cross-domain concept.
- Clever design of coupled denoising: Achieving cross-view interaction by simply restructuring the input layout, without modifying model architectures or weights.
- Plug-and-play general framework: Can be integrated with any geometric descriptor method, providing a "free" color enhancement.
- First work to exploit the zero-shot paired image generation capability of pre-trained ControlNet.
- First realization of synthesizing panoramic images from LiDAR point clouds.

## Limitations & Future Work

- Significant increase in inference time: Requires running the complete diffusion denoising process (multi-step iterations), which limits real-time capability.
- Coupled denoising doubles the latent space height, increasing GPU memory overhead.
- Generated image quality is limited by the pre-training capabilities of ControlNet and Stable Diffusion.
- Texture consistency in large-scale outdoor scenes remains challenging (generation quality degrades in distant regions).
- In theory, the method works as long as the generation quality is high, but degraded generations might conversely introduce noise matchings.

## Related Work & Insights

- **Zhang et al. ControlNet**: The foundation of depth-conditioned image generation, upon which this work builds registration-specific variants.
- **Rombach et al. Stable Diffusion**: Latent diffusion model, providing infrastructure for feature extraction and generation.
- **Oquab et al. DINOv2**: Zero-shot vision features, used for semantic encoding of generated images.
- **Qin et al. GeoTransformer**: Geometric Transformer registration, one of the baseline methods enhanced in this work.
- **Insights**: Utilizing 2D generative models to augment 3D vision tasks is a promising paradigm. Cross-modal and cross-domain generative enhancement warrants further research (e.g., generative SLAM, generative reconstruction).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The generative registration paradigm is brand new, and the coupled denoising for zero-shot consistent generation is extremely clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both depth camera and LiDAR scenarios, integrated and validated across multiple baselines, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed method elaboration, and theoretical analyses that enhance persuasiveness.
- Value: ⭐⭐⭐⭐ Highly practical plug-and-play framework, opening up new directions with LiDAR panorama generation, though inference speed remains a bottleneck for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](../../CVPR2025/autonomous_driving/unlocking_generalization_power_in_lidar_point_cloud_registration.md)
- [\[ICCV 2025\] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion](../../ICCV2025/autonomous_driving/mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)
- [\[ICCV 2025\] SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis](../../ICCV2025/autonomous_driving/leveraging_bev_paradigm_for_ground-to-aerial_image_synthesis.md)
- [\[ICML 2025\] Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance](dont_be_so_negative_score-based_generative_modeling_with_oracle-assisted_guidanc.md)
- [\[ICML 2025\] InfoCons: Identifying Interpretable Critical Concepts in Point Clouds via Information Theory](infocons_identifying_interpretable_critical_concepts_in_point_clouds_via_informa.md)

</div>

<!-- RELATED:END -->
