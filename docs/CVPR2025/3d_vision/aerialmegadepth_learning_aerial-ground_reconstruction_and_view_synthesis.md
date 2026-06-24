---
title: >-
  [Paper Note] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis
description: >-
  [CVPR 2025][3D Vision][Joint Aerial-Ground Reconstruction] This paper proposes the AerialMegaDepth dataset generation framework. By co-registering pseudo-synthetic aerial renders from Google Earth and real ground-level images from MegaDepth into a unified coordinate system, it constructs a large-scale training dataset of 132k mixed-altitude images. Fine-tuning DUSt3R on this dataset improves the camera rotation estimation accuracy for aerial-ground pairs from 5% to 56%…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Joint Aerial-Ground Reconstruction"
  - "Cross-View Matching"
  - "Pseudo-Synthetic Data"
  - "Multi-View Geometry"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: efb285e82f004825
---

# AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2504.13157](https://arxiv.org/abs/2504.13157)  
**Code**: [https://aerial-megadepth.github.io](https://aerial-megadepth.github.io)  
**Area**: 3D Vision  
**Keywords**: Joint Aerial-Ground Reconstruction, Cross-View Matching, Pseudo-Synthetic Data, Multi-View Geometry, Novel View Synthesis

## TL;DR
This paper proposes the AerialMegaDepth dataset generation framework. By co-registering pseudo-synthetic aerial renders from Google Earth and real ground-level images from MegaDepth into a unified coordinate system, it constructs a large-scale training dataset of 132k mixed-altitude images. Fine-tuning DUSt3R on this dataset improves the camera rotation estimation accuracy for aerial-ground pairs from 5% to 56%, while also significantly enhancing the quality of novel view synthesis.

## Background & Motivation
Multi-view 3D reconstruction and camera registration are fundamental tasks in computer vision. Recently, learning-based approaches (such as DUSt3R and MASt3R) have made remarkable progress in estimating geometry from "in-the-wild" images. However, they almost completely fail when encountering a critical scenario: **extreme viewpoint variations between ground and aerial perspectives**. Evaluations indicate that the pretrained DUSt3R achieves only about 5% accuracy (within 5°) for camera rotation estimation on aerial-ground image pairs.

The authors' core hypothesis is that this failure stems from the lack of co-registered aerial-ground training data. Existing datasets, such as MegaDepth, primarily consist of tourist photos taken from the ground and rarely cover continuous viewpoint transitions from drones to the ground. While obtaining camera poses for ground and aerial images independently is straightforward, merging them into the same coordinate system requires specialized sensors or extensive manual alignment, making it difficult to scale.

The proposed solution is ingenious: leverage 3D textured meshes from geospatial platforms like Google Earth to render multi-altitude images (pseudo-synthetic data), and then register a large number of real ground-level images into the same coordinate system via a visual localization pipeline to establish a hybrid dataset.

## Method

### Overall Architecture
The data generation consists of two main steps: 1) rendering pseudo-synthetic images from Google Earth and constructing a 3D reconstruction; 2) registering real ground-level images from MegaDepth to the pseudo-synthetic reconstruction. The resulting AerialMegaDepth dataset comprises 137 landmark scenes and 132,137 co-registered images. This dataset is then utilized to fine-tune models like DUSt3R and MASt3R to enhance aerial-ground geometric estimation.

### Key Designs

1. **Pseudo-Synthetic Data Generation**:
    - Google Earth is chosen as the data source due to its coverage of numerous landmarks and high-quality textures.
    - **Automatic query viewpoint generation**: EXIF GPS tags of MegaDepth images are used to geo-register the SfM reconstructions into the global coordinate system (ECEF), from which 200 points are sampled as look-at targets.
    - For each landmark, 600 rendered images are generated at various altitudes (ranging from 1m to 350m), totaling 82,220 images.
    - It is referred to as "pseudo-synthetic" rather than "synthetic" because the underlying 3D meshes are textured with real photographs.
    - Although Google Earth does not provide direct access to the underlying 3D meshes, 3D point clouds are reconstructed via feature extraction, matching, and triangulation using the known camera intrinsics and extrinsics.

2. **Co-Registration of Real Images**:
    - Key observation: Despite the domain gap between pseudo-synthetic and real images (e.g., lack of transient objects, simplified lighting), state-of-the-art feature matching methods can still establish reliable correspondences.
    - A standard visual localization pipeline is adopted: retrieve the most similar pseudo-synthetic images for each real image $\rightarrow$ lift 2D feature matches to 2D-3D correspondences $\rightarrow$ solve 6-DoF poses using RANSAC PnP.
    - COLMAP MVS is further applied to generate semi-dense depth maps for supervision.
    - A total of 49,937 MegaDepth images are registered alongside 82,200 pseudo-synthetic images.

3. **Training Pair Construction — Asymmetric Co-visibility Matrix**:
    - An $N \times N$ co-visibility matrix $\mathcal{C}$ is computed, where $\mathcal{C}[i,j]$ represents the fraction of 3D points visible in image $i$ that are also visible in image $j$.
    - For the aerial-ground setup, image pairs with high asymmetry are prioritized (where the ground image observes only a small fraction of the large scene, while the aerial image observes a large portion).
    - A score $s = \text{AM}/\text{HM}$ (Arithmetic Mean / Harmonic Mean) is designed, where a high score indicates a large viewpoint discrepancy.
    - A total of 1.5 million training image pairs are generated.

4. **Downstream Task Fine-Tuning**:
    - **Multi-view Pose and Geometry Estimation**: Fine-tuning is conducted on DUSt3R and MASt3R to regress the 3D pointmaps of the image pairs.
    - **Novel View Synthesis**: ZeroNVS is fine-tuned using a single aerial image as reference to generate ground-level views. During training, a 3:1 mixture with MegaScenes is used to prevent overfitting.

### Loss & Training
- Fine-tuning is performed using the original pointmap regression loss from DUSt3R.
- Training starts from the DUSt3R checkpoint pre-trained on 8 public datasets.
- For novel view synthesis, ZeroNVS's standard diffusion loss is used, fine-tuning from the MegaScenes pre-trained checkpoint.

## Key Experimental Results

### Main Results — Aerial-Ground Camera Registration

| Method | RRA@5° | RRA@10° | RRA@15° | RTA@5° |
|------|--------|---------|---------|--------|
| DUSt3R (baseline) | 5.20 | 7.95 | 9.48 | 2.75 |
| DUSt3R + MatrixCity | 17.85 | 37.28 | 42.80 | 11.33 |
| DUSt3R + PSynth | 31.28 | 47.63 | 51.61 | 28.78 |
| DUSt3R + Hybrid | **55.96** | **71.25** | **76.15** | **46.48** |
| MASt3R (baseline) | 3.36 | 3.36 | 4.59 | 2.45 |
| MASt3R + Hybrid | **49.54** | **66.36** | **72.48** | **42.51** |

### Multi-View Scene Registration (1 Aerial + N Ground Images, RRA@15°)

| Method | N=2 | N=3 | N=4 | N=5 |
|------|-----|-----|-----|-----|
| DUSt3R-GA (baseline, ground-only) | 12.20 | 32.21 | 38.31 | 43.98 |
| DUSt3R-GA + Hybrid (incl. 1 aerial) | **56.10** | **55.28** | **57.72** | **59.27** |

### Novel View Synthesis (Aerial $\rightarrow$ Ground)

| Method | DreamSim↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|-----------|--------|-------|-------|
| ZeroNVS (MegaScenes) - Pseudo-synth | 0.448 | 0.413 | 10.85 | 0.416 |
| ZeroNVS (Ours) - Pseudo-synth | **0.377** | **0.359** | **12.38** | **0.484** |
| ZeroNVS (MegaScenes) - Real | 0.550 | 0.639 | 7.48 | 0.183 |
| ZeroNVS (Ours) - Real | **0.442** | **0.580** | **8.22** | **0.218** |

### Ablation Study

| Configuration | RRA@5° | Description |
|------|--------|------|
| Pseudo-synthetic only (PSynth) | 31.28 | 6x improvement compared to baseline |
| Synthetic only (MatrixCity) | 17.85 | Domain gap exists in purely synthetic data |
| Hybrid data (Hybrid) | **55.96** | Real ground images bridge the domain gap |

### Key Findings
- Hybrid data almost doubles the performance compared to using pseudo-synthetic data alone (RRA@5°: 31% $\rightarrow$ 56%), proving that real ground-level images are crucial for bridging the domain gap.
- Pseudo-synthetic data is more effective than fully synthetic data (MatrixCity) because its textures are derived from real photographs.
- Adding just one aerial reference image significantly improves pose estimation across multiple ground images (acting as a "bird's eye map" to link ground perspectives).
- The fine-tuned models show no noticeable performance degradation on intra-view pairs (ground-ground, aerial-aerial).
- 3D pointmap accuracy is also substantially improved (the proportion of points within a 1m error increases from 42% to 62%).

## Highlights & Insights
- The data-driven perspective is highly instructive: rather than designing complex cross-view matching algorithms, it is better to provide appropriate cross-view training data.
- The strategy of mixing real and pseudo-synthetic data expertly combines the strengths of both: pseudo-synthetic data offers aerial viewpoint coverage, while real data provides visual fidelity.
- The insight of using a single aerial image as a "map" to connect non-overlapping ground images is highly inspiring for real-world drone applications.
- The image pair selection strategy based on the asymmetric co-visibility matrix (AM/HM score) elegantly characterizes the aerial-ground viewpoint differences.
- The framework is highly scalable and can easily integrate other crowdsourced datasets and geospatial platforms.

## Limitations & Future Work
- The quality of Google Earth 3D meshes varies, and ground textures for some landmarks are relatively poor.
- GPS tag accuracy is limited, which may introduce initial alignment errors.
- The dataset covers 137 landmarks, which may introduce geographical bias (primarily tourist attractions).
- Although novel view synthesis quality has improved, visual artifacts remain; generation under extreme viewpoint changes remains an open challenge.
- Currently, the approach relies on COLMAP's MVS to generate depth supervision, the accuracy of which is bounded by the quality of SfM reconstruction.
- Bridging the satellite perspective to the drone perspective has not yet been explored.

## Related Work & Insights
- MegaDepth pioneered the paradigm of training geometric estimation models using SfM reconstructions from internet crowdsourced images; this work extends it to the joint aerial-ground setting.
- DUSt3R and MASt3R demonstrate the strong capabilities of end-to-end 3D geometry learning, but are limited by the viewpoint distribution of the training data.
- BlendedMVS mixed rendered environments with real images for MVS training; this work generalizes a similar concept to cross-view settings.
- Visual localization methods (such as localization using 3D city meshes) validate the feasibility of feature matching from pseudo-synthetic to real images.
- Future work could further bridge the satellite $\rightarrow$ drone $\rightarrow$ ground three-tier viewpoints, moving toward planetary-scale 3D reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ The data construction framework is novel (mixing pseudo-synthetic and real data), though the downstream methodology mainly leverages existing models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-task evaluation (pose estimation, multi-view registration, NVS) with zero-shot evaluation on real-world data.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear motivation, sound experimental design, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ The dataset will drive development in the entire field of joint aerial-ground 3D reconstruction, and the framework is highly generalizable to other scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)

</div>

<!-- RELATED:END -->
