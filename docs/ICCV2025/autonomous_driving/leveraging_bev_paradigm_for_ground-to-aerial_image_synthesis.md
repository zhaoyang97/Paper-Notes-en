---
title: >-
  [Paper Note] SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis
description: >-
  [ICCV 2025][Autonomous Driving][Cross-view image synthesis] This paper proposes SkyDiffusion, which combines a Curved-BEV transformation with a BEV-guided diffusion model to achieve high-quality cross-view synthesis from ground-level street view images to aerial/satellite imagery, and introduces the Ground2Aerial-3 multi-scene dataset.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Cross-view image synthesis"
  - "Bird's-eye view transformation"
  - "Diffusion models"
  - "Ground-to-aerial generation"
  - "Street view to satellite imagery"
date: 2026-05-08
content_hash: 314c7e3709139a69
---

# SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis

**Conference**: ICCV 2025  
**arXiv**: [2408.01812](https://arxiv.org/abs/2408.01812)  
**Code**: [https://opendatalab.github.io/skydiffusion/](https://opendatalab.github.io/skydiffusion/)  
**Area**: Autonomous Driving  
**Keywords**: Cross-view image synthesis, Bird's-eye view transformation, Diffusion models, Ground-to-aerial generation, Street view to satellite imagery

## TL;DR

This paper proposes SkyDiffusion, which combines a Curved-BEV transformation with a BEV-guided diffusion model to achieve high-quality cross-view synthesis from ground-level street view images to aerial/satellite imagery, and introduces the Ground2Aerial-3 multi-scene dataset.

## Background & Motivation

Cross-view ground-to-aerial image synthesis aims to generate bird's-eye aerial images of corresponding locations from street view images, with important applications in land cover classification, urban planning, and disaster emergency response. Street view images are readily accessible via platforms such as Google Street View, offering high temporal coverage and flexibility. Synthesizing aerial imagery from street views thus provides an alternative when satellite imagery is unavailable.

**However, the task faces two core challenges:**

**Large viewpoint domain gap**: Street view images primarily capture ground-level details and building facades, whereas satellite images reveal rooftops and macro-level layout. Even with state-of-the-art diffusion models, aerial images directly generated from street views may appear photorealistic but are severely inconsistent with actual ground content—critical structural features such as road orientation and building placement are misaligned.

**Dense occlusion**: In urban scenes, tall buildings and trees significantly limit the observable range of street views. A single street view image is far from sufficient to cover the full area corresponding to a satellite image. Multiple street view images from different positions must be fused to fill in blind zones.

**Limitations of existing methods**: Early GAN-based methods (X-Seq, SelectionGAN) produce poor and blurry results; diffusion-based methods (AerialDiffusion, Instr-p2p) generate visually realistic images but **lack content consistency**; methods relying on semantic maps are impractical in real-world scenarios; the concurrent work GPG2A uses existing BEV methods combined with text guidance and semantic segmentation, requiring additional data and multi-stage processing.

**Starting Point of SkyDiffusion**: The BEV paradigm has been widely adopted in autonomous driving to unify multi-view perception. Applying BEV transformation to cross-view synthesis naturally bridges the viewpoint gap between street views and aerial images—by first transforming street views into BEV space (achieving domain alignment), and then using the BEV map as a condition to guide a diffusion model in generating aerial images.

## Method

### Overall Architecture

SkyDiffusion comprises two core modules: (1) the Curved-BEV transformation, which maps street view panoramas to a bird's-eye perspective and supports both One-to-One and Multi-to-One mapping modes; and (2) the BEV-guided diffusion model, which takes BEV maps as conditional input to control the diffusion model in generating content-consistent satellite images.

### Key Designs

#### 1. Curved-BEV Transformation

- **Function**: Converts panoramic street view images into a bird's-eye view (BEV) perspective while preserving information from the upper region (e.g., distant roads, building tops).
- **Mechanism**: Improves upon the conventional BEV "ground plane assumption" ($z=0$) by proposing an **upward curved surface assumption**—the height of the BEV plane increases rapidly with distance from the center:
  $z = d_{norm}^4 \times \lambda = \left(\frac{\sqrt{x^2+y^2}}{d_{max}}\right)^4 \times \lambda$
  
  The 3D point $P(x,y,z)$ on the curved surface is then converted to spherical coordinates $P(\theta,\varphi)$, and subsequently mapped to pixel coordinates $P(u,v)$ on the panorama via equirectangular projection:
  $u = [\text{arctan2}(y,x) + \pi] \frac{w}{2\pi}, \quad v = [\frac{\pi}{2} + \text{arctan2}(z-H, \sqrt{x^2+y^2})] \frac{h}{\pi}$
- **Design Motivation**: Conventional ground-plane BEV cannot map content in the upper portion of street view images (e.g., distant roads, upper building facades), leading to information loss. The curved surface design causes the mapping of distant regions to "look upward," thereby capturing more cross-view correlated information. Since the mapping relationship is fixed, the computational overhead is negligible.

#### 2. Multi-to-One BEV Mapping

- **Function**: Unifies BEV mapping results from multiple street views captured at different positions into a common satellite coordinate system.
- **Mechanism**: Based on camera position relationships, each street view's BEV mapping result $\text{BEV}_{cam_k}$ is shifted into the satellite coordinate system. For overlapping regions, the BEV mapping result closest to the respective capture point is selected:
  $k = \arg\min_i \sqrt{(x - x_{cam_i})^2 + (y - y_{cam_i})^2}$
- **Design Motivation**: In dense urban scenes, the BEV perceptual range of a single street view is highly limited. By fusing BEV information from multiple nearby street views, the perceptual coverage is effectively expanded, addressing the large-scale blind zone problem caused by building occlusion.

#### 3. BEV-Controlled Diffusion Model

- **Function**: Uses the BEV-transformed image as a condition to guide a pretrained diffusion model in generating content-consistent satellite images.
- **Mechanism**: A conditional encoder encodes the BEV map $I_{bev}$ into features $c_{bev} = \mathcal{E}(I_{bev})$ (with spatial attention to enhance features and suppress distorted regions), which are then injected into the diffusion model via zero-convolution layers and duplicated SD encoder/middle blocks (following a ControlNet-style architecture). The training loss follows the standard denoising objective:
  $\mathcal{L} = \mathbb{E}_{sat_0, t, c_{bev}, \epsilon \sim \mathcal{N}(0,1)} \left[\|\epsilon - \epsilon_\theta(sat_t, t, c_{bev})\|_2^2\right]$
- **Design Motivation**: The BEV map is approximately aligned with the satellite view but contains incomplete information and distorted regions. The diffusion model can "fill in" missing content in the BEV map (e.g., rooftop textures), while spatial attention automatically suppresses the influence of severely distorted regions.

### Loss & Training

- Based on Stable Diffusion v1.5 pretrained weights; the diffusion decoder is unfrozen.
- Classifier-free guidance scale set to 9.0.
- DDIM sampling with 50 steps; 8 A100 GPUs, batch size 128, trained for 100 epochs.
- Text prompts are not the optimization target—the task is inherently image-conditioned generation, and text cannot adequately describe the complexity of street view scenes.

## Key Experimental Results

### Main Results — Comparison with SOTA Methods

| Method | CVUSA FID↓ | CVUSA SSIM↑ | CVACT FID↓ | VIGOR FID↓ | VIGOR SSIM↑ |
|--------|-----------|-------------|-----------|-----------|-------------|
| X-Seq (GAN) | 161.16 | 0.084 | 190.12 | — | — |
| SelectionGAN | 116.57 | 0.129 | 100.21 | 149.53 | 0.127 |
| CUT | 72.83 | 0.121 | 62.22 | 69.42 | 0.169 |
| ControlNet | 32.45 | 0.149 | 62.21 | 53.27 | 0.170 |
| GPG2A | 58.80 | 0.135 | 63.50 | 70.19 | 0.159 |
| **SkyDiffusion** | **29.18** | **0.168** | **36.48** | **45.29** | **0.186** |

SkyDiffusion achieves comprehensive improvements across all datasets: FID is reduced by 25.72% and SSIM improved by 7.68% on CVUSA; on the urban dataset VIGOR, FID is reduced by 14.98% and SSIM improved by 9.41%.

### Ablation Study — Curved-BEV Module

| Configuration | CVACT FID↓ | CVACT SSIM↑ | VIGOR FID↓ | VIGOR SSIM↑ |
|---------------|-----------|-------------|-----------|-------------|
| Direct Street View → Diffusion (Baseline) | 62.21 | 0.115 | 53.27 | 0.170 |
| Standard BEV + Diffusion | 42.84 | 0.117 | 48.63 | 0.175 |
| Curved-BEV + Diffusion | 36.48 | 0.118 | 45.29 | 0.186 |
| Curved-BEV Multi-to-One | — | — | **31.90** | **0.205** |

Incrementally adding each component yields consistent improvements in consistency. Multi-to-One on the VIGOR urban dataset further reduces FID from 45.29 to 31.90 (+29.6%) and improves SSIM from 0.186 to 0.205.

### Key Findings

- **GAN-based methods generate significant artifacts and blurriness**, making them unsuitable for cross-view synthesis.
- **Existing diffusion-based methods (AerialDiffusion, Instr-p2p) generate visually realistic but content-inconsistent images**—the lack of BEV domain alignment leads to loss of structural information such as road orientation and building placement.
- On the **G2A-3 dataset**, SkyDiffusion significantly outperforms the ControlNet baseline across all three application scenarios (disaster emergency response, low-altitude UAV, historical satellite imagery), with an average FID reduction of 25.81% and average SSIM improvement of 12.88%.
- In disaster scene generation, critical damaged areas are clearly visible in the synthesized satellite images, supporting post-disaster assessment.

## Highlights & Insights

- **Applying the BEV paradigm to generative tasks** represents an elegant cross-domain transfer. While BEV is well-established in autonomous driving perception, employing it for cross-view domain alignment in image generation is a novel direction.
- **The curved surface assumption in Curved-BEV** addresses the inherent deficiency of conventional ground-plane BEV, which discards upper-field-of-view information, at virtually zero additional computational cost due to the fixed mapping relationship.
- **The G2A-3 dataset** introduces three practically valuable new scenarios—particularly disaster emergency response and historical image completion—addressing the limitation of existing cross-view datasets that are designed solely for retrieval tasks.

## Limitations & Future Work

- When missing regions in the BEV map are excessively large (e.g., rooftop information of tall buildings), the diffusion model can only "hallucinate" plausible content, with no guarantee of fidelity.
- Multi-to-One mapping requires precise geographic position relationships among multiple street view images, which may be costly to obtain in practice.
- Experiments are conducted only at $512\times512$ resolution; synthesis quality at higher resolutions remains to be validated.
- The hyperparameter $\lambda$ in the BEV transformation may require different settings for different scenes, and no adaptive mechanism is provided.

## Related Work & Insights

- Related to general image translation frameworks such as Pix2Pix and ControlNet, though SkyDiffusion is specifically designed to address the large domain gap inherent to cross-view synthesis.
- Methodologically connected to autonomous driving BEV perception methods (BEVFormer, LSS, etc.), extending BEV from a perception tool to a generative tool.
- Inspiration: Could the approach be reversed—synthesizing street views from aerial imagery? Or applied to scene generation in autonomous driving simulators?

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification](../../NeurIPS2025/autonomous_driving/gsalign_geometric_and_semantic_alignment_network_for_aerial-ground_person_re-ide.md)
- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](../../ICML2025/autonomous_driving/geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[ICCV 2025\] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)

</div>

<!-- RELATED:END -->
