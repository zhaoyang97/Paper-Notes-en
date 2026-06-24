---
title: >-
  [Paper Note] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][Text-to-3D Generation] Proposes DreamScene360, which utilizes panoramic images as an intermediate representation, combined with a GPT-4V self-refinement mechanism and panoramic 3D Gaussian Splatting, to achieve rapid generation of immersive 360° 3D scenes from text.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Text-to-3D Generation"
  - "Panorama Images"
  - "3D Gaussian Splatting"
  - "Scene Generation"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 4c244cc50e55c791
---

# DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2404.06903](https://arxiv.org/abs/2404.06903)  
**Code**: [https://github.com/dreamscene360/dreamscene360](https://github.com/dreamscene360/dreamscene360)  
**Area**: 3D Vision  
**Keywords**: Text-to-3D Generation, Panorama Images, 3D Gaussian Splatting, Scene Generation, Diffusion Models

## TL;DR

Proposes DreamScene360, which utilizes panoramic images as an intermediate representation, combined with a GPT-4V self-refinement mechanism and panoramic 3D Gaussian Splatting, to achieve rapid generation of immersive 360° 3D scenes from text.

## Background & Motivation

**Background**: Text-to-3D scene generation primarily follows two technical pathways: (a) Score Distillation Sampling (SDS)-based methods (e.g., DreamFusion), which optimize NeRF/3DGS representations by distilling priors from 2D diffusion models; and (b) progressive methods based on explicit representations (e.g., LucidDreamer, Text2Room), which gradually expand 3D representations to cover a wider field of view.

**Limitations of Prior Work**:
   - SDS-based methods struggle with low rendering quality, are limited by multi-view inconsistency of 2D models, and cannot easily scale to scene-level 3D structures.
   - Progressive methods perform poorly when filling in large missing areas, leading to severe distortions and incoherent structures in 360° scenes.
   - The challenge of prompt engineering in text-to-image is even more pronounced in 3D generation, requiring extensive trial and error.

**Key Challenge**: Existing methods lack a globally consistent 2D scene representation, making it impossible to maintain semantic and geometric consistency across a full 360° range.

**Goal**: Generate globally consistent, immersive 360° 3D scenes from arbitrary text prompts.

**Key Insight**: Adopting panoramic images as an intermediate representation ensures global consistency while enabling automatic prompt optimization using GPT-4V.

**Core Idea**: Panoramic images provide a globally consistent 2D representation of complete 360° scenes. Combined with monocular depth initialization and semantic/geometric regularization, they can be efficiently elevated to 3D Gaussian Splatting representations.

## Method

### Overall Architecture

DreamScene360 consists of three stages: (1) generating 360° panoramas using a diffusion model, and iteratively optimizing them through GPT-4V self-refinement; (2) performing 2D-to-3D initialization of the panorama using monocular depth estimation and a learnable geometric field; and (3) optimizing panoramic 3D Gaussians via semantic and geometric regularization to fill in invisible regions from the single-view input.

### Key Designs

1. **Text-to-360° Panorama Generation + Self-Refinement Mechanism**:

    - **Function**: Generate high-quality, globally consistent 360° panoramic images from text.
    - **Mechanism**: Based on the MultiDiffusion sliding window process, StitchDiffusion is employed to ensure left-right boundary continuity. It generates a panorama of resolution $H \times 2H$, where each patch's update during the denoising process is merged through a weighted average:
    $\Phi(I_{t-1}) = \sum_{i=1}^{n} \frac{P_i^{-1}(W_i)}{\sum_{j=1}^{n} P_j^{-1}(W_j)} \otimes P_i^{-1}(\Phi(P_i(I_t)))$
    - In each denoising timestep, diffusion is performed not only on the original resolution but also on the concatenated leftmost and rightmost regions to ensure boundary consistency.
    - **Self-Refinement**: Integrates GPT-4V for multi-round self-improvement. Starting from a simple user prompt, GPT-4V evaluates the generated image on aspects such as object counts, attributes, relationships, and appearance with a score (0-10). It then provides improvement suggestions to refine the prompt, ultimately selecting the highest-scoring panorama.
    - **Design Motivation**: The panorama provides a globally consistent 2D representation for subsequent 3D generation, while GPT-4V eliminates the need for manual prompt engineering.

2. **Panoramic Geometric Field Initialization**:

    - **Function**: Elevate the 2D panorama into a consistent 3D point cloud as Gaussian initialization.
    - **Mechanism**: Project the panorama into N=20 overlapping perspective tangent images and obtain the depth map $D_i^{\text{Mono}}$ for each view using a DPT monocular depth estimator. To address the affine ambiguity inherent in monocular depth, a learnable global geometric field (MLP) and per-view scale/shift parameters are introduced for global alignment:
    $\min_{\alpha,\beta,\Theta} \left\{ \|\alpha \cdot D^{\text{Mono}} + \beta - \text{MLPs}(v;\Theta)\|_2^2 + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}(\beta) + \lambda_\alpha \|\gamma(\alpha) - 1\|^2 \right\}$
    - Where $\alpha_i$ is the scale parameter for each view, $\beta_i$ is the pixel-wise shift parameter, $\Theta$ denotes the MLP parameters, and $\gamma(\cdot)$ is the softplus function.
    - The TV loss ensures the spatial smoothness of the shift parameters.
    - **Design Motivation**: Outdoor scenes lack structured layout priors, necessitating deformable alignment to achieve scale consistency across different views.

3. **Virtual Camera Synthesis Parallax + Semantic/Geometric Regularization**:

    - **Function**: Solve the lack of parallax information in single-view panoramas and fill in invisible areas.
    - **Mechanism**: Synthesize virtual cameras by introducing progressive perturbations to the panoramic viewpoint coordinates:
    $(x', y', z') = (x, y, z) + \delta(d_x, d_y, d_z)$
      The perturbation range is $[-0.05, +0.05] \times \gamma$, where $\gamma \in \{1, 2, 4\}$ denotes the three-stage progressive perturbation.
    - **Semantic Regularization**: Uses the [CLS] feature of DINOv2 to constrain semantic consistency between the training view and the virtual view:
    $\mathcal{L}_{\text{sem}} = 1 - \text{Cos}([\text{CLS}](I_i), [\text{CLS}](I_i'))$
    - **Geometric Regularization**: Uses DPT to estimate the depth of the rendered image, regularizing the relative relationships of the rendered depth via Pearson correlation:
    $\mathcal{L}_{\text{geo}}(I_i, D_i) = 1 - \frac{\text{Cov}(D_i, \text{DPT}(I_i))}{\sqrt{\text{Var}(D_i) \cdot \text{Var}(\text{DPT}(I_i))}}$
    - **Design Motivation**: Single-view panoramas lack parallax information (cannot perceive depth via binocular disparity), which needs to be compensated for using virtual views paired with 2D model priors.

### Loss & Training

Overall loss function:
$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_1 \cdot \mathcal{L}_{\text{sem}} + \lambda_2 \cdot \mathcal{L}_{\text{geo}}$$

- $\mathcal{L}_{\text{RGB}}$: Photometric loss, including L1 and D-SSIM terms.
- $\lambda_1 = \lambda_2 = 0.05$
- Input panorama resolution: $1024 \times 2048$
- Disable 3DGS adaptive density control (densification) since high-quality point cloud initialization is already available.
- Set FoV to 80° for geometric field optimization.

## Key Experimental Results

### Main Results

| Metric | DreamScene360 | LucidDreamer | Description |
|-----------|--------------|--------------|------|
| CLIP Distance ↓ | **0.8732** | 0.8900 | Text-Image Alignment |
| Q-Align ↑ | **3.1094** | 3.0566 | SOTA Perceptual Quality Assessment |
| NIQE ↓ | **4.9165** | 6.2305 | No-Reference Image Quality |
| BRISQUE ↓ | **38.3911** | 51.9764 | No-Reference Image Quality |
| Runtime | 7min 20sec | 6min 15sec | Slightly slower but acceptable |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Photometric Loss Only | Artifacts in virtual views | Lack of constraints in occluded regions |
| + Geometric Regularization | Reduced artifacts | Improved depth consistency |
| + Semantic Regularization | Reduced artifacts | High-level semantic complement |
| Full Model | **Optimal visual quality** | Mutual complement of both regularizations |
| Random Initialization | Blurry results | Lack of geometric priors |
| Monocular Depth + Alignment | **Clear and consistent** | Key to a good initialization |

### Key Findings
- LucidDreamer's progressive inpainting tends to generate repetitive content in complex scenes (e.g., repeating a bedroom multiple times).
- GPT-4V self-refinement significantly improves the visual quality and detail richness of the panorama.
- Disabling 3DGS densification actually helps improve quality and accelerate convergence.

## Highlights & Insights
- **Using a panorama as an intermediate representation** is an excellent design choice—it naturally addresses the global consistency issue of 360° scenes while enabling GPT-4V's quality evaluation (prior methods without a global 2D representation could not perform such evaluations).
- The entire pipeline enables "one-click" 3D scene generation in about 7 minutes, offering a user experience significantly superior to SDS-based methods that require laborious parameter tuning.
- Formulating the affine ambiguity of monocular depth as a learnable geometric field optimization is a highly practical and elegant solution.

## Limitations & Future Work
- The generation resolution is limited by the default resolution of the pre-trained panoramic diffusion model ($512 \times 1024$).
- Future work can explore higher-resolution generation and extension to 4D dynamic scenes.
- The geometric accuracy of the scene remains bounded by the quality of monocular depth estimation.

## Related Work & Insights
- **vs. LucidDreamer**: LucidDreamer extends views via progressive inpainting but cannot guarantee 360° global consistency. DreamScene360 solves this fundamental issue through the panoramic intermediate representation.
- **vs. DreamFusion**: DreamFusion utilizes SDS distillation, which makes generation slow and low quality. DreamScene360 avoids this time-consuming score distillation process.
- **vs. Text2Room**: Text2Room uses a mesh representation to progressively construct indoor scenes, whereas DreamScene360 supports unconstrained both indoor and outdoor scenes.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of using a panorama as an intermediate representation + GPT-4V self-refinement is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐ Only compared with LucidDreamer; lacks more baselines and quantitative ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-explained motivation, high-quality illustrations.
- Value: ⭐⭐⭐⭐ Provides a practical end-to-end solution for 360° scene generation, indicating clear industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)

</div>

<!-- RELATED:END -->
