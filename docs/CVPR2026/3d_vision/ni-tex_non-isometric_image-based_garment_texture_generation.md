---
title: >-
  [Paper Note] NI-Tex: Non-isometric Image-based Garment Texture Generation
description: >-
  [CVPR 2026][3D Vision][Garment texture generation] NI-Tex is proposed as a framework that, through the construction of a 3D Garment Videos dataset, image-editing-based cross-topology augmentation, and an uncertainty-guided iterative baking algorithm, achieves for the first time high-quality feed-forward generation of PBR textures for 3D garments from a single image under non-isometric conditions.
tags:
  - CVPR 2026
  - 3D Vision
  - Garment texture generation
  - PBR materials
  - non-isometric deformation
  - uncertainty-guided baking
  - cross-topology augmentation
date: 2026-05-08
content_hash: 82696cfb1a1f7389
---

# NI-Tex: Non-isometric Image-based Garment Texture Generation

**Conference**: CVPR 2026
**arXiv**: [2511.18765](https://arxiv.org/abs/2511.18765)
**Code**: [Available](https://github.com/SII-Hui/NI-Tex)
**Area**: 3D Vision
**Keywords**: Garment texture generation, PBR materials, non-isometric deformation, uncertainty-guided baking, cross-topology augmentation

## TL;DR

NI-Tex is proposed as a framework that, through the construction of a 3D Garment Videos dataset, image-editing-based cross-topology augmentation, and an uncertainty-guided iterative baking algorithm, achieves for the first time high-quality feed-forward generation of PBR textures for 3D garments from a single image under non-isometric conditions.

## Background & Motivation

Existing industrial-grade 3D garment meshes cover most real-world garment geometries, yet texture diversity remains limited. To obtain more realistic textures, generative methods commonly extract PBR (physically based rendering) textures from large collections of real images and project them onto garment meshes. However, existing image-conditioned texture generation methods face two core limitations:

**Topology consistency requirement**: Most methods require strict topological consistency between the input image and the target 3D mesh. For example, Hunyuan3D and Meshy exhibit severe quality degradation when image–mesh topology is mismatched.

**Mesh deformation dependency**: Some methods (e.g., Pix2Surf, Cloth2Tex) rely on accurate mesh deformation to match image poses, but the deformation process introduces cumulative errors and limits flexibility.

In practice, significant topological differences (e.g., generating trousers mesh textures from a skirt image) and geometric differences (different poses, different body shapes) frequently exist between user-provided images and target meshes, making existing methods ill-suited for non-isometric scenarios. The paper's core starting point is to reframe the non-isometric problem as a data augmentation problem: image editing models are used to synthesize cross-topology training pairs, while physics simulation data covers cross-pose scenarios.

## Method

### Overall Architecture

NI-Tex adopts a three-stage design:

- **Data space**: 3D Garment Videos are constructed from BEDLAM to provide cross-pose training pairs; Nano Banana image editing generates cross-topology training pairs.
- **Generation network**: A dual-branch feed-forward architecture — a guidance branch extracts reference image features, and a generation branch outputs multi-view PBR textures conditioned on normal/position maps.
- **Iterative baking**: A UQ (uncertainty quantification) model is trained to iteratively select new viewpoints and perform weighted fusion into mesh textures.

The input is a single RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and a target garment mesh; the output is albedo ($C=3$), roughness ($C=1$), and metallic ($C=1$) texture maps in UV space.

### Key Designs

#### 1. **Cross-Pose Augmentation via 3D Garment Videos**

**Mechanism**: Physics simulation data from BEDLAM is used to construct cross-pose training pairs, eliminating the effect of pose variation on texture generation.

- Per-frame garment geometry is extracted from each motion sequence in BEDLAM, forming a sequence $V = \{M_1, M_2, \ldots, M_n\}$ where all frames share the same albedo texture map.
- PBR material properties are supplemented for each frame: $\text{roughness} \sim \mathcal{U}(0,1)$, $\text{metallic} = 0$.
- During training, two frames are randomly sampled: one serves as the **condition frame** (a single illuminated view is selected as the input image prompt, with normal and position maps from 10 viewpoints as geometric constraints), and the other as the **supervision frame** (PBR texture attributes from 10 viewpoints serve as supervision signals).
- This cross-frame supervision combinatorially expands the dataset from hundreds of thousands of frames to tens of billions of training samples.

**Design Motivation**: Different frames within the same sequence maintain texture consistency while exhibiting different poses, naturally forming cross-pose training pairs without additional paired annotation. Various light source types (point, area, and environment lights) are randomly applied to enhance illumination diversity.

#### 2. **Cross-Topology Augmentation via Nano Banana**

**Mechanism**: An image editing model modifies the garment topology of the condition image while preserving the original texture information, constructing cross-topology training pairs.

- Rendered views are randomly sampled from 3D Garment Videos, and Nano Banana edits the garment topology (e.g., editing trousers into shorts, or skirts into trousers).
- Illumination-rendered images (rather than albedo images) are used as editing inputs to reduce the domain gap at inference time.
- The edited image replaces the original condition image, while supervision still comes from the original supervision frame — in essence, texture identity consistency is distilled from Nano Banana.

To avoid erroneous distillation, three semantic integrity constraints are enforced:

- **Category consistency**: When editing full-body garments, upper and lower garment textures must not drift or be swapped.
- **Layer consistency**: In layered outfits, the outer garment texture must remain distinguishable from the inner garment.
- **Permissibility of auxiliary body parts**: Occasional generation of extra body regions is permitted, encouraging the model to focus on garment material itself.

Approximately 50K edited images are generated in total for cross-topology training.

#### 3. **Dual-Branch Generation Network and Switchable Multi-Channel U-Net**

**Mechanism**: A feed-forward dual-branch architecture is adopted, and a switchable mechanism is introduced to handle inconsistent MR channels.

**Dual-branch architecture**: The guidance branch extracts hierarchical features from the input image; the generation branch receives multi-view normal and position maps and produces textures. The two branches are connected via **Multi-Channel Alignment Attention (MCAA)**:

$$\text{Attn}_{albedo} = \text{Softmax}\left(\frac{Q_{albedo} K_{ref}^T}{\sqrt{d}}\right) \cdot V_{ref}$$

Albedo attention is injected into the MR latent representation for spatial and geometric alignment:

$$z_{MR}^{new} = z_{MR} + \text{Attn}_{albedo}$$

**Switchable mechanism**: Since images generated by Nano Banana cannot maintain consistent MR properties, a switchable U-Net is designed — when processing edited images, the MR channel is disabled and single-channel attention is used; for normal frames, multi-channel alignment attention is applied. This prevents inconsistent MR supervision from contaminating training.

#### 4. **Uncertainty-Guided Iterative Baking**

**Mechanism**: A UQ model is trained to predict per-pixel uncertainty of texture maps; high-uncertainty viewpoints are iteratively selected and re-generated to repair baking artifacts.

**UQ model training**: A ResNet-50 architecture is adopted. Training data is collected through an error simulation pipeline — 10 views of a GT mesh are rendered, random views are edited with Nano Banana, and the texture generation model reconstructs the result by optimizing a latent code such that front/back view textures match the GT:

$$\min_{\boldsymbol{z}} \| \Gamma^{\text{front}}(\boldsymbol{z}) - T_{\text{gt}}^{\text{front}} \|^2 + \| \Gamma^{\text{back}}(\boldsymbol{z}) - T_{\text{gt}}^{\text{back}} \|^2$$

Prediction–GT pairs from all viewpoints are collected, per-pixel uncertainty labels are computed using SSIM, and the supervision loss is:

$$\sum_{p_i} \| \text{UQ}(p_i) - y^{\text{SSIM, GT}}(p_i) \|_2^2$$

**Viewpoint selection**: The viewpoint with the highest average uncertainty is selected from candidates and re-inferred. Iteration continues until the maximum number of viewpoints $N_{view}$ is reached or the uncertainty of a new viewpoint falls below threshold $\epsilon$.

**Multi-view reweighted fusion**: The final texture is computed by jointly weighting uncertainty and viewpoint importance:

$$t_i^{\star} = \frac{\sum_j (1 - \text{UQ}(p_{ij})) c_j p_{ij}}{\sum_j (1 - \text{UQ}(p_{ij})) c_j + \epsilon_1}$$

where $c_j$ denotes the viewpoint weight: front and back views are set to 1, with remaining views decaying by distance to 0.5, 0.25, 0.125, and 0.1.

### Loss & Training

Multi-channel optimization stage (joint albedo + MR supervision):

$$\mathcal{L}_1 = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_t^{MR} \|_2^2 + \| \epsilon - \epsilon_t^{Albedo} \|_2^2 \right]$$

Single-channel optimization stage (albedo supervision only, for Nano Banana edited samples):

$$\mathcal{L}_2 = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1), t} \left[ \alpha \cdot \| \epsilon - \epsilon_t^{Albedo} \|_2^2 \right]$$

The two losses are optimized alternately, with balancing factor $\alpha = 2$ to smooth the training loss curve. **MR Rectification**: Due to inconsistent MR values across frames, representative foreground pixels are sampled from the condition frame MR map and used to replace all foreground pixel values in the supervision frame MR map, enabling consistent cross-frame MR supervision.

Training is based on Stable Diffusion 2.1, conducted on 8×H200 GPUs for approximately 10 days with batch size 2 and resolution 512×512. Data scale: 100K Objaverse + 90K TexVerse (general 3D data) + 150K BEDLAM (garment simulation) + 50K edited images (cross-topology). To prevent MR overfitting to uniform values, Objaverse/TexVerse data are additionally incorporated via cross-mixing training.

## Key Experimental Results

### Main Results

| Method | KID ↓ | FID ↓ |
|--------|-------|-------|
| Paint3D | 0.0695 | 293.45 |
| Hyper3D OmniCraft | 0.0471 | 285.45 |
| Hunyuan3D | 0.0528 | 272.34 |
| Meshy 6 Preview | 0.0383 | 246.39 |
| **NI-Tex (Ours)** | **0.0364** | **0.0364** |

Experimental setup: 10 industrial/generated meshes × 10 image prompts × multi-view rendering × 42 random seeds. NI-Tex achieves the best performance on both KID and FID: KID is 5.0% lower than the second-best Meshy, and FID is reduced by 3.6%.

### Baking Strategy Comparison

| Baking Strategy | Mesh Coverage | Artifact Handling | PSNR |
|-----------------|---------------|-------------------|------|
| 6 orthogonal views | Significant self-occlusion gaps | None | Baseline |
| Coverage-based view selection | Improved but minor gaps remain | None | Medium |
| **UQ iterative baking (Ours)** | **Full coverage** | **Actively repairs blur/holes** | **Highest** |

### Key Findings

1. **Cross-topology robustness**: NI-Tex generates high-quality textures even under significant image–mesh topology discrepancy (e.g., skirt → trousers), whereas Hunyuan3D and Meshy exhibit severe texture distortion or generation failure.
2. **In-the-wild image adaptability**: On real DeepFashion2 images (masked with SAM2), NI-Tex effectively captures correct texture information including logos and fine patterns.
3. **Cross-pose consistency**: Validated on the 4D-Dress dataset, texture generation remains consistent across different poses of the same subject.
4. **UQ baking outperforms coverage baking**: Uncertainty-guided viewpoint selection captures intermediate baking artifacts (blur, seams, holes, etc.) missed by conventional coverage-based methods, yielding significantly higher PSNR on worst-case viewpoints.
5. **Generalization across industrial and generated meshes**: The method works stably on Hunyuan3D-generated meshes with more complex wrinkles, preserving logos, patterns, and fine details.

## Highlights & Insights

- **Image editing tools as a data augmentation engine**: The non-isometric problem is reframed as an image editing problem, using Nano Banana to cost-effectively generate cross-topology training pairs from existing 3D assets. This strategy is transferable to any 3D generation task requiring geometric diversity.
- **Combinatorial data expansion**: 3D Garment Videos expand the dataset from hundreds of thousands of frames to tens of billions of training samples through frame-pair combinations, representing a highly efficient data augmentation paradigm.
- **Pragmatic switchable architecture design**: Rather than forcing unified supervision, a switchable U-Net is designed to address MR inconsistency in edited images, reflecting a pragmatic engineering mindset.
- **Closed-loop uncertainty**: The UQ model is not only used for quality assessment but also directly drives viewpoint selection and fusion weights, forming a complete quality-detection–repair loop.
- **Cross-topology augmentation as distillation**: In essence, the approach distills Nano Banana's texture identity consistency capability into the texture generation model.

## Limitations & Future Work

- Generalization to **complex rigid deformations** is limited, as physics simulation data for general objects is unavailable; the method is currently best suited for flexible objects such as garments.
- The approach depends on the quality of external image editing models such as Nano Banana; editing failures introduce training noise.
- Training cost is high (approximately 10 days on 8×H200 GPUs), and inference requires multiple rounds of iterative baking.
- Quantitative evaluation relies primarily on KID/FID, lacking dedicated metrics for texture consistency and PBR material accuracy.
- MR Rectification assumes globally uniform MR properties per garment, which may not apply to complex multi-material garments.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | First feed-forward architecture for non-isometric texture generation; image-editing-driven cross-topology augmentation is novel; however, the backbone network is borrowed from Hunyuan3D and the UQ component draws on AVS. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comparisons against multiple commercial models (Hyper3D, Meshy, Hunyuan3D) covering both industrial and generated mesh scenarios, with quantitative ablation of baking strategies; quantitative metrics are limited to KID/FID. |
| Writing Quality | ⭐⭐⭐⭐ | Architecture diagrams are clear and comprehensive, problem formulation is precise, and the logic distinguishing and handling cross-topology versus cross-pose scenarios is coherent; the appendix provides thorough supplementary material. |
| Value | ⭐⭐⭐⭐⭐ | Directly addresses industrial-grade 3D garment design needs; generated PBR materials are applicable to real rendering pipelines; code will be open-sourced; high practical value. |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](text-image_conditioned_3d_generation.md)
- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting](fact-gs_frequency-aligned_complexity-aware_texture_reparameterization_for_2d_gau.md)

<!-- RELATED:END -->
