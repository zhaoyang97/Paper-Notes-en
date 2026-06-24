---
title: >-
  [Paper Note] FlashTex: Fast Relightable Mesh Texturing with LightControlNet
description: >-
  [ECCV 2024][3D Vision][Texture Generation] Proposed LightControlNet, an illumination-aware variant of ControlNet. Combined with a two-stage texture optimization pipeline, it generates high-quality, relightable PBR textures for 3D meshes in approximately 4 minutes, which is 3-10 times faster than existing methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Texture Generation"
  - "Relightability"
  - "ControlNet"
  - "Score Distillation Sampling"
  - "Material Decoupling"
date: 2026-05-08
content_hash: a134ea8e0108b042
---

# FlashTex: Fast Relightable Mesh Texturing with LightControlNet

**Conference**: ECCV 2024  
**arXiv**: [2402.13251](https://arxiv.org/abs/2402.13251)  
**Code**: No public code  
**Area**: 3D Vision  
**Keywords**: Texture Generation, Relightability, ControlNet, Score Distillation Sampling, Material Decoupling

## TL;DR

Proposed LightControlNet, an illumination-aware variant of ControlNet. Combined with a two-stage texture optimization pipeline, it generates high-quality, relightable PBR textures for 3D meshes in approximately 4 minutes, which is 3-10 times faster than existing methods.

## Background & Motivation

**Background**: Text-driven 3D mesh texture generation has made significant progress in recent years. Methods based on Score Distillation Sampling (SDS) (such as DreamFusion and Fantasia3D) and view-projection-based methods (such as Text2tex and TEXTure) each have their own strengths.

**Limitations of Prior Work**: Current methods suffer from three major issues: (1) slow generation speed (typically requiring tens of minutes); (2) visual artifacts such as seams and blurriness; (3) illumination is baked into the texture (baked-in lighting), making rendering unnatural under new lighting environments.

**Key Challenge**: Existing methods are either fast but not relightable (e.g., Text2tex), or relightable but extremely slow (e.g., Fantasia3D requires 5000 iterations taking 30 minutes). There is a fundamental trade-off between speed, quality, and relightability.

**Goal**: To simultaneously address the three issues of speed, quality, and relightability. While maintaining high-quality textures, it decouples lighting and material properties (albedo, metallic, roughness, etc.) and compresses the generation time to approximately 4 minutes.

**Key Insight**: Starting from the ControlNet architecture, an illumination condition image is introduced as an additional control signal to achieve illumination-aware text-to-image generation, which is then combined with a reference-view-guided optimization strategy to dramatically reduce the number of SDS iterations.

**Core Idea**: Utilizing a LightControlNet conditioned on three predefined material renders to achieve illumination-controlled generation, combined with multi-view visual prompting and SDS optimization of only 400 iterations for rapid relightable texture generation.

## Method

### Overall Architecture

FlashTex adopts a two-stage pipeline: **Stage 1** generates four sparse but visually consistent reference views using LightControlNet via Multi-view Visual Prompting; **Stage 2** texture optimization is guided by the reference views, representing the texture with a multi-resolution hash grid and optimized via a joint reconstruction loss + SDS loss + smoothness regularization, completing texture generation in only 400 iterations.

### Key Designs

1. **LightControlNet (Light Control Network)**:

    - **Function**: Introduces illumination control capabilities based on ControlNet, enabling the generated images to specify particular lighting environments.
    - **Mechanism**: The conditioning image is composed of three predefined material renders: (1) non-metallic + rough; (2) semi-metallic + semi-rough; (3) fully metallic + highly smooth. The three renders are stacked into a three-channel conditioning image $I_{\text{cond}}(L, C)$. These three materials span the full range from diffuse to specular reflections, fully encoding the illumination information.
    - **Training**: Trained on 40K objects from Objaverse, with 12 views and 6 environmental lights randomly sampled per object, resulting in 480K training pairs.
    - **Design Motivation**: Rendering with only a single material is insufficient to control illumination direction and intensity. Three complementary materials can fully describe the light-geometry interaction.

2. **Distilled Encoder**:

    - **Function**: Distills and accelerates the image encoder of Stable Diffusion.
    - **Mechanism**: Removes the attention modules from the encoder and trains it on the COCO dataset to match the outputs of the original encoder. The distilled encoder runs 5x faster, accelerating the overall pipeline by approximately 2x.
    - **Design Motivation**: The original SD encoder accounts for nearly 50% of the forward/backward pass time of SDS, acting as the performance bottleneck.

3. **Multi-view Visual Prompting**:

    - **Function**: Concatenates condition images from 4 canonical views into a $2 \times 2$ grid as a single image input for LightControlNet.
    - **Mechanism**: $I_{\text{ref}} = \text{ControlNet}(I_{\text{cond}}(L^*, C^*), y)$, where $C^*$ represents the 4 canonical views (front, back, left, right), and $L^*$ represents fixed lighting.
    - **Design Motivation**: Generating 4 images independently leads to appearance inconsistency (Figure 5). Placing the 4 views into a grid allows the diffusion model to leverage the grid-like priors from the training data, naturally producing style-consistent 4-views. This is a key insight that exploits the data distribution characteristics of the Stable Diffusion training set.

4. **Texture Optimization**:

    - **Function**: Optimizes joint reference views and SDS to generate PBR material textures decoupled from lighting.
    - **Mechanism**: Uses a multi-resolution hash grid $\beta$ + a two-layer MLP $\Gamma$ to represent the 3D texture: $(k_c, k_m, k_r, k_n) = \Gamma(\beta(p))$, which respectively output base color, metallicness, roughness, and bump vectors. Rendered into a 2D image via the nvdiffrast differentiable renderer, while optimizing both reconstruction loss and SDS loss:
        - Reconstruction loss: $\mathcal{L}_{\text{recon}} = \|I_{\text{ref}} - \mathcal{R}(\Gamma(\beta(\cdot)), L^*, C^*)\|_2 + \mathcal{L}_{\text{perceptual}}$
        - SDS loss: Uses LightControlNet as the diffusion model to randomly sample views and lightings.
    - **Design Motivation**: Direct back-projection yields seams and baked-in lighting issues. SDS optimization fills in unobserved regions between views and decouples lighting. Reference-view guidance allows optimization to take only 400 iterations (vs 5000 in Fantasia3D).

### Loss & Training

- **Reconstruction loss**: L2 + perceptual loss, weight $\lambda_{\text{recon}} = 1000$
- **SDS loss**: Diffusion model gradients based on LightControlNet
- **Smoothness regularization**: $\mathcal{L}_{\text{reg}} = \sum_{p \in S} |k_c(p) - k_c(p + \epsilon)|$, weight $\lambda_{\text{reg}} = 10$
- **Optimization schedule**: Warm-up for the first 50 iterations with reconstruction loss only; thereafter, alternate between SDS and reconstruction loss; noise level linearly decreases $t: 0.1 \to 0.02$; ControlNet conditioning strength $s$ linearly decreases from 1 to 0.
- Total iteration steps are only 400.

## Key Experimental Results

### Main Results

| Method | FID↓ (Objaverse) | KID(×10⁻³)↓ | FID↓ (Game) | KID(×10⁻³)↓ | Time (min) |
|------|------------------|-------------|-------------|-------------|-----------|
| Latent-Paint | 73.65 | 7.26 | 204.43 | 9.25 | 10 |
| Fantasia3D | 120.32 | 8.34 | 164.32 | 9.34 | 30 |
| TEXTure | 71.64 | 5.43 | 103.49 | 5.64 | 6 |
| Text2tex | 95.59 | 4.71 | 119.98 | 5.21 | 15 |
| **Ours (w/ depth)** | **60.49** | 3.96 | **85.92** | 3.87 | **2** |
| **Ours (LightControlNet)** | 62.67 | **2.69** | 83.32 | **3.34** | 4 |

### Ablation Study

| Configuration | FID↓ | KID(×10⁻³)↓ | Time (min) | Description |
|------|------|-------------|-----------|------|
| Without distilled encoder | 60.34 | 2.84 | 8 | Time doubles, no significant improvement in quality |
| Without multi-view prompting | 74.23 | 3.54 | 19 | Requires 2000 iterations to converge, 5× slower |
| **Full method** | **62.67** | **2.69** | **4** | Optimal balance |

### Key Findings

- Removing any material base degrades quality (Table 4), demonstrating that the complementarity of the three materials is crucial.
- Four canonical views (front, back, left, right) are the optimal choice; 2 views are insufficient, and 6 views (adding top, bottom) are actually worse—due to the poor ability of 2D diffusion models to generate top/bottom viewpoints and the decreased resolution after grid concatenation.
- In a user study, 30 participants preferred the proposed method across all three aspects—realism, text consistency, and relighting plausibility (vs. all baselines, with a >57% preference rate).

## Highlights & Insights

- **Consistency Trick via Multi-view Concatenation**: Concatenating 4 views into a $2\times2$ grid leverages the priors in the SD training set to achieve appearance consistency—a neat and effective discovery corporate into subsequent works.
- **Encoder Distillation**: Simply removing modules and retraining yields a 2× speedup without compromising quality.
- **Feasibility of Few-step SDS**: By providing strong priors via reference views, SDS iterations are reduced from 5000 (Fantasia3D) to 400 (a 10× speedup).
- **PBR Material Output**: Directly outputs PBR parameters such as metallicness, roughness, and base color, facilitating direct usage in downstream rendering engines.

## Limitations & Future Work

- Illumination baking issues still persist for some out-of-distribution (OOD) meshes.
- Material maps are occasionally not fully decoupled into interpretable metallicness/roughness.
- LightControlNet is trained only on Objaverse, limiting its generalization capability to real-world objects.
- More advanced SDS variants (e.g., VSD) could be explored to further enhance quality.

## Related Work & Insights

- **vs. Fantasia3D**: Also generates PBR materials, but suffers from severe baked-in lighting and requires 30 minutes. FlashTex improves lighting decoupling and runs 7.5× faster.
- **vs. TEXTure/Text2tex**: Fast view-projection-based methods, but textures contain baked-in lighting and are not relightable.
- **vs. TANGO**: Uses a Spherical Gaussian renderer but struggles with complex texture generation.
- **vs. Paint3D (concurrent)**: Generates unlit textures but does not output material maps.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The design of LightControlNet and the multi-view grid consistency trick are highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Quantitative evaluation + user study + multiple ablation studies cover almost all design choices.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with intuitive pipeline diagrams and complete methodology descriptions.
- **Value**: ⭐⭐⭐⭐ Achieves a good trade-off in the speed-quality-relightability trilemma, holding practical significance for 3D content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting](repaint123_fast_and_high-quality_one_image_to_3d_generation_with_progressive_con.md)

</div>

<!-- RELATED:END -->
