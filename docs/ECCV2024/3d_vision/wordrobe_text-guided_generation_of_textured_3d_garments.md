---
title: >-
  [Paper Note] WordRobe: Text-Guided Generation of Textured 3D Garments
description: >-
  [ECCV 2024][3D Vision][Text-to-3D Garment] Proposes WordRobe, which learns a 3D garment UDF latent space through a coarse-to-fine two-stage encoder-decoder framework. It utilizes a weakly supervised CLIP mapping network to achieve text-driven 3D garment generation and editing, and leverages the view-composited property of ControlNet to generate view-consistent texture maps in a single forward inference pass, running 13 times faster than Text2Tex.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Text-to-3D Garment"
  - "Latent Space"
  - "UDF"
  - "ControlNet Texture"
  - "CLIP-guided Generation"
date: 2026-05-08
content_hash: 52ea6a8cf483fd2d
---

# WordRobe: Text-Guided Generation of Textured 3D Garments

**Conference**: ECCV 2024  
**arXiv**: [2403.17541](https://arxiv.org/abs/2403.17541)  
**Code**: Planned to be open-source (project homepage available)  
**Area**: 3D Garment Generation / Text-driven 3D Content Creation  
**Keywords**: Text-to-3D Garment, Latent Space, UDF, ControlNet Texture, CLIP-guided Generation

## TL;DR

Proposes WordRobe, which learns a 3D garment UDF latent space through a coarse-to-fine two-stage encoder-decoder framework. It utilizes a weakly supervised CLIP mapping network to achieve text-driven 3D garment generation and editing, and leverages the view-composited property of ControlNet to generate view-consistent texture maps in a single forward inference pass, running 13 times faster than Text2Tex.

## Background & Motivation

**Background**: 3D garment modeling is in high demand for virtual try-on, game characters, and AR/VR experiences. Traditional methods rely on manual modeling using design tools like CLO or high-end 3D scanners, which are costly and difficult to scale.

**Limitations of Prior Work**:

- **Parametric methods** (BCNet, SMPLicit): Restricted by human body templates such as SMPL, they can only handle tight-fitting clothes, leading to limited garment varieties.

- **Non-parametric methods** (ReEF, xCloth): Can model various styles but output posed meshes with low-quality textures, making them difficult to use directly in graphics pipelines.

- **DrapeNet**: Although capable of learning a latent space, it lacks support for textures, offers uncontrollable generation, and requires explicit labels for editing.

- **General text-to-3D** (DreamBooth3D, etc.): Generate garments with poor surface quality. NeRF/SDF cannot handle open surfaces (cuffs, necklines), and multi-view optimization is extremely slow.

**Key Challenge**: A method is required that can both generate high-quality, unposed (canonical T-pose) 3D garment meshes and intuitively control shape and texture via text.

**Key Insight**: A divide-and-conquer strategy—first learn the latent space of garment geometry, then achieve text control through CLIP mapping, and finally utilize the zero-shot capability of ControlNet to efficiently generate textures.

**Core Idea**: Learning a robust 3D garment UDF latent space + weakly supervised CLIP-to-Latent mapping + view-composited ControlNet to generate textures in a single formulation.

## Method

### Overall Architecture

Input text prompt $\rightarrow$ CLIP text encoder yields embedding $\psi$ $\rightarrow$ Mapping Network predicts garment latent code $\phi \in \Omega$ $\rightarrow$ Coarse Decoder + Fine Decoder decode to UDF in two steps $\rightarrow$ Marching Cubes extracts mesh $\rightarrow$ UV parameterization $\rightarrow$ Front and back depth maps are merged and fed into ControlNet $\rightarrow$ Generates a view-composited RGB image $\rightarrow$ Projected onto the UV texture map $\rightarrow$ Final textured 3D garment.

### Key Designs

1. **Coarse-to-Fine 3D Garment Latent Space**:

    - **Function**: Encodes multi-category 3D garments into a 32-dimensional latent space $\Omega$, allowing decoding back to high-quality geometry.
    - **Mechanism**: Uses a DGCNN as the encoder $\xi$ to encode the garment surface point cloud into $\phi \in \mathbb{R}^{32}$; employs two task-specific MLP decoders: $D_{coarse}$ predicts smooth UDF, and $D_{fine}$ predicts residuals to correct high-frequency details.
    - **Key Formula**: $\sigma_{fine} = D_{coarse}(\phi) + D_{fine}(\phi) = \sigma_{coarse} + \sigma_{delta}$
    - **Design Motivation**: A single decoder cannot simultaneously learn a regularized latent space and high-frequency geometric details (e.g., wrinkles, folds). The coarse decoder is responsible for the overall shape + latent space regularization, while the fine decoder is responsible for detailing.
    - **Latent space disentanglement loss**: $\mathcal{L}_{latent} = \|\Sigma_b - \mathbf{I}_k\|$, which encourages independence among latent dimensions, allowing single-dimension manipulation for a single attribute change.
    - **Coarse stage loss**: $\mathcal{L}_{coarse} = \lambda_{dist}\mathcal{L}_{dist} + \lambda_{grad}\mathcal{L}_{grad} + \lambda_{latent}\mathcal{L}_{latent}$
    - **Fine stage loss**: $\mathcal{L}_{fine} = \lambda_{dist}\mathcal{L}_{dist} + \lambda_{grad}\mathcal{L}_{grad}$

2. **CLIP-Guided Weakly Supervised Mapping Network**:

    - **Function**: Trains an $MLP_{map}$ to map CLIP embeddings to garment latent codes, enabling text control.
    - **Mechanism**: Eliminates the need for manual text labeling—renders depth maps with random rotations for each training garment, feeds them into ControlNet to generate garment images, and then obtains $\psi_i$ via a CLIP image encoder; simultaneously, $\phi_i$ is obtained via the encoder $\xi$. The $MLP_{map}$ is trained on $(\psi_i, \phi_i)$ pairs.
    - **Training loss**: Simple L1 loss $\|\text{MLP}_{map}(\psi_i) - \phi_i\|_1$
    - **Design Motivation**: 3D garment data lacks text annotations. Leveraging ControlNet to generate realistic garment images and passing them through a CLIP image encoder cleverly eliminates the annotation requirement.
    - **Prompt template construction**: Random combinations of "a garment made of {silk/cotton/wool/leather}, with {vibrant/dull/bright/shiny/matte} colors"

3. **View-Composited ControlNet Texture Synthesis**:

    - **Function**: Generates view-consistent, high-quality texture maps for 3D garments in a single forward inference pass.
    - **Mechanism**: Discovered a key property of ControlNet—when multi-view depth maps are concatenated into a single image input, the generated RGB image maintains color and lighting consistency across different views. Front-back orthographic projections are used to render depth maps $\pi_{depth}$, which are concatenated into a single 1024x1024 image. ControlNet generates $\pi_{rgb}$, which is then projected onto the UV texture map.
    - **Design Motivation**: Multi-view optimization methods like Text2Tex are slow (~5 min/prompt) and suffer from view inconsistencies and patchy artifacts. The proposed method only requires ~22 seconds.
    - **Orthographic projection selection**: Perspective projection loses more information in tangent regions; front-back division is a natural choice for garments, reducing visible seams.

### Loss & Training

- Encoder $\xi$ + $D_{coarse}$ are jointly trained for 20 epochs ($\lambda_{dist}=1.0, \lambda_{grad}=0.3, \lambda_{latent}=0.2$)
- $D_{fine}$ is trained separately for 10 epochs
- $MLP_{map}$: 10-layer MLP with skip connections, optimized with AdamW
- Training data: Approximately 20,000 unposed garments from [20], belonging to 19 categories, with 12 categories for training and 7 for testing.

## Key Experimental Results

### Main Results — Garment Geometry Quality

| Method | CD↓ | P2S↓ |
|------|-----|------|
| DrapeNet | 1.796 | 0.573 |
| Ours (Single Stage) | 1.631 | 0.494 |
| **Ours (Full)** | **1.078** | **0.329** |

Compared to DrapeNet, CD is reduced by approximately 40% and P2S is reduced by approximately 42%.

### Cross-Dataset Generalization (CLOTH3D)

| Method | CD (topwear)↓ | P2S (topwear)↓ |
|------|--------------|----------------|
| DrapeNet (trained on CLOTH3D) | 1.522 | **0.631** |
| Ours (trained on [20]) | **1.491** | 0.635 |

Even when trained only on the [20] dataset and tested on CLOTH3D, the model still achieves performance comparable to or even better than DrapeNet trained directly on CLOTH3D.

### Texture Synthesis Comparison

| Method | CLIP Score (ViT-H/14)↑ | Speed |
|------|----------------------|------|
| Text2Tex | 0.263 ± 0.047 | ~5 min |
| **WordRobe** | **0.304 ± 0.043** | **~22 sec** |

13 times faster, with a higher CLIP score and better view consistency.

### Ablation Study

| Configuration | CD↓ | P2S↓ | Description |
|------|-----|------|------|
| w/o $\mathcal{L}_{grad}$ | 1.886 | 0.612 | Lacks gradient regularization |
| w/o $\mathcal{L}_{latent}$ | 1.094 | 0.331 | Disentanglement loss is auxiliary but not decisive |
| Full | **1.078** | **0.329** | Optimal |

| Interpolation Evaluation | $\Delta_{area}$↓ | $\Delta_{vol}$↓ |
|---------|-----------------|----------------|
| w/o $\mathcal{L}_{latent}$ | 0.028 | 1.275 |
| with $\mathcal{L}_{latent}$ | **0.022** | **1.206** |

### Key Findings
- The coarse-to-fine two-stage decoding significantly reduces surface noise and holes.
- $\mathcal{L}_{grad}$ plays a critical regularizing role in reducing high-frequency noise.
- $\mathcal{L}_{latent}$ has little impact on CD/P2S but significantly improves the quality of interpolation.
- In user studies, 63% of users prefer WordRobe (vs. 27% preferring the DreamFusion variant).

## Highlights & Insights

- **Weakly supervised CLIP mapping scheme**: Utilizes ControlNet generation $\rightarrow$ CLIP encoding to construct training pairs, completely avoiding manual text annotation. The mechanism is clever and generalizable.
- **View-composited property of ControlNet**: This empirical finding is highly practical—rendering multi-view depth maps into a single composite image and feeding it to ControlNet naturally preserves view consistency in the output. This represents a new paradigm for texture generation without requiring multi-view optimization.
- **Practicality of canonical T-pose generation**: Directly interfaces with standard animation/simulation pipelines (rigging, skinning, cloth simulation), offering high industrial application value.
- **CLIP arithmetic for latent editing**: Leverages CLIP text-text vector arithmetic to automatically locate dimensions in the latent space that need modification, eliminating the need for explicit labeling.

## Limitations & Future Work

- Front-back orthographic projection loses texture information in tangent regions, requiring inpainting which can lead to blurry seams.
- The implicit UDF representation struggles to model fine-grained geometric details (pockets, buttons, etc.).
- Texture synthesis contains baked-in shadows/lighting/edge hallucinations, which limits its applicability to new illumination environments.
- Handles only single-piece garments, with no support for layered clothing yet.
- Both training and evaluation data are synthetic (CLOTH3D, [20]), and generalization to real-world garments remains to be verified.

## Related Work & Insights

- **vs DrapeNet**: Both learn a garment UDF latent space, but DrapeNet has no textures, no text control, and requires explicit labels for editing. WordRobe is a comprehensive upgrade in all aspects.
- **vs Text2Tex**: Text2Tex uses progressive multi-view inpainting for texture generation, which is slow and view-inconsistent. WordRobe is a single-pass forward approach and is 13x faster.
- **vs DreamBooth3D/DreamFusion**: General text-to-3D methods using SDF cannot handle the open surfaces of clothing, and their geometric quality is far inferior to specialized methods.

## Rating

- Novelty: ⭐⭐⭐⭐ The first text-driven unposed textured 3D garment generation framework, featuring three major innovations: coarse-to-fine, weakly supervised CLIP mapping, and view-composited texture.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparisons, user studies, cross-dataset generalization, and ablation analyses are comprehensively covered.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete formulation of the method, and rich illustrations.
- Value: ⭐⭐⭐⭐ High industrial application value; the concept of view-composited texture generation can be widely generalized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
