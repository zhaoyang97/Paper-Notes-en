---
title: >-
  [Paper Note] Blended Point Cloud Diffusion for Localized Text-guided Shape Editing
description: >-
  [ICCV 2025 (Highlight)][3D Vision][Point cloud editing] This paper proposes BlendedPC, which reformulates localized text-guided 3D shape editing as a semantic inpainting problem. By fine-tuning an Inpaint-E model on top of Point·E and introducing an inversion-free coordinate blending mechanism at inference time, BlendedPC achieves precise local editing while preserving the identity of the original shape, outperforming existing methods comprehensively on the ShapeTalk dataset.
tags:
  - ICCV 2025 (Highlight)
  - 3D Vision
  - Point cloud editing
  - text-guided 3D editing
  - diffusion model inpainting
  - coordinate blending
  - localized shape editing
date: 2026-05-08
content_hash: d7b21c0e702fcb7f
---

# Blended Point Cloud Diffusion for Localized Text-guided Shape Editing

**Conference**: ICCV 2025 (Highlight)
**arXiv**: [2507.15399](https://arxiv.org/abs/2507.15399)
**Code**: [https://github.com/TAU-VAILab/BlendedPC](https://github.com/TAU-VAILab/BlendedPC) (Available, MIT License)
**Area**: 3D Vision / Diffusion Models / Shape Editing
**Keywords**: Point cloud editing, text-guided 3D editing, diffusion model inpainting, coordinate blending, localized shape editing

## TL;DR

This paper proposes BlendedPC, which reformulates localized text-guided 3D shape editing as a semantic inpainting problem. By fine-tuning an Inpaint-E model on top of Point·E and introducing an inversion-free coordinate blending mechanism at inference time, BlendedPC achieves precise local editing while preserving the identity of the original shape, outperforming existing methods comprehensively on the ShapeTalk dataset.

## Background & Motivation

Text-guided 3D shape editing is a rapidly evolving research direction. Existing methods such as ChangeIt3D and Spice·E enable semantic fine-grained editing based on text, yet they face a fundamental tension: **editing a local region often fails to preserve the structural consistency of the remaining regions**. For instance, attempting to make the legs of a chair thinner may inadvertently alter the backrest as well. This occurs because these methods operate on shapes globally, lacking explicit control over edited versus preserved regions.

The 2D image editing community has developed mature solutions for localized editing (e.g., Blended Diffusion, Blended Latent Diffusion), achieving precise region control through the inpainting paradigm. However, an analogous efficient solution for 3D point clouds remains absent. Furthermore, inversion-based methods commonly employed in 3D editing are computationally expensive and inaccurate, particularly in the context of conditional models.

## Core Problem

How to achieve **precise localized editing** on 3D point clouds—modifying only the region specified by text (e.g., "thinner legs") while **perfectly preserving** the structure of all other regions, without resorting to computationally expensive inversion?

The difficulty is threefold: (1) inpainting models inherently cannot "see" masked regions, making it difficult to reference the original shape for fine-grained editing; (2) textual descriptions in existing datasets tend to be global, making it hard to localize editing effects; (3) 3D point cloud inversion is far less mature than its 2D image counterpart.

## Method

### Overall Architecture

BlendedPC adopts a two-stage strategy:

**Training stage**: An **Inpaint-E** model is obtained by fine-tuning the Point·E base diffusion model using Cross-Entity Attention (following the design of Spice·E). The model takes three inputs—a masked partial point cloud $x_M$, a text prompt $C$, and a timestep $t$—and outputs a denoised prediction of the complete shape. During training, $(x_M, C)$ is occasionally replaced with the complete point cloud $x$ paired with an empty text prompt $C_0=""$, teaching the model high-fidelity reconstruction.

**Inference stage**: A **Coordinate Blending** mechanism is employed. The complete shape with an empty prompt is first used for reconstruction-driven denoising from $t=T$ to $t=t_r$. Starting from $t=t_r$, two branches run in parallel: (1) a reconstruction branch (complete shape + empty prompt); and (2) an inpainting branch (masked shape + editing prompt). At each step, the outputs of both branches are blended according to the mask—the edited region takes the inpainting result, and the remaining region takes the reconstruction result.

### Key Designs

1. **Inpaint-E — Point Cloud Inpainting Model with Cross-Entity Attention**: The self-attention layers in Point·E's 1024-point generator are replaced with cross-entity attention (from Spice·E), enabling structural information from the partial point cloud to be propagated to the denoising process. The 4096-point upsampler is left unmodified. Masked point coordinates are set to $(0,0,0)$ with color $(1,1,1)$, while unmasked points use color $(0,0,0)$ to eliminate ambiguity. The training loss is $\mathcal{L}_{Inpaint-E} = \|\varepsilon - \varepsilon_\theta(x_t, t, C, x_M)\|_2^2$.

2. **Inversion-Free Coordinate Blending**: The core innovation. This approach leverages Inpaint-E's reconstruction capability: when provided with a complete point cloud and an empty prompt, the model can faithfully reconstruct the original shape from arbitrary noise. During inference, $T - t_r$ steps of pure reconstruction are first performed (to establish a coarse shape prior for the edited region), after which the inpainting branch is activated starting from $t_r$. The per-step blending formula is: $\hat{x}_{t-1} \leftarrow \hat{x}_{t-1} \odot M + \hat{x}_{recon,t-1} \odot (1-M)$. This entirely bypasses inversion while ensuring that unedited regions remain nearly identical to the original shape.

3. **l-ShapeTalk — Localized Editing Data Construction**: Many textual descriptions in the ShapeTalk dataset are global (e.g., "looks more comfortable"), and shape differences are not confined to specific parts. The authors use LLaMA 3 to extract concrete part names (e.g., "seat") from each description, employ a PointNet-based segmentation model trained on PartNet to generate editing masks, and filter for localized samples to construct the l-ShapeTalk subset. Evaluation is conducted on three categories: Chair, Table, and Lamp.

### Loss & Training

- **Loss function**: Standard denoising objective $\mathcal{L}_{Inpaint-E} = \|\varepsilon - \varepsilon_\theta(x_t, t, C, x_M)\|_2^2$
- **Reconstruction training augmentation**: During training, $(x_M, C)$ is occasionally replaced by $(x, C_0="")$, enabling the model to perform high-fidelity reconstruction from a complete point cloud—a prerequisite for coordinate blending at inference time
- **Inference hyperparameters**: Heun sampler (Karras schedule), total steps $T=64$, transition timestep $t_r=20$ (empirically set to balance identity preservation and editing flexibility)
- **Two-stage generation**: The base model generates 1024 points; the upsampler expands them to 4096 points (upsampler unchanged)
- **Post-processing**: After inference, a segmentation model re-segments the output point cloud and applies nearest-neighbor replacement for points outside the edited region to further improve identity preservation

## Key Experimental Results

### Main Results (l-ShapeTalk test set, Chair + Table + Lamp)

| Method | CLIP Sim↑ | CLIP Dir↓ | GD↓ | CD↓ | FPD↓ | l-GD↓ |
|--------|-----------|-----------|-----|-----|------|-------|
| Changeit3D | 0.21 | 1.02 | 0.87 | 0.19 | 217.28 | 0.20 |
| Spice-E | 0.25 | 1.01 | 2.26 | 0.26 | 487.71 | 0.39 |
| **Ours** | **0.27** | **0.99** | **0.29** | **0.04** | **13.51** | **0.05** |

> Metric descriptions: GD = Chamfer distance (global identity preservation), l-GD = Chamfer distance outside the edited region (local identity preservation), FPD = Fréchet Point Distance (structural quality), CD = category distortion (classifier confidence difference), CLIP Sim = semantic similarity between output and text, CLIP Dir = alignment between editing direction and text direction

### Full ShapeTalk Test Set

| Method | CLIP Sim↑ | CLIP Dir↓ | GD↓ | CD↓ | FPD↓ | l-GD↓ |
|--------|-----------|-----------|-----|-----|------|-------|
| Changeit3D | 0.21 | 1.02 | 0.65 | 0.18 | 183.02 | 0.19 |
| Spice-E | 0.25 | 1.01 | 1.84 | 0.24 | 390.02 | 0.31 |
| **Ours** | **0.26** | **0.99** | **0.34** | **0.05** | **33.64** | **0.07** |

### User Study (60 participants, 15 questions each)

| Method | Changeit3D | Spice-E | Ours |
|--------|-----------|---------|------|
| Preference | 9% | 16% | **75%** |

### Ablation Study

| Variant (l-ShapeTalk) | CLIP Sim↑ | CLIP Dir↓ | GD↓ | CD↓ | FPD↓ | l-GD↓ |
|-----------------------|-----------|-----------|-----|-----|------|-------|
| Inpaint-E Only (w/o coordinate blending) | 0.26 | 1.01 | 1.31 | 0.12 | 73.45 | 0.39 |
| $t_r=T$ (blending at every step) | 0.25 | 1.01 | 0.92 | 0.13 | 102.52 | 0.14 |
| **Ours** ($t_r=20$) | **0.27** | **0.99** | **0.29** | **0.04** | **13.51** | **0.05** |

- **Coordinate blending is critical**: Removing coordinate blending (Inpaint-E Only) raises GD from 0.29 to 1.31 (4.5×) and l-GD from 0.05 to 0.39 (7.8×), indicating a severe degradation in identity preservation.
- **The transition timestep $t_r$ matters**: Setting $t_r=T$ (blending at every step) causes the model to remain "blind" to the edited region, yielding the worst FPD (102.52 vs. 13.51) and visible artifacts, as the inference procedure deviates substantially from the training distribution.
- **$t_r=20$ is the sweet spot**: The first 44 steps of pure reconstruction establish a shape prior for the edited region; the subsequent 20 steps perform inpainting with blending, balancing identity preservation and editing flexibility.

## Highlights & Insights

- **Reformulating localized 3D editing as inpainting** draws naturally on the paradigm of 2D Blended Diffusion and extends it elegantly to 3D point clouds.
- **Inversion-free coordinate blending**: By decomposing inference into reconstruction and inpainting branches, the method entirely circumvents the unreliable inversion process in 3D, requiring no additional training.
- **Clever reconstruction training trick**: Occasionally replacing inputs with a complete point cloud paired with an empty prompt during training endows the model with reconstruction capability, which is the fundamental prerequisite for coordinate blending at inference time.
- **Data construction pipeline**: Part extraction via LLaMA 3 → segmentation via PointNet → construction of the l-ShapeTalk subset constitutes a reusable pipeline for other localized editing tasks.
- **FPD drops from 217–488 to 13.5; GD and l-GD improve by over 75%**, representing substantial quantitative gains; a 75% user preference further corroborates the method's effectiveness.

## Limitations & Future Work

- **Category scope**: Training and evaluation are limited to Chair, Table, and Lamp; generalization to other categories requires additional data and segmentation model support.
- **Segmentation dependency**: The method relies on a pretrained PointNet segmentation model to obtain masks; segmentation failures directly cause editing failures, and the granularity of PartNet segmentation is limited.
- **Two-branch inference overhead**: Each timestep requires two forward passes (reconstruction + inpainting), roughly doubling inference time compared to the base model.
- **Limitations of the point cloud representation**: Point clouds lack topological structure and texture information; editing results must be converted to meshes for downstream applications.
- **Fixed $t_r=20$**: Different editing tasks (large modifications vs. subtle adjustments) may require different transition timesteps; adaptive selection of $t_r$ is a potential direction for improvement.
- **Training code not released**: As of now, only inference code and model weights are publicly available (HuggingFace: noamatia/BPCDiff).

## Related Work & Insights

| Dimension | Changeit3D | Spice·E | BlendedPC (Ours) |
|-----------|-----------|---------|-----------------|
| Editing scope | Global | Global | Local (mask-controlled) |
| Identity preservation | Moderate (GD=0.87) | Poor (GD=2.26) | Excellent (GD=0.29) |
| Base model | Independently trained | Point·E + CEA | Point·E + CEA + inpainting |
| Requires inversion | No | No | No |
| Dataset | ShapeTalk | ShapeTalk | l-ShapeTalk (localized subset) |
| Editing quality | CLIP Sim=0.21 | CLIP Sim=0.25 | CLIP Sim=0.27 |

- vs. **Changeit3D**: Identity preservation is moderate, but editing capability is weak (CLIP Sim=0.21); high FPD (217 vs. 13.5) indicates limited generation quality.
- vs. **Spice·E**: A prior work from the same group. CLIP Sim is comparable (0.25 vs. 0.27), but identity preservation is poor (GD=2.26, l-GD=0.39), as global editing inevitably disrupts structure.
- The proposed method inherits Spice·E's cross-entity attention design but introduces the inpainting paradigm and coordinate blending at inference time, substantially improving editing quality.

**Methodological transfer from 2D to 3D**: Blended Latent Diffusion → BlendedPC demonstrates that mature solutions from 2D image editing can be systematically transferred to the 3D domain. **The power of inference-time algorithms**: Without modifying the training procedure, introducing coordinate blending solely at inference time yields substantial performance gains—a paradigm applicable to other conditional generation tasks. **Reconstruction as the foundation for editing**: Teaching the model to reconstruct before editing is more elegant than DDIM inversion and can be generalized to NeRF/3DGS editing. **Data construction paradigm**: The pipeline of LLM-based part extraction followed by segmentation-based mask generation is transferable to a broader range of 3D understanding and editing tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ [Coordinate blending is elegant and effective, though the core idea is fundamentally a 3D adaptation of Blended Diffusion and is relatively straightforward in principle]
- Experimental Thoroughness: ⭐⭐⭐⭐ [Six metrics + user study + complete ablation, but limited to three categories with only two baselines]
- Writing Quality: ⭐⭐⭐⭐⭐ [Motivation is clear, method description is fluent, and the combination of algorithm pseudocode + method diagram + inference diagram is well integrated; project page is well-presented]
- Value: ⭐⭐⭐⭐ [ICCV Highlight; addresses a practical pain point in localized 3D editing; open-source code; quantitative improvements are highly significant]

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Image-Guided Shape-from-Template Using Mesh Inextensibility Constraints](image-guided_shape-from-template_using_mesh_inextensibility_constraints.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)
- [\[ICCV 2025\] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](guiding_diffusion-based_articulated_object_generation_by_partial_point_cloud_ali.md)

<!-- RELATED:END -->
