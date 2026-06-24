---
title: >-
  [Paper Note] ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance
description: >-
  [ECCV 2024][3D Vision][Compositional 3D Generation] This paper proposes ComboVerse, a compositional 3D asset generation framework. It first decomposes an input image containing multiple objects into individual elements and reconstructs them independently as single-object 3D models. Then, it optimizes the position, scale, and rotation parameters of the objects guided by Spatially-Aware Score Distillation Sampling (SSDS), enabling high-quality multi-object compositional 3D asse…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Compositional 3D Generation"
  - "Multi-Object 3D Reconstruction"
  - "Diffusion Model Guidance"
  - "Spatially-Aware SDS"
  - "Single-Image 3D Reconstruction"
date: 2026-05-08
content_hash: f70d72be076c7e63
---

# ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance

**Conference**: ECCV 2024  
**arXiv**: [2403.12409](https://arxiv.org/abs/2403.12409)  
**Code**: [https://cyw-3d.github.io/ComboVerse/](https://cyw-3d.github.io/ComboVerse/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Compositional 3D Generation, Multi-Object 3D Reconstruction, Diffusion Model Guidance, Spatially-Aware SDS, Single-Image 3D Reconstruction

## TL;DR

This paper proposes ComboVerse, a compositional 3D asset generation framework. It first decomposes an input image containing multiple objects into individual elements and reconstructs them independently as single-object 3D models. Then, it optimizes the position, scale, and rotation parameters of the objects guided by Spatially-Aware Score Distillation Sampling (SSDS), enabling high-quality multi-object compositional 3D asset creation. It significantly outperforms existing methods in both CLIP Score and human evaluation.

## Background & Motivation

Generating high-quality 3D assets from a single image is a core demand in fields such as AR/VR, gaming, film, and television. Recently, feed-forward single-image 3D generation models have made significant progress, but they face a systemic issue:

**"Multi-Object Gap"**—Current mainstream models perform well on single objects, but their performance drops sharply when confronting complex scenes containing multiple objects. Through in-depth analysis, three root causes are revealed:

**Camera Setting Bias**: Most models assume normalized sizes and centered positions for objects. Consequently, the reconstruction quality of small or off-center objects in multi-object scenes degrades significantly.

**Dataset Bias**: The training dataset, Objaverse, is dominated by single-object assets and contains almost no occlusions. This prevents models from generalizing to multi-object compositions and occluded scenes, resulting in "fusion" artifacts in the generated outputs.

**Leaking Pattern**: When generating multiple objects simultaneously, the geometry and appearance of one object leak into another (e.g., the colors of an owl leaking onto the back of a tiger).

**Key Insight**: Since existing methods work well on single-object reconstruction, why not reconstruct each object independently first and then automatically compose them? This exactly mirrors the workflow of professional 3D artists, who model individual objects first before assembling them into a complete scene.

## Method

### Overall Architecture

ComboVerse consists of two stages:
1. **Single-object reconstruction stage**: Decompose individual objects from the input image, remove occlusions, and perform single-image 3D reconstruction independently.
2. **Multi-object composition stage**: Keep the geometry and texture of each object fixed, and only optimize their scale $s_i$, rotation $r_i$, and translation $t_i$ parameters, guided by spatially-aware SDS loss and reference view loss to resolve the spatial layout.

### Key Designs

1. **Components Decomposition & Object Inpainting**:

    - Use SAM to segment each object based on 2D bounding boxes: $O_i, M_i = \text{SAM}(I, b_i)$.
    - **Occlusion Inpainting Strategy**:
        - Replace the object background with random noise (to prevent white/black borders during-inpainting): $I_i = O_i + noise \cdot (\sim M_i)$.
        - Construct a bounding box-aware mask: $m_i = (\sim M_i) \cap b_i$ to mark the regions to be inpainted.
        - Use Stable Diffusion with the text prompt "a complete 3D model" for inpainting.
    - **Design Motivation**: The noise background, bounding box-aware mask, and text guidance are all indispensable, as confirmed by ablation studies showing the necessity of each component.

2. **Spatially-Aware Score Distillation Sampling (SSDS)**:

    - **Limitations of Prior Work**: When the image content already matches the text prompt, standard SDS does not drive positional adjustments—it prioritizes content matching over spatial relationships.
    - **Mechanism**: In the cross-attention of UNet, enhance the attention weights of tokens that describe spatial relationships (e.g., "sitting on", "riding", "front").
    $$M := \begin{cases} c \cdot M_j & \text{if } j = j^\star \\ M_j & \text{otherwise} \end{cases}$$
   where $c > 1$ is an amplification constant ($e.g., c=25$ in experiments), and $j^\star$ is the index of the spatial relationship token.
    - SSDS Gradient:
    $$\nabla_\theta \mathcal{L}_{\text{SSDS}}(\phi^\star, x) = \mathbb{E}_{t,\epsilon}[w(t)(\hat{\epsilon}_{\phi^\star}(x_t;y,t) - \epsilon)\frac{\partial x}{\partial \theta}]$$
    - Timestep sampling range: [800, 900] (high noise levels), as these steps have the most significant impact on spatial layout.
    - **Spatial Token Extraction**: Can be automatically extracted by an LLM or specified by the user.

3. **Combine the Objects**:

    - **Coarse Initialization**:
        - Scale: $s_i = \max\{W_{b_i}/W_I, H_{b_i}/H_I\}$, based on the ratio of bounding box dimensions to the image dimensions.
        - Translation: x/y coordinates are determined by the center of the bounding box, and z is determined by the mean of monocular depth estimation.
        - Rotation: Initialized to (0,0,0).
    - **Fine Optimization**: Uses SSDS as novel-view supervision + reference view reconstruction loss.
    $$\mathcal{L}_{\text{Ref}} = \lambda_{\text{RGB}}|\hat{I}_{\text{RGB}} - I_{\text{RGB}}| + \lambda_A|\hat{I}_A - I_A|$$
    - The total loss is a weighted sum: $\mathcal{L}_{\text{Ref}} + \mathcal{L}_{\text{SSDS}}$.

### Loss & Training

- Rendering engine: PyTorch3D
- Optimizer: Adam
- Learning rate for z-translation: 0.01, other parameters: 0.001
- Loss weights: $\lambda_{\text{Ref}} = 1$, $\lambda_{\text{SSDS}} = 1$, $\lambda_{\text{RGB}} = 1000$, $\lambda_A = 1000$
- Attention amplification constant $c = 25$
- Stable Diffusion inpainting settings: guidance scale = 7.5, inference steps = 30
- Renders 10 views per iteration
- Decimates each object mesh to 50,000 faces

## Key Experimental Results

### Main Results

**Quantitative Comparison (100 test images)**:

| Method | CLIP-Score↑ | GPT-3DScore↑ |
|------|------------|-------------|
| SyncDreamer | 81.47% | 13.54% |
| OpenLRM | 83.65% | 53.12% |
| Wonder3D | 85.57% | 56.25% |
| **ComboVerse** | **86.58%** | **65.63%** |

- User Study: 990 responses collected from 22 evaluators show that ComboVerse consistently outperforms all baseline methods in both geometry and texture quality.

### Ablation Study

**SSDS Ablation (Effect of Attention Amplification)**:

| Guidance Method | CLIP B/16 Color↑ | CLIP B/16 Geo↑ | ResNet50 Color↑ | ResNet50 Geo↑ |
|---------|-----------------|---------------|----------------|--------------|
| No Guidance (Base) | 86.62% | 75.24% | 80.35% | 74.19% |
| Depth Loss | 84.57% | 78.42% | 81.69% | 75.83% |
| Standard SDS | 84.16% | 78.25% | 84.08% | 74.66% |
| SSDS (Uniform Noise) | 85.33% | 78.49% | 85.55% | 75.85% |
| SSDS (Low Noise) | 84.86% | 79.03% | 84.42% | 75.44% |
| **SSDS (Full)** | **89.01%** | **79.66%** | **86.60%** | **78.10%** |

**Object Inpainting Ablation**:
- Removing noisy background $\rightarrow$ Black borders appear in the inpainted results.
- Replacing bounding box-aware mask with background mask $\rightarrow$ Extraneous parts are generated in the inpainted results.
- Removing text guidance $\rightarrow$ Inpainting quality degrades.

### Key Findings

- SSDS performs best in the high-noise interval [800, 900], consistent with the intuition that early steps in the diffusion denoising process determine the global layout.
- Standard SDS even underperforms the unguided Base on certain metrics, indicating that standard SDS indeed has inherent limitations in task-oriented position adjustment.
- The spatial token attention amplification coefficient $c=25$ is highly effective, indicating that diffusion models indeed encode knowledge of spatial relationships, which is otherwise overshadowed by content matching in standard SDS.
- The method is scalable to scene reconstruction with $>2$ objects (experiments demonstrate a 4-object scene).

## Highlights & Insights

1. **"Divide-and-Conquer" Compositional Paradigm**: Mimicking the workflow of human 3D artists, decomposing a complex problem into resolved sub-problems followed by automatic assembly is a pragmatic and scalable approach.
2. **In-depth Analysis of the "Multi-Object Gap"**: Systematically diagnoses the failure modes of existing methods across three dimensions—Camera Setting Bias, Dataset Bias, and Leaking Pattern—providing a clear direction for future research.
3. **Simple and Effective SSDS**: Significantly improves spatial layout guidance merely through simple scaling of attention weights, offering low implementation cost with remarkable gains.
4. **Optimizing Only Spatial Parameters without Modifying Geometry and Texture**: Greatly accelerates the optimization process while preventing the geometric and texture degradation typically induced by SDS.

## Limitations & Future Work

1. Applicable to scenes with 2 to 5 objects, while more complex scenes remain challenging.
2. The composition stage does not optimize geometry and texture, meaning the final quality is bottlenecked by the capability of the single-image 3D backbone model.
3. Depth-scale ambiguity: Inaccurate depth estimation in single-view images can lead to suboptimal initialization.
4. Explorable paths include utilizing physical constraints (such as gravity or contact forces) during the composition stage to further improve spatial relationships.
5. Modeling of object-to-object interactions (such as deformation or contact surface blending) has not yet been addressed.
6. Future work can integrate multi-view diffusion models (such as Zero123++) to replace monocular depth estimation and improve compositional accuracy.

## Related Work & Insights

- Comparison with text-based compositional 3D generation such as Set-the-Scene: Images describe spatial relationships more precisely than text, thereby imposing higher requirements on composition quality.
- The SSDS concept can be generalized to other SDS applications requiring precise spatial alignment (e.g., text-to-3D scene generation).
- Attention map manipulation methods (originating from Attend-and-Excite) have wide applicability in 3D generation.
- Advancements in single-object 3D generation models (such as more powerful backbones) will directly improve the performance of ComboVerse.

## Rating

- Novelty: ⭐⭐⭐⭐ (The compositional paradigm and SSDS designs are novel, though individual components reuse existing methods.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Self-built benchmark, comparisons with multiple baselines, user study, and extensive ablations, though the benchmark scale is relatively small.)
- Writing Quality: ⭐⭐⭐⭐ (Clear analysis, rich illustrations, and well-articulated motivations.)
- Value: ⭐⭐⭐⭐ (Fills the gap in multi-object 3D generation; the compositional paradigm holds promising prospects for practical application.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](../../ICCV2025/3d_vision/reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[ECCV 2024\] 4Diff: 3D-Aware Diffusion Model for Third-to-First Viewpoint Translation](4diff_3d-aware_diffusion_model_for_third-to-first_viewpoint_translation.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
