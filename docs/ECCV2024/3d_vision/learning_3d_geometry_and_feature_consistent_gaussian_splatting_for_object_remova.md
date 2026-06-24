---
title: >-
  [Paper Note] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This paper proposes the GScream framework, which achieves high-quality object removal in the 3D Gaussian Splatting representation using monocular depth-guided training and cross-attention feature regularization while maintaining geometric consistency and texture coherence.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Object Removal"
  - "Depth Guidance"
  - "Cross-Attention"
  - "Radiance Field Editing"
date: 2026-05-08
content_hash: 36766a2ee5115055
---

# Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal

**Conference**: ECCV 2024  
**arXiv**: [2404.13679](https://arxiv.org/abs/2404.13679)  
**Code**: [Available](https://w-ted.github.io/publications/gscream)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Object Removal, Depth Guidance, Cross-Attention, Radiance Field Editing

## TL;DR

This paper proposes the GScream framework, which achieves high-quality object removal in the 3D Gaussian Splatting representation using monocular depth-guided training and cross-attention feature regularization while maintaining geometric consistency and texture coherence.

## Background & Motivation

3D object removal is a complex yet crucial task in 3D vision, with widespread applications in virtual reality and content generation. Unlike 2D image inpainting, which primarily focuses on texture filling, 3D object removal also requires addressing geometric completion, making it significantly more challenging.

**Limitations of Prior Work**:

**Inefficiency of NeRF-based methods**: Although methods like SPIn-NeRF and OR-NeRF achieve reasonable results, the inherent drawbacks of implicit representations—such as slow training and rendering speeds—significantly limit their practical application. For instance, OR-NeRF requires approximately 6 hours of training.

**Inconsistent multi-view inpainting**: Existing methods typically rely on 2D inpainting models to generate multi-view pseudo ground truth. However, the inpainting results across different views are often inconsistent, resulting in "floater" or "ghosting" artifacts in the removed regions.

**Difficulties in directly applying 3DGS**: Despite the efficient rendering advantages of 3DGS, applying it to object removal faces two major challenges: (a) the large number of discrete Gaussian primitives leads to imprecise underlying geometry, making geometric completion difficult; and (b) filling consistent textures within the 3DGS framework remains under-explored.

**Key Insight**: Promote information interaction between visible regions and invisible regions (regions occluded by the removed object) to achieve content recovery across both geometric and texture dimensions. The explicit representation of 3DGS naturally facilitates this interaction, allowing direct manipulation of Gaussian primitives in 3D space rather than relying solely on supervision signals in the 2D image domain.

## Method

### Overall Architecture

GScream is built upon Scaffold-GS (a lightweight 3DGS architecture). Given multi-view masked images, it selects a reference view for 2D inpainting, and then learns the scene representation after object removal via two core modules:

1. **Monocular Depth-Guided Training**: Leverages monocular depth estimation as an auxiliary geometric constraint to optimize the positions of Gaussian primitives.
2. **Cross-Attention Feature Regularization**: Propagates texture information via 3D Gaussian sampling and bidirectional cross-attention.

### Key Designs

1. **Online Depth Alignment and Supervision**

   **Function**: Provides geometric constraints for 3DGS using monocular depth estimation, addressing the geometric inconsistency of the removed regions.

   **Mechanism**: Since the scale of monocular depth estimation differs from the rendered depth of 3DGS, an online least-squares alignment is adopted. A weighted depth loss is designed to distinguish between the reference view (including the inpainted region) and other views (only background regions):

   $\mathcal{L}_{\text{depth}} = \frac{1}{HW} \sum M'_i \| (w\hat{D}_i + q) - D_i \|$

   where $w, q$ are alignment parameters solved via least squares, and $M'_i$ is a weight mask designed for different views: for the reference view, the removed region has weight $\lambda_1$ and the visible region has weight $\lambda_2$; for other views, the weight $\lambda_3$ is applied only to the background region.

   **Design Motivation**: The inpainted image of the reference view contains complete depth data (including the removed region), whereas the masked regions of other views still contain the original object, meaning depth supervision can only be applied to the background region. Through this patch-wise weighted strategy, the method utilizes the complete depth of the inpainted reference image while avoiding interference from the object region in other views.

2. **Cross-Attention Feature Regularization**

   **Function**: Propagates accurate texture information from visible regions to inpainted regions to improve texture coherence.

   **Mechanism**: Leveraging the explicit nature of 3DGS, Gaussian anchors in the inpainted region and surrounding visible regions are first sampled in 3D space, and then bidirectional cross-attention is employed to promote feature interaction:

   $\hat{f}_{in} = \text{Attention}(\mathbf{Q}=f_{in}, \mathbf{K}=f_{sur}, \mathbf{V}=f_{sur})$
   $\hat{f}_{sur} = \text{Attention}(\mathbf{Q}=f_{sur}, \mathbf{K}=f_{in}, \mathbf{V}=f_{in})$

   where $f_{in}$ and $f_{sur}$ are the Gaussian anchor features of the inpainted region and the surrounding region, respectively.

   **Design Motivation**: Unlike NeRF methods that rely on multi-view pseudo ground truth or view-dependent effect simulation, the explicit representation of 3DGS permits direct manipulation of Gaussian features in 3D space. The bidirectional attention ensures reciprocal information flow: reliable textures from visible regions are propagated to the inpainted region, while the updates in the inpainted region feedback to influence the surrounding region, achieving a smoother transition.

3. **3D Gaussian Sampling Strategy**

   **Function**: Determines which Gaussian anchors belong to the inpainted region and which belong to the visible region for the cross-attention module.

   **Mechanism**: For each view $i$, patches overlapping the mask boundary are sampled. The center coordinates of the 3D anchors are projected onto the 2D plane of the current view, dividing them into two groups depending on whether their projections lie within the mask.

   **Design Motivation**: The strategy based on 2D mask back-projection is simple yet effective. Furthermore, sampling from different views covers distinct 3D regions, enhancing the comprehensiveness of feature interaction.

### Loss & Training

The total loss is a weighted sum of the depth loss, total variation (TV) smoothness loss, and color reconstruction loss:

$$\mathcal{L}_{\text{total}} = \lambda_{depth} \mathcal{L}_{\text{depth}} + \lambda_{tv} \mathcal{L}_{\text{tv}} + \mathcal{L}_{\text{color}}$$

- $\mathcal{L}_{\text{color}}$: $L_1$ + SSIM reconstruction loss
- $\mathcal{L}_{\text{tv}}$: Total variation loss of depth differences, steering depth smoothness
- The base model adopts Scaffold-GS, which decodes Gaussian attributes from anchor features to reduce storage requirements.

## Key Experimental Results

### Main Results

Quantitative comparison on the SPIn-NeRF dataset:

| Method | PSNR↑ | Masked PSNR↑ | SSIM↑ | Masked LPIPS↓ | FID↓ | Training Time |
|------|-------|-------------|-------|--------------|------|---------|
| SPIn-NeRF | 20.18 | 15.80 | 0.46 | 0.58 | 58.78 | ~3.0h |
| OR-NeRF | 20.32 | 15.74 | 0.54 | 0.56 | 38.69 | ~6.0h |
| **GScream** | **20.49** | **15.84** | **0.58** | **0.54** | **36.72** | **~1.2h** |

GScream matches or outperforms baseline methods across all metrics, with a training speed 2.5 times faster than SPIn-NeRF and 5 times faster than OR-NeRF.

### Ablation Study

| Configuration | PSNR↑ | Masked PSNR↑ | SSIM↑ | Masked SSIM↑ | Masked LPIPS↓ |
|------|-------|-------------|-------|-------------|--------------|
| w/o Cross-Attn & Mono-Depth | 20.12 | 14.87 | 0.58 | 0.19 | 0.56 |
| w/o Cross-Attn | 20.47 | 15.63 | 0.58 | 0.20 | 0.50 |
| **Full Model** | **20.49** | **15.84** | **0.58** | **0.21** | **0.54** |

- Removing depth supervision: Masked PSNR drops from 15.84 to 14.87 (-0.97), and geometric quality degrades significantly.
- Removing cross-attention: Masked PSNR drops to 15.63, with reduced texture coherence.

### Key Findings

- **Depth supervision dictates geometric quality**: Without depth supervision, Gaussian primitives "float in the air," causing severe texture-floating artifacts during rendering.
- **Cross-attention compensates for insufficient 2D priors**: Relying solely on 2D priors fails to mend texture holes in unobserved areas, whereas 3D feature interaction can propagate appropriate textures to the occluded regions.
- **The choice of depth estimation model impacts performance**: Marigold estimates more continuous depth maps than MiDaS, leading to more continuous GScream results.
- **The choice of 2D inpainting model is not critical**: Both LaMa and Stable Diffusion provide reasonable reference images; the key is simply obtaining a reasonable reference.

## Highlights & Insights

1. **Inherent Advantages of Explicit Representation**: This work is the first to fully exploit the explicit representation of 3DGS to perform direct feature interaction within 3D space, which is difficult to achieve with NeRF's implicit representations.
2. **Cooperative Geometry-Texture Optimization**: Improving the geometry through depth supervision first, then propagating textures over the refined geometry, establishing a virtuous cycle.
3. **Outstanding Efficiency**: The training time is only about 1.2 hours, which is several times faster than NeRF-based alternatives, demonstrating high practical value.

## Limitations & Future Work

- Relying on a 2D inpainting result from a single reference view may be insufficient for large-area occlusions.
- Cross-attention has a minor negative impact on LPIPS metrics, suggesting that feature propagation might introduce some deviation in high-frequency details.
- Video scenes or dynamic object removal have not yet been explored.
- Potential future work could integrate diffusion models to generate more diverse inpainting priors.

## Related Work & Insights

- **Scaffold-GS** provides an efficient 3DGS foundational architecture; its anchor-based design reduces storage and computational overhead.
- **GaussianEditor** is a concurrent 3DGS editing work, but lacks specific constraints in the 3D domain.
- The depth-guidance strategy can be extended to other 3DGS editing tasks, such as scene completion and style transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Conducting object removal on 3DGS is a relatively new direction. Designing bidirectional cross-attention to propagate information among 3D Gaussian features is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two datasets with multiple baseline comparisons, detailed ablation studies, and auxiliary experiments (different depth/inpainting models).
- **Writing Quality**: ⭐⭐⭐⭐ — Well-defined problem formulation, convincing motivation, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — An efficient 3D object removal solution that is significantly superior to and faster than NeRF-based methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GOR-IS: 3D Gaussian Object Removal In the Intrinsic Space](../../CVPR2026/3d_vision/gor-is_3d_gaussian_object_removal_in_the_intrinsic_space.md)
- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[ECCV 2024\] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally](flashsplat_2d_to_3d_gaussian_splatting_segmentation_solved_optimally.md)
- [\[ECCV 2024\] SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)

</div>

<!-- RELATED:END -->
