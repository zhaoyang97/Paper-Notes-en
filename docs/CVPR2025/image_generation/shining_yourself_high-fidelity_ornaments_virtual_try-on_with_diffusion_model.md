---
title: >-
  [Paper Note] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model
description: >-
  [CVPR 2025][Image Generation][Virtual Try-on] This work introduces diffusion models to the virtual try-on task of ornaments (bracelets, rings, earrings, necklaces) for the first time. It proposes an iterative pose-aware wearing-mask prediction and a mask-guided attention mechanism, achieving high-fidelity geometric structure preservation under challenging pose and scale variations.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Virtual Try-on"
  - "Ornament Try-on"
  - "Diffusion Models"
  - "Pose Alignment"
  - "Attention Guidance"
date: 2026-05-08
content_hash: 5417da10f049a0e8
---

# Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2503.16065](https://arxiv.org/abs/2503.16065)  
**Code**: [Project Page](https://shiningyourself.github.io/)  
**Area**: Image Generation  
**Keywords**: Virtual Try-on, Ornament Try-on, Diffusion Models, Pose Alignment, Attention Guidance

## TL;DR

This work introduces diffusion models to the virtual try-on task of ornaments (bracelets, rings, earrings, necklaces) for the first time. It proposes an iterative pose-aware wearing-mask prediction and a mask-guided attention mechanism, achieving high-fidelity geometric structure preservation under challenging pose and scale variations.

## Background & Motivation

- While virtual try-on for apparel has been heavily researched and matured, virtual ornament try-on remains virtually unexplored despite substantial commercial demand.
- Ornament try-on faces three unique challenges: (1) Ornaments possess delicate and fine-grained geometric structures (e.g., ring structures, repeating sub-structures) that are difficult to preserve; (2) Ornaments are typically rigid objects, making any deformation or artifact immediately noticeable (unlike clothing, which can hide distortions behind natural wrinkles); (3) They cannot rely on skeletons and semantic maps like apparel try-on, as the contour masks for ornaments are extremely hard to obtain due to pose occlusion.
- Existing apparel try-on methods (e.g., OOTDiffusion, IDM-VTON) require auxiliary inputs (such as pose maps and semantic maps) and are inapplicable to ornaments.
- Reference images of ornaments are usually close-up shots, exhibiting much larger scale differences from the target model than apparel, thus requiring more precise pose alignment.
- General image editing methods (e.g., AnyDoor, Paint-by-Example) can insert objects but lack pose alignment capabilities.

## Method

### Overall Architecture

The framework is constructed based on a Latent Diffusion Model and ReferenceNet. The input requires only a reference ornament image, a target model image, and a coarse bounding box. ReferenceNet extracts ornament features to inject into the denoising U-Net. The core innovation comprises two modules: (1) Iterative pose-aware mask prediction, which progressively refines a coarse bounding box into an accurate wearing mask; and (2) Mask-guided attention, which utilizes the implicit mapping constraint from the reference mask to the wearing mask to preserve geometric structures.

### Key Designs

**1. Iterative Pose-Aware Mask Refinement**
- **Function**: Progressively estimates the precise wearing mask from a coarse bounding box to align the pose and scale of the ornament with the target model.
- **Mechanism**: A linear layer is integrated into ReferenceNet to predict the wearing mask $\hat{M}_p^t = \text{MLP}([f_m^t \odot \hat{M}_p^{t-1}, f_o^t])$ from intermediate features. The predicted mask is blended with the initial bounding box using a dynamic weight $\alpha_t$ as the input for the next step, enabling iterative refinement. The mask prediction is aligned with the ground truth using an $L_2$ loss.
- **Design Motivation**: Single-step mask prediction is often too coarse as intermediate features in early stages contain limited semantic info. Iterative refinement allows the mask quality to improve progressively during the denoising process. Graduating $\alpha_t$ from small to large ensures reliance on boundary box stability in early stages and prediction accuracy in later stages.

**2. Mask-guided Attention**
- **Function**: Preserves fine-grained geometric structures of the ornament (e.g., repetitive patterns, ring structures, the number of sub-components).
- **Mechanism**: Attention maps $\{M_a^i\}$ are extracted from the U-Net layers. Using the reference ornament mask $M_o^i$ to mask one dimension and summing over the other, the result is upsampled to a mask $\tilde{M}_o$ of the same size. Forcing $\tilde{M}_o$ to be consistent with the ground-truth wearing mask (via an $L_2$ loss) implicitly constrains the attention maps to learn the mapping from the reference mask to the wearing mask.
- **Design Motivation**: Direct masking on attention maps blocks too much information, causing performance degradation. The indirect approach, by constraining the consistency of the output mask of attention maps, forces the attention to automatically learn geometric correspondences. This preserves structures without severely restricting generation flexibility.

**3. Zero-shot Inference Design**
- **Function**: Enables inference without external auxiliary inputs.
- **Mechanism**: During inference, only the reference ornament image, model image, and a bounding box (indicating the wearing position) are required. The mask is automatically estimated by the iterative prediction module, requiring no skeletons, semantic maps, or exact contours.
- **Design Motivation**: The wearing position of ornaments is highly user-dependent (e.g., which finger to wear a ring on), making a bounding box the simplest and most intuitive way of interaction.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_3$$

where $\mathcal{L}_1$ is the diffusion noise prediction loss, $\mathcal{L}_2 = \|\hat{M}_p^T - M_o^{gt}\|_2^2$ is the mask prediction loss, and $\mathcal{L}_3 = \|\tilde{M}_o - M_o^{gt}\|_2^2$ is the mask mapping loss guided by attention. $\lambda_1, \lambda_2$ decay over training.

## Key Experimental Results

### Main Results

| Method | FID↓ | LPIPS↓ | CLIP Score↑ | DINO Score↑ |
|------|------|--------|-------------|-------------|
| Paint-by-Example | 23.49 | 0.0789 | 85.6 | 64.8 |
| AnyDoor | 28.28 | 0.1029 | 85.1 | 67.2 |
| IDM-VTON | 22.99 | 0.0709 | 85.9 | 65.0 |
| **Ours** | **19.00** | **0.0593** | **88.7** | **74.5** |

*Achieves leading performance across all metrics, with FID 17% lower than the best baseline.*

### Ablation Study

| Configuration | FID↓ | DINO Score↑ |
|------|------|-------------|
| W/o Mask Prediction | ~24 | ~66 |
| W/o Mask-guided Attention | ~21 | ~70 |
| **Ours (Full)** | **19.0** | **74.5** |

### Key Findings

- The mask prediction module is crucial for pose alignment, and mask-guided attention is critical for preserving geometric details.
- Existing methods fail to preserve the appearance and structural consistency of ornaments, especially regarding geometric details and the number of components.
- The training dataset consists of approximately 64K image triplets, uniformly distributed across four types of ornaments.

## Highlights & Insights

1. **Opening a New Task**: Systematically defines and solves the virtual ornament try-on problem for the first time, filling an important gap in the virtual try-on field.
2. **Implicit Mask Mapping**: Instead of directly manipulating attention maps, constraining their output mask consistency serves as an elegant solution for preserving fine structures.
3. **Minimal Input Requirement**: Inference requires only a bounding box, which is much simpler than apparel try-on methods that demand skeletons, semantic maps, and DensePose.

## Limitations & Future Work

- Limited handling capacity under extreme occlusions and extreme pose differences.
- The dataset scale (64K) is relatively small, which might limit generalization ability.
- Scenario of simultaneous styling of multiple ornaments remains unexplored.
- Future work can extend this to the video domain for dynamic ornament try-on.

## Related Work & Insights

- Compared to apparel methods like OOTDiffusion and IDM-VTON, this approach requires no extra inputs and is better suited for small-scale detailed objects.
- The concept of iterative mask refinement can be generalized to other image editing tasks that require precise spatial alignment.
- The mask-guided attention mechanism provides a new paradigm for geometry preservation in diffusion models.

## Rating

⭐⭐⭐⭐ — Pioneeringly defines the virtual ornament try-on task. The two core modules (iterative mask prediction and mask-guided attention) are designed ingeniously and complement each other. The experimental results are convincing with high practical application value. However, the scale of the dataset and evaluation is relatively limited, and the robustness under extreme scenarios requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](../../CVPR2026/image_generation/promo_promptable_virtual_tryon_efficient.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[ICCV 2025\] OmniVTON: Training-Free Universal Virtual Try-On](../../ICCV2025/image_generation/omnivton_training-free_universal_virtual_try-on.md)
- [\[CVPR 2025\] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [\[CVPR 2025\] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)

</div>

<!-- RELATED:END -->
