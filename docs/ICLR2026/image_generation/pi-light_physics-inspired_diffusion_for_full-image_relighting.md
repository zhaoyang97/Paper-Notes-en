---
title: >-
  [Paper Note] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting
description: >-
  [ICLR 2026][Image Generation][Diffusion Models] This paper proposes π-Light (PI-Light), a two-stage full-image relighting framework. Stage 1 performs intrinsic property decomposition (albedo, normals, roughness, etc.) via a physics-guided diffusion model; Stage 2 synthesizes the relit image under target illumination via a physics-guided neural rendering module. Batch-aware attention and physics-inspired losses are introduced to achieve strong generalization to real-world scenes.
tags:
  - ICLR 2026
  - Image Generation
  - Diffusion Models
  - Image Relighting
  - Inverse Rendering
  - Physics-Guided
  - Intrinsic Decomposition
date: 2026-05-08
content_hash: 46f2fb0bd928da89
---

# PI-Light: Physics-Inspired Diffusion for Full-Image Relighting

**Conference**: ICLR 2026
**arXiv**: [2601.22135](https://arxiv.org/abs/2601.22135)
**Code**: None
**Area**: Image Generation / Image Relighting
**Keywords**: Diffusion Models, Image Relighting, Inverse Rendering, Physics-Guided, Intrinsic Decomposition

## TL;DR

This paper proposes π-Light (PI-Light), a two-stage full-image relighting framework. Stage 1 performs intrinsic property decomposition (albedo, normals, roughness, etc.) via a physics-guided diffusion model; Stage 2 synthesizes the relit image under target illumination via a physics-guided neural rendering module. Batch-aware attention and physics-inspired losses are introduced to achieve strong generalization to real-world scenes.

## Background & Motivation

Full-Image Relighting is a long-standing challenge in computer vision and graphics, aiming to alter the illumination of a scene while preserving its content. The task faces three core difficulties:

**Scarcity of large-scale paired data**: High-quality paired images of the same scene under different lighting conditions are extremely difficult to collect, severely limiting data-driven approaches.

**Difficulty in ensuring physical plausibility**: End-to-end learning tends to produce physically implausible lighting effects (e.g., incorrect shadow directions, unnatural specular reflections).

**Synthetic-to-real domain gap**: Models trained on rendered data often fail to generalize to real-world scenes, and existing attempts to bridge this gap remain unsatisfactory.

Existing methods fall broadly into two categories: (1) direct end-to-end learning of image-level black-box transformations, lacking physical constraints; (2) relighting via explicit 3D geometry reconstruction followed by re-rendering, which is computationally expensive and sensitive to reconstruction quality. The key contribution of this paper lies in embedding physical constraints into the diffusion model, achieving physically plausible relighting without explicit 3D reconstruction.

## Method

### Overall Architecture

PI-Light adopts a two-stage design, decomposing full-image relighting into two physically well-defined sub-problems: **Inverse Rendering** and **Forward Rendering**:

- **Stage 1 — Physics-Guided Intrinsic Decomposition**: Given an input image, a fine-tuned diffusion model predicts its intrinsic properties, including albedo, normals, roughness, and others.
- **Stage 2 — Physics-Guided Neural Forward Rendering**: Given the decomposed intrinsic properties and target illumination conditions, a physics-guided neural rendering module synthesizes the image under the target lighting.

### Key Designs

1. **Batch-Aware Attention**:

    - *Function*: During diffusion model fine-tuning, multiple images of the same scene/object under different illumination conditions share attention computation.
    - *Mechanism*: Intrinsic properties (albedo, normals, etc.) of the same object should remain identical across different lighting conditions; cross-image attention is therefore exploited to enforce prediction consistency.
    - *Design Motivation*: Single-image intrinsic decomposition is highly ill-posed; sharing information within a batch effectively reduces ambiguity. This is analogous to multi-view consistency, applied here in a multi-illumination setting.

2. **Physics-Guided Neural Rendering Module**:

    - *Function*: Receives intrinsic properties and a target environment lighting map to synthesize the relit image.
    - *Mechanism*: The module design follows the physical rendering equation, handling diffuse and specular reflections separately.
    - *Design Motivation*: Enforces adherence to physical light transport laws, preventing physically implausible lighting effects. Unlike purely neural rendering, the imposed physical structure enables the model to correctly generate specular highlights and diffuse reflections.

3. **Physics-Inspired Losses**:

    - *Function*: Impose physics-based constraints during training.
    - *Mechanism*: In addition to minimizing pixel-level reconstruction loss, physics constraints regularize the training dynamics, guiding optimization toward physically meaningful solutions.
    - *Design Motivation*: Prevents the diffusion model from forgetting physical priors during fine-tuning and enhances generalization from synthetic data to real-world scenes.

4. **Curated Dataset**:

    - Collects diverse objects and scenes under controlled illumination conditions.
    - Covers a wide range of materials (metal, plastic, fabric, glass, etc.) to encompass various light–material interactions.
    - Serves as a standardized benchmark for relighting research.

### Loss & Training

**Stage 1 Training**: Efficient fine-tuning on top of a pre-trained diffusion model.
- Reconstruction loss for intrinsic decomposition (L1/L2 over each intrinsic property channel).
- Cross-illumination consistency loss (implicitly enforced via batch-aware attention).
- Possible adversarial loss to ensure generation quality.

**Stage 2 Training**:
- Pixel-level reconstruction loss: L1/L2 under target illumination conditions.
- Perceptual Loss: quality measured in VGG feature space.
- Physics constraint loss: regularization terms based on the rendering equation, ensuring physical plausibility of diffuse and specular reflections.
- Auxiliary losses such as SSIM/LPIPS.

## Key Experimental Results

### Main Results

| Method | Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Highlights |
|--------|---------|--------|--------|---------|------------|
| Prev. SOTA | Synthetic test set | — | — | — | Poor physical plausibility |
| PI-Light | Synthetic test set | **Best** | **Best** | **Best** | Surpasses all baselines |
| Prev. SOTA | Real-world scenes | Poor generalization | — | — | Significant domain gap |
| PI-Light | Real-world scenes | **Best generalization** | — | — | Maintains physical plausibility |

**Intrinsic Decomposition Quality**:
- Albedo prediction: excellent consistency across different illumination conditions.
- Normal prediction: high agreement with ground truth.
- Material properties: correctly distinguishes metallic vs. non-metallic materials.

**Relighting Quality**:
- Correctly generates specular highlights.
- Correctly handles diffuse reflections.
- Performs well across diverse materials (metal, plastic, fabric, etc.).
- Significantly better real-world generalization compared to prior methods.

### Ablation Study

| Component | Effect of Removal | Notes |
|-----------|------------------|-------|
| Batch-Aware Attention | Degraded intrinsic decomposition consistency | Albedo predictions become inconsistent across illuminations |
| Physics-Guided Rendering Module | Reduced physical plausibility | Increased errors in highlight direction and intensity |
| Physics-Inspired Losses | Degraded generalization | Performance deteriorates on real-world scenes |
| Curated Dataset | Insufficient material coverage | Performance degrades on certain materials (e.g., translucent) |

### Key Findings

1. **Physics guidance is critical for generalization**: With physical constraints, models trained on synthetic data generalize well to real-world scenes.
2. **Batch-aware attention substantially improves consistency**: Predictions of intrinsic properties for the same object under different illuminations become significantly more consistent, which is crucial for downstream rendering quality.
3. **Two-stage design outperforms end-to-end approaches**: Decomposing the problem into physically well-defined stages of inverse and forward rendering is more controllable than direct end-to-end mapping.
4. **Synergy between diffusion priors and physical knowledge**: The combination of rich visual priors from pre-trained diffusion models and physical constraints underlies the method's success.

## Highlights & Insights

1. **Elegant integration of physical priors and generative models**: Rather than simply appending physics losses to a diffusion model, physical priors are embedded at three levels simultaneously: model architecture (batch-aware attention), training objective (physics-inspired losses), and inference pipeline (two-stage physical decomposition).
2. **Innovative batch-aware attention design**: Leveraging the physical invariance of intrinsic properties across different illuminations of the same scene, constraints are imposed at the attention mechanism level—a compelling example of injecting domain knowledge into Transformer architectures.
3. **Strong practical applicability**: The method has direct utility in film visual effects, virtual reality, and augmented reality, enabling high-quality relighting from a single input image.
4. **Dataset contribution**: The curated controlled-illumination dataset provides a standardized benchmark for the community.

## Limitations & Future Work

1. **Error accumulation in the two-stage pipeline**: Errors from the intrinsic decomposition stage propagate to the rendering stage; end-to-end joint optimization may further improve performance.
2. **Limitations of the lighting representation**: Environment maps may be insufficient to represent complex near-field lighting, area lights, or multi-source illumination scenarios.
3. **Computational cost**: Two-stage diffusion-based inference may be slower than single-stage methods, limiting real-time applications.
4. **Outdoor scene generalization**: The curated dataset primarily covers indoor/object-level scenes; generalization to complex outdoor environments remains to be validated.
5. **Editing flexibility**: The fixed two-stage pipeline may not readily support more flexible editing requirements (e.g., local lighting adjustment, lighting interpolation).
6. **Resolution constraints**: Bounded by the generation resolution of diffusion models, high-resolution images may require additional super-resolution post-processing.

## Related Work & Insights

- **Intrinsic Image Decomposition**: The evolution from Retinex theory to deep learning approaches; PI-Light upgrades this paradigm with diffusion model-driven decomposition.
- **NeRF-based Relighting**: Methods such as NeRFactor and NVDiffrec achieve relighting via 3D reconstruction; PI-Light avoids explicit 3D reconstruction.
- **Diffusion for Inverse Problems**: DDPM/DDIM applied to denoising, super-resolution, and other inverse problems; PI-Light extends this to relighting.
- **Multi-view consistency attention** in works such as Zero-1-to-3 and Wonder3D inspired the concept of batch-aware attention.

## Rating

- Novelty: ⭐⭐⭐⭐ — The systematic embedding of physics guidance into a diffusion-based relighting framework is a novel architectural design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on both synthetic and real-world scenes with complete ablations; dataset contribution adds value.
- Writing Quality: ⭐⭐⭐⭐ — The two-stage framework is described clearly, and the physical motivation is well articulated.
- Value: ⭐⭐⭐⭐ — A practically strong relighting solution; the physics-guided diffusion model paradigm has broad transferability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](../../NeurIPS2025/image_generation/unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[CVPR 2026\] Learning Latent Proxies for Controllable Single-Image Relighting](../../CVPR2026/image_generation/learning_latent_proxies_for_controllable_single-image_relighting.md)
- [\[CVPR 2026\] Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](../../CVPR2026/image_generation/physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)

<!-- RELATED:END -->
