---
title: >-
  [Paper Note] CObL: Toward Zero-Shot Ordinal Layering without User Prompting
description: >-
  [ICCV 2025][Self-Supervised Learning][object layers] This paper presents CObL, an architecture based on multiple frozen Stable Diffusion UNets operating in parallel…
tags:
  - "ICCV 2025"
  - "Self-Supervised Learning"
  - "object layers"
  - "amodal completion"
  - "diffusion model"
  - "zero-shot generalization"
  - "occlusion ordering"
  - "scene decomposition"
date: 2026-05-08
content_hash: 214a7b75938794ff
---

# CObL: Toward Zero-Shot Ordinal Layering without User Prompting

**Conference**: ICCV 2025
**arXiv**: [2508.08498](https://arxiv.org/abs/2508.08498)  
**Code**: [Project Page](https://vision.seas.harvard.edu/cobl/)  
**Area**: Self-Supervised Learning / Scene Decomposition / Diffusion Models / Perceptual Organization
**Keywords**: object layers, amodal completion, diffusion model, zero-shot generalization, occlusion ordering, scene decomposition

## TL;DR

This paper presents CObL, an architecture based on multiple frozen Stable Diffusion UNets operating in parallel, capable of inferring an occlusion-ordered object layer representation (one amodally-completed object per layer) from a single image without any user prompts or prior knowledge of object count. Trained on only a few thousand synthetic tabletop scenes, CObL generalizes zero-shot to real-world photographs.

## Background & Motivation

**Background**: Perceptual organization is a foundational capability in computer vision — humans naturally decompose images into multiple objects, judge occlusion relationships, and even infer the shape and color of occluded regions (amodal completion). Enabling machines to do this, especially in zero-shot settings, remains highly challenging.

**Limitations of Prior Work**:
   - **Amodal completion methods** (e.g., pix2gestalt): handle only one object at a time and require manually provided masks to specify the target object
   - **Inpainting methods** (e.g., LaMa, SDXL Inpainting): similarly require manual occlusion masks
   - **Object-centric representation learning** (e.g., MONet, IODINE): can decompose scenes in an unsupervised manner but are confined to closed-world settings seen during training (e.g., CLEVR) and cannot generalize to novel objects
   - **Lack of suitable training data**: existing datasets either provide only amodal boundaries (without appearance), only occlusion ordering (without amodal completion), or depict insufficiently realistic scenes

**Key Challenge**: No prior method simultaneously achieves all four of the following: (1) no user prompts, (2) automatic processing of all objects, (3) inference of occlusion order and amodal completion, and (4) zero-shot generalization to novel objects.

**Goal**:
   - Design a model that automatically infers a complete object layer representation from a single image
   - Require no knowledge of object count and no manual prompts of any kind
   - Generalize zero-shot to real-world scenes when trained exclusively on synthetic data

**Key Insight**: Leverage Stable Diffusion as a strong prior over natural images. A single SD instance may struggle with basic perceptual organization tasks (e.g., recognizing disconnected visible parts of an occluded object), but multiple SD instances communicating via cross-layer attention can address this more effectively.

**Core Idea**: Deploy $N$ frozen Stable Diffusion UNets to jointly denoise $N$ object layers in parallel, synchronize generation across layers via learnable cross-layer attention, and enforce physical consistency during inference through compositional guidance ensuring the layers composite back to the original image.

## Method

### Overall Architecture

CObL comprises three core components:
1. **Synthetic data generation pipeline**: Creates synthetic tabletop scenes with corresponding object layer annotations via 3D modeling combined with ControlNet text-to-image generation
2. **Model architecture**: $N$ frozen SD UNet copies connected via cross-layer attention, with a conditioning adapter injecting input image information
3. **Guided sampling**: During inference, a compositional loss and a prior score matching (PSM) loss guide the diffusion denoising process

The input is a multi-object image; the output is $N$ RGBA object layers arranged in occlusion order, each containing one amodally-completed object, whose composition reconstructs the original image.

### Key Designs

1. **Synthetic Data Generation Pipeline**:

    - **Function**: Generate synthetic tabletop scene images with corresponding ground-truth object layers for training
    - **Mechanism**: A two-stage "partial rendering + text-to-image generation" approach. In the first stage, 3D assets are arranged in Blender to render depth maps, masks, and shadow maps (without texture rendering). In the second stage, ControlNet-depth converts the depth map and text prompts into naturally textured images, which are then composited using the masks.
    - **Design Motivation**: Compared to fully synthetic rendering (e.g., CLEVR), this approach produces images within the natural image distribution of SD, effectively bridging the sim-to-real gap. A single geometry can yield multiple textures, providing efficient data augmentation. In total, only 600 3D assets are used to generate 2,250 training scenes.

2. **Parallel Multi-Layer Diffusion Architecture**:

    - **Function**: Simultaneously generate all object layers rather than processing them sequentially
    - **Mechanism**: $N$ frozen SD 2.1 UNet copies (only cross-layer attention weights $\phi$ and conditioning adapter $\psi$ are trained). Cross-layer attention enables each UNet to communicate with others via learnable lateral attention, inspired by inter-frame attention in video generation. The conditioning adapter $c_\psi(I)$ injects input image features alongside a pseudo-depth map generated by frozen MiDaS.
    - **Training Objective**: Standard diffusion denoising loss $\mathcal{L}(\phi,\psi) = \mathbb{E}_{t,\epsilon,(I,z_0)} \|\epsilon - \epsilon_\phi(z_0, t, c_\psi(I))\|^2$

3. **Inference-Time Guided Sampling**:

    - **Function**: Apply additional constraints to guide the denoising process and ensure physical consistency of the output layers
    - **Mechanism**: Two complementary guidance losses:
        - **Compositional Loss** $\mathcal{L}_c$: Composites the generated layers in occlusion order using $\alpha$-blending and requires the composite to match the input image $\|I - \bar{x}^N\|^2$. Masks are estimated automatically via a frozen foreground segmentation model.
        - **Prior Score Matching Loss (PSM Loss)** $\mathcal{L}_{psm}$: Encourages the latent of each layer to remain within the natural image distribution of the original SD $\|\hat{\epsilon}_t - \epsilon_\phi(\hat{z}_{0;t}, t, c_\psi(I))\|^2$, preventing unnatural generation artifacts.
    - **Additional non-differentiable operations**: Periodic layer permutation optimization, empty-layer pruning, and re-ordering.

### Loss & Training

- **Training**: Standard diffusion denoising loss with 10% conditioning embedding dropout (for classifier-free guidance)
- **Inference**: DDIM with 30 steps, guidance parameters $w = 10^4$, $\lambda = 10^{-7}$
- Trained on a single H100 GPU for approximately one day with batch size 2

## Key Experimental Results

### Main Results

Evaluated on the authors' TABLETOP real-world dataset (100 tabletop photographs, 2–6 objects), compared against baselines that require additional manually provided masks:

| Model | LPIPS ↓ (Best/Avg) | CLIP ↑ (Best/Avg) | Requires Extra Masks? |
|------|---------|---------|------|
| LaMa (Inpainting-GAN) | .113 / — | .914 / — | ✓ (oracle) |
| SDXL Inpainting | .373 / .384 | .760 / .758 | ✓ (oracle) |
| pix2gestalt (Completion) | .128 / .153 | .889 / .872 | ✓ (oracle) |
| **CObL (Ours)** | **.094 / .122** | **.935 / .914** | ✗ |

CObL comprehensively outperforms all baselines that rely on oracle masks, while requiring **no additional masks whatsoever**.

Visible segmentation quality (TABLETOP subset):
- CObL ARI = **83.5%**
- Mask2Former (without fine-tuning) ARI = 66.3%

### Ablation Study

| Configuration | LPIPS ↓ | Notes |
|------|---------|------|
| Full CObL | .094 | Complete model |
| w/o depth cues (MiDaS) | +4% | Removing depth conditioning causes a slight performance drop |
| w/o frozen SD prior (UNet unfrozen) | +9% | Unfreezing the UNet leads to overfitting and severely degraded generalization |
| w/o guidance (PSM + compositional) | +8% | Without guidance, layers fail to correctly composite back to the original image |

### Key Findings

- **Freezing SD is critical**: Unfreezing the UNet may improve fitting on synthetic data but fails to generalize to real-world scenes. The frozen natural image prior is the key to zero-shot generalization.
- **Compositional guidance ensures quality**: Inference-time compositional guidance enables CObL to better preserve fine details in visible regions (e.g., patterns on a teapot, text on book covers), whereas competing methods tend to hallucinate.
- **Performance degrades noticeably beyond 4 objects**: This is the primary limitation of the current architecture; a greater number of objects increases the difficulty of layer separation.
- **Depth prior is helpful but not essential**: SD already encodes a strong implicit depth prior.

## Highlights & Insights

- **Prompt-free multi-object decomposition**: CObL is the first method to simultaneously perform amodal completion and occlusion ordering for all objects without any user-provided masks or prompts — a property of significant practical value for interactive applications.
- **Elegant sim-to-real bridging**: The data pipeline is cleverly designed — 3D models supply precise geometry, ControlNet generates natural textures, and textures are never baked into the render. Generating multiple textures from the same geometry serves as an effective data augmentation strategy.
- **Collaborative multi-SD-instance generation**: Inspired by inter-frame attention in video generation, though object layers are far more heterogeneous and loosely related than consecutive video frames. CFG and PSM provide two complementary guidance signals to regulate the collaboration.
- **Inference-time compositional constraint**: The compositional loss directly enforces the constraint that "composited layers = original image," providing a physically grounded consistency guarantee that is central to the method's reliability.

## Limitations & Future Work

- **Restricted to tabletop scenes**: Training and evaluation are currently limited to top-down tabletop viewpoints; applicability to more complex natural scenes (e.g., indoor or outdoor multi-layer occlusion) has not been validated.
- **Object count constraint**: Storing $N=7$ SD UNet copies incurs substantial memory overhead, and performance degrades with more than 4 objects. Gradients for all 7 UNets must be retained during guided sampling.
- **Slow inference**: 30-step DDIM combined with guidance computation is considerably slower than guidance-free methods.
- **Inherited SD biases**: The model inherits generative biases from Stable Diffusion, potentially leading to systematically poor completion for certain object categories.
- **Directions for improvement**: Replacing SD with a lightweight diffusion model could reduce computational cost; an iterative, adaptive layer-count strategy could be explored as an alternative to a fixed $N$.

## Related Work & Insights

- **vs. pix2gestalt**: pix2gestalt uses SD for amodal completion but requires a manually provided visible mask for the target object and processes one object at a time. CObL requires no prompts and processes all objects in parallel.
- **vs. Slot Attention / SLATE and related object-centric methods**: These methods are effective in closed-world settings such as CLEVR, but produce unordered slots rather than ordered layers with clear boundaries, do not perform amodal completion, and fundamentally cannot generalize to objects outside their training distribution.
- **vs. inter-frame attention in video diffusion models**: CObL borrows this architectural idea, but object layers are far more heterogeneous and loosely coupled than temporally adjacent video frames.
- The proposed framework has direct downstream applicability to scene editing in AR/VR and robotic grasp planning (which requires understanding object stacking relationships).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Novel problem formulation; the multi-SD-instance collaboration with inference-time guidance is an elegant and original design
- Experimental Thoroughness: ⭐⭐⭐⭐ Introduces a new dataset, conducts multi-faceted evaluation with complete ablations, though scenarios are limited to tabletop settings
- Writing Quality: ⭐⭐⭐⭐ Well-organized paper with clear and visually intuitive figures
- Value: ⭐⭐⭐⭐ Opens a new direction for zero-shot perceptual organization; broader scene generalization remains to be demonstrated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](../../CVPR2026/self_supervised/las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[AAAI 2026\] From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models without Finetuning](../../AAAI2026/self_supervised/from_pretrain_to_pain_adversarial_vulnerability_of_video_foundation_models_witho.md)
- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](../../CVPR2026/self_supervised/zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)
- [\[ICCV 2025\] From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations](from_linearity_to_non-linearity_how_masked_autoencoders_capture_spatial_correlat.md)
- [\[ICCV 2025\] Manual-PA: Learning 3D Part Assembly from Instruction Diagrams](manual-pa_learning_3d_part_assembly_from_instruction_diagrams.md)

</div>

<!-- RELATED:END -->
