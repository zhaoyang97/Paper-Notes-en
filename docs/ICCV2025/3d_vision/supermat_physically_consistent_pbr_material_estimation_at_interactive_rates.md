---
title: >-
  [Paper Note] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates
description: >-
  [3D Vision] This paper proposes SuperMat, a single-step inference framework for PBR material decomposition. Through structured expert branches and scheduler correction…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: f728435e1065ec03
---

# SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2411.17515](https://arxiv.org/abs/2411.17515)
- **Code**: [Project Page](https://hyj542682306.github.io/SuperMat/)
- **Area**: 3D Vision
- **Keywords**: PBR material decomposition, single-step inference, end-to-end training, re-render loss, 3D material estimation

## TL;DR

This paper proposes SuperMat, a single-step inference framework for PBR material decomposition. Through structured expert branches and scheduler correction, it enables end-to-end training and introduces a re-render loss to enforce physical consistency, accelerating inference from seconds to milliseconds.

## Background & Motivation

Decomposing PBR materials (albedo, metallic, roughness) from images is a core challenge in 3D asset creation. Existing methods suffer from three major bottlenecks:

**Model redundancy**: Each material property requires a separate diffusion model, doubling training and inference costs.

**Slow inference**: DDIM requires 30–50 denoising steps, making it unsuitable for interactive applications.

**Insufficient decomposition**: Noise-prediction-based training objectives cannot directly supervise final material outputs, hindering the application of advanced techniques such as perceptual loss and re-render loss.

## Method

### Overall Architecture

SuperMat is built upon fine-tuned Stable Diffusion and incorporates three core designs:

### 1. Structured Expert Branches

The last UpBlock of the UNet is duplicated into two expert branches:
- **Albedo Branch**: dedicated to predicting the diffuse albedo map.
- **RM Branch**: dedicated to predicting the joint roughness-metallic map.

Shared modules extract general features, while expert branches capture material-specific representations. The added parameters account for only 2.23% of total UNet parameters (19.3M), enabling multi-material output from a single model.

### 2. Single-Step Inference and End-to-End Training

**Scheduler correction**: The default leading timestep setting in DDIM is found to be flawed — during single-step prediction, the timestep presented to the model ($t=1$, implying nearly noise-free input) is inconsistent with the actual input (pure noise). Correcting this to a trailing setting ($t=T$) enables genuine single-step inference.

**End-to-end training**: Single-step inference makes backpropagation tractable, allowing losses to be computed directly on the final predicted materials:

$$\mathcal{L}_m = \mathcal{L}_p(\hat{k}_d, k_d) + \mathcal{L}_p(\hat{k}_{rm}, k_{rm})$$

where $\mathcal{L}_p$ denotes a VGG-16-based perceptual loss.

### 3. Re-render Loss

The predicted materials are used to render images under novel lighting conditions, which are then compared against ground-truth renderings:

$$\mathcal{L}_{re} = \mathcal{L}_p(\mathcal{R}_{\hat{k}_n, \hat{k}_p, \hat{c}}(\hat{k}_d, \hat{k}_m, \hat{k}_r), \mathcal{R}_{\hat{k}_n, \hat{k}_p, \hat{c}}(k_d, k_m, k_r))$$

This enforces physical consistency across different material attributes — even when individual material maps appear close to the ground truth, the composite rendering result may still be incorrect without such a constraint.

### Multi-view Extension (SuperMatMV)

Built upon the MVDream architecture, 3D self-attention and camera extrinsic conditioning are incorporated to enable simultaneous decomposition across 6 views, ensuring cross-view consistency.

### UV Refinement Network (3D Extension)

Multi-view decomposition results from SuperMatMV are back-projected into UV space, where a UV refinement network completes uncovered regions and improves overall quality. The complete 3D pipeline requires approximately 3 seconds.

## Experiments

### Main Results: Image-Space Material Decomposition

| Method | Albedo PSNR↑ | Metallic PSNR↑ | Roughness PSNR↑ | Relighting PSNR↑ | Time (s)↓ |
|------|------|------|------|------|------|
| IIR | 21.94 | 17.95 | 19.73 | 20.98 | 0.04 |
| RGB→X | 22.30 | 15.36 | 20.40 | 21.51 | 3.32 |
| StableMaterial | 23.44 | 20.29 | 21.01 | 22.56 | 0.53 |
| SuperMat w/o e2e | 24.26 | 20.79 | 20.81 | 23.90 | 3.09 |
| SuperMat w/o re-render | 26.70 | 24.54 | 23.52 | 26.41 | 0.07 |
| **SuperMat** | **27.68** | **25.48** | **24.25** | **27.66** | **0.07** |
| SuperMatMV | 27.56 | 26.11 | 24.84 | 27.64 | 0.09 |

### Ablation Study

| Configuration | Albedo PSNR | Relighting PSNR | Time |
|------|------|------|------|
| w/o scheduler correction (multi-step inference) | 24.26 | 23.90 | 3.09s |
| w/ scheduler correction, w/o re-render | 26.70 | 26.41 | 0.07s |
| Full SuperMat | **27.68** | **27.66** | 0.07s |

### Key Findings
1. **Scheduler correction is the central contribution**: It accelerates inference by approximately 40× (3.09s → 0.07s) while improving PSNR by 2–4 dB.
2. **Re-render loss contributes substantially**: Relighting PSNR improves by 1.25 dB, validating the importance of cross-attribute interaction supervision.
3. **Single model vs. multiple models**: The structured expert branches achieve dual-model equivalent functionality with only a 2.23% increase in parameters.

## Highlights & Insights

1. **Identification and correction of a DDIM scheduler flaw**: This simple fix unlocks the substantial potential of single-step diffusion models and has broad implications.
2. **Re-render loss enables cross-attribute physical constraints**: This is the first introduction of rendering consistency constraints in diffusion-based material decomposition.
3. **Millisecond-level inference**: SuperMat advances material decomposition from academic research toward practical deployment.

## Limitations & Future Work

- Performance depends on the diversity of lighting conditions and material coverage in the training data.
- The generalization of the UV refinement network to unseen complex geometries remains to be validated.
- Single-step inference may sacrifice fine details under extreme lighting conditions.

## Related Work & Insights

- **Diffusion-based material decomposition**: RGB→X, IntrinsicAnything, StableMaterial
- **Traditional methods**: Derender3D, IIR
- **Single-step diffusion**: DMD, InstaFlow

## Rating

- Novelty: ⭐⭐⭐⭐ (combined innovation of scheduler correction and re-render loss)
- Technical Depth: ⭐⭐⭐⭐ (systematically addresses three major bottlenecks)
- Experimental Quality: ⭐⭐⭐⭐⭐ (comprehensive ablation with substantial SOTA improvements)
- Practical Value: ⭐⭐⭐⭐⭐ (millisecond inference, highly applicable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

</div>

<!-- RELATED:END -->
