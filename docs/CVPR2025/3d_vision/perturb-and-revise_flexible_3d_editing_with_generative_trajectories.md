---
title: >-
  [Paper Note] Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories
description: >-
  [CVPR 2025][3D Vision][3D Editing] Perturb-and-Revise accomplishes flexible 3D editing by applying adaptive perturbations in the NeRF parameter space to allow parameters to escape local minima, optimizing along the generative trajectory using score distillation from multi-view diffusion models, and integrating identity-preserving gradients. This represents the first method to support large-scale geometric and appearance modifications, including pose changes and the addition o…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Editing"
  - "NeRF Editing"
  - "Score Distillation"
  - "Parameter Perturbation"
  - "Multi-view Consistency"
date: 2026-05-08
content_hash: 6d516acb904076fa
---

# Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories

**Conference**: CVPR 2025  
**arXiv**: [2412.05279](https://arxiv.org/abs/2412.05279)  
**Code**: [https://susunghong.github.io/Perturb-and-Revise](https://susunghong.github.io/Perturb-and-Revise)  
**Area**: 3D Vision  
**Keywords**: 3D Editing, NeRF Editing, Score Distillation, Parameter Perturbation, Multi-view Consistency

## TL;DR

Perturb-and-Revise accomplishes flexible 3D editing by applying adaptive perturbations in the NeRF parameter space to allow parameters to escape local minima, optimizing along the generative trajectory using score distillation from multi-view diffusion models, and integrating identity-preserving gradients. This represents the first method to support large-scale geometric and appearance modifications, including pose changes and the addition of new objects.

## Background & Motivation

**Background**: Text-driven 3D editing has emerged as a major research focus. Existing methods, such as Instruct-NeRF2NeRF and Posterior Distillation, achieve text-guided 3D editing by using diffusion models to guide the update of NeRF parameters.

**Limitations of Prior Work**: While existing methods perform well in color, texture, and style modifications, they struggle with editing tasks that require substantial geometric or appearance changes (such as changing poses, adding new objects, or altering species). This limitation stems from the fact that optimized NeRF parameters reside in a low-energy local minimum, and the gradients from Score Distillation are insufficient to push the parameters out of the current basin of attraction, even when the editing prompt is updated.

**Key Challenge**: NeRF editing faces a dilemma: (1) editing from the source NeRF traps the parameters in a local minimum, preventing large-scale modifications; (2) generating from dummy/random initialization completely loses the identity and connection to the source object.

**Goal**: Enable flexible 3D editing supporting a spectrum of changes from simple color/texture modifications to complex geometric/pose variations while preserving similarity to the source object.

**Key Insight**: View NeRF parameters as "particles" in a generative ODE. The optimized NeRF represents particles that have reached the target distribution and are in a low-energy state. Introducing perturbations in the parameter space is equivalent to rewinding these particles to an intermediate state of the optimization process, allowing them to re-converge along a new generative trajectory specified by the editing prompt.

**Core Idea**: Perturb NeRF parameters via linear interpolation in the parameter space to escape local minima, adaptively determine the perturbation magnitude, guide the editing using multi-view Score Distillation, and utilize identity-preserving gradients to align the edited results with the source object in the final stages.

## Method

### Overall Architecture

Perturb-and-Revise consists of three stages: (1) Parameter Perturbation: linearly interpolating the source NeRF parameters with a random initialization and adaptively determining the interpolation ratio $\eta$; (2) Multi-view Consistent Editing: optimizing along the generative trajectory using multi-view diffusion models (e.g., MVDream) via Score Distillation; (3) Identity-Preserving Finetuning: introducing IPG (Identity-Preserving Gradient) in the later stages to balance editing effectiveness with source object fidelity.

### Key Designs

1. **Parameter Perturbation**:

    - **Function**: Allows NeRF parameters to escape the current local minimum, granting sufficient "flexibility" for large-scale editing.
    - **Mechanism**: Linearly interpolate the source NeRF parameters $\theta_{\text{src}}$ and the randomly initialized parameters $\theta_0$: $\theta_{\text{perturbed}} = (1-\eta) \cdot \theta_{\text{src}} + \eta \cdot \theta_0$. Here, $\eta \in [0,1]$ controls the perturbation magnitude—$\eta=0$ represents no perturbation, and $\eta=1$ represents a complete random initialization. The perturbed parameters reside in an "intermediate state" of the optimization process, enabling re-optimization to converge along the generative trajectory guided by the new prompt.
    - **Design Motivation**: Analogous to adding noise in diffusion models—adding more noise retracts more of the completed generation, thereby allowing larger modifications. Perturbing in the parameter space is a natural extension of this concept to 3D editing.

2. **Adaptive $\eta$ Selection (Loss Landscape Analysis)**:

    - **Function**: Automatically determine the appropriate perturbation magnitude, avoiding manual grid searching.
    - **Mechanism**: Utilize the loss function as a proxy to measure the depth of the basin of attraction. Specifically, simulate several Score Distillation steps using the editing prompt, then compute the difference between the average loss of the initial steps and the later steps. If the loss barely decreases or even increases, it indicates that the parameters are trapped in a deep local minimum, requiring a larger $\eta$. An inverse exponential decay function is used to map the loss difference to the $\eta$ value.
    - **Design Motivation**: Different types of editing demand different levels of perturbation—color modifications require a very small $\eta$, whereas pose changes necessitate a larger $\eta$. Analyzing the loss landscape can automatically adapt the parameters, eliminating the need for expensive grid searches.

3. **Identity-Preserving Gradient (IPG)**:

    - **Function**: Align the edited results with the source object in the later stages of editing, balancing editing quality and source object fidelity.
    - **Mechanism**: Introduce an auxiliary gradient term in the later optimization phase to keep the rendered images of the edited NeRF similar to those of the source NeRF: $d\theta_\tau^{\text{refine}} = d\theta_\tau + \lambda_d \nabla_\theta d(\theta_\tau, \theta_{\text{src}})$, where $d(\cdot, \cdot)$ is a combination of L1 and perceptual losses. This forms a "tug-of-war" between two gradient forces—Score Distillation pushes toward the editing goal, while IPG pulls back toward the source object.
    - **Design Motivation**: The parameter perturbation and Score Distillation phases might introduce estimation errors or biases from the diffusion model. IPG corrects these biases in the later stage. Simultaneously, applying preservation constraints from the beginning would conflict with the generative ODE, so it is introduced only in the late stages.

### Loss & Training

A multi-view diffusion model (MVDream) is utilized to generate consistent predictions across N different viewpoints simultaneously for Score Distillation updates. The noise level employs a timestep annealing strategy (from high to low), performing low-frequency edits first followed by fine-grained modifications. IPG utilizes a combination of L1 and LPIPS perceptual losses.

## Key Experimental Results

### Main Results

| Method | CLIP-Dir-Sim ↑ | LPIPS_vgg ↓ | LPIPS_alex ↓ | Description |
|------|---------------|------------|-------------|------|
| SDS (MVDream) | 0.0438 | 0.1273 | 0.1533 | Blurry, unable to make large changes |
| PDS | 0.0285 | **0.0337** | **0.0215** | Under-edited, overly conservative |
| IN2N | 0.0557 | 0.1065 | 0.1112 | Only changes texture, unable to modify geometry |
| **PnR (Ours)** | **0.0565** | 0.1060 | 0.1034 | **Best balance between editing and fidelity** |

### Ablation Study

| Configuration | CLIP-Dir-Sim ↑ | CLIP-Dir-Con ↑ | LPIPS ↓ | Description |
|------|---------------|---------------|---------|------|
| Without IPG Refinement | 0.0624 | 0.7572 | 0.1147 | More aggressive editing but deviates from the source |
| With IPG Refinement | 0.0565 | 0.7642 | 0.1047 | Significant reduction in LPIPS |

### Key Findings

- The effect of parameter perturbation is remarkably significant: when $\eta=0$, pose changes and new object addition are unattainable; as $\eta$ increases appropriately, progressively larger modifications can be achieved.
- Adaptive $\eta$ selection performs close to or on par with the optimal fixed $\eta$ across the average of all editing types, while ensuring a high experimental success rate and keeping computational overhead much lower than grid search.
- Different editing types require different $\eta$: color/texture edits require a small $\eta$, while pose/object addition requires a large $\eta$.
- IPG delivers a significant improvement in LPIPS (from 0.1147 to 0.1047), while the CLIP direction consistency slightly increases, indicating that IPG effectively corrects biases.
- Though PDS achieves the lowest LPIPS, it barely performs any effective edits due to its overly conservative nature (having the lowest CLIP-Dir-Sim).

## Highlights & Insights

- The concept of **parameter-space perturbation** is highly elegant. By viewing NeRF parameters as endpoints of particle flows and "rewinding" the optimization process via interpolation with an initial distribution, it allows the parameters to evolve along a new trajectory. This migration of diffusion-aligned adding/removing noise concepts into the parameter space is highly inspiring.
- **Automatic $\eta$ selection via loss landscape analysis** is a practical algorithmic innovation that avoids costly hyperparameter search. The idea of "simulating a few steps to observe trends" can be extended to other optimization scenarios demanding adaptive control.
- The **staged strategy** (aggressive perturbation for coarse editing first $\rightarrow$ IPG refinement for fidelity later) is intuitive and experimentally validated as effective.

## Limitations & Future Work

- Optimization after perturbation still requires a considerable number of iterative steps, making the editing efficiency potentially inferior to single-step inference-based methods.
- For editing tasks requiring precise spatial control (e.g., relocating an object to a specific position), text prompts may lack sufficient control ability.
- Currently, the method is primarily validated on object-level editing, showing limited support for local editing in complex scenes (e.g., editing only one object in a scene).
- The 3D consistency of multi-view diffusion models themselves is still imperfect, which may introduce artifacts like the Janus problem.

## Related Work & Insights

- **vs Instruct-NeRF2NeRF**: IN2N edits by iteratively updating the training views with InstructPix2Pix, excelling at color/texture modifications but failing entirely at geometric changes. PnR fundamentally resolves this limitation through parameter perturbation.
- **vs PDS (Posterior Distillation)**: PDS maintains similarity to the source by matching random latents but is overly conservative, yielding barely any effective edits. PnR's "bold perturbation first, fine-grained correction later" strategy is more flexible.
- **vs SDS**: SDS optimizes directly from the source NeRF; although it can change textures, it struggles to break through local minima and is prone to generating blurry textures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of parameter-space perturbation is novel and elegant, successfully migrating diffusion concepts to NeRF editing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple editing types, multiple baselines, quantitative + qualitative + ablation + user study.
- Writing Quality: ⭐⭐⭐⭐⭐ Demystified with clear motivation illustrations and profound theoretical insights.
- Value: ⭐⭐⭐⭐⭐ Achieves 3D editing with major geometric changes for the first time, filling an important gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variation-Aware Flexible 3D Gaussian Editing](../../ICLR2026/3d_vision/variation-aware_flexible_3d_gaussian_editing.md)
- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)
- [\[CVPR 2025\] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)
- [\[ICCV 2025\] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior](../../ICCV2025/3d_vision/driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)
- [\[ICCV 2025\] 3D Mesh Editing using Masked LRMs](../../ICCV2025/3d_vision/3d_mesh_editing_using_masked_lrms.md)

</div>

<!-- RELATED:END -->
