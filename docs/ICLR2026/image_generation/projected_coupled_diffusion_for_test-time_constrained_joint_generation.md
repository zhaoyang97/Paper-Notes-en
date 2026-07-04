---
title: >-
  [Paper Note] Projected Coupled Diffusion for Test-Time Constrained Joint Generation
description: >-
  [ICLR2026][Image Generation][Test-time Constrained Generation] **Background**: Diffusion models have become universal modeling tools for generation tasks involving images, videos, language, graphs, and robot trajectories. Many practical systems require more than just "unconditional sample generation" but involve incorporating additional objectives during inference—such as classifier guidance, inpainting, reward guidance, or projected diffusion—to guide existing models toward…
tags:
  - "ICLR2026"
  - "Image Generation"
  - "Test-time Constrained Generation"
  - "Joint Generation"
  - "Projected Diffusion"
  - "Coupled Diffusion"
  - "Multi-model Collaboration"
date: 2026-05-08
content_hash: 5daf669d0a2fb631
---

# Projected Coupled Diffusion for Test-Time Constrained Joint Generation

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=1FEm5JLpvg](https://openreview.net/forum?id=1FEm5JLpvg)
**Code**: https://github.com/EdmundLuan/pcd  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Test-time Constrained Generation, Joint Generation, Projected Diffusion, Coupled Diffusion, Multi-model Collaboration
**Code**: TBD  
**Area**: generative models  
This paper proposes Projected Coupled Diffusion (PCD), which enables joint sampling of multiple pre-trained marginal diffusion models via a coupling cost without retraining. It performs projection at each diffusion step to strictly satisfy test-time hard constraints, simultaneously improving correlation and constraint satisfaction across robot trajectories, face pair generation, and object manipulation tasks.

## TL;DR
**Background**: Diffusion models have become universal modeling tools for generation tasks involving images, videos, language, graphs, and robot trajectories. Many practical systems require more than just "unconditional sample generation" but involve incorporating additional objectives during inference—such as classifier guidance, inpainting, reward guidance, or projected diffusion—to guide existing models toward specific conditions or constraints without retraining.

## Background & Motivation
### Overall Architecture
The input to PCD is a set of pre-trained diffusion models or score models, such as two image latent diffusion models, multiple robot trajectory diffusion models, or diffusion policies for two manipulation trajectories. Each model remains responsible for its own marginal distribution; PCD does not modify model parameters but adds a coupling cost gradient at each step of reverse diffusion. The updated samples are then projected into a feasible set provided at test time, ultimately outputting a group of joint samples that satisfy constraints and are mutually coordinated.

## Method
### Main Results
The paper covers three application scenarios: multi-robot navigation, PushT object manipulation, and face pair generation. The following extracts results from the main text that best illustrate PCD's ability to handle "correlation + hard constraints" simultaneously.

## Key Experimental Results
The most elegant aspect of PCD is the decoupling of "jointness" and "feasibility" into two pluggable operators. The coupling cost does not need to know all constraints, and the projection does not need to understand generative semantics; the two meet at each diffusion step to form a concise test-time control loop.

## Highlights & Insights
PCD relies on the gradient or sub-gradient of the coupling cost. If task constraints involve discrete logic, non-differentiable simulation metrics, or black-box safety rules, differentiable approximations, sampling-based estimation, or other surrogates are required, and performance may depend on the quality of these approximations.

## Limitations & Future Work
**vs Classifier Guidance**: Classifier guidance uses the gradient of the conditional likelihood to drive a single diffusion model toward target attributes; PCD treats this as a special case where $Y$ is fixed as a condition and $X$ has no projection. The advantage of PCD is its ability to move multiple variables simultaneously while supporting hard constraint projection.

## Related Work & Insights
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICLR 2026\] MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning](milr_improving_multimodal_image_generation_via_test-time_latent_reasoning.md)
- [\[CVPR 2026\] Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation](../../CVPR2026/image_generation/progress_by_pieces_test-time_scaling_for_autoregressive_image_generation.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICML 2026\] Linearizing Vision Transformer with Test-Time Training](../../ICML2026/image_generation/linearizing_vision_transformer_with_test-time_training.md)

</div>

<!-- RELATED:END -->
