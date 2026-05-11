---
title: >-
  [Paper Note] PHAC: Promptable Human Amodal Completion
description: >-
  [CVPR 2026][Object Detection][human amodal completion] This paper introduces Promptable Human Amodal Completion (PHAC), a novel task that accepts point-based user prompts (pose/bounding box) via dedicated ControlNet modu…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "human amodal completion"
  - "diffusion model"
  - "ControlNet"
  - "pose-guided generation"
  - "image inpainting"
date: 2026-05-08
content_hash: 1f24b044cb682a55
---

# PHAC: Promptable Human Amodal Completion

**Conference**: CVPR 2026
**arXiv**: [2603.14741](https://arxiv.org/abs/2603.14741)
**Code**: None
**Area**: Object Detection
**Keywords**: human amodal completion, diffusion model, ControlNet, pose-guided generation, image inpainting

## TL;DR

This paper introduces Promptable Human Amodal Completion (PHAC), a novel task that accepts point-based user prompts (pose/bounding box) via dedicated ControlNet modules to inject conditional signals, and designs an inpainting-based refinement module to preserve the appearance of visible regions, achieving high-quality and controllable completion of occluded human images.

## Background & Motivation

**Limitations of State of the Field**: Existing Human Amodal Completion (HAC) methods can only hallucinate invisible regions from occluded images without accepting user-specified constraints (e.g., target pose or spatial extent), forcing users to repeatedly sample to obtain satisfactory results.

**Limitations of Prior Work**: Pose-Guided Person Image Synthesis (PGPIS) supports pose conditioning but struggles to preserve instance-specific visible appearance, tending to generate content biased toward the training distribution (e.g., garment characteristics of the DeepFashion dataset).

**Visible Appearance Degradation**: Latent-space denoising and VAE reconstruction discard fine-grained visible details; existing decoder fine-tuning approaches introduce blurring and boundary artifacts, while UV-coordinate schemes lose details under coordinate noise.

**Lack of Multi-Type Prompt Support**: Existing methods do not support multiple types of user prompts (pose + bounding box), limiting flexible trade-offs between performance and interaction cost.

**Weak Zero-Shot Generalization**: PGPIS baselines (e.g., PIDM, MCLD) frequently exhibit severe appearance hallucinations on real-world images outside the training set, producing extreme failures such as transforming elderly men into white women.

**Boundary Artifacts**: Directly compositing visible and generated regions introduces conspicuous artifacts at mask boundaries due to the absence of smooth transition mechanisms.

## Method

### Overall Architecture

The PHAC framework consists of two stages: (A) coarse image generation and (B+C) refinement.

- **Input**: Occluded human image $I_{ic}$ + user prompt $P$ (five supported types: pose $p_{po}$, interest-region bbox $p_{ib}$, entire-region bbox $p_{eb}$, pose + interest bbox $p_{poib}$, pose + entire bbox $p_{poeb}$)
- **Coarse Generation**: User prompts are rendered into a prompt image $I_p$, encoded by a dedicated ControlNet $\Phi_{CN}$ into a conditional signal $c_{pr}$, and injected into the denoising U-Net $\epsilon_{cig}$, which denoises for $T$ steps from random noise to produce the coarse completion $I_{cc}$
- **Refinement Stage**: A lightweight U-Net $\mathcal{U}_{iv}$ predicts the invisible-region mask $M_{iv}$ and dilates it; a baseline composite image is constructed, and low-magnitude noise is injected only into the masked region; the refinement network $\Phi_{RF}$ performs a small number of denoising steps to produce the final output $I_{rc}$

### Key Designs

1. **Multi-Type Point Prompts**: Users need only provide a small number of points—pose prompts specify extra joint coordinates, while bbox prompts specify two corner points. Prompts are rendered as images and fed into the corresponding ControlNet module.
2. **Fine-Tuning Cross-Attention Blocks Only**: To preserve the generative prior of the pretrained diffusion model, only the cross-attention blocks of the denoising U-Net are fine-tuned; all remaining parameters are frozen, balancing prompt alignment with generation quality.
3. **SAM-Based Visible Mask Prediction**: Rather than relying on ground-truth visible region masks, SAM is used to automatically predict $M_v$ from the input image, improving practical applicability.
4. **Invisible Mask Prediction and Dilation**: The lightweight U-Net takes $I_{ic}$, $I_{cc}$, and $M_v$ as inputs to predict the invisible mask $M_{iv}$, which is then dilated to expand the masked region and avoid missing boundary pixels.
5. **Inpainting-Based Refinement**: Instead of regenerating RGB values within the masked region from scratch, the method injects low-magnitude noise ($s=0.5$) into the coarse completion and performs approximately 40% of the standard denoising steps (20 steps), preserving the visible region and eliminating boundary artifacts.
6. **Plug-and-Play Refinement Module**: The refinement network can be directly applied to outputs from other diffusion models as a general post-processing component to boost performance.

### Loss & Training

- **Coarse Generation Loss**: Standard diffusion denoising objective $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_{cig}(z_t, t, c_{te}, c_{pr})\|_2^2]$
- **Mask Prediction Loss**: Weighted combination of BCE and Dice losses $\mathcal{L} = \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$
- **Refinement Network**: Directly employs the pretrained SDXL Inpainting model without additional training
- **Training Setup**: Diffusion model learning rate $5 \times 10^{-6}$, ControlNet learning rate $5 \times 10^{-5}$, batch size 14, trained for 1750 epochs (4×A6000, ~16 hours); mask U-Net trained for 40 epochs (1×RTX 3090, ~30 minutes)
- **Stochastic Inference**: The diffusion model generates $N=16$ coarse outputs per training image; the mask U-Net randomly samples one of these during training for supervision

## Key Experimental Results

### Main Results

**Datasets**: OccThuman2.0 (synthetic, 5,260 images) + AHP (real-world, 56 images)

| Method | Prompt Type | LPIPS*↓ | SSIM↑ | KID*↓ | PSNR↑ | Joint Err.↓ |
|--------|-------------|---------|-------|-------|-------|-------------|
| PIDM | 2D pose | 126.33 | 0.797 | 56.91 | 16.80 | 113.72 |
| MCLD | UV map | 115.90 | 0.833 | 41.11 | 18.37 | 53.38 |
| pix2gestalt | - | 90.11 | 0.911 | 16.51 | 22.63 | 36.65 |
| SDHDO | 2D pose | 81.39 | 0.924 | 16.41 | 23.80 | 43.49 |
| **Ours** | **2D pose** | **49.47** | **0.948** | **6.12** | **25.86** | **23.33** |

*Results on OccThuman2.0; values ×10³*

| Method | LPIPS*↓ | SSIM↑ | KID*↓ | PSNR↑ | Joint Err.↓ |
|--------|---------|-------|-------|-------|-------------|
| SDHDO | 64.19 | 0.956 | 6.05 | 24.45 | 9.24 |
| **Ours** | **38.77** | **0.970** | **1.25** | **26.93** | **6.37** |

*Results on the AHP real-world dataset*

### Ablation Study

**Comparison of Different Prompt Types (OccThuman2.0)**:

| Prompt Type | LPIPS*↓ | SSIM↑ | PSNR↑ | Joint Err.↓ |
|-------------|---------|-------|-------|-------------|
| Pose $p_{po}$ | 49.47 | 0.948 | 25.86 | 23.33 |
| Interest bbox $p_{ib}$ | 51.83 | 0.942 | 24.99 | 24.01 |
| Entire bbox $p_{eb}$ | 52.28 | 0.941 | 25.07 | 28.23 |
| Pose + interest bbox | 49.35 | 0.947 | 25.69 | 22.15 |
| Pose + entire bbox | 49.42 | 0.946 | 25.49 | 21.96 |

**Noise Strength Ablation ($s$ parameter)**: $s=0.5$ achieves the best performance on both OccThuman2.0 and AHP; values that are too small (0.1) yield insufficient denoising, while values that are too large (0.9) degrade the original appearance.

**Plug-and-Play Refinement Effect**: When applied to MCLD, MSE decreases by ~60% and KID by ~71%; when applied to pix2gestalt, LPIPS decreases by 37%.

### Key Findings

1. Pose prompts provide the most effective single-prompt guidance; bbox prompts primarily constrain spatial extent but leave pose ambiguity.
2. Combined prompts (pose + bbox) consistently reduce joint error while maintaining perceptual quality.
3. The refinement module further improves all metrics even though the proposed method already surpasses all baselines prior to refinement.
4. The interest-region bbox achieves the highest per-point gain ($\Delta$JE pp=2.86), offering the best user interaction efficiency.

## Highlights & Insights

- **Novel Task Definition**: This work is the first to propose promptable human amodal completion, establishing a natural bridge between HAC and PGPIS.
- **Practical Prompt Design**: Point-based prompts are extremely lightweight (only a few points required), avoiding hard-to-obtain inputs such as 3D information or dense masks.
- **Generality of the Refinement Module**: The inpainting-based refinement serves as a universal plug-and-play component that yields significant gains for other methods as well.
- **Cross-Attention-Only Fine-Tuning**: Strong prompt alignment is achieved with minimal parameter modification while preserving the pretrained generative prior.
- **Training Efficiency**: The main model requires 4 GPUs for 16 hours, and the mask network requires a single GPU for 30 minutes, keeping overall training costs manageable.

## Limitations & Future Work

- Training is conducted only on synthetic data (OccThuman2.0, 526 3D human subjects), limiting diversity in real-world scenarios.
- The AHP test set contains only 56 images, making real-world evaluation limited in scale.
- The refinement network relies on pretrained SDXL Inpainting and requires approximately 4 seconds per image, which is insufficient for real-time applications.
- Pose prompts require users to manually annotate missing joints, increasing interaction burden under severe occlusion.
- Comparisons with recent DiT architectures or video diffusion models are absent.
- Integration of text prompts or other high-level semantic conditions has not been explored.

## Related Work & Insights

- **HAC Methods**: pix2gestalt (general amodal completion with diffusion prior) and SDHDO (human-specific with 2D pose prior) both lack user prompt support and suffer from visible appearance degradation.
- **PGPIS Methods**: PIDM (2D pose map-conditioned diffusion) and MCLD (UV map-conditioned) both exhibit severe appearance hallucinations due to training data bias.
- **ControlNet**: Serves as the foundational architecture for multi-type prompt injection; a dedicated ControlNet is trained for each prompt type.
- **SAM**: Used to automatically predict visible region masks, replacing traditional approaches that depend on ground-truth masks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel task formulation with a well-designed prompting mechanism
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-baseline comparison and multi-dimensional ablations (prompt type / noise strength / plug-and-play), though real-world evaluation scale is limited
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete formulations, and rich figures and tables
- Value: ⭐⭐⭐⭐ — The generality of the refinement module and the practicality of the prompt design offer promising application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[NeurIPS 2025\] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](../../NeurIPS2025/object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)
- [\[ICCV 2025\] Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding](../../ICCV2025/object_detection/intervening_in_black_box_concept_bottleneck_model_for_enhancing_human_neural_net.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)

</div>

<!-- RELATED:END -->
