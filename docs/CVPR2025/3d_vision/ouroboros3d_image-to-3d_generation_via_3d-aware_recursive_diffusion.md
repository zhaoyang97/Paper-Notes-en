---
title: >-
  [Paper Note] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion
description: >-
  [CVPR 2025][3D Vision][Image-to-3D generation] The paper proposes Ouroboros3D, which integrates multi-view generation and 3D reconstruction into a recursive diffusion process. By utilizing a 3D-aware feedback mechanism (rendering CCM and color maps as denoising conditions) and a joint training strategy, it resolves the issues of insufficient 3D consistency and domain gaps in two-stage methods, achieving state-of-the-art performance on the GSO dataset.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Image-to-3D generation"
  - "recursive diffusion"
  - "3D-aware feedback"
  - "multi-view consistency"
  - "joint training"
date: 2026-05-08
content_hash: e24fe71687da3a20
---

# Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2406.03184](https://arxiv.org/abs/2406.03184)  
**Code**: [Project Page](https://costwen.github.io/Ouroboros3D/)  
**Area**: 3D Vision / Image-to-3D  
**Keywords**: Image-to-3D generation, recursive diffusion, 3D-aware feedback, multi-view consistency, joint training

## TL;DR

The paper proposes Ouroboros3D, which integrates multi-view generation and 3D reconstruction into a recursive diffusion process. By utilizing a 3D-aware feedback mechanism (rendering CCM and color maps as denoising conditions) and a joint training strategy, it resolves the issues of insufficient 3D consistency and domain gaps in two-stage methods, achieving state-of-the-art performance on the GSO dataset.

## Background & Motivation

1. **Background**: The dominant methods for single-image-to-3D generation follow a two-stage approach: generating multi-view images using multi-view diffusion models, followed by recovering the 3D representation via feed-forward reconstruction models. This pipeline has achieved promising results, with representative methods including InstantMesh, LGM, CRM, etc.

2. **Limitations of Prior Work**: (a) The multi-view generation stage optimizes in 2D image space rather than 3D space, making it difficult to guarantee geometric consistency; (b) Reconstruction models are primarily trained on synthetic data, leading to a domain gap when processing generated multi-view images; (c) The two models are designed and trained independently, preventing them from mutually benefiting each other.

3. **Key Challenge**: When optimized as independent components, the multi-view diffusion model and the 3D reconstruction model lack information exchange—the diffusion model does not know whether the generated images can be reconstructed correctly, and the reconstruction model adapts poorly to out-of-distribution generated images.

4. **Goal**: Integrate the two stages into an end-to-end trainable recursive diffusion process to enable mutual enhancement between the two models.

5. **Key Insight**: Feed the rendering outputs of the reconstruction model back into the denoising loop of the diffusion model as 3D-aware conditions, while jointly training both models to eliminate the domain gap.

6. **Core Idea**: Recursive diffusion—at each denoising step, first predict clean multi-views $\rightarrow$ feed them into the reconstruction model $\rightarrow$ render 3D-aware maps $\rightarrow$ use them as conditioning for the next denoising step, repeatedly iterating to form a self-optimizing closed loop.

## Method

### Overall Architecture

Based on Stable Video Diffusion (SVD) as the multi-view generator and Large Gaussian Model (LGM) as the 3D reconstructor. In the denoising sampling loop, each step first decodes the predicted $\tilde{\mathbf{x}}_0^f$ into multi-view images and feeds them into LGM to reconstruct 3D Gaussians. It then renders color maps and CCMs (Canonical Coordinates Maps) from the reconstructed 3D model, which are encoded and injected into the denoising network for the next step.

### Key Designs

1. **3D-aware Feedback Mechanism**:
    - **Function**: Injecting explicit 3D geometric information into the denoising process of the multi-view diffusion model.
    - **Mechanism**: At each denoising step, two types of maps are rendered from the 3D Gaussians reconstructed by LGM: (a) RGB color maps (retaining texture information); (b) Canonical Coordinates Maps (CCMs, where each pixel corresponds to a globally normalized vertex coordinate on the 3D model). Two lightweight convolutional encoders (similar to T2I-Adapter) are used to encode these maps into features with the same spatial dimensions as the intermediate features of the U-Net encoder, which are then added to the U-Net encoder at each resolution level.
    - **Design Motivation**: Choosing CCM over depth/normal maps because CCM captures global vertex coordinates (consistent across views), whereas depth maps are normalized relative to individual camera perspectives. CCM naturally encodes cross-view geometric correspondences, providing a stronger constraint for multi-view consistency.

2. **Joint Training Strategy**:
    - **Function**: Simultaneously training the multi-view diffusion model and the 3D reconstruction model to eliminate the domain gap between the two stages.
    - **Mechanism**: During training, the reconstruction model does not use the original GT multi-view images but instead takes the recovered images $\tilde{\mathbf{x}}_0$ from the diffusion process as input. The reconstruction loss consists of an RGB L2 loss and an LPIPS perceptual loss. A zero-initialized time embedding layer is introduced in LGM to perceive different noise levels. Self-conditioning is applied with a probability of 0.5 (using the 3D feedback from the previous step half of the time and not using it the other half) to prevent the model from over-relying on 3D information.
    - **Design Motivation**: Independently trained reconstruction models only see "clean" rendered images and suffer performance degradation when processing generated multi-view images with "generative noise". Joint training allows the reconstruction model to adapt to the output distribution of the diffusion model, while the backpropagation of the reconstruction loss to the diffusion model provides implicit supervision for 3D consistency.

3. **3D-aware Recursive Inference Strategy**:
    - **Function**: Progressively optimizing multi-view images and the 3D model through an iterative loop during inference.
    - **Mechanism**: The initial condition is set to zero (no 3D feedback), and each subsequent denoising step updates the 3D condition using the reconstruction results from the previous step. As denoising progresses, the signal-to-noise ratio improves $\rightarrow$ reconstruction quality increases $\rightarrow$ feedback conditions become more accurate $\rightarrow$ denoising results become more consistent $\rightarrow$ forming a virtuous spiral.
    - **Design Motivation**: Compared to combining them only during inference (e.g., the re-sampling strategy in VideoMV), joint training ensures that the model learns to utilize 3D feedback during training, and it prevents deviation from the input image due to inaccurate early-stage 3D feedback.

### Loss & Training

The diffusion model uses standard denoising loss. The reconstruction model uses $\mathcal{L}_G = \mathcal{L}_{rgb} + \lambda \mathcal{L}_{LPIPS}$, with the input being the images $\tilde{\mathbf{x}}_0$ recovered by the diffusion model (incorporating noise level embeddings). The training data consists of ~80K objects filtered from Objaverse, with rendered 16-frame orbit videos at 512×512 resolution. The self-conditioning probability is set to 0.5.

## Key Experimental Results

### Main Results

GSO dataset (100 objects, zero-shot evaluation):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Type |
|------|-------|-------|--------|------|
| VideoMV (Multi-view) | 18.605 | 0.841 | 0.155 | Two-stage |
| SV3D (Multi-view) | 21.042 | 0.850 | 0.130 | Two-stage |
| InstantMesh (3D) | 19.948 | 0.873 | 0.121 | Two-stage |
| LGM (3D) | 17.716 | 0.832 | 0.189 | Two-stage |
| **Ouroboros3D (Multi-view)** | **21.770** | **0.887** | **0.109** | Unified |
| **Ouroboros3D (3D)** | **21.761** | **0.889** | **0.109** | Unified |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| No feedback (Baseline SVD+LGM) | 20.5 | 0.870 | 0.125 | Standard two-stage |
| +Joint training (No 3D feedback) | 21.0 | 0.878 | 0.118 | Joint training alone reduces domain gap |
| +RGB feedback | 21.3 | 0.882 | 0.114 | Color guidance provides appearance information |
| +CCM feedback | 21.5 | 0.885 | 0.111 | Coordinate map provides stronger geometric constraint |
| +RGB+CCM feedback (Full) | **21.8** | **0.887** | **0.109** | Full scheme |

### Key Findings

- Ouroboros3D outperforms all two-stage methods in both multi-view quality and 3D reconstruction quality, demonstrating the advantages of the unified framework.
- CCM feedback contributes more than RGB feedback—global coordinates provide explicit geometric correspondences across views.
- Joint training improves performance (PSNR +0.5) even without 3D feedback, indicating that eliminating the domain gap is valuable on its own.
- Compared with VideoMV(GS), which combines them only during inference, Ouroboros3D's joint training strategy yields superior performance.

## Highlights & Insights

- The concept of **recursive diffusion** lives up to its name (Ouroboros)—the output of 3D reconstruction is fed back into the generation input, forming a self-improving closed loop. This "generation $\rightarrow$ understanding $\rightarrow$ feedback" paradigm has broader applicability.
- **CCM as a 3D-aware condition** is an overlooked yet highly valuable choice—it is more global than depth maps (independent of a single perspective) and more unique than normal maps (global coordinates vs. local orientations).
- The training strategy of **self-conditioning with a 0.5 probability** is noteworthy—it allows the model to learn scenarios both with and without 3D feedback, enhancing robustness.

## Limitations & Future Work

- Relies on the reconstruction capability of LGM, which has limited capacity for recovering details.
- The recursive process increases inference time, as reconstruction and rendering are required at each step.
- Only 8-frame multi-views are used, covering a limited range of angles.
- Future work can extend this to more frames or video diffusion models to further improve coverage and consistency.

## Related Work & Insights

- **vs InstantMesh/CRM**: These two-stage methods train the two modules independently, whereas Ouroboros3D achieves mutual enhancement through joint training and 3D feedback.
- **vs IM-3D/VideoMV**: These methods introduce 3D information via re-sampling during inference but lack joint training, while Ouroboros3D integrates them more thoroughly during training.
- **vs DMV3D**: DMV3D treats 3D reconstruction as a diffusion denoiser but trains from scratch, resulting in poor generalization; Ouroboros3D retains generalization by building upon pre-trained SVD.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified framework of recursive diffusion + 3D feedback is elegantly designed, and the choice of CCM conditioning is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparison on GSO + thorough ablation studies + qualitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ The conceptual framework diagram is clear, and the comparison illustration is intuitive.
- Value: ⭐⭐⭐⭐ Provides an effective paradigm for the unification of "multi-view generation + 3D reconstruction."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)

</div>

<!-- RELATED:END -->
