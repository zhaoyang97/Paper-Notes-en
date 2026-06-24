---
title: >-
  [Paper Note] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion
description: >-
  [ECCV 2024][Image Generation][Image Interpolation] DreamMover is proposed to perform image interpolation between image pairs with large motions based on pre-trained text-to-image diffusion models. By utilizing three core components—diffusion-aware optical flow estimation, two-level latent space fusion, and self-attention concatenation and replacement—it generates semantically consistent intermediate frames.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image Interpolation"
  - "Diffusion Model Prior"
  - "Optical Flow Estimation"
  - "Semantic Consistency"
  - "Large Motion"
date: 2026-05-08
content_hash: 5048d999443e1483
---

# DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion

**Conference**: ECCV 2024  
**arXiv**: [2409.09605](https://arxiv.org/abs/2409.09605)  
**Code**: [Project Page](https://dreamm0ver.github.io)  
**Area**: Image Generation  
**Keywords**: Image Interpolation, Diffusion Model Prior, Optical Flow Estimation, Semantic Consistency, Large Motion

## TL;DR

DreamMover is proposed to perform image interpolation between image pairs with large motions based on pre-trained text-to-image diffusion models. By utilizing three core components—diffusion-aware optical flow estimation, two-level latent space fusion, and self-attention concatenation and replacement—it generates semantically consistent intermediate frames.

## Background & Motivation

Generating intermediate transition frames from two images with large motion is a highly valuable yet extremely challenging task. Existing methods exhibit obvious limitations:

- **Video frame interpolation methods** (e.g., FILM, LDMVFI): Mainly designed to increase video frame rates, handling small motion discrepancies between adjacent frames. They lack semantic understanding in large-motion scenarios, easily leading to artifacts and object tearing.
- **Image morphing methods** (e.g., DiffInterp, DiffMorpher): Focus primarily on transitions between topologically similar objects (such as faces with different expressions), demonstrating limited capability in modeling the semantic consistency of the same object under large-motion scenarios.

The key challenge lies in the fact that when the motion discrepancy between the two input images is immense, the semantic information of the intermediate frames may not exist in either of the input images (e.g., a semi-open state during an animal opening its mouth). The authors propose to leverage the rich implicit semantic information of pre-trained diffusion models to supplement the missing intermediate semantic representations, ensuring the generated results remain consistent with the inputs.

## Method

### Overall Architecture

DreamMover is based on Stable Diffusion 1.5, and its pipeline consists of three steps:
1. **Diffusion-Aware Optical Flow Estimation**: Establishes pixel correspondences between two images from the U-Net feature maps.
2. **Two-Level Latent Space Fusion**: Processes high-level semantics and low-level details separately to avoid the loss of high-frequency information caused by weighted averaging.
3. **Reference-Guided Consistency Enhancement**: Ensures semantic consistency between the output and input images via self-attention concatenation and replacement + LoRA fine-tuning.

Given an input image pair $\mathcal{I}^0$ and $\mathcal{I}^1$, the goal is to generate an intermediate image $\mathcal{I}^\delta$ ($\delta \in (0,1)$) to construct a semantically consistent video.

### Key Designs

**1. Diffusion-Aware Optical Flow Estimation**

Leverages the feature maps extracted from the diffusion model U-Net during the inversion process to implicitly establish semantic correspondences, without requiring an additional optical flow prediction module:

- Maps both images to the latent space and performs DDIM inversion to feed them into U-Net for noise addition.
- Extracts the feature maps $f^0, f^1$ of the second upsampling block at step 14.
- Finds the corresponding position with the highest cosine similarity in the other feature map by traversing all pixels, obtaining the bidirectional optical flow:

$$F^{0 \to 1}(x,y) = \arg\max_{i,j} \langle f^0(x,y), f^1(i,j) \rangle$$

- Obtains the optical flow at any intermediate timestamp through linear scaling: $F^{0 \to \delta} = \delta \cdot F^{0 \to 1}$.

The authors utilize PCA visualization to verify that the spatial layout of the diffusion U-Net feature maps is highly consistent with the original images, providing a reliable foundation for optical flow estimation.

**2. Two-Level Latent Space Fusion**

Directly performing weighted average fusion in the latent space leads to severe loss of high-frequency information (blurring), as both softmax splatting and temporal interpolation introduce averaging operations.

Key Observation: The two components of the DDIM denoising process exhibit distinct frequency characteristics:
- $z_{t \to 0}$ (the single-step predicted clean latent): Primarily contains high-level contextual information, lacking high-frequency details.
- $\epsilon_\theta(z_t, t)$ (the predicted noise): Contains high-frequency components with more low-level textures.

Based on this, a two-level fusion strategy is proposed:
- **High-level Information**: Employs softmax splatting + temporal weighted interpolation in the $z_{T \to 0}$ space.

$$z^\delta_{T \to 0} = (1-\delta) \cdot \vec{\sigma}(z^0_{T \to 0}, F^{0 \to \delta}) + \delta \cdot \vec{\sigma}(z^1_{T \to 0}, F^{1 \to \delta})$$

- **Low-level Information**: Utilizes a Winner-Takes-All (WTA) operation in the $\epsilon_\theta$ space to select the value with the highest weight, avoiding the high-frequency loss caused by averaging.

$$\epsilon^\delta = WTA(\epsilon_\theta(z^0_T), \epsilon_\theta(z^1_T))$$

Finally, the two parts are combined: $z^\delta_T = \sqrt{\alpha_T} \cdot z^\delta_{T \to 0} + \sqrt{1-\alpha_T} \cdot \epsilon^\delta$.

Spectrum analysis confirms that two-level fusion preserves significantly more high-frequency energy.

**3. Self-Attention Concatenation and Replacement**

During the denoising process, the self-attention features of the input image pairs are injected into the denoising process of the intermediate image:

- The noisy latents of the two input images are fed into the U-Net to extract the Key and Value matrices of each layer.
- For the self-attention block of the intermediate image, its Query is retained, while the Key/Value are replaced with the concatenated Key/Value of the input images:

$$Q = Q^\delta, \quad K = (K^0 \oplus K^1), \quad V = (V^0 \oplus V^1)$$

In this way, the intermediate latents can query relevant local structures and textures from both input images, enhancing consistency.

In addition, a single LoRA (rank 16, fine-tuned for 80 steps, taking about 40 seconds) is used to adapt to the input image pair, further improving semantic consistency. Note that unlike DiffMorpher, DreamMover only requires one LoRA to simultaneously adapt to both images.

### Loss & Training

- DDIM is performed with 50 steps, with latent optimization executed at step 30 during noise addition.
- Generates 32 intermediate interpolated images.
- LoRA uses the AdamW optimizer with a learning rate of $5 \times 10^{-4}$.
- Does not use classifier-free guidance (CFG accumulates numerical errors leading to oversaturation).
- The entire pipeline runs on a single NVIDIA RTX 3090.

## Key Experimental Results

### Main Results

Quantitative comparison on the self-built InterpBench benchmark (100 pairs of large-motion images):

| Method | FID ↓ | LPIPS ↓ | WE ↓ | WE_mid ↓ |
|------|-------|---------|------|----------|
| DiffInterp | 185.78 | 0.5375 | 0.5112 | 0.9573 |
| DiffMorpher | 68.23 | 0.3061 | 0.2673 | 0.7784 |
| Film | 54.28 | 0.2313 | **0.1244** | 0.4176 |
| LDMVFI | 48.35 | 0.2347 | 0.1453 | 0.4373 |
| **DreamMover** | **43.18** | **0.2227** | 0.2069 | **0.3687** |

### Ablation Study

User preference study (pairwise comparison, percentage of preferring the proposed method):

| Compared Method | Preference Rate for DreamMover |
|----------|------------------------|
| vs DiffInterp | Significant Advantage |
| vs DiffMorpher | Significant Advantage |
| vs Film | Clear Advantage |
| vs LDMVFI | Advantage |

Key Ablations:
- Two-level fusion vs. Direct fusion: Two-level fusion significantly reduces blur and preserves more high-frequency details.
- Self-attention replacement: Critical for maintaining appearance consistency.
- LoRA fine-tuning: Further enhances semantic identity consistency.

### Key Findings

1. The U-Net features of diffusion models are naturally suitable for semantic correspondence and optical flow estimation without additional training.
2. Intuitive weighted average fusion destroys high-frequency information—decomposing the signals into high-level and low-level for separate processing is a key innovation.
3. FILM and LDMVFI perform better on temporal consistency metrics, but the generated intermediate content is not guaranteed to be semantically correct, affecting the actual video quality.
4. The proposed inference pipeline requires no training, needing only LoRA fine-tuning (~40 seconds) and optimization.

## Highlights & Insights

1. **Utilizing diffusion model prior for optical flow**: No optical flow network is required; correspondences are established directly from U-Net features, which is an underestimated capability.
2. **Frequency analysis insights for two-level fusion**: Clearly reveals the differences in frequency characteristics between $z_{t \to 0}$ and $\epsilon_\theta$ in the latent space, providing theoretical support for the fusion strategy.
3. **InterpBench Benchmark**: The first benchmark dataset specifically targeting the evaluation of semantic consistency in image interpolation under large motions.
4. **Lightweight pipeline**: Based on pre-trained models, requiring no training from scratch, running easily on a single GPU.

## Limitations & Future Work

1. The optical flow estimation is based on maximum cosine similarity matching, which may fail in heavily occluded or symmetric scenes.
2. The temporal consistency metric (WE) is inferior to FILM/LDMVFI, indicating that the smoothness of frame-by-frame generation still has room for improvement.
3. Generating 32 frames requires multiple DDIM inversions and denoising steps, incurring high computational costs.
4. It relies on the accuracy of DDIM inversion, where inversion errors can accumulate into the final results.
5. It is only based on SD 1.5, and its performance on stronger diffusion models has not been verified.

## Related Work & Insights

- **Diffusion Features for Correspondence**: Shares the insight that "diffusion features are suitable for correspondence" with works like DIFT, but is the first to apply it to optical flow estimation for image interpolation.
- **Video Frame Interpolation**: FILM works well for near-duplicate frames but lacks semantic understanding; DreamMover fills the gap in large-motion scenarios.
- **Insights**: The two-level fusion strategy can be generalized to other latent space operations requiring detail preservation (e.g., editing, inpainting).

## Rating

- Novelty: ⭐⭐⭐⭐ — Each of the three components features innovations, with the two-level fusion being particularly ingenious.
- Technical Depth: ⭐⭐⭐⭐ — Frequency analysis provides a solid theoretical foundation for the design choices.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Self-built benchmark + multi-method comparisons + user study.
- Practical Value: ⭐⭐⭐⭐ — Direct application value in short video creation scenarios.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](../../CVPR2025/image_generation/eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation](coin_control-inpainting_diffusion_prior_for_human_and_camera_motion_estimation.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)

</div>

<!-- RELATED:END -->
