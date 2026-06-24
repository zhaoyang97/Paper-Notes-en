---
title: >-
  [Paper Note] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction
description: >-
  [CVPR 2025][3D Vision][HDR Novel View Synthesis] This paper proposes InstantHDR, the first feed-forward HDR novel view synthesis method. It performs multi-exposure fusion via geometry-guided appearance modeling and employs a MetaNet to predict a scene-adaptive tone mapper. It reconstructs HDR 3D Gaussians from uncalibrated multi-exposure LDR images in a single forward pass, achieving a speedup of ~700x compared to optimization-based methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "HDR Novel View Synthesis"
  - "Feed-forward 3D Reconstruction"
  - "3D Gaussian Splatting"
  - "Tone Mapping"
  - "Multi-exposure Fusion"
date: 2026-05-08
content_hash: e3a0dbc5317c8434
---

# InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2603.11298](https://arxiv.org/abs/2603.11298)  
**Code**: To be open-sourced  
**Area**: 3D Vision  
**Keywords**: HDR Novel View Synthesis, Feed-forward 3D Reconstruction, 3D Gaussian Splatting, Tone Mapping, Multi-exposure Fusion

## TL;DR
This paper proposes InstantHDR, the first feed-forward HDR novel view synthesis method. It performs multi-exposure fusion via geometry-guided appearance modeling and employs a MetaNet to predict a scene-adaptive tone mapper. It reconstructs HDR 3D Gaussians from uncalibrated multi-exposure LDR images in a single forward pass, achieving a speedup of ~700x compared to optimization-based methods.

## Background & Motivation
**Background**: HDR novel view synthesis aims to reconstruct HDR scenes from multi-exposure LDR images. Existing methods (such as HDR-GS, GaussianHDR) rely on per-scene optimization, requiring accurate camera poses, dense point cloud initialization, and long training times.

**Limitations of Prior Work**: Optimization-based methods are computationally expensive and fail to generalize. Directly applying a feed-forward model (such as AnySplat) to multi-exposure inputs leads to severe artifacts due to: (a) exposure inconsistencies causing ghosting artifacts, (b) difficult geometric correspondence under extreme brightness variations, (c) inconsistent camera response functions across images, and (d) the scarcity of multi-view HDR datasets.

**Key Challenge**: Feed-forward models assume appearance consistency across views and cannot handle exposure-induced appearance inconsistencies, whereas optimization-based models can handle them but are prohibitively slow.

**Goal**: How to handle multi-exposure fusion and camera response function estimation within a feed-forward reconstruction framework?

**Key Insight**: The cross-attention maps of a geometry encoder naturally encode cross-view geometric correspondences, which can be reused to guide appearance fusion. Furthermore, tone mapping parameters can be predicted in a single forward step by a MetaNet instead of being optimized per scene.

**Core Idea**: Freeze the geometry branch to provide structural guidance, train an appearance branch to handle exposure inconsistency, and integrate a MetaNet to predict the tone mapper.

## Method

### Overall Architecture
The input consists of $V$ uncalibrated LDR images of different exposures and their exposure times. The output is a set of HDR 3D Gaussians, camera parameters, and scene-level tone mapping parameters. The framework contains two branches: (1) a frozen geometry branch predicting depth and poses; (2) a trainable appearance branch merging multi-exposure inputs into a unified HDR representation. The Gaussian HEAD combines outputs from both branches to obtain HDR-aware Gaussian attributes, while MetaNet predicts the tone mapper to enable controllable-exposure rendering.

### Key Designs

1. **Exposure Normalization $F_E$**:

    - **Function**: Aligns the appearance features of all views to a shared reference exposure level.
    - **Mechanism**: Compute the relative log-exposure $\tilde{\ell}_v = \ell_v - \bar{\ell}$ and transform it into an embedding $\mathbf{e}_v$ via sinusoidal position encoding. Then, predict view-specific affine parameters $(\gamma_v, \beta_v)$ using a FiLM layer to modulate the appearance tokens: $\hat{t}_v^A = t_v^A \odot (1+\gamma_v) + \beta_v$.
    - **Design Motivation**: Eliminate exposure-induced brightness variations before fusion, ensuring subsequent cross-attention operations occur in a unified irradiance space.

2. **Geometry-guided Cross-view Attention $F_A$**:

    - **Function**: Utilizes the attention maps of the frozen geometry encoder to guide the fusion of multi-exposure appearance features.
    - **Mechanism**: Reuse the Q and K matrices from the 14th layer of the geometry encoder, and use the normalized appearance tokens as V to compute attention: $\tilde{t}_v^A = \text{softmax}(QK^\top/\sqrt{d}) \hat{t}_v^A$.
    - **Design Motivation**: The cross-attention of the geometry encoder naturally learns cross-view geometric correspondences (matching objects like leaves, cups, or door frames even under extreme exposure differences) without requiring extra training, thereby providing reliable guidance for fusion.

3. **High-Resolution Upsampling $F_U$**:

    - **Function**: Restores pixel-level texture details from patch-level resolution.
    - **Mechanism**: Apply a Difference-of-Gaussians (DoG) strategy: encode full-resolution LDR feature maps $\mathbf{g}_v$ with a shallow CNN, downsample and then upsample to obtain a low-frequency version $\mathbf{g}_v^{\downarrow\uparrow}$, and add the residual $\mathbf{g}_v - \mathbf{g}_v^{\downarrow\uparrow}$ (which provides high-frequency details) to the upsampled fused features.
    - **Design Motivation**: Fusion operations at the patch level lose high-frequency information. The DoG residual recovers this information simply and effectively.

4. **MetaNet Tone Mapping $F_M$**:

    - **Function**: Predicts scene-specific tone mapper parameters, eliminating per-scene optimization.
    - **Mechanism**: Take LDR features, exposure embeddings, and HDR Gaussian attributes, compress them into a scene descriptor $\boldsymbol{\theta}$ using a strided convolutional encoder followed by global pooling, and use it as the weights and biases of a two-layer MLP tone mapper $g_\theta$. During rendering, $\mathbf{L}_v(\ell) = g_\theta(\log \mathbf{H}_v + (\ell - \bar{\ell}) \cdot \log 2)$ supports arbitrary exposure control.
    - **Design Motivation**: Different cameras and software use different CRFs (AgX, Filmic, etc.). MetaNet infers the mapping from the scene context rather than overfitting to it.

### Loss & Training
The geometry encoder and decoding heads are frozen, while only the appearance branch, the Gaussian Head, and MetaNet are trained. The model is trained end-to-end without 3D or HDR supervision, relying solely on multi-exposure LDR image reconstruction loss. An HDR-Pretrain dataset (comprising 168 synthetic indoor scenes with various tone mapping operators and 5 exposure levels) is constructed to support pretraining.

## Key Experimental Results

### Main Results (HDR-NeRF Real Scenes)

| Method | 4-view PSNR↑ | 4-view SSIM↑ | 18-view PSNR↑ | Time (s)↓ |
|------|-----------|----------|-----------|---------|
| AnySplat (Zero-shot) | 12.10 | 0.517 | 13.91 | ~2s |
| **InstantHDR (Zero-shot)** | **18.44** | **0.721** | **19.48** | **~2.5s** |
| GaussianHDR (Optimization) | 19.26 | 0.691 | 29.36 | 1891s |
| HDR-GS (Optimization) | 15.40 | 0.622 | 27.42 | 815s |
| **InstantHDR_1K (Post-opt.)** | **22.16** | **0.762** | **29.19** | **~39s** |

Zero-shot performance outperforms AnySplat by +6.3dB on 4-view real scenes. With post-optimization taking only ~39s (50x faster than GaussianHDR), the method outperforms reference optimization pipelines by +2.9dB in sparse-view settings.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Description |
|------|-------|-------|------|
| Full model | 18.44 | 0.721 | Full model |
| w/o Exposure Norm | — | — | Without Exposure Norm -> noisy fusion |
| w/o Geo-attention | — | — | Without geometry attention reuse -> inaccurate correspondences |
| w/o DoG upsampling | — | — | Loss of high-frequency details |
| w/o MetaNet (Fixed TM) | — | — | Degradation in scenes with different CRFs |

### Key Findings
- Reusing geometric attention is critical: the cross-attention from the 14th layer of the frozen encoder successfully matches structures even under extreme exposure differences.
- The scarcity of HDR data is a bottleneck; the proposed HDR-Pretrain dataset (168 scenes $\times$ 3 TMs $\times$ 5 exposures) effectively supports pretraining.
- Feed-forward initialization followed by lightweight post-optimization (1K iterations) represents the best practice, bypassing the expensive densification step.
- Real-world scenes still require fine-tuning on HDR-Plenoxels to bridge the domain gap.

## Highlights & Insights
- **Reusing geometric attention** is exceptionally clever: instead of training a new correspondence module, it directly reuses the QK matrices of the frozen backbone, letting the appearance branch focus solely on learning the mapping of V. This "free geometric structure" design is transferable to other feed-forward reconstruction tasks requiring multi-modal fusion.
- **MetaNet predicting tone mapper weights** rather than learning a fixed mapping is an elegant application of meta-learning in 3D reconstruction.
- The **HDR-Pretrain dataset** fills a vacancy in the research community, with multiple TM operators enhancing system generalization.

## Limitations & Future Work
- Pretraining on synthetic data still requires fine-tuning for real scenes, meaning the end-to-end generalization capability is limited.
- The method is only validated on indoor scenes; outdoor HDR (with overexposed skies and complex lighting) has not been addressed.
- The tone mapper is a simple two-layer MLP, which has limited capacity to fit complex non-linear CRFs.
- Post-optimization still requires 30-40 seconds, which remains a bottleneck for real-time deployment.

## Related Work & Insights
- **vs GaussianHDR**: Optimization-based, 50x slower, but yields higher synthesis quality in dense-view settings.
- **vs AnySplat**: Feed-forward but neglects exposure inconsistency, leading to severe degradation under multi-exposure inputs.
- **vs HDR-NeRF**: NeRF-based and even slower; this work adopts its evaluation protocol.

## Rating
- Novelty: ⭐⭐⭐⭐ First feed-forward HDR NVS; geometric attention reuse is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-setup comparisons, ablations, and dataset construction, though real-world scenarios are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear and comprehensive, with a well-structured problem breakdown.
- Value: ⭐⭐⭐⭐ Significantly drives forward real-time HDR reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)
- [\[ICML 2025\] High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)
- [\[CVPR 2025\] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](../../ICLR2026/3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)

</div>

<!-- RELATED:END -->
