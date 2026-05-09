---
title: >-
  [Paper Note] CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][Gaussian Splatting] CLIPGaussian proposes the first unified style transfer framework based on Gaussian Splatting, supporting text- and image-guided stylization of 2D images, videos, 3D objects, and 4D dynamic scenes. It integrates as a plug-and-play module into existing GS pipelines without requiring large generative models or retraining from scratch, and without altering model size.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Gaussian Splatting
  - style transfer
  - CLIP
  - multi-modal
  - 3D/4D
date: 2026-05-08
content_hash: 341689bd2a6a5cf8
---

# CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2505.22854](https://arxiv.org/abs/2505.22854)
**Code**: Coming soon (promised in paper)
**Area**: 3D Vision / Style Transfer
**Keywords**: Gaussian Splatting, style transfer, CLIP, multi-modal, 3D/4D

## TL;DR

CLIPGaussian proposes the first unified style transfer framework based on Gaussian Splatting, supporting text- and image-guided stylization of 2D images, videos, 3D objects, and 4D dynamic scenes. It integrates as a plug-and-play module into existing GS pipelines without requiring large generative models or retraining from scratch, and without altering model size.

## Background & Motivation

**Background**: Gaussian Splatting (GS) has become an efficient scene representation for rendering 3D scenes from 2D images, and has been extended to images, videos, and 4D dynamic content. Style transfer, a well-established task in 2D visual editing, remains challenging in the context of GS representations.

**Limitations of Prior Work**:
- Existing GS style transfer methods (StyleGaussian, ReGS, InstantStyleGaussian, etc.) modify only color and opacity, without altering geometry.
- G-Style jointly optimizes appearance and geometry but significantly increases model size (more than doubling the number of Gaussians).
- Diffusion-based methods (e.g., Morpheus, Style3D) rely on large models, incur high computational cost, and struggle to ensure multi-view consistency.
- No unified framework exists that handles style transfer across 2D, video, 3D, and 4D scenes simultaneously.
- Text-guided GS editing methods (I-GS2GS, DGE) primarily target general editing rather than style transfer.

**Key Challenge**: GS style transfer involves a trade-off between appearance modification and geometric deformation—modifying geometry typically requires increasing the number of Gaussians, sacrificing the compactness of the original model.

**Goal**:
- How to jointly optimize color and geometry without changing the number of Gaussians?
- How to support unified style transfer across 2D/video/3D/4D modalities?
- How to accommodate both text and image as style conditioning?

**Key Insight**: Style transfer is framed as a fine-tuning problem over GS parameters. CLIP's cross-modal alignment capability is leveraged to unify text and image as style conditions, enabling optimization of all parameters while preserving the original number of Gaussians.

**Core Idea**: The patch-based CLIP loss from CLIPStyler is extended to the GS framework. By combining a global directional CLIP loss with a local patch CLIP loss, multimodal style transfer is achieved without increasing model size.

## Method

### Overall Architecture

CLIPGaussian adopts a two-stage training pipeline:

**Stage 1**: A standard GS method (3DGS / D-MiSo / MiRaGe / VeGaS) is used to train a base model on the input data, yielding a Gaussian representation of the scene $\mathcal{G} = \{(\mathcal{N}(m_i, \Sigma_i), \sigma_i, c_i, \theta_i)\}_{i=1}^n$, where $m_i$ denotes the mean position, $\Sigma_i$ the covariance matrix, $\sigma_i$ the opacity, $c_i$ the spherical harmonic color, and $\theta_i$ the modality-specific additional parameters.

**Stage 2**: With the number of Gaussians frozen, all Gaussian parameters are fine-tuned using style condition $\mathcal{S}$ (image or text) guided by CLIP and VGG losses. At each step, a training view $I_l$ is selected, the reconstruction $R_\mathcal{G}(I_l)$ is rendered, random patches with random perspective augmentation are extracted, and $\mathcal{G}$ is updated via a multi-component loss.

### Key Designs

**Unified Gaussian Representation**: Different modalities use different GS base models:
- 3D scenes: standard 3DGS
- 4D dynamic scenes: D-MiSo (multi-Gaussian + deformation network)
- 2D images: MiRaGe (planar Gaussians + 3D latent space)
- Video: VeGaS (3D folded Gaussians)

Crucially, all modalities can be unified under the formulation of "a set of Gaussian parameters + a renderer," reducing style transfer to a parameter optimization problem.

**Plug-and-Play Design**: CLIPGaussian does not modify the base model architecture and performs no densification or pruning—only the existing Gaussian parameters are fine-tuned. This implies:
- The stylized model has exactly the same size as the original.
- Style interpolation can be achieved via linear interpolation of Gaussian parameters.
- The approach is non-invasive and compatible with any GS method as the backbone.

**Joint Color and Geometry Optimization**: For 3D and 4D scenes, CLIPGaussian jointly optimizes position, color, scale, rotation, and opacity, enabling genuine geometric deformation rather than mere color modification. For the video modality, only color is optimized to preserve temporal consistency.

### Loss & Training

The total loss is a weighted combination of four terms:

$$L_{total} = \lambda_d L_d + \lambda_p L_p + \lambda_c L_c + \lambda_b L_b$$

1. **Content Loss $L_c$**: MSE between VGG-19 features (conv4_2 and conv5_2) of the original image $I_l$ and the rendered image $R_\mathcal{G}(I_l)$, preserving content structure:

$$L_c(R_\mathcal{G}(I_l), I_l) = MSE(\Phi_{VGG}(R_\mathcal{G}(I_l)), \Phi_{VGG}(I_l))$$

2. **Directional CLIP Loss $L_d$** (global style): Measures whether the direction of change in CLIP embeddings between the rendered and original image aligns with the direction from the negative prompt "Photo" to the style condition:

$$L_d = 1 - \cos(\Phi_{CLIP}(R_\mathcal{G}(I_l)) - \Phi_{CLIP}(I_l), \Phi_{CLIP}(\mathcal{S}) - \Phi_{CLIP}(\text{"Photo"}))$$

3. **Patch CLIP Loss $L_p$** (local style): Computes the mean directional CLIP loss over randomly sampled and perspective-augmented patches of the rendered image, focusing on local details:

$$L_p = \frac{1}{n}\sum_{i=1}^n L_d(p_i(R_\mathcal{G}(I_l)), I_l)$$

4. **Background Loss $L_b$**: Constrains background regions from being corrupted by stylization, computed as the L1 distance within the background mask.

Default hyperparameters: $\lambda_b=1000$, $\lambda_p=90$, $\lambda_d=5$, $\lambda_c=0.8$, patch\_size=128, num\_patch=64, trained for 5,000 steps without densification or pruning.

## Key Experimental Results

### Main Results

**3D Text-Guided Style Transfer** (NeRF-Synthetic + Mip-NeRF 360):

| Method | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ | Model Size Change |
|------|---------|----------|------------|---------|-----------|
| I-GS2GS | 16.80 | 12.03 | 99.19 | 13.53 | −36% |
| DGE | 17.59 | 12.27 | 99.31 | 12.46 | −5% |
| **CLIPGaussian** | **26.86** | **26.31** | 98.80 | 2.34 | **+0%** |

**3D Image-Guided Style Transfer**:

| Method | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ | Model Size Change |
|------|---------|----------|------------|---------|-----------|
| StyleGaussian | 63.69 | 13.07 | 98.87 | 1.36 | +0% |
| G-Style | 76.94 | 24.94 | 98.94 | 1.31 | **+126%** |
| **CLIPGaussian** | 72.65 | 20.72 | 98.78 | 1.77 | **+0%** |

**Video Style Transfer** (DAVIS dataset, image-guided):

| Method | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ |
|------|---------|----------|------------|---------|
| CCPL | 18.89 | 8.20 | 97.92 | −0.02 |
| UniST | 15.93 | 3.85 | 99.36 | 5.16 |
| **CLIPGaussian** | **74.31** | **17.60** | 99.18 | 1.27 |

**Video Style Transfer** (text-guided):

| Method | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F ↑ |
|------|---------|----------|------------|---------|
| Rerender | 19.40 | 9.83 | 98.23 | −0.03 |
| Text2Video | 26.05 | 24.99 | 93.63 | 0.03 |
| **CLIPGaussian** | **26.25** | 24.53 | **99.00** | 1.92 |

### Ablation Study

**Effect of Feature Learning Rate** (3D, CLIP metrics):

| feature_lr | CLIP-S ↑ | CLIP-SIM ↑ | CLIP-CONS ↑ | CLIP-F |
|-----------|---------|----------|------------|--------|
| 32 | 18.53 | 19.51 | 98.08 | 15.02 |
| 128 | 25.60 | 29.78 | 97.87 | 6.93 |
| 256 | 27.45 | 32.65 | 97.96 | 4.42 |

**Effect of $\lambda_p$ and $\lambda_d$**:

| $\lambda_p$ | CLIP-S | CLIP-SIM | CLIP-CONS | CLIP-F |
|------------|--------|---------|----------|--------|
| 0 | 16.00 | 14.12 | 98.74 | 11.03 |
| 90 (default) | 23.67 | 25.07 | 97.95 | 2.36 |
| 180 | 24.36 | 25.27 | 97.78 | 1.48 |

**Training Time Comparison**:

| Scene | # Gaussians | Stylization Time |
|------|------------|-----------|
| hotdog | 0.14M | 11m 29s |
| lego | 0.31M | 11m 36s |
| bonsai | 1.35M | 11m 37s |
| garden | 4.48M | 21m 03s |

### Key Findings

1. **Text-guided superiority**: CLIPGaussian significantly outperforms all baselines in text-guided style transfer (CLIP-S improved by 50%+), owing to CLIP's native text–image alignment.
2. **Zero model bloat**: Unlike G-Style (+126% Gaussians), CLIPGaussian preserves the original model size, offering practical deployment advantages.
3. **Temporal consistency advantage**: The GS representation inherently shares parameters across frames, so style modifications propagate automatically to all related frames, yielding greater consistency than inter-frame propagation methods.
4. **User study validation**: In a formal user study (CLICKworker platform, 30 participants per survey), CLIPGaussian ranked highest in text-guided scenarios.

## Highlights & Insights

- **Unified design philosophy**: The first framework to unify style transfer across four modalities (2D / video / 3D / 4D) within a single pipeline, demonstrating the potential of GS as a universal backbone.
- **Practical plug-and-play utility**: The design—modifying neither the base architecture nor the model size—is well-suited for engineering deployment.
- **Style interpolation capability**: Since the number of Gaussians remains unchanged, smooth style transitions can be achieved via linear interpolation of parameters.
- **Extension of patch-based CLIP loss**: The 2D methodology of CLIPStyler is elegantly generalized to 3D/4D domains.

## Limitations & Future Work

- 2D image stylization quality falls short of large-model or diffusion-based approaches (e.g., ChatGPT-4o), which is an inherent limitation of GS-based methods.
- Stylization quality depends on the reconstruction quality of the base model—poor reconstruction may yield unreasonable stylization results.
- Training time (~11 minutes per scene) is reasonable but not real-time.
- CLIP's semantic understanding is limited, potentially insufficient for complex or abstract style descriptions.
- Comparative experiments on 4D scenes are constrained (4DStyleGaussian had no public code at the time; only qualitative comparison was conducted).

## Related Work & Insights

- **3DGS series**: The foundational framework for CLIPGaussian; the core contribution lies in introducing style transfer into the GS ecosystem.
- **CLIPStyler / FastCLIPstyler**: 2D text-guided style transfer methods; the patch CLIP loss in CLIPGaussian is directly inspired by these works.
- **AdaIN / StyTr2**: Classic image style transfer methods serving as 2D baselines.
- **G-Style**: The closest 3D competitor, which however requires increasing the number of Gaussians.
- **D-MiSo / VeGaS / MiRaGe**: The base GS models used by CLIPGaussian for each respective modality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First unified GS style transfer framework spanning four modalities.
- **Technical Depth**: ⭐⭐⭐ — Core technique is a combination of CLIP losses, relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across four modalities, user studies, and extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rich visualizations.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play design holds practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](../../ICCV2025/3d_vision/tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[NeurIPS 2025\] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)
- [\[NeurIPS 2025\] LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](langsplatv2_high-dimensional_3d_language_gaussian_splatting_with_450_fps.md)
- [\[NeurIPS 2025\] MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)

</div>

<!-- RELATED:END -->
