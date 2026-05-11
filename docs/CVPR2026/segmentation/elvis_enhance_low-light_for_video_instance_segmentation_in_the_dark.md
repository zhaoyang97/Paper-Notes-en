---
title: >-
  [Paper Note] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark
description: >-
  [CVPR 2026][Segmentation][Low-light video instance segmentation] ELVIS proposes the first low-light video instance segmentation (VIS) framework…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Low-light video instance segmentation"
  - "synthetic low-light pipeline"
  - "degradation estimation"
  - "domain adaptation"
  - "enhancement decoder"
date: 2026-05-08
content_hash: 647132a00d5b7442
---

# ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark

**Conference**: CVPR 2026
**arXiv**: [2512.01495](https://arxiv.org/abs/2512.01495)
**Code**: [joannelin168.github.io/research/ELVIS](https://joannelin168.github.io/research/ELVIS)
**Area**: Image Segmentation
**Keywords**: Low-light video instance segmentation, synthetic low-light pipeline, degradation estimation, domain adaptation, enhancement decoder

## TL;DR

ELVIS proposes the first low-light video instance segmentation (VIS) framework, comprising a physics-driven synthetic low-light video pipeline (with motion blur modeling), a calibration-free degradation parameter estimation network (VDP-Net), and an enhancement decoder integrated into the VIS architecture for degradation-content decoupling. It achieves gains of +3.7 AP and +2.8 AP on synthetic and real low-light videos, respectively.

## Background & Motivation

Video instance segmentation under low-light conditions is an important yet underexplored problem with broad applications in autonomous driving, wildlife conservation, and surveillance. The field faces multiple challenges:

**Lack of annotated data**: Degradations in low-light conditions make both manual and automatic annotation extremely difficult, and no public benchmark specifically for low-light VIS exists.

**Inadequate synthetic pipelines**: Existing synthetic low-light methods are primarily designed for images and neglect motion blur degradation caused by long shutter exposure times in low-light videos.

**Lack of robustness in existing VIS methods**: State-of-the-art VIS methods are not designed for low-light degradations and perform poorly even after fine-tuning on synthetic low-light data.

**Limitations of two-stage approaches**: Enhancement-then-segmentation pipelines are constrained by the immaturity of low-light video enhancement itself.

**Mechanism**: Design an end-to-end domain adaptation framework incorporating a physically realistic synthetic low-light video pipeline and a degradation-content decoupling mechanism to adapt existing VIS models to low-light scenarios.

## Method

### Overall Architecture

The ELVIS framework consists of two major components:
1. **Unsupervised synthetic low-light video pipeline** (green panel): degrades normal-light videos into low-light counterparts.
2. **Enhanced instance segmentation module**: integrates an enhancement decoder head into the VIS network to decouple degradation from scene content.

### Key Designs

1. **Synthetic low-light video degradation model**: Fully models the physical process from normal light to low light.

   Final degradation formula: $X^{low} = Deg(X^{high}, \phi) = H * (2^\epsilon X^{high}) + N$

   Three categories of degradation are modeled:
    - **Illumination adjustment**: The image is first converted to the XYZ color space to ensure linearity, then brightness is reduced by exposure value $\epsilon$: $X' = 2^\epsilon X$.
    - **Blur degradation** (introduced for the first time in synthetic low-light video pipelines): The joint effect of motion blur and defocus blur is modeled using a multivariate Gaussian distribution with only three parameters $(\sigma_{Hx}, \sigma_{Hy}, \theta_H)$. When $\sigma_{Hx} = \sigma_{Hy}$, only defocus blur is present.
    - **Physical noise**: Four types — read noise (Gaussian), shot noise (Poisson), quantization noise (uniform distribution), and banding noise (Gaussian, supporting both horizontal and vertical directions).

   Degradation parameter vector: $\phi = \{\epsilon, \sigma_r, K, \lambda_q, \sigma_b, \theta_b, \sigma_{Hx}, \sigma_{Hy}, \theta_H\}$

2. **VDP-Net (Video Degradation Parameter Network)**:

    - Estimates degradation parameters $\phi$ from real low-light videos in an **unsupervised** manner, without requiring camera calibration.
    - Architecture: lightweight ResNet-18 backbone + temporal fusion convolutional block + two MLP prediction heads.
    - The two prediction heads separately handle: one for exposure and noise (global degradation), and one for blur (local degradation).
    - **Unsupervised training strategy**: Degradation parameters are uniformly sampled to synthesize low-light inputs; the network learns to infer parameters from degraded videos.
    - Loss function: $\mathcal{L}_{total} = \lambda_1 \|\phi - \phi'\|_1 + \lambda_2 (1 - \cos(|\theta_H - \theta_H'|))$, where the cosine angle loss handles the periodicity of blur orientation.

3. **Enhancement decoder integration**:

    - An enhancement decoder head is integrated into the segmentation module of Mask2Former.
    - The decoder uses a multi-scale deformable attention pixel decoder (10 Transformer decoder layers + bilinear upsampling) to reconstruct normal-light frames.
    - An auxiliary L1 loss (clean frame vs. reconstructed frame) is added during training to guide the network to decouple scene content from degradation in the latent feature space.
    - At inference, only the segmentation output is used; the decoder introduces no additional computational overhead.

### Loss & Training

- During VIS training, degradation parameters are sampled from a pre-generated real degradation parameter set $\Phi$ (estimated from four datasets: SDSD, DID, BVI-RLV, and LMOT) to synthesize low-light versions of training videos on the fly.
- The auxiliary L1 loss from the enhancement decoder is jointly optimized with the original VIS segmentation loss.
- VDP-Net is trained using uniformly sampled degradation parameters within expert-defined reasonable upper bounds.

## Key Experimental Results

### Main Results

**Synthetic low-light YouTube-VIS 2019 validation set**

| Method | Backbone | ELVIS | AP | AP50 | AP75 |
|--------|----------|-------|----|------|------|
| MinVIS | ResNet-50 | ✗ | 36.4 | 57.3 | 36.4 |
| MinVIS | ResNet-50 | ✓ | 37.2 | 57.0 | 39.6 |
| GenVIS | ResNet-50 | ✗ | 39.1 | 58.4 | 42.7 |
| GenVIS | ResNet-50 | ✓ | **41.0** | **59.8** | **46.2** |
| DVIS++ | ResNet-50 | ✗ | 38.8 | 59.9 | 42.8 |
| DVIS++ | ResNet-50 | ✓ | **42.5** | **63.8** | **46.6** |
| DVIS++ | ViT-L | ✗ | 55.2 | 77.2 | 62.1 |
| DVIS++ | ViT-L | ✓ | **56.9** | **78.7** | **65.3** |

Maximum gain: **+3.7 AP** (DVIS++ R50).

**Real low-light video evaluation (LMOT-S)**

| Method | ELVIS | AP | AP50 | AR10 |
|--------|-------|----|------|------|
| GenVIS R50 | ✗ | 6.6 | 14.5 | 9.8 |
| GenVIS R50 | ✓ | 6.7 | 15.5 | 12.1 |
| DVIS++ ViT-L | ✗ | 10.0 | 21.4 | 13.1 |
| DVIS++ ViT-L | ✓ | **10.5** | **22.6** | **14.5** |

### Ablation Study

**Comparison with two-stage baselines (ELVIS-S and LMOT-S)**

| Method | ELVIS-S AP | LMOT-S AP |
|--------|-----------|-----------|
| SDSD-Net (enhance→segment) | 46.7 | 2.5 |
| StableLLVE (enhance→segment) | 57.3 | 3.9 |
| DarkIR (enhance→segment) | 55.9 | 3.8 |
| **ELVIS** | **58.0** | **6.7** |

Outperforms the best two-stage method on LMOT-S by **+2.8 AP**.

**Synthetic pipeline comparison**

| Synthetic Pipeline | ELVIS-S AP | LMOT-S AP |
|--------------------|-----------|-----------|
| Lv et al. | 53.5 | 5.1 |
| Cui et al. | 51.1 | 5.7 |
| Ours (random ϕ) | 39.9 | 4.7 |
| **Ours (VDP-Net ϕ)** | **54.5** | **6.6** |

VDP-Net-estimated parameters outperform random sampling by +14.6 AP / +1.9 AP, demonstrating the importance of matching the real degradation distribution.

### Key Findings

- ELVIS yields consistent improvements across all VIS methods and backbone architectures, demonstrating the generality of the framework.
- The enhancement decoder significantly boosts AP75 (the strictest metric) through degradation-content decoupling, indicating the greatest improvement in fine-grained segmentation quality.
- Incorporating blur modeling into the synthetic pipeline is critical — existing pipelines overlook this inherent degradation in low-light video.
- VDP-Net's unsupervised training strategy is effective at extracting real degradation distributions from real low-light videos.

## Highlights & Insights

- **Physics-driven low-light video synthesis**: This is the first synthetic pipeline to model motion blur (multivariate Gaussian kernel), addressing the gap in prior methods that only account for noise. The constraint of blur orientation to $[0, \pi]$ is a thoughtful design that accounts for the bidirectionality of motion blur kernels.
- **Degradation-content decoupling**: The auxiliary reconstruction task via the enhancement decoder compels the VIS backbone to learn degradation-invariant feature representations, a more elegant approach than two-stage methods.
- **Calibration-free degradation estimation**: VDP-Net requires no camera metadata (model, ISO, etc.) and can be applied to any dataset. The cosine angle loss for handling periodic parameters is an elegant design choice.
- **Zero inference overhead**: The enhancement decoder is used only during training and introduces no additional computational cost at inference time.

## Limitations & Future Work

- Real low-light VIS evaluation is limited in scale (ELVIS-S contains only 250 frames; LMOT-S uses pseudo-labels), and a larger-scale real low-light VIS benchmark is needed.
- The synthetic pipeline does not model spatially correlated artifacts introduced by the ISP (e.g., compression, demosaicing, in-camera denoising), which may be significant in real-world scenarios.
- VDP-Net assumes uniform degradation parameters throughout a video clip, whereas real low-light video degradation may vary spatially and temporally.
- Validation is currently limited to Mask2Former-based VIS methods; applicability to other architectures (e.g., tracking-based methods) remains unexplored.
- Absolute AP on real low-light data remains low (<11%), indicating that low-light VIS remains an extremely challenging problem.

## Related Work & Insights

- The synthetic low-light pipeline design is generalizable to other low-light downstream tasks (detection, tracking, depth estimation, etc.).
- The degradation-content decoupling approach can be applied to downstream task adaptation under other degradation conditions (haze, rain, underwater, etc.).
- VDP-Net's unsupervised degradation estimation can be used independently to provide scene-adaptive degradation parameters for low-light enhancement methods.
- Complementary to RAW-domain methods (which require raw sensor data), ELVIS operates directly in the sRGB domain, making it more practical.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First low-light VIS framework; novel inclusion of blur modeling in the synthetic pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across multiple VIS methods, backbones, and both synthetic and real settings, though real data scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Physical model derivation is clear and the framework is presented comprehensively.
- **Value**: ⭐⭐⭐⭐ — Fills the gap in low-light VIS and provides a reusable synthetic pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2026\] Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)
- [\[CVPR 2026\] MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events](a_dataset_of_medication_images_with_instance_segme.md)

</div>

<!-- RELATED:END -->
