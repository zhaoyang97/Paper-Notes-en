---
title: >-
  [Paper Note] UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation
description: >-
  [CVPR 2026][Image Restoration][Low-light pose estimation] UDAPose achieves a 56.4% AP improvement on the low-light hard set by combining stable diffusion-based low-light image synthesis (with preserved high-frequency low-light characteristics) and a dynamic attention control module (adaptively balancing visual cues and pose priors).
tags:
  - CVPR 2026
  - Image Restoration
  - Low-light pose estimation
  - domain adaptation
  - stable diffusion
  - attention control
  - high-frequency injection
date: 2026-05-08
content_hash: 712abb8a1113f792
---

# UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2604.10485](https://arxiv.org/abs/2604.10485)
**Code**: VMIL/UDAPose
**Area**: Image Restoration
**Keywords**: Low-light pose estimation, domain adaptation, stable diffusion, attention control, high-frequency injection

## TL;DR

UDAPose achieves a 56.4% AP improvement on the low-light hard set by combining stable diffusion-based low-light image synthesis (with preserved high-frequency low-light characteristics) and a dynamic attention control module (adaptively balancing visual cues and pose priors).

## Background & Motivation

**State of the Field**: Human pose estimation performs well under normal lighting but degrades significantly in low-light conditions. Annotating low-light datasets is extremely costly, making domain adaptation an appealing alternative.

**Limitations of Prior Work**: (1) Manual augmentations (e.g., Gaussian noise) oversimplify real low-light noise, which comprises complex components such as photon noise, thermal noise, and quantization noise; (2) Learning-based image translation methods (CycleGAN/StyleID) fail to preserve high-frequency low-light characteristics; (3) Modern one-stage pose estimators query image features via cross-attention but continue to over-rely on image features even when visual cues are unreliable under low-light conditions.

**Root Cause**: The effectiveness of domain adaptation depends on the fidelity of synthesized low-light images, yet existing methods are either overly simplistic or discard critical high-frequency low-light features. Furthermore, pose models themselves lack the ability to fall back on pose priors when visual information degrades.

**Paper Goals**: (1) Synthesize training data that preserves high-frequency low-light characteristics; (2) Enable pose models to adaptively balance visual cues and pose priors.

**Starting Point**: Use stable diffusion as the generative backbone to extract and inject high-frequency features from unannotated low-light reference images; modify the fusion mechanism of DETR-like pose estimators.

**Core Idea**: DHF preserves high-frequency low-light features → LCIM injects them at multiple scales → DCA adaptively controls the balance between visual cues and pose priors.

## Method

### Overall Architecture

During training, a stable diffusion model converts annotated normal-light images into low-light versions (inheriting their annotations), with DHF and LCIM injecting realistic low-light features. The DCA module replaces the rigid summation in the pose estimator. At inference, the model is applied directly to real low-light images.

### Key Designs

1. **DC High-Pass Filter (DHF)**:

    - Function: Extracts and preserves high-frequency information from low-light images.
    - Mechanism: The high-pass filtered image $I_{HP}$ has a mean close to zero; directly clipping it to $[0,1]$ discards negative dark-region information. DHF resolves this by realigning the mean: $I_{DHF} = I_{HP} + (mean(I_{LL}) - mean(I_{HP}))$, ensuring $mean(I_{DHF}) = mean(I_{LL})$ and reducing information loss during clipping.
    - Design Motivation: The SD encoder expects inputs in the $[0,1]$ range; directly clipping negative high-frequency values discards critical dark-region noise patterns.

2. **Low-light Characteristic Injection Module (LCIM)**:

    - Function: Injects high-frequency low-light features into the decoding process at multiple scales.
    - Mechanism: Features $\{z_1,...,z_4\}$ are extracted at different scales of the VAE encoder from the DHF-processed high-frequency image, then processed by lightweight convolutions and injected additively at corresponding decoder scales: $\hat{I}'_{LL} \leftarrow d_{final}(d_4(d_3(d_2(d_1(z_0)+f_1)+f_2)+f_3)+f_4)$. Channel statistics are aligned at the end.
    - Design Motivation: Multi-scale injection ensures fine-grained low-light noise is rendered at appropriate spatial resolutions. LCIM is trained under a reconstruction objective but captures transferable noise patterns.

3. **Dynamic Attention Control (DCA) Module**:

    - Function: Adaptively balances image visual cues and pose priors.
    - Mechanism: In DETR-like pose estimators, $\mathbf{Q}_{pose}$ (pose priors) and $\mathbf{Q}_{image}$ (visual cues) are typically summed directly. Analysis reveals that under low-light conditions the ratio $\|\mathbf{Q}_{image}\|_2/\|\mathbf{Q}_{pose}\|_2$ remains approximately constant (≈1.7) even when keypoints are invisible. DCA implements adaptive weighting via concatenation → lightweight network → sigmoid gating.
    - Design Motivation: Rigid summation causes visual cues to persistently dominate; unreliable visual features under low-light conditions lead to erroneous predictions.

### Loss & Training

LCIM is trained with MSE and a frequency-domain loss: $\mathcal{L}_\mathcal{D} = \mathcal{L}_{MSE}(I, \hat{I}) + \lambda\mathcal{L}_{freq}(I, \hat{I})$, where the frequency-domain loss uses sinusoidal weighting to emphasize mid-to-high frequencies. The pose model is trained on synthesized low-light data with normal-light annotations.

## Key Experimental Results

### Main Results

| Dataset | Metric | UDAPose | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| ExLPose-test LL-H | AP | +10.1 | Previous best | 56.4% |
| EHPT-XC (cross-dataset) | AP | +7.4 | Previous best | 31.4% |

### Ablation Study

| Configuration | AP | Note |
|------|-----|------|
| w/o DHF | Drops | High-frequency information lost |
| w/o LCIM | Drops | Low-light features not injected |
| w/o DCA | Drops | Visual cues persistently dominate |
| Gaussian noise substitute | Much lower | Manual augmentation insufficiently realistic |
| CycleGAN substitute | Lower | Over-darkening and lighting artifacts |
| Full UDAPose | Best | All three components work synergistically |

### Key Findings

- DHF's mean alignment is simple yet critical — omitting it causes substantial loss of dark-region high-frequency information.
- DCA enables the model to automatically fall back on pose priors when keypoints are invisible, significantly improving prediction of difficult keypoints.
- Cross-dataset evaluation (EHPT-XC) validates the generalization capability of the synthesized data.

## Highlights & Insights

- **Simplicity of DHF**: A single mean-alignment operation resolves the high-frequency information preservation problem — minimal yet highly effective.
- **DCA exposes a design flaw in DETR-like architectures**: The fragility of rigid summation under degraded conditions is a general issue.
- **No low-light annotations required**: Noise patterns are extracted solely from unannotated low-light reference images, lowering the barrier for practical deployment.

## Limitations & Future Work

- Relies on stable diffusion as the generative backbone; SD weights are not needed at inference but are required during training.
- LCIM may lack sufficient low-light references in extremely dark scenarios.
- The gating mechanism of DCA may require adaptation for different pose estimator architectures.

## Related Work & Insights

- **vs. ELLA**: ELLA simulates low-light conditions with Gaussian white noise, oversimplifying real noise patterns.
- **vs. CycleGAN/StyleID**: Learning-based translation methods alter global appearance but discard high-frequency low-light details.

## Rating

- Novelty: ⭐⭐⭐⭐ The DHF+LCIM+DCA three-component design offers targeted and well-motivated innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ The 56.4% AP improvement is highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis (e.g., Frobenius norm ratio analysis) is rigorous and in-depth.
- Value: ⭐⭐⭐⭐ Directly applicable to real-world low-light scenarios such as security surveillance.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[CVPR 2026\] BluRef: Unsupervised Image Deblurring with Dense-Matching References](bluref_unsupervised_image_deblurring_with_dense-matching_references.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](raw-domain_degradation_models_for_realistic_smartphone_super-resolution.md)
- [\[CVPR 2026\] IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE](ia_clahe_image_adaptive_clip_limit.md)
- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)

<!-- RELATED:END -->
