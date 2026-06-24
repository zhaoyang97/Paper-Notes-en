---
title: >-
  [Paper Note] Asymmetric Mask Scheme for Self-supervised Real Image Denoising
description: >-
  [ECCV2024][Image Restoration][self-supervised denoising] Proposed the asymmetric mask scheme AMSNet, which utilizes a single mask during training and complementary multiple masks during inference, breaking the structural requirements and receptive field constraints of blind spot networks, and achieving SOTA performance in self-supervised real image denoising.
tags:
  - "ECCV2024"
  - "Image Restoration"
  - "self-supervised denoising"
  - "blind spot network"
  - "mask strategy"
  - "real image denoising"
  - "asymmetric scheme"
date: 2026-05-08
content_hash: fb9f1bbb0c4b830b
---

# Asymmetric Mask Scheme for Self-supervised Real Image Denoising

**Conference**: ECCV2024  
**arXiv**: [2407.06514](https://arxiv.org/abs/2407.06514)  
**Code**: [lll143653/amsnet](https://github.com/lll143653/amsnet)  
**Area**: Image Restoration  
**Keywords**: self-supervised denoising, blind spot network, mask strategy, real image denoising, asymmetric scheme

## TL;DR

Proposed the asymmetric mask scheme AMSNet, which utilizes a single mask during training and complementary multiple masks during inference, breaking the structural requirements and receptive field constraints of blind spot networks, and achieving SOTA performance in self-supervised real image denoising.

## Background & Motivation

Self-supervised denoising methods have attracted significant attention due to their exemption from paired training data, with the Blind Spot Network (BSN) representing the most classic paradigm. The core assumption of BSN is that noise is zero-mean and pixel-wise independent, preventing identity mapping (noise-to-noise) by excluding the center pixel via blind-spot convolution. However, BSN imposes strict constraints on network architectures:

1. **Restricted Receptive Field**: After blind-spot convolution, dilated convolutions or other specialized structures must be employed to further restrict the receptive field; otherwise, the center pixel information leaks back to the output via neighboring pixels, leading to identity mapping.
2. **Loss of Structural Information**: Excluding the center pixel inevitably damages structural details.
3. **Limited Choice of Denoisers**: State-of-the-art denoisers (such as Restormer and NAFNet) cannot be easily integrated directly into the BSN framework.

These limitations severely restrict the performance upper bound of BSN-based methods. Inspired by Masked Autoencoders (MAE), the authors explore whether masking operations can replace blind-spot convolutions to resolve identity mapping, thereby freeing the framework from structural limitations.

## Core Problem

How to prevent identity mapping in self-supervised denoising without restricting the receptive field and architecture of the denoising networks, thereby allowing the flexible adoption of high-performance modern denoisers?

## Method

### Training Phase: Single Mask Scheme

The core idea is to mask raw pixels directly at the input stage, fundamentally preventing them from participating in their own reconstruction process:

1. Generate a random binary mask matrix $M$ (where approximately 50% of the pixels are masked to zero) for the noisy image $I_N$.
2. Feed the masked image $M \odot I_N$ into the denoiser $D_E$.
3. Compute the reconstruction loss only on the masked locations (as indicated by the complementary mask $\tilde{M}$).

Mask self-supervised loss:

$$\mathcal{L}_m(M_s, I_s) = \|\tilde{M_s} \odot (D_E(M_s \odot I_s, \theta) - I_s)\|_1$$

Key point: Since the masked pixels are entirely reconstructed from surrounding unmasked pixels, identity mapping is inherently avoided, and **restricting the receptive field of the network is no longer necessary**.

### Handling Spatial Correlation of Real Noise

Real-world noise typically violates the pixel-wise independence assumption. Following AP-BSN, a Pixel Downsampling (PD) strategy is introduced: the input image is downshifted and sampled with a stride $s$ to obtain $s^2$ sub-samples $I_s$. PD breaks the spatial correlation of noise, rendering sub-samples virtually independent. Masks are then independently generated and applied to each sub-sample.

### Inference Phase: Multi Mask Scheme

During training, a single branch only restores a subset of the masked pixels. To achieve full-image denoising during inference, a Multi-branch Mask complementary Denoising Block (MMDB) is proposed:

- Employ $k$ denoising branches (default $k=2$), sharing the same denoiser $D_E$.
- The masks applied to the branches are mutually complementary and non-overlapping: $\sum_{i=1}^{k} \tilde{M}_s^i = \mathbb{I}$.
- The final denoised sub-sample is obtained by summing the outputs: $D_M(I_s) = \sum_{i=1}^{k} \tilde{M}_s^i \odot D_E(M_s^i \odot I_s, \theta)$.
- Finally, the original resolution is reconstructed via inverse pixel downsampling $P_s^{-1}$.

### Checkerboard Effect Elimination

The PD strategy disrupts structural consistency, often introducing checkerboard artifacts into the denoised outputs. A two-stage remedy is introduced:

1. **Prior Smoothing Loss $\mathcal{L}_p$**: Fine-tune the base model AMSNet-B with the total loss $\mathcal{L}_t = \lambda \mathcal{L}_p(I_{DN}) + \|I_{DN} - I_N\|_1$, where $\lambda=0.01$.
2. **Random Replacing Refinement Strategy $\mathcal{R}^3$**: Further suppress checkerboard patterns during inference.

This yields four model variants: AMSNet-B (Base), AMSNet-P (+ smoothing loss fine-tuning), AMSNet-B-E (+ refinement), and AMSNet-P-E (+ both, final version).

## Key Experimental Results

### Main Results (SIDD / DND / PolyU)

| Method | SIDD Val (PSNR/SSIM) | SIDD Bench | DND Bench |
|---|---|---|---|
| AP-BSN+$\mathcal{R}^3$ | 36.74/0.850 | 36.91/0.931 | 38.09/0.937 |
| LG-BPN+$\mathcal{R}^3$ | 37.31/0.886 | 37.28/0.936 | 38.43/0.942 |
| BNN-LAN | 37.39/0.883 | 37.41/0.934 | 38.18/0.939 |
| **AMSNet-P-E** | **37.93/0.895** | **37.87/0.941** | **38.70/0.947** |

AMSNet-P-E also achieves state-of-the-art performance on the PolyU dataset with 37.92 dB / 0.9645 SSIM.

### Ablation Study & Key Findings

- **Identity Mapping Validation**: When AP-BSN uses a denoiser with an unrestricted receptive field, its PSNR plummets to 20.91 dB (suffering from identity mapping). In contrast, AMSNet maintains 37.11 dB, proving that the input masking effectively prevents identity mapping.
- **Denoiser Versatility**: Restormer (37.93) > DeamNet (37.80) > NAFNet (37.10) > UNet (36.94) $\approx$ DnCNN (36.93). This demonstrates the freedom to easily integrate modern, powerful denoisers.
- **Optimal Masking Ratio**: A masking ratio of approximately 50% (corresponding to $k=2$ branches) yields the best performance.
- **Smoothing Loss Fine-Tuning**: Introducing $\mathcal{L}_t$ brings an improvement of $\approx$ 0.1 dB.

## Highlights & Insights

1. **Elegant Formulation**: Drawing inspiration from MAE, this work brings the masking concept to self-supervised denoising. Utilizing input masks instead of blind-spot convolutions completely liberates the network from receptive field and structural constraints.
2. **Asymmetric Train-Test Strategy**: Simple single masks are used during training to minimize optimization and training costs, while complementary multiple masks are applied during inference for seamless whole-image reconstruction.
3. **Denoiser Agnostic**: The framework allows modern architectures like Restormer and NAFNet to be incorporated in a plug-and-play manner, establishing high extensibility.
4. **Rigorous Ablation**: The identity mapping experiments explicitly illustrate the limitations of classic BSNs and highlight the core advantages of the proposed masking scheme.

## Limitations & Future Work

1. **Doubled Inference Overhead**: The $k=2$ setting requires two separate forward passes during inference, doubling the computational cost.
2. **Post-Processing for Artifacts**: The checkerboard artifacts originating from PD still necessitate extra smooth-loss fine-tuning and refinement strategies, adding complexity to the pipeline.
3. **Asymmetric PD Strides**: Running training with $P_5$ but inference with $P_2$ is highly empirical and requires heuristic tuning.
4. **Restricted to sRGB Denoising**: The generalization capability has not been verified on RAW image denoising or other low-level restoration tasks.
5. The masking ratio is fixed at 50%, leaving adaptive or learnable masking strategies unexamined.

## Related Work & Insights

| Aspect | BSN-based Methods (AP-BSN, LG-BPN) | AMSNet |
|---|---|---|
| Identity Mapping Avoidance | Blind-spot + dilated conv. to restrict receptive field | Input masks to directly obstruct identity mappings |
| Denoiser Constraints | Strictly restricted, standard convolutions cannot be used | Unconstrained, any off-the-shelf denoiser can be integrated |
| Real Noise Handling | PD + BSN | PD + Mask |
| Inference Cost | Single forward pass | $k$ forward passes (default 2) |
| SIDD Val PSNR | 36.74 / 37.31 | 37.93 |

Compared to older self-supervised baseline frameworks (e.g., Noise2Void, Self2Self), AMSNet provides a substantial performance leap in real-world scenarios, primarily driven by enabling the usage of advanced modern denoisers.

## Insights & Connections

- The success of this masking strategy demonstrates that key design principles of MAE are highly generalizable to image restoration tasks, pointing to potential future adoption in super-resolution and deblurring.
- The asymmetric train-test paradigm is highly powerful: simplifying the reconstruction task during training and assembling multiple complementary predictions during inference to obtain the final outcome.
- The checkerboard artifacts expose persistent drawbacks within the PD strategy; future works may explore alternative approaches to decouple spatial noise correlation without spatial downsampling.

## Rating
- **Novelty**: 8/10 — Seamlessly transfers MAE's masking concept to self-supervised denoising to lift the structural limitations of BSNs, with a clear and effective execution.
- **Experimental Thoroughness**: 8/10 — Multiple evaluation benchmarks, five different denoisers, and extensive ablation studies, particularly the identity mapping experiment.
- **Writing Quality**: 7/10 — Lucid explanations, though the formulation of some mathematical derivations could be further streamlined.
- **Value**: 7/10 — Offers a highly flexible framework and opens new paths for self-supervised restoration, though doubled inference cost remains a minor pain point for real-life applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising](../../CVPR2026/image_restoration/next-scale_prediction_a_self-supervised_approach_for_real-world_image_denoising.md)
- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2026\] Convexity-Aware Noise Calibration: A Self-Supervised Framework for Noise-Level-Unknown Image Denoising](../../CVPR2026/image_restoration/convexity-aware_noise_calibration_a_self-supervised_framework_for_noise-level-un.md)

</div>

<!-- RELATED:END -->
