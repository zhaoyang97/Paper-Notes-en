---
title: >-
  [Paper Note] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking
description: >-
  [CVPR 2025][3D Vision][Watermarking] This paper proposes 3D-GSW, the first robust digital watermarking method designed specifically for 3D Gaussian Splatting. It enhances watermark robustness by removing redundant Gaussians and splitting Gaussians in high-frequency regions via Frequency-Guided Densification (FGD). Combined with a gradient mask and wavelet sub-band loss to maintain rendering quality, 3D-GSW achieves superior watermark robustness and rendering quality across th…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Watermarking"
  - "3D-GS"
  - "Frequency-Guided Densification"
  - "Gradient Mask"
  - "DWT"
date: 2026-05-08
content_hash: 1463a4d05b50d426
---

# 3D-GSW: 3D Gaussian Splatting for Robust Watermarking

**Conference**: CVPR 2025  
**arXiv**: [2409.13222](https://arxiv.org/abs/2409.13222)  
**Code**: [https://kuai-lab.github.io/cvpr20253dgsw/](https://kuai-lab.github.io/cvpr20253dgsw/)  
**Area**: 3D Vision / Digital Watermarking / 3D Gaussian Splatting  
**Keywords**: Watermarking, 3D-GS, Frequency-Guided Densification, Gradient Mask, DWT  

## TL;DR
This paper proposes 3D-GSW, the first robust digital watermarking method designed specifically for 3D Gaussian Splatting. It enhances watermark robustness by removing redundant Gaussians and splitting Gaussians in high-frequency regions via Frequency-Guided Densification (FGD). Combined with a gradient mask and wavelet sub-band loss to maintain rendering quality, 3D-GSW achieves superior watermark robustness and rendering quality across the Blender, LLFF, and Mip-NeRF 360 datasets.

## Background & Motivation
3D Gaussian Splatting (3D-GS) is becoming the mainstream representation for 3D content creation due to its real-time rendering and high quality. With its increasing commercial use, copyright protection has become an urgent demand. Existing NeRF watermarking methods (such as WateRF and CopyRNeRF) cannot be directly applied to 3D-GS because: (1) Pre-trained 3D-GS contains a large number of redundant Gaussians, leading to high computational overhead for watermark embedding and fragile embedding (Gaussians that do not contribute to rendering might also carry watermark signals, leading to weak signals); (2) The parameter structure of explicit representation differs significantly from the implicit network weights of NeRF.

## Core Problem
How can a robust digital watermark be embedded into a 3D-GS model such that the correct watermark message can be extracted from images rendered from any perspective, while remaining robust to both image-level attacks (noise, cropping, JPEG compression, etc.) and model-level attacks (removing/cloning Gaussians, parameter noise) without degrading rendering quality and real-time rendering speed?

## Method

### Overall Architecture
Fine-tune the pre-trained 3D-GS to embed the watermark. The workflow is: Pre-trained 3D-GS $\rightarrow$ FGD preprocessing (removing redundant Gaussians + splitting Gaussians in high-frequency regions) $\rightarrow$ Constructing the gradient mask $\rightarrow$ Fine-tuning optimization (message loss + reconstruction loss + LPIPS loss + wavelet sub-band loss) $\rightarrow$ Watermarked 3D-GS. During extraction: Rendered image $\rightarrow$ DWT transform to extract the LL2 sub-band $\rightarrow$ Pre-trained HiDDeN decoder to extract the message.

### Key Designs
1. **Frequency-Guided Densification (FGD)**: This consists of two phases. Phase 1: Measures each Gaussian's contribution $V_\pi$ to the rendering quality via an auxiliary loss function, and removes redundant Gaussians with extremely small contributions ($V_\pi < 10^{-8}$) (reducing by ~28%). This ensures that the remaining Gaussians significantly impact the rendering, making the embedded watermark more robust. Phase 2: Performs DFT on patches of the rendered image, selects the top-K% patches with the highest high-frequency intensity, and splits Gaussians with smaller contributions within these patches into smaller Gaussians to enhance the rendering quality of high-frequency details. Since the number of removed Gaussians is far larger than that of split ones, the total count decreases, thereby improving rendering FPS.

2. **Gradient Mask**: The FGD-processed 3D-GS already has high rendering quality. During fine-tuning, parameter changes must be minimized. An exponentially decaying gradient mask $z = \frac{1}{e^{|\theta|^\beta}}$ (after normalization) is designed to propagate smaller gradients to Gaussians with large parameter values. $\beta=4$ achieves the optimal accuracy-quality trade-off. No masking is applied to the positions $\mu$ (since their gradients are already close to zero).

3. **Wavelet Sub-band Loss**: Since FGD modifies high-frequency regions, high-frequency details must be protected. DWT decomposes both the rendered image and the original image, and L1 loss is calculated only on the high-frequency sub-bands {LH, HL, HH} to specifically enhance rendering fidelity in high-frequency regions.

4. **DWT Low-frequency Sub-band Watermark Embedding**: Computes 2-level DWT on the rendered image, and feeds the LL2 (low-frequency) sub-band into a pre-trained HiDDeN decoder to extract the message. The low-frequency sub-band is more robust to image attacks (JPEG compression, noise, etc.).

### Loss & Training
$$\mathcal{L}_{total} = \lambda_{rec}\mathcal{L}_1(I_w, I_o) + \lambda_{lpips}\mathcal{L}_{LPIPS} + \lambda_w\mathcal{L}_{wavelet} + \lambda_m\mathcal{L}_{BCE}(M, M')$$
$\lambda_{rec}=1, \lambda_{lpips}=0.2, \lambda_w=0.3, \lambda_m=0.4$. Adam optimizer, 2-10 epochs. The HiDDeN decoder is frozen after being pre-trained on COCO.

## Key Experimental Results

### Rendering Quality and Watermark Accuracy (32-bit, Average of Three Datasets)

| Method | Bit Acc↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|----------|-------|-------|--------|
| StegaNeRF+3D-GS | 93.15 | 32.68 | 0.953 | 0.049 |
| WateRF+3D-GS | 93.42 | 30.49 | 0.956 | 0.050 |
| 3D-GSW w/o FGD | 94.60 | 34.27 | 0.975 | 0.047 |
| **3D-GSW (Ours)** | **97.37** | **35.08** | **0.978** | **0.043** |

### Image Attack Robustness (32-bit)

| Attack | StegaNeRF | WateRF | 3D-GSW |
|------|-----------|--------|--------|
| No Attack | 93.15 | 93.42 | **97.37** |
| Gaussian Noise | 54.48 | 73.85 | ~90+ |
| JPEG Compression | 73.28 | 80.58 | ~95+ |
| Cropping 40% | 75.87 | 82.32 | ~90+ |

### Model Attack Robustness (32-bit)

| Attack | StegaNeRF | WateRF | 3D-GSW |
|------|-----------|--------|--------|
| Remove 20% Gaussians | 60.24 | 80.58 | **87.99** |
| Clone 20% Gaussians | 75.56 | 82.32 | **89.87** |
| Parameter Noise | 61.82 | 73.85 | ~90+ |

### Efficiency (Mip-NeRF 360, 64-bit)

| Method | Embedding Time | FPS | Storage |
|------|---------|-----|------|
| WateRF+3D-GS | 6h 47m | 56.65 | 833.89MB |
| StegaNeRF+3D-GS | 58h 56m | 56.65 | 833.89MB |
| **3D-GSW** | **21m 03s** | **72.68** | **640.21MB** |

### Ablation Study
- **FGD**: Without it, the bit accuracy drops by ~3%. Including it also improves FPS (+28% on Mip-NeRF 360) and reduces storage.
- **Gradient Mask**: Without it, PSNR drops from 35.08 to 33.26 (-1.82), showing a significant degradation in rendering quality.
- **Wavelet Sub-band Loss**: Without it, PSNR drops from 35.08 to 33.56 (-1.52), showing degraded high-frequency details.
- **All Removed**: PSNR drops to 29.96, and the quality collapses.
- **Optimal DWT Level 2**: Level 1/3/4 all perform worse than Level 2 in either bit accuracy or PSNR.

## Highlights & Insights
- **Frequency-Guided Gaussian Management**: FGD simultaneously addresses three issues: reducing redundancy to improve efficiency, enhancing watermark robustness, and maintaining rendering quality through high-frequency splitting. It kills three birds with one stone.
- **Exponentially Decaying Gradient Mask**: Cleverly handles the zero-value issue in 3D-GS parameters (normal mask would divide by zero), while ensuring minimal parameter variation.
- **Both Model and Image Attack Robustness**: For the first time systematically evaluates attacks on the 3D-GS model itself (removing/cloning Gaussians, parameter noise adding), which is more comprehensive than methods that only consider image post-processing.
- **Massive Efficiency Advantages**: Embedding time is reduced from several hours to 21 minutes, FPS is boosted by 28%, and storage is reduced by 23%.

## Limitations & Future Work
- The pre-trained decoder needs to be individually trained for each bit length (though it only needs to be trained once).
- Not robust to 3D-GS compression attacks—all methods lose watermarks after compression (bit accuracy drops to ~50%), which remains a common challenge in radiance field watermarking research.
- Only embeds binary messages, without extending to multi-modal information such as embedding images.
- Relies on a fixed HiDDeN architecture as the decoder.

## Related Work & Insights
- **WateRF**: SOTA in NeRF watermarking, embedding messages in DWT low frequencies. 3D-GSW inherits the DWT concept but adds FGD and exponential gradient mask tailored to 3D-GS properties, achieving 4.6 dB higher PSNR and 4% higher bit accuracy.
- **StegaNeRF**: A steganography method prioritizing imperceptibility over robustness. Extremely fragile to various attacks when applied to 3D-GS, and takes an extremely long embedding time (59 hours vs. 21 minutes).
- **CopyRNeRF**: A pioneer in NeRF watermarking, but only supports implicit NeRF and cannot handle explicit 3D-GS.

## Inspirations & Connections
- FGD's strategy of "cleaning up redundancy before executing the task" can be generalized to other 3D-GS fine-tuning scenarios (e.g., style transfer, editing).
- Analyzing 3D-GS contributions in the frequency domain is a promising inspiration for 3D-GS compression and pruning research.

## Rating
- Novelty: ⭐⭐⭐⭐ First watermarking method designed specifically for 3D-GS, with creative FGD design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets + three bit lengths + image & model attacks + efficiency analysis + rich ablations + failure cases + 4D extensions.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, complete formulas.
- Value: ⭐⭐⭐⭐ Addresses the practical need for copyright protection in commercial 3D-GS, though vulnerability to compression attacks limits some applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [\[ICLR 2026\] CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting](../../ICLR2026/3d_vision/compmarkgs_robust_watermarking_for_compressed_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] Robust3DGSW: Toward Robust Watermarking for Quantization-Aware 3D Gaussian Splatting](../../CVPR2026/3d_vision/robust3dgsw_toward_robust_watermarking_for_quantization-aware_3d_gaussian_splatt.md)

</div>

<!-- RELATED:END -->
