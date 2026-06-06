---
title: >-
  [Paper Note] Beyond the Ground Truth: Enhanced Supervision for Image Restoration
description: >-
  [CVPR 2026][Image Restoration][Supervision Enhancement] This paper proposes to enhance the perceptual quality of suboptimal ground-truth images in existing datasets via super-resolution combined with frequency-domain ada…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Supervision Enhancement"
  - "Frequency-Domain Mixup"
  - "Super-Resolution"
  - "Output Refinement Network"
  - "Perceptual Quality"
date: 2026-05-08
content_hash: ae346eb081f819d7
---

# Beyond the Ground Truth: Enhanced Supervision for Image Restoration

**Conference**: CVPR 2026
**arXiv**: [2512.03932](https://arxiv.org/abs/2512.03932)  
**Code**: [Project Page](https://hij1112.github.io/beyond-the-ground-truth/)  
**Area**: Image Restoration
**Keywords**: Supervision Enhancement, Frequency-Domain Mixup, Super-Resolution, Output Refinement Network, Perceptual Quality

## TL;DR
This paper proposes to enhance the perceptual quality of suboptimal ground-truth images in existing datasets via super-resolution combined with frequency-domain adaptive mixing, and trains a lightweight Output Refinement Network (ORNet) that improves the perceptual quality of restoration outputs without modifying any pretrained restoration model.

## Background & Motivation
**Background**: Deep learning-based image restoration has achieved remarkable success under supervised training paradigms, yet model performance is fundamentally bounded by the quality of ground-truth (GT) images.

**Limitations of Prior Work**: GT images in real-world datasets are often **far from ideal** due to acquisition constraints:
   - Deblurring datasets (e.g., GoPro): GT frames are selected from video sequences and still contain residual camera shake.
   - Denoising datasets (e.g., SIDD): GT images are obtained by averaging multiple noisy frames, introducing blur.
   - Models trained on suboptimal GT inevitably inherit these artifacts.

**Existing Attempts**: Diffusion models can improve perceptual quality but incur high inference cost and tend to produce hallucinations.

**Core Idea**: Rather than improving model architectures, this work **enhances the supervision signal itself** — adaptively fusing the semantic structure of the original GT with the perceptual details of its super-resolved variants in the frequency domain to produce an "enhanced GT."

## Method

### Overall Architecture
1. **Supervision Enhancement Stage**: Original GT $I_0^{GT}$ → bicubic upsampling at $N$ scales → one-step diffusion super-resolution model → downsampling back to original resolution to obtain variants $\{I_i^{GT}\}_{i=1}^N$ → frequency-domain mixup to generate enhanced GT $\hat{I}^{GT}$.
2. **Output Refinement Stage**: Lightweight ORNet learns the mapping $I_0^{GT} \rightarrow \hat{I}^{GT}$ and is appended after any pretrained restoration model.

### Key Designs
1. **Frequency-Domain Mixup**:

    - Enhanced GT = $\mathcal{F}^{-1}\left(\sum_{i=0}^{N} M_i \odot \mathcal{F}(I_i^{GT})\right)$
    - $M_i$ is predicted by a conditional frequency mask generator, satisfying $\sum_i M_i = 1$.
    - **Why frequency domain instead of spatial domain**: Spatial mixing struggles to simultaneously preserve high-level semantic structure and enhance fine details. The frequency domain enables precise control — retaining the low-frequency components (semantics) from the original GT while selectively incorporating high-frequency components (details) from super-resolved variants.

2. **Conditional Frequency Mask Generator**:

    - $B$ predefined annular Gaussian basis masks $R_b$: $(R_b)_{h,w} = \exp(-(d(h,w)-\mu_b)^2/2\sigma_b^2)$
    - The network predicts coefficients $c_{i,b} = g(I_0^{GT}, \lambda)$
    - Final mask: $M_i = \text{softmax}_i(\sum_b c_{i,b} R_b)$
    - **Annular Gaussian design**: Bandpass control from low to high frequencies; the Gaussian shape ensures smooth transitions between frequency bands to avoid artifacts.
    - The network takes both RGB and FFT representations as input, jointly exploiting spatial and frequency-domain information.

3. **Output Refinement Network (ORNet)**:

    - Training objective: $\mathcal{L}_{ref} = \|R_\theta(I_0^{GT}, \lambda) - \hat{I}^{GT}\|_2^2$
    - Leverages the prior $R_\phi(I^{LQ}) \approx I_0^{GT}$, so ORNet only needs to learn the residual mapping from GT to enhanced GT.
    - **Model-agnostic**: Can be appended after any pretrained restoration model without modifying its architecture or retraining.

### Loss & Training
- Mask generator training: $\mathcal{L} = (1-\lambda)\mathcal{L}_{recon} + \lambda\mathcal{L}_{percep}$
    - $\mathcal{L}_{recon} = \|\hat{I}^{GT} - I_0^{GT}\|_2^2$ (preserves semantic consistency)
    - $\mathcal{L}_{percep} = -\sum_k \text{IQA}_k(\hat{I}^{GT})$ (improves perceptual quality using MUSIQ/MANIQA/TOPIQ)
    - $\lambda$ controls the fidelity–perception trade-off.

## Key Experimental Results

### Main Results (GoPro Deblurring + SIDD Denoising)

| Method | MUSIQ↑ | MANIQA↑ | TOPIQ↑ | LIQE↑ |
|------|--------|---------|--------|-------|
| AdaRevD | 45.49 | 0.5363 | 0.3393 | 1.566 |
| **+ORNet** | **64.25** | **0.5916** | **0.4880** | **2.429** |
| FFTformer | 46.47 | 0.5420 | 0.3456 | 1.613 |
| **+ORNet** | **64.57** | **0.5949** | **0.4924** | **2.466** |
| NAFNet (SIDD) | 22.73 | 0.3937 | 0.2458 | 1.219 |
| **+ORNet** | **35.87** | **0.4380** | **0.3776** | **1.959** |

### Ablation Study / OOD Robustness

| Setting | Method | MUSIQ↑ | LPIPS↓ | Note |
|------|------|--------|--------|------|
| +Gaussian blur σ=2.5 | FFTformer | 22.38 | 0.471 | OOD degradation |
| | +ORNet | **42.91** | **0.343** | Strong robustness |
| +White noise σ=9 | FFTformer | 30.14 | 0.446 | OOD degradation |
| | +ORNet | **41.88** | **0.407** | Removes residual artifacts |

### Key Findings
- ORNet yields **substantial gains** across all perceptual metrics (MUSIQ: 45→64, >40% improvement).
- Remains effective under OOD conditions (unseen degradation types during training), demonstrating that ORNet learns generalizable perceptual enhancement capabilities.
- VLM-based evaluations (VisualQuality-R1, Q-Insight) further confirm quality improvements.
- User studies additionally validate the perceptual superiority of enhanced GT and ORNet outputs.

## Highlights & Insights
- **Novel perspective**: Improving the supervision signal rather than the model architecture represents an important complementary paradigm for image restoration.
- Frequency-domain mixup elegantly mitigates hallucination artifacts from super-resolution — low frequencies are preserved from the original GT while high frequencies are selectively enhanced.
- The model-agnostic nature of ORNet offers strong practical value: train once, apply to any restoration model.
- OOD robustness suggests that ORNet learns generalizable perceptual enhancement beyond task-specific degradation correction.

## Limitations & Future Work
- Fidelity metrics such as PSNR/SSIM may exhibit slight degradation (perception–distortion trade-off).
- Performance depends on the quality of the pretrained super-resolution model.
- How to rigorously define the "correctness" of enhanced GT remains an open question.
- Extension to temporally sequential scenarios such as video restoration has not been explored.

## Related Work & Insights
- Complementary to diffusion-prior restoration methods such as StableSR: the latter replaces the entire restoration pipeline, whereas this work only enhances the supervision signal.
- The frequency-domain mixup concept is generalizable to other supervised learning tasks (e.g., GT enhancement for medical image segmentation).
- Offers methodological implications for dataset construction: GT quality can be systematically improved through post-processing.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Novel supervision-enhancement perspective with an elegant frequency-domain mixup design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple tasks (deblurring + denoising), multiple models, OOD settings, and user studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architectural diagrams and detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ High practical utility; ORNet serves as a plug-and-play module to boost existing systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)

</div>

<!-- RELATED:END -->
