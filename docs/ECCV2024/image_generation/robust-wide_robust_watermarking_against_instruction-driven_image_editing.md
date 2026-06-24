---
title: >-
  [Paper Note] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing
description: >-
  [ECCV 2024][Image Generation][Robust Watermarking] This paper proposes Robust-Wide, the first robust watermarking method against instruction-driven image editing. The core innovation is the Partial Instruction-driven Denoising Sampling Guidance (PIDSG) module, which opens the gradient flow of the last $k$ steps of the editing process during training. This forces the watermark to be embedded into semantic-aware areas, achieving a bit error rate (BER) of only about 2.6% for 64-…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Robust Watermarking"
  - "Image Editing"
  - "InstructPix2Pix"
  - "Semantic-level Perturbation"
  - "Diffusion Models"
date: 2026-05-08
content_hash: c46b4966bd99e866
---

# Robust-Wide: Robust Watermarking against Instruction-driven Image Editing

**Conference**: ECCV 2024  
**arXiv**: [2402.12688](https://arxiv.org/abs/2402.12688)  
**Code**: [https://github.com/hurunyi/Robust-Wide](https://github.com/hurunyi/Robust-Wide)  
**Area**: Image Generation  
**Keywords**: Robust Watermarking, Image Editing, InstructPix2Pix, Semantic-level Perturbation, Diffusion Models

## TL;DR
This paper proposes Robust-Wide, the first robust watermarking method against instruction-driven image editing. The core innovation is the Partial Instruction-driven Denoising Sampling Guidance (PIDSG) module, which opens the gradient flow of the last $k$ steps of the editing process during training. This forces the watermark to be embedded into semantic-aware areas, achieving a bit error rate (BER) of only about 2.6% for 64-bit watermarks after editing.

## Background & Motivation

**Background**: Instruction-driven image editing models such as InstructPix2Pix allow users to quickly edit images through text instructions. While serving as a convenient tool, they are also prone to malicious usage (e.g., generating fake news, style theft, etc.). Watermarking is a widely used approach for tracking and traceability.

**Limitations of Prior Work**: Existing SOTA watermarking methods (HiDDeN, MBRS, PIMoG, SepMark, etc.) predominantly focus on robustness against pixel-level perturbations (e.g., JPEG compression, blurring, geometric transformations). However, instruction-driven editing drastically alters images at the **semantic level** (such as changing expressions, replacing objects, and style transfer). Consequently, the bit error rate (BER) of these methods after editing approaches 50%—equivalent to random guessing, rendering them completely ineffective.

**Key Challenge**: Information embedded via pixel-level watermarking cannot survive semantic-level editing because editing models regenerate most of the image content. The watermark must be embedded into "semantic anchor" regions that remain unchanged during the editing process.

**Goal**: How to simulate perturbations from instruction-driven editing within an end-to-end training framework, enabling the watermark encoder to learn to embed information into semantically robust regions.

**Key Insight**: The editing process involves numerous denoising sampling steps, making the direct propagation of gradients through all steps computationally prohibitive in terms of GPU memory. The authors observe that propagating gradients through only the last $k$ steps is sufficient for the encoder to perceive features of the editing process.

**Core Idea**: Selectively propagate gradient flows through the last $k$ steps of the editing sampling process, coupled with the injection of diverse instructions, to force the watermark encoder to embed information into semantically robust regions.

## Method

### Overall Architecture
Robust-Wide follows the classic Encoder-Noise Layer-Decoder framework. Given an original image $I_{ori}$ and a random $L$-bit message $m$, the encoder $E_m$ generates the watermarked image $I_{wm}$. The PIDSG module simulates instruction-driven editing to produce the edited image $I_{wm}^{edit}$. The decoder $E_x$ then extracts the watermark message from the edited image. These three components are jointly trained in an end-to-end manner.

### Key Designs

1. **Watermark Encoder $E_m$ (U-Net structure)**:

    - **Function**: Embeds the $L$-bit binary message into the image to generate a watermarked image that is visually consistent with the original image.
    - **Mechanism**: The message of shape $1 \times \sqrt{L} \times \sqrt{L}$ is first expanded to $C \times H \times W$ via transposed convolutions, and then concatenated with the original image before being fed into the U-Net.
    - **Embedding Constraints**: Pixel-level $L_2$ loss $L_{em_1} = L_2(I_{ori}, I_{wm})$ and latent-level $L_2$ loss $L_{em_2} = L_2(\mathcal{E}(I_{ori}), \mathcal{E}(I_{wm}))$ are applied to ensure that the watermarked image remains editable.

2. **PIDSG (Partial Instruction-driven Denoising Sampling Guidance)**:

    - **Function**: Simulates semantic-level perturbations from instruction-driven editing during training, allowing gradients to propagate back from the decoder to the encoder.
    - **Mechanism**: All parameters of InstructPix2Pix are frozen. Out of $T$ total sampling steps during editing, gradients are truncated in the first $T-k$ steps to obtain the partially denoised latent variable $Z_k$, while gradients are retained in the remaining $k$ steps to make the entire pipeline differentiable. A CLIP encoder processes the editing instructions $Ins$ to guide the sampling.
    - **Design Motivation**: (a) Flowing gradients through all $T$ steps is computationally infeasible due to GPU memory limitations; (b) the final $k$ steps are sufficient to capture key features of the editing process; (c) injecting diverse instructions forces the encoder to focus on robust semantic regions rather than specific editing patterns.
    - **Difference from Prior Methods**: Existing noise layers only simulate pixel-level perturbations such as JPEG and blur. PIDSG introduces the complete diffusion editing process into training for the first time.

3. **Watermark Decoder $E_x$ (Residual block structure)**:

    - **Function**: Extracts the embedded message from either the edited or unedited watermarked image.
    - **Key Findings**: The decoder must be trained simultaneously on both edited and unedited watermarked images. Training solely on edited images fails to converge because the decoder cannot locate the watermarked regions.

### Loss & Training
The total loss is formulated as:

$$L_{total} = L_{em_1} + 0.001 \cdot L_{em_2} + 0.1 \cdot L_{ex_1} + 1.0 \cdot L_{ex_2}$$

where $L_{ex_1} = \text{MSE}(m, E_x(I_{wm}^{edit}))$ is the post-editing extraction loss, and $L_{ex_2} = \text{MSE}(m, E_x(I_{wm}))$ is the pre-editing extraction loss.

## Key Experimental Results

### Main Results

The training set contains 20k image-instruction pairs, and the test set contains 1.2k samples plus 1.44k real-world samples.

| Method | Image Size | Watermark Bits | BER% (No Edit) | BER% (Post-Edit) | PSNR↑ | SSIM↑ |
|------|----------|----------|-------------|-------------|-------|-------|
| DWT-DCT | 512 | 32 | 11.94 | 49.23 | 38.71 | 0.966 |
| DWT-DCT-SVD | 512 | 32 | 0.03 | 47.57 | 38.65 | 0.973 |
| RivaGAN | 512 | 32 | 0.63 | 40.53 | 40.61 | 0.972 |
| MBRS | 256 | 256 | 0.00 | 46.77 | 43.98 | 0.987 |
| PIMoG | 256 | 64 | 0.00 | 49.96 | 35.32 | 0.921 |
| SepMark | 256 | 128 | 0.01 | 28.15 | 36.43 | 0.919 |
| **Robust-Wide** | **512** | **64** | **0.00** | **2.66** | **41.91** | **0.991** |

All baseline methods exhibit a post-editing BER close to 50% (approx. random guessing), whereas Robust-Wide achieves only 2.66%.

### Robustness to Pixel-level Perturbations (Unseen during Training)

| Perturbation Type | Pre-editing Perturbation | Post-editing Perturbation (Watermarked) | Post-editing Perturbation (Edited) |
|----------|-----------|-------------------|-------------------|
| None | 2.66% | 0.00% | 2.66% |
| JPEG | 2.73% | 0.00% | 2.79% |
| Gaussian Noise | 3.07% | 0.07% | 6.05% |
| Brightness | 12.29% | 0.47% | 9.48% |
| Noise + Denoise | 8.64% | 3.91% | 9.37% |

### Key Findings
- **PIDSG is the Core**: Removing PIDSG causes the post-editing BER to surge from 2.66% to 50.16%, rendering the method completely ineffective.
- **Inherent Pixel-level Robustness**: Even though traditional perturbations like JPEG compression and blurring are never seen during training, Robust-Wide remains robust to them. This suggests that semantic-level embedding naturally covers pixel-level robustness.
- **Generalization across Editing Models**: Effective on ControlNet-InstructPix2Pix (BER 0.96%), MagicBrush (BER 9.34%), Inpainting, and DDIM Inversion.
- **Robustness to Sequential Editing**: Watermarks can still be accurately extracted after 3 rounds of sequential editing.
- Visualizations demonstrate that the watermark is primarily embedded in the main subject outlines and conceptual background regions.

## Highlights & Insights
- **Exquisite Design of PIDSG**: Solves the non-differentiable challenge within diffusion sampling. Propagating gradients through only the last $k$ steps saves GPU memory while remaining sufficient to guide semantic-aware embedding. This "partial gradient propagation" concept can be transferred to any scenario that requires incorporating non-differentiable generative processes into training.
- **Semantic-level Robustness Encompasses Pixel-level Robustness**: A profound insight—if a watermark can survive extensive semantic modifications, minor pixel-level perturbations will naturally not affect it.
- **Insight from $L_{ex_2}$**: The decoder must be trained on unedited images as well to converge. This reveals that the decoder needs to first "learn to locate the watermark" before identifying it in complex post-editing scenarios.

## Limitations & Future Work
- The BER rises significantly when editing is extremely severe (e.g., $s_I = 1$, where the image is almost entirely regenerated).
- Training relies on a specific editing model (InstructPix2Pix), leading to reduced generalization performance on models with larger architectural differences, such as MagicBrush.
- Evaluated on a single GPU (A6000) with a fixed image size of $512 \times 512$; the applicability to higher resolutions has not been verified.
- PIDSG introduces heavy training overhead, as it requires running $k$ steps of denoising sampling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first watermarking method tailored for instruction-driven editing, featuring a highly innovative PIDSG module.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceedingly comprehensive evaluations spanning multiple editing models, sampling configurations, pixel-level perturbations, and sequential editing scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and articulation of motivations.
- Value: ⭐⭐⭐⭐⭐ Resolves an urgent problem in the field of AI safety, carrying significant value for watermarking and AI content governance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](../../CVPR2026/image_generation/rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] Shedding More Light on Robust Classifiers under the lens of Energy-based Models](shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)
- [\[CVPR 2025\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](../../CVPR2025/image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[ECCV 2024\] RegionDrag: Fast Region-Based Image Editing with Diffusion Models](regiondrag_fast_region-based_image_editing_with_diffusion_models.md)

</div>

<!-- RELATED:END -->
