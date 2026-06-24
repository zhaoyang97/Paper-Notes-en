---
title: >-
  [Paper Note] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement
description: >-
  [ECCV 2024][Image Generation] A universal frequency-disentangled learning paradigm is proposed. By leveraging Laplacian decomposition and low-frequency consistency constraints, it decouples low-frequency (illumination recovery) and high-frequency (denoising) enhancement into two independent sub-tasks. With only 88K additional parameters, it delivers up to 7.68dB PSNR improvement across 6 SOTA low-light enhancement models.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: d06163a81bbd16ed
---

# Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement

**Conference**: ECCV 2024  
**arXiv**: [2409.01641](https://arxiv.org/abs/2409.01641)  
**Area**: Image Generation

## TL;DR

A universal frequency-disentangled learning paradigm is proposed. By leveraging Laplacian decomposition and low-frequency consistency constraints, it decouples low-frequency (illumination recovery) and high-frequency (denoising) enhancement into two independent sub-tasks. With only 88K additional parameters, it delivers up to 7.68dB PSNR improvement across 6 SOTA low-light enhancement models.

## Background & Motivation

Low-light image enhancement (LLIE) faces the challenge of coupled optimization of low-frequency (illumination recovery) and high-frequency (noise removal) components. Existing methods usually address both types of degradation within a unified framework. However, low-frequency adjustments can amplify noise, while high-frequency restoration can affect illumination intensity recovery, leading to sub-optimal results.

**Key Challenge**: How to design a universal frequency disentanglement paradigm that can (1) seamlessly integrate with existing LLIE methods, (2) boost frequency restoration capabilities, and (3) require minimal computational overhead?

Unlike existing frequency decomposition methods, this work not only decomposes image frequencies but, more importantly, **decouples the optimization process of low-frequency and high-frequency components**, achieving effective disentangled learning through a low-frequency consistency loss.

## Method

### Overall Architecture

A two-stage framework:
1. **Coarse Phase**: The ACCA module primarily restores low-frequency information (illumination), producing a preliminary enhanced result $I_l$.
2. **Coarse-to-Fine Phase**: The LDRM utilizes Laplacian decomposition representations to perform fine-grained high-frequency restoration by combining the original input and the coarse result.

### Key Designs

**ACCA (Adaptive Convolution Composition Aggregation) Module**:
- Dual-branch structure: local branch (W-CCA) + global ISP branch.
- W-CCA splits 2D features into non-overlapping patches through step-wise convolutions and uses tensor decomposition techniques (combining three 1D tensors) to generate a 3D Omni-similarity map, achieving efficient spatial-channel aggregation.
- The computational complexity is $O(4HWC + 2HWC^2/s)$, which scales linearly with image resolution.
- With only 88K parameters, it outperforms Retinexformer on LOL-v2 by 1dB using only 5.5% of its parameters.

**LDRM (Laplacian Disentangled Restoration Model)**:
- Utilizes a Laplacian pyramid to decompose the image into multi-scale high-frequency and low-frequency components.
- Integrates into SOTA models by only modifying their initial and final convolutional layers.
- Takes stacked Laplacian decomposition maps of the original input and the coarse estimation to generate the enhanced Laplacian output.
- Reconstructs the final output via inverse Laplacian transform, with $K=4$ levels yielding the optimal performance.

### Loss & Training

Total loss function: $L_{total} = L_r + \alpha \cdot L_i$

- **Reconstruction Loss** $L_r$: The sum of the L1 norms of the multi-scale predictions and their corresponding ground truth decomposition maps.
- **Low-Frequency Consistency Loss** $L_i$: Constrains the coarsest level feature map of the LDRM output to be consistent with the low-frequency component from the coarse phase, defined as $L_i = \|m_d^K - m_l^K\|_1$.

The low-frequency consistency loss is the core of achieving frequency-disentangled optimization—it forces LDRM to preserve the recovered low-frequency information from the coarse phase, allowing it to focus strictly on high-frequency enhancement.

## Key Experimental Results

### Main Results

Improvements on 6 SOTA models across 5 benchmarks (PSNR(dB)/SSIM):

| Method | LOL-v2 | SID | SDSD-in | SDSD-out | SMID |
|------|--------|-----|---------|----------|------|
| MIR-Net → MIR-Net-De | +4.17/+0.062 | +3.34/+0.075 | +3.76/+0.036 | +2.32/+0.047 | +1.16/+0.072 |
| Restormer → Restormer-De | +4.62/+0.066 | +2.49/+0.045 | **+6.11/+0.091** | **+7.68/+0.102** | +2.14/+0.058 |
| LLFlow → LLFlow-De | +1.70/+0.020 | +2.92/+0.064 | +5.09/+0.034 | +8.83/+0.057 | +1.53/+0.017 |
| SNR → SNR-De | +2.52/+0.023 | +0.68/+0.042 | +0.87/+0.007 | +3.32/+0.031 | +1.99/+0.017 |
| Retinexformer → Retinexformer-De | +1.41/+0.041 | +0.20/+0.014 | +0.77/+0.013 | +3.67/+0.028 | +1.70/+0.013 |
| Diff-L → Diff-L-De | +4.98/+0.081 | +2.03/+0.104 | +4.80/+0.031 | +4.14/+0.056 | +1.31/+0.034 |

Extra model overhead: only +88K parameters, +2.53 GFLOPS, and +0.008s inference time (for $256 \times 256$ input), accounting for 0.2%–5.5% of the original model parameters.

### Ablation Study

Effectiveness validation of ACCA and low-frequency consistency loss (based on Restormer, LOL-v2):

| Configuration | PSNR/SSIM | Params/FLOPs |
|------|-----------|------------|
| Restormer (Baseline) | 19.94/0.827 | 26.13M/144.25G |
| + ACCA, w/o $L_i$ | 20.21/0.837 | 26.22M/146.78G |
| + ACCA + $L_i$ (Ours) | **24.56/0.893** | 26.22M/146.78G |

Restormer improvements when replacing ACCA with different coarse estimation models:

| Coarse Method | PSNR Gain | SSIM Gain |
|----------|----------|----------|
| ZeroDCE | +1.59 | +0.034 |
| Star | +2.46 | +0.039 |
| PairLIE | +2.42 | +0.033 |
| IAT | +4.03 | +0.060 |
| ACCA (Ours) | **+4.62** | **+0.066** |

### Key Findings

1. The low-frequency consistency loss $L_i$ is crucial for the performance gain; removing it drops the PSNR from 24.56 to 20.21.
2. The framework is effective for CNNs, Transformers, flow-based models, and diffusion models, proving its strong universality.
3. Despite having only 88K parameters, ACCA achieves performance that matches or even outperforms the coarse adjustment of large LLIE models.
4. The coarse adjustment module can be flexibly replaced; different lightweight models can all bring significant improvements.

## Highlights & Insights

- **Plug-and-Play Paradigm**: Boosts any LLIE model with almost zero extra cost, boasting an elegant framework design.
- **Key Insight of Frequency Disentanglement**: The low-frequency consistency constraint decomposes a complex joint optimization problem into two simpler sub-problems.
- **Efficient Design of ACCA**: Tensor decomposition techniques decompose the regression of the 3D similarity map into three 1D tensors, significantly reducing the computational workload.
- A PSNR boost of up to 7.68dB is an exceptionally rare margin of improvement in the image restoration field.

## Limitations & Future Work

- The proposed method relies on the frequency decomposition assumption of the Laplacian pyramid, which may yield limited effectiveness for degradations lacking distinct frequency features.
- It requires a two-stage training process (ACCA first, followed by LDRM), which complicates the training workflow.
- The improvements are relatively moderate for models that are already well-optimized in the frequency domain (e.g., SNR, Retinexformer).

## Rating

- **Novelty**: 8/10 — The frequency-disentangled optimization paradigm is simple yet effective, and the low-frequency consistency loss is an ingenious design.
- **Technical Depth**: 7/10 — Clear theoretical analysis, but the main contribution lies at the paradigm level rather than model architecture innovation.
- **Experimental Thoroughness**: 9/10 — Comprehensive validation across 6 baselines and 5 datasets, with detailed ablation studies.
- **Value**: 9/10 — The plug-and-play property makes real-world deployment extremely easy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Shedding More Light on Robust Classifiers under the lens of Energy-based Models](shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)

</div>

<!-- RELATED:END -->
