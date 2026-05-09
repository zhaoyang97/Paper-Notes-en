---
title: >-
  [Paper Note] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction
description: >-
  [CVPR 2026][Image Restoration][Radio interferometric imaging] Building upon the POLISH framework, this work proposes POLISH+ and POLISH++, which employ a patch-based training-and-stitching strategy and an arcsinh-based nonlinear transform to achieve radio interferometric image reconstruction and super-resolution under wide-field (12,960×12,960 pixels) and high-dynamic-range ($\sim 10^6$) conditions. The paper also presents the first demonstration that deep learning methods can super-resolve strong gravitational lens systems.
tags:
  - CVPR 2026
  - Image Restoration
  - Radio interferometric imaging
  - deep learning deconvolution
  - super-resolution
  - high dynamic range
  - strong gravitational lensing
date: 2026-05-08
content_hash: 12b8514446c0a79f
---

# POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction

**Conference**: CVPR 2026  
**arXiv**: [2603.09162](https://arxiv.org/abs/2603.09162)  
**Code**: None (extension of the POLISH framework)  
**Area**: Radio Astronomical Image Reconstruction / Image Deconvolution  
**Keywords**: Radio interferometric imaging, deep learning deconvolution, super-resolution, high dynamic range, strong gravitational lensing

## TL;DR

Building upon the POLISH framework, this work proposes POLISH+ and POLISH++, which employ a patch-based training-and-stitching strategy and an arcsinh-based nonlinear transform to achieve radio interferometric image reconstruction and super-resolution under wide-field (12,960×12,960 pixels) and high-dynamic-range ($\sim 10^6$) conditions. The paper also presents the first demonstration that deep learning methods can super-resolve strong gravitational lens systems.

## Background & Motivation

Radio interferometric imaging achieves high-resolution astronomical imaging by synthesizing large apertures from antenna arrays, with image deconvolution as its core problem. The forthcoming **DSA-2000** (1,650 antennas) will produce images exceeding $10,000 \times 10,000$ pixels with a dynamic range of $\sim 10^6$ at data throughput rates exceeding 80 Tb/s.

Limitations of existing methods:

**CLEAN**: The standard approach, but resolution is limited to the PSF scale with no super-resolution capability; poorly suited to recovering complex source morphologies.

**RML optimization methods**: High computational cost of iterative solving makes them unsuitable for DSA's real-time requirements.

**Existing deep learning methods**:
   - Small image sizes (< 1,024 pixels) and low dynamic ranges (< $10^3$)
   - Tested only on simple Gaussian sources, without addressing complex celestial morphologies (e.g., strong lenses)
   - PSF mismatch (calibration errors) not considered

Core challenges: (1) extremely high dimensionality imposed by wide fields of view; (2) extremely high dynamic ranges of real skies.

## Method

### Overall Architecture

The POLISH family employs an end-to-end CNN (based on the WDSR architecture) that directly learns the mapping from low-resolution dirty images to high-resolution clean sky images. This work extends that framework with:
- **POLISH+**: Patch-based training and inference
- **POLISH++**: POLISH+ augmented with an arcsinh nonlinear transform

Forward model: $I_{\text{dirty}} = [k * (I_{\text{true}} + n)]{\downarrow_s}$, where $k$ is the PSF and $s=2$ is the downsampling factor.

### Key Designs

1. **Patch-wise Processing**: The full 12,960×12,960 field-of-view image is divided into $J$ non-overlapping 324×324 patches, each forming an individual training pair. Key insight:

    - Patches cropped from full-field dirty images contain **cross-patch contamination** (PSF sidelobe artifacts from bright sources in neighboring patches)
    - This is fundamentally different from training directly on small images — the network must learn to handle these non-local artifacts
    - 18 full-field images → 28,800 training pairs (containing 6 million detectable galaxies)
    - At inference, each patch is predicted independently and then stitched back into the full field

   Design Motivation: GPU memory cannot accommodate training on complete images of $10^8$ pixels; patch-wise processing makes large-scale training feasible.

2. **Arcsinh Nonlinear Transform (AsinhStretch)**: Addresses the extreme dynamic range of $10^4 \sim 10^6$:
    $$\text{AsinhStretch}(x; a) = \frac{\operatorname{arcsinh}(x/a)}{\operatorname{arcsinh}(1/a)}$$
    - Logarithmic in form: compresses pixel values spanning multiple orders of magnitude into a comparable range
    - Handles positive and negative values (dirty images may contain negative values), making it appropriate for interferometric imaging
    - Training loss is computed in the transformed space:
    $$\theta^* = \arg\min_\theta \frac{1}{NJ}\sum_{i,j} \|\text{G}_\theta(\text{AsinhStretch}(I_{\text{dirty}}^{[j]}; a_d)) - \text{AsinhStretch}(I_{\text{true}}^{[j]}; a_t)\|_1$$
    - At inference, original intensity scales are recovered via the inverse transform $\text{AsinhStretch}^{-1}$

   Design Motivation: When training directly in the original intensity space, the $\ell_1$ loss is dominated by a small number of bright sources, leading to poor recovery of faint sources.

3. **Model Robustness and Adaptability**:

    - Robustness: A model trained only on ideal PSFs maintains visual consistency when faced with randomly perturbed PSFs ($\gamma \in [0, 30]$)
    - Adaptability: Fine-tuning converges more than 5× faster than training from scratch (11 vs. 57 epochs), enabling rapid adaptation to different observing conditions

### Loss & Training

- Loss: $\ell_1$ loss (computed in the AsinhStretch-transformed space)
- Optimizer: Adam, lr 0.0001, batch size 12
- $a_{\text{dirty}} = a_{\text{true}} = 0.1$
- Training data: T-RECS sky simulations, 18 training images + 5 test images
- Noise: Gaussian noise $\sigma = 1\,\mu\text{Jy}$

## Key Experimental Results

### Main Results

| Method | Precision↑ | Recall↑ | F1↑ | Major-axis FWHM RMSE↓ | Minor-axis FWHM RMSE↓ |
|--------|-----------|---------|-----|----------------------|----------------------|
| CLEAN | 0.3612 | 0.2220 | 0.2750 | 1.0046″ | 0.7862″ |
| POLISH | 0.5560 | 0.4612 | 0.5042 | 0.9642″ | 0.3219″ |
| POLISH+ | 0.8744 | 0.5751 | 0.6938 | 0.4335″ | 0.1889″ |
| POLISH++ | **0.8433** | **0.6142** | **0.7107** | 0.4654″ | 0.2056″ |

Note: POLISH++ improves F1 by 159% over CLEAN, with more than 2× improvement in shape estimation accuracy.

### Ablation Study

| Configuration | Key Metric | Note |
|--------------|-----------|------|
| POLISH (full-image training) | F1=0.5042 | Baseline |
| POLISH+ (patch-based training) | F1=0.6938 | Patch strategy yields large gains |
| POLISH++ (patch + arcsinh) | F1=0.7107 | Nonlinear transform further improves recall by +4% |
| PSF perturbation γ=0→30 | PSNR drops but visually consistent | Robust to calibration errors |
| Fine-tuning vs. training from scratch | 11 vs. 57 epochs | 5× speedup |

### Key Findings

- **Super-resolution capability**: POLISH++ accurately estimates galaxy shape parameters below the PSF scale (≈3.3″), where CLEAN fails entirely
- **Strong lens discovery**: A lens-finding CNN trained on POLISH++ super-resolved images can lower the discoverable lens threshold from 3× the PSF resolution limit to near the PSF scale, yielding an approximately **10×** increase in the expected strong lens yield from DSA surveys
- **Dynamic range**: POLISH++ successfully handles a dynamic range of $\sim 10^6$, three orders of magnitude higher than existing DL methods (< $10^3$)
- CLEAN still outperforms POLISH in flux estimation (model-based methods retain better absolute flux calibration)

## Highlights & Insights

1. **Deployment-oriented design**: Rather than optimizing benchmark PSNR on small images, the method is designed for DSA's practical requirements (12,960×12,960 pixels, $10^6$ dynamic range)
2. **Discovery of cross-patch contamination**: Dirty image patches contain PSF sidelobe artifacts from neighboring bright sources — a unique domain-specific insight
3. **Scientific application value**: Super-resolution directly enables strong gravitational lens discovery, increasing DSA's lens yield by 10×
4. **Honest limitation analysis**: The paper explicitly acknowledges that CLEAN remains superior for flux estimation, and that DL methods lack an explicit flux calibration mechanism
5. **From robustness to adaptability**: The paper not only validates robustness under PSF mismatch but also demonstrates rapid fine-tuning adaptability

## Limitations & Future Work

1. Flux estimation accuracy is inferior to CLEAN; no explicit flux calibration mechanism is present
2. Operations are confined to the image plane (not the visibility domain), potentially losing phase information
3. Training data are based on T-RECS simulations, which may not fully represent real skies
4. Patch stitching may introduce discontinuities at patch boundaries (not discussed in detail)
5. Future directions: flux post-processing calibration, end-to-end visibility-domain methods, simulation of more complex celestial morphologies

## Related Work & Insights

- **POLISH (Connor et al. 2022)**: The baseline method of this work, tested only on 2,048-pixel images with a dynamic range of $\sim 10^2$
- **R2D2**: An unrolled network approach supporting 512-pixel images and a dynamic range of $5 \times 10^5$
- **CLEAN**: The standard method in radio astronomy, limited by PSF resolution
- Insight: The "killer application" of deep learning deconvolution in astronomical imaging is the scientific discovery capability enabled by super-resolution

## Rating

- Novelty: ⭐⭐⭐ Technically an engineering improvement over existing methods (patch-based training + nonlinear transform); the core contribution lies in engineering scale and application impact
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers source detection, shape estimation, flux estimation, strong lens discovery, and PSF robustness/adaptability
- Writing Quality: ⭐⭐⭐⭐ Seamless integration of astronomical background and DL methodology; problem formulation is clear
- Value: ⭐⭐⭐⭐ High practical value for DSA deployment; the 10× improvement in strong lens yield constitutes an important scientific contribution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2026\] Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)

</div>

<!-- RELATED:END -->
