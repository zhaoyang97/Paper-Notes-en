---
title: >-
  [Paper Note] OptiFusion: Towards Universal Computational Aberration Correction in Photographic Cameras
description: >-
  [CVPR 2025][Image Restoration][Computational Aberration Correction] By extending OptiFusion to automatically design 120 diverse lenses, this work proposes the ODE comprehensive evaluation metric and a large-scale benchmark. Systematically comparing 24 algorithms, it reveals that CNN models provide the best speed-accuracy trade-off for aberration correction, counter-intuitively outperforming Transformers.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Computational Aberration Correction"
  - "Lens Design"
  - "Benchmark"
  - "CNN vs Transformer"
  - "Optical Aberration"
  - "Universal Correction"
date: 2026-05-08
content_hash: 474dd191a3024ee1
---

# OptiFusion: Towards Universal Computational Aberration Correction in Photographic Cameras

**Conference**: CVPR 2025  
**arXiv**: [2603.12083](https://arxiv.org/abs/2603.12083)  
**Code**: None  
**Area**: Image Restoration / Optical Aberration Correction  
**Keywords**: Computational Aberration Correction, Lens Design, Benchmark, CNN vs Transformer, Optical Aberration, Universal Correction

## TL;DR

By extending OptiFusion to automatically design 120 diverse lenses, this work proposes the ODE comprehensive evaluation metric and a large-scale benchmark. Systematically comparing 24 algorithms, it reveals that CNN models provide the best speed-accuracy trade-off for aberration correction, counter-intuitively outperforming Transformers.

## Background & Motivation

### Background

**Background**: **Importance of Computational Aberration Correction (CAC)**: Camera lenses inevitably introduce various aberrations (e.g., spherical aberration, chromatic aberration, field curvature). Computational correction compensates for these optical defects at the software level.

**Limitations of Prior Work**: Existing CAC methods are typically trained for specific lenses, demonstrating poor generalization across different lenses. Thus, retraining is required for every new lens.

**Lack of Comprehensive Benchmark**: Existing literature usually evaluates methods on only a few lenses, lacking a large-scale and diverse benchmark for comparison.

**Incomplete Evaluation Metrics**: PSNR/SSIM cannot fully reflect the quality of aberration correction, with a notable lack of metrics specialized for spatial uniformity and chromatic aberration.

**Unknowns of CNN vs Transformer**: While Transformers gradually dominate most low-level vision tasks, a systematic comparison in the CAC domain is still lacking.

**Core Idea**: Establishing a large-scale, diverse lens benchmark alongside comprehensive evaluation metrics to systematically reveal the best practices in the CAC domain.

### Goal

**Goal**: ### Overall Architecture

This paper is primarily a benchmark paper, with core contributions including:

1. **Automated Lens Design**: Extending OptiFusion to automatically design 120 diverse lenses, covering different focal lengths, apertures, and fields of view.
2. **ODE Evaluation Metric**: A unified evaluation framework combining PSNR/SSIM/OIQE with spatial uniformity and chromatic aberration metrics.
3. **Comprehensive Comparison of 24 Algorithms**: Covering various architectures, including CNN, Transformer, and hybrid models.

## Method

### Overall Architecture

This paper is primarily a benchmark paper, with core contributions including:

1. **Automated Lens Design**: Extending OptiFusion to automatically design 120 diverse lenses, covering different focal lengths, apertures, and fields of view.
2. **ODE Evaluation Metric**: A unified evaluation framework combining PSNR/SSIM/OIQE with spatial uniformity and chromatic aberration impairment metrics.
3. **Comprehensive Comparison of 24 Algorithms**: Covering various architectures, including CNN, Transformer, and hybrid models.

### Key Design 1: OptiFusion Automated Lens Design

- Automatically designs a diverse group of lenses based on ray tracing and computational optics principles.
- The 120 lens designs cover various parameter combinations from wide-angle to telephoto, and large to small apertures.
- Generates corresponding aberration PSF data for each lens, which is used for training and evaluation.
- Addresses the challenge of obtaining diverse lens data in the real world.

### Key Design 2: ODE Evaluation Metric

- Integrates standard image quality metrics (PSNR, SSIM, OIQE).
- Evaluates spatial uniformity: Measures the consistency of correction between the center and the edges of the image (aberrations are typically more severe at the edges).
- Includes specific chromatic aberration metrics: Measures the correction quality of longitudinal and lateral chromatic aberrations.
- The ODE metric provides a more comprehensive reflection of the actual performance of CAC.

### Key Design 3: Discovery of Three Key Performance Factors

Through systematic experiments, three key factors affecting CAC performance are identified:
1. **Prior Utilization** (Optical Prior / Image Prior)
2. **CNN-based Architectures**
3. **Training Strategies** (Learning rate, data augmentation, loss functions)

## Key Experimental Results

### Main Results

| Architecture Type | Representative Method | Comprehensive ODE | Speed | Remarks |
|---------|---------|---------|------|---------|
| CNN | Restormer-CNN, etc. | **Optimal** | **Fastest** | Best speed-accuracy |
| Transformer | SwinIR, etc. | Medium | Slower | Accuracy slightly lower than CNN |
| Hybrid | — | Medium | Medium | No clear advantage |
| Traditional | Wiener, etc. | Lower | Fastest | Weak |

- Among the 24 algorithms, CNN-based methods provide the best speed-accuracy trade-off.

### Ablation Study

| Factor | Impact | Description |
|------|------|------|
| Optical Prior Utilization | Significant Positive | Methods with PSF knowledge perform significantly better |
| CNN vs Transformer | CNN is better | Counter-intuitive, but reproducible in this domain |
| Training Strategy | Significant Impact | Training details are more important than architecture |
| Lens Diversity | Critical | Training on diverse lenses significantly improves generalization |

### Key Findings

- **CNN Outperforms Transformer**: In the CAC task, CNN models outperform Transformers in both accuracy and speed, likely because CAC is more dependent on local information rather than long-range dependencies.
- The impact of training strategies is greater than that of architectural choices.
- Optical prior (PSF knowledge) is the most critical underlying factor.
- Large-scale training on 120 lenses significantly enhances generalization across different lenses.

## Highlights & Insights

1. **Highly Systematic**: High credibility of findings backed by a large-scale comparison of 24 algorithms across 120 lenses.
2. **Counter-intuitive Finding**: CNNs outperform Transformers in CAC, which is contrary to the trend in most vision tasks, providing valuable insights.
3. **Practicality of the ODE Metric**: The comprehensive metric reflects actual performance much better than a single PSNR.
4. **Automated Lens Design Approach**: Resolves the data bottleneck in optical AI research.

## Limitations & Future Work

- All 120 lenses are computationally designed, which may differ from real-world physical lenses.
- Real-world factors such as lens aging and temperature variance are not considered.
- The weight settings of the ODE metric may need to be adjusted based on specific application scenarios.
- The conclusions may be limited by current model architectures and training strategies; future advanced Transformers might still surpass CNNs.

## Related Work & Insights

- **Aberration Correction**: Unlike traditional deblurring or super-resolution tasks, CAC needs to handle specific issues such as spatially varying PSFs and chromatic aberrations.
- **Image Restoration Benchmark**: Similar to composite benchmarks like Rain13K, but it is the first of its kind in the optics domain.
- **Insights**: For other image restoration tasks strongly correlated with physical priors, CNNs may still remain a better choice.

## Rating

- **Novelty**: ⭐⭐⭐☆ — Primarily a benchmark contribution with limited methodological innovation
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely thorough, featuring 24 algorithms across 120 lenses
- **Writing Quality**: ⭐⭐⭐⭐ — Deep analysis and clear conclusions
- **Value**: ⭐⭐⭐⭐ — Direct guiding value for fields such as computational photography and mobile photography
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](../../CVPR2026/image_restoration/unicac_universal_computational_aberration_correction_benchmark.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](dpir_dual_prompting_restoration_dit.md)
- [\[CVPR 2025\] DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables](dnlut_ultra-efficient_color_image_denoising_via_channel-aware_lookup_tables.md)
- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)
- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)

</div>

<!-- RELATED:END -->
