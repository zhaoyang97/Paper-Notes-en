---
title: >-
  [Paper Note] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis
description: >-
  [CVPR 2026][Image Restoration][Computational Aberration Correction] This work constructs UniCAC, the first large-scale universal computational aberration correction (CAC) benchmark. It proposes the Optical Degradation Evaluator (ODE) to quantify aberration difficulty and provides a comprehensive evaluation of 24 image restoration and CAC algorithms. The study reveals how prior utilization, network architecture, and training strategies influence CAC performance.
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Computational Aberration Correction"
  - "Optical Degradation Evaluation"
  - "Benchmarking"
  - "Automatic Optical Design"
date: 2026-05-08
content_hash: c5588edcfd341702
---

# Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis

**Conference**: CVPR 2026  
**arXiv**: [2603.12083](https://arxiv.org/abs/2603.12083)  
**Code**: [https://github.com/XiaolongQian/UniCAC](https://github.com/XiaolongQian/UniCAC)  
**Area**: Image Restoration  
**Keywords**: Computational Aberration Correction, Optical Degradation Evaluation, Benchmarking, Automatic Optical Design, Image Restoration

## TL;DR

This work constructs UniCAC, the first large-scale universal computational aberration correction (CAC) benchmark. It proposes the Optical Degradation Evaluator (ODE) to quantify aberration difficulty and provides a comprehensive evaluation of 24 image restoration and CAC algorithms. The study reveals how prior utilization, network architecture, and training strategies influence CAC performance.

## Background & Motivation

1. **Background**: Computational aberration correction (CAC) is a classic problem in computational imaging. Existing methods typically target specific optical systems and require retraining for new lenses.
2. **Limitations of Prior Work**: There is a lack of comprehensive benchmarks covering diverse optical aberrations, which limits the development of universal CAC. Furthermore, traditional metrics like the RMS radius fail to accurately quantify the difficulty of CAC tasks.
3. **Key Challenge**: Universal CAC requires zero-shot generalization to unseen lenses. However, design parameters for commercial lenses are usually proprietary, making it difficult to build large-scale, diverse training and testing datasets.
4. **Goal**: (1) Construct a large-scale CAC benchmark including both spherical and aspherical lenses; (2) Propose a more reliable aberration quantization framework; (3) Systematically evaluate existing methods and summarize key findings.
5. **Key Insight**: Leverage Automatic Optical Design (AOD) methods to generate a large number of lens description files that adhere to physical constraints, bypassing the unavailability of commercial lens data.
6. **Core Idea**: By extending the OptiFusion automatic design method to generate a diverse lens library and proposing the ODE framework to quantify aberration severity, the UniCAC benchmark is established for thorough evaluation.

## Method

### Overall Architecture

UniCAC connects "data generation, difficulty measurement, and evaluation" into a unified pipeline. It addresses the lack of benchmarks by using extended AOD to batch-generate lenses with physical constraints. The proposed ODE assigns a "degradation severity" score to each lens, enabling stratified sampling. 120 lenses were sampled and categorized into 5 difficulty levels. 24 image restoration and CAC algorithms were evaluated on this standardized benchmark. The input consists of aberration-degraded images, and the output is the corrected clean image.

### Key Designs

**1. Extended Automatic Optical Design: Supplementing Unavailable Lenses with Physical Simulation**

The main obstacle for universal CAC is that commercial lens design files are proprietary. This work extends the OptiFusion method by redefining "spherical parameters" to incorporate aspherical parameters, allowing the search process to generate both types. Four key specifications—lens count, aperture position, half field-of-view (HFOV), and F-number—are explicitly locked, followed by a heuristic global search. Validation against Zemax ray tracing confirms a mean error of only 1μm, ensuring simulated aberrations match real optical behavior.

**2. Optical Degradation Evaluator (ODE): A Scalable Predictor for CAC Difficulty**

Traditional RMS spot radius correlates poorly with actual CAC difficulty ($R^2$ of only 0.30). ODE characterizes degradation across three orthogonal dimensions:

$$ODE = \lambda_{oiq} \cdot OIQ + \lambda_s \cdot U_s + \lambda_c \cdot U_c$$

Where $OIQ$ evaluates overall quality by blending fidelity metrics with MTF-based scores ($OIQ = \alpha \frac{PSNR}{50} + \beta \frac{SSIM-0.5}{0.5} + \gamma \cdot OIQE$); $U_s$ measures spatial uniformity across different fields of view; and $U_c$ measures chromatic consistency. Both $U_s$ and $U_c$ use $U_{s,c} = e^{-\sigma \cdot CV_{s,c}}$ to map variation coefficients to a 0–1 range. The resulting $R^2$ with CAC performance is 0.84, enabling much more accurate stratified sampling.

**3. Overall Performance (O.P.): Combining Fidelity, Optical, and Perceptual Metrics**

Existing metrics often fail to capture all aspects of CAC performance. O.P. aggregates six dimensions using empirical weights:

$$O.P. = 4 \times \frac{PSNR}{50} + 3 \times \frac{SSIM-0.5}{0.5} + 4 \times \frac{1-LPIPS}{0.4} + 3 \times OIQE + 1 \times \frac{100-FID}{100} + 1 \times ClipIQA$$

This allowed for a balanced comparison between regression-based methods (focusing on fidelity) and GAN/Diffusion-based methods (focusing on perception).

### Loss & Training

This benchmark evaluation covers three training paradigms: regression training for image fidelity (high PSNR/SSIM, average perception), GAN training for better perceptual quality, and diffusion training for enhanced perception, though it shows limited improvement in optical quality (OIQE). achieving high scores across all three dimensions simultaneously remains an open challenge.

## Key Experimental Results

### Main Results

| Method | Type | PSNR↑ | SSIM↑ | OIQE↑ | O.P.↑ |
|------|------|-------|-------|-------|-------|
| PART (Non-blind CAC) | Transformer + Reg | 28.10 | 0.866 | 0.608 | 1.494 |
| FOV-KPN (Blind CAC) | CNN + Reg | 26.34 | 0.824 | 0.631 | 1.502 |
| MPRNet (Blind IR) | CNN + Reg | 27.64 | 0.860 | 0.651 | 1.519 |
| FeMaSR (Blind IR) | Transformer + GAN | 23.65 | 0.749 | 0.501 | 1.363 |
| DiffBIR (Blind IR) | CNN + Diffusion | 22.50 | 0.706 | 0.455 | 1.394 |

### Ablation Study

| Config | Key Metric | Description |
|------|---------|------|
| ODE vs RMS Radius | R²=0.84 vs 0.30 | ODE has a much stronger correlation with performance than RMS |
| With FoV Prior vs Without | Significant Gain | Field-of-view info is crucial for spatially varying aberrations |
| With PSF Prior vs Without | Significant Gain | PSF cues assist in understanding aberration patterns |
| CNN vs Transformer | CNN Efficiency | CNNs provide better trade-offs as local features match aberration nature |

### Key Findings

- **Optical Priors (FoV and PSF)** are critical for handling spatially varying aberrations; both FoV and PSF information significantly boost performance.
- **Clean Image Priors** (e.g., FeMaSR's codebook or DiffBIR's diffusion prior) are highly beneficial for CAC.
- **CNN Architectures** offer a better balance between CAC performance and inference time, as convolution efficiently captures local characteristics inherent in optical degradation.
- Regression training improves fidelity, while GAN/Diffusion improves perceptual quality; achieving a holistic improvement remains an area for exploration.
- Extending AOD allows for the generation of physically valid lenses beyond the limitations of proprietary data.
- The benchmark categorizes 120 lenses into 5 difficulty levels based on the ODE score.

## Highlights & Insights

- **ODE Framework Design**: Decomposing optical degradation into overall quality, spatial uniformity, and chromatic variation is more comprehensive and accurate than traditional single-value metrics.
- **AOD-based Benchmark Construction**: This approach is highly transferable. When real data is scarce, physical simulations can be used to generate large-scale benchmarks.
- **IR to CAC Transferability**: General Image Restoration (IR) methods can be directly applied to CAC, often outperforming specialized CAC methods, suggesting that general image restoration knowledge is effective here.

## Limitations & Future Work

- The benchmark primarily covers consumer-grade photographic lenses and does not include specialized systems like microscopes or telescopes.
- A gap exists between simulated and real-captured images; future work requires more precise simulation or more real-world data validation.
- Integrating regression and GAN/Diffusion training to achieve simultaneous gains in fidelity, perception, and optical quality is a key open problem.
- The weight settings for the O.P. metric require further experimental validation.
- Detailed lens diversity is currently limited by the 4 key specifications used in the global search.

## Related Work & Insights

- **vs. Traditional CAC (e.g., FOV-KPN)**: While traditional methods are trained for specific lenses, this work proves the feasibility and necessity of universal training.
- **vs. General IR (e.g., NAFNet/Restormer)**: IR methods can match or exceed specialized CAC methods under unified training, though they still lack explicit utilization of optical priors.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale universal CAC benchmark; ODE framework is well-designed and highly correlated with performance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 24 methods across 120 sampled lenses with multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, in-depth analysis, and informative visualizations.
- Value: ⭐⭐⭐⭐ Lays a significant foundation for universal CAC research; datasets and code are accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OptiFusion: Towards Universal Computational Aberration Correction in Photographic Cameras](../../CVPR2025/image_restoration/towards_universal_computational_aberration_correction_in_photographic_cameras_a_.md)
- [\[CVPR 2026\] Learning Personalized Photographic Style from Pairwise User Preferences](learning_personalized_photographic_style_from_pairwise_user_preferences.md)
- [\[CVPR 2026\] Distilling Quasi-Conformal Mapping: A Generalizable and Efficient Solution for Wide-Angle Correction](distilling_quasi-conformal_mapping_a_generalizable_and_efficient_solution_for_wi.md)
- [\[CVPR 2026\] OMoBlur: An Object Motion Blur Dataset and Benchmark for Real-World Local Motion Deblurring](omoblur_an_object_motion_blur_dataset_and_benchmark_for_real-world_local_motion_.md)
- [\[CVPR 2026\] Human-Centric Multi-Exposure Fusion: Benchmark and Bi-level Cognition Distillation Framework](human-centric_multi-exposure_fusion_benchmark_and_bi-level_cognition_distillatio.md)

</div>

<!-- RELATED:END -->
