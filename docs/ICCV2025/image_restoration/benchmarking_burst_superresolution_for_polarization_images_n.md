---
title: >-
  [Paper Note] Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis
description: >-
  [ICCV 2025][Image Restoration][Polarization Imaging] To address the hardware bottleneck of polarization cameras—low light efficiency, low spatial resolution, and high noise—this work constructs two dedicated datasets (PolarNS for noise statistics analysis and PolarBurstSR for burst super-resolution training/evaluation), proposes a polarimetric noise propagation analysis model, and adapts five SOTA burst super-resolution methods to the polarization domain. Results demonstrate that polarization-specific training significantly outperforms generic RGB training in reconstructing both intensity maps (s0) and angle of linear polarization (AoLP).
tags:
  - ICCV 2025
  - Image Restoration
  - Polarization Imaging
  - Burst Super-Resolution
  - Noise Modeling
  - Dataset Benchmark
  - Stokes Parameters
date: 2026-05-08
content_hash: cb644954594bfa77
---

# Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis

**Conference**: ICCV 2025
**arXiv**: [2503.18705](https://arxiv.org/abs/2503.18705)
**Code**: [KAIST-VCLAB/polarns](https://github.com/KAIST-VCLAB/polarns) (open-source, includes dataset + 5 model implementations + pretrained weights)
**Area**: Image Restoration
**Keywords**: Polarization Imaging, Burst Super-Resolution, Noise Modeling, Dataset Benchmark, Stokes Parameters

## TL;DR
To address the hardware bottleneck of polarization cameras—low light efficiency, low spatial resolution, and high noise—this work constructs two dedicated datasets (PolarNS for noise statistics analysis and PolarBurstSR for burst super-resolution training/evaluation), proposes a polarimetric noise propagation analysis model, and adapts five SOTA burst super-resolution methods to the polarization domain. Results demonstrate that polarization-specific training significantly outperforms generic RGB training in reconstructing both intensity maps (s0) and angle of linear polarization (AoLP).

## Background & Motivation

### Limitations of Prior Work

**Background**: Snapshot polarization cameras use a **dual-Bayer-pattern sensor** to simultaneously capture color and polarization information. Each pixel is covered by a micropolarizer (oriented at 0°/45°/90°/135°) overlaid with a Bayer color filter array, forming a 4×4 super-pixel structure. This design introduces two fundamental problems:
1. **Extremely low light efficiency**: Each pixel receives only light of a specific polarization direction and color, reducing actual photon flux to 1/4–1/16 of a conventional sensor, leading to severe noise.
2. **Low spatial resolution**: Stokes parameters (s0, s1, s2) must be estimated from 4×4 super-pixel pools, substantially reducing effective resolution.

Burst super-resolution is an effective technique for jointly denoising and upsampling by fusing multiple low-resolution frames. However, applying it to polarization imaging faces two critical gaps: (1) the absence of ground-truth noise statistics for polarization images—accurate noise modeling requires knowledge of the noise characteristics; and (2) the lack of dedicated polarimetric burst super-resolution datasets and benchmarks—prior work resorted to RGB datasets, ignoring the unique noise propagation mechanisms of polarization imaging.

### Starting Point

**Goal**:
1. What are the noise statistical characteristics of polarization cameras? How does noise propagate from raw sensor data to Stokes parameters and polarimetric quantities (DoLP, AoLP)?
2. How can a reliable polarimetric burst super-resolution benchmark be constructed? How much does polarization-specific training improve over generic RGB training?

## Method

### Overall Architecture
The core contribution of this paper is a systematic effort combining **dataset construction, noise analysis, and benchmark evaluation**, rather than a single novel model. The work is organized into three parts:
1. **PolarNS dataset** → polarimetric noise statistics analysis
2. **Polarimetric noise propagation model** → quantifying noise from raw pixels to Stokes parameters
3. **PolarBurstSR dataset** + 5 adapted models → burst super-resolution benchmark

### Key Designs

1. **PolarNS Dataset (Noise Statistics Dataset)**: Contains 244 scenes captured in dark-room environments as well as real indoor/outdoor scenes. Each scene provides: (a) a denoised ground truth obtained by averaging a large number of frames of the same static scene; and (b) pixel-wise mean and standard deviation statistics (raw values in [0, 4095], 12-bit sensor). Using this dataset, the authors verify that polarization sensor noise follows a heteroscedastic Gaussian model (variance grows linearly with signal intensity, i.e., the classic shot noise + read noise model), and further analyze how noise propagates from the four raw polarization channels to the Stokes vector s0, s1, s2, as well as to physical quantities such as DoLP (degree of linear polarization) and AoLP (angle of linear polarization).

2. **Polarimetric Noise Propagation Model**: Stokes parameters are computed via linear combinations: $s_0 = I_0 + I_{90}$, $s_1 = I_0 - I_{90}$, $s_2 = I_{45} - I_{135}$. Because s1 and s2 involve subtraction, noise is amplified. The authors derive noise propagation formulas from raw pixel noise to Stokes parameter noise and validate the model on the dark-room data from PolarNS. This analysis reveals a key finding: **polarimetric quantities (especially AoLP) are extremely sensitive to noise**—even when noise in the raw image appears small, the error in the polarization angle is significantly amplified after the Stokes transformation.

3. **PolarBurstSR Dataset (Burst Super-Resolution Dataset)**: Designed specifically for polarimetric burst super-resolution, with train/val/test splits. Ground truth is obtained by averaging multiple frames (providing denoised, high-resolution references). Each sample contains 14 burst frames as input (simulating hand-held micro-jitter) and corresponding Stokes parameter ground truth (s0, s1, s2). Data is collected under diverse real-world scene conditions. Synthetic data is generated from the Sony RSP dataset via polarization mosaicking and noise addition.

4. **Five Polarization-Adapted Burst Super-Resolution Models**: Five existing RGB burst super-resolution methods (BSRT, BurstM, Burstormer, FBAnet, MFIR) are adapted to the polarization domain, uniformly named p-BSRT, p-BurstM, p-Burstormer, p-FBAnet, and p-MFIR. Key adaptations include: (a) input channels changed from 3 (RGB) to 16 (4 polarization orientations × 4 Bayer colors); (b) output changed to 9 Stokes parameter channels (s0/s1/s2 × 3 color channels); (c) noise simulation uses polarization-specific shot noise + read noise (noise levels sampled from a log-log linear distribution).

### Loss & Training
- **Loss Function**: L1 loss applied between predicted Stokes parameters and ground truth (L2 and polarimetric-space loss variants are also supported).
- **Two-Stage Training**: Pre-training on synthetic data (polarimetric data generated from the Sony RSP dataset), followed by fine-tuning on PolarBurstSR real data.
- **Training Configuration**: Adam optimizer, ExponentialLR learning rate schedule, DDP multi-GPU training (4 GPUs), burst size = 14 frames, 2× super-resolution.
- **Evaluation Metrics**: Standard metrics (PSNR, SSIM, LPIPS) for synthetic data; Aligned Metrics for real data (due to sub-pixel misalignment between the ground truth and inputs in real captures, alignment is performed prior to metric computation).

## Key Experimental Results

| Setting | Metric | Polar Training (p-BSRT) | RGB Training (BSRT) | Difference |
|---|---|---|---|---|
| Synthetic data (s0 intensity) | PSNR | Significantly better than RGB | Baseline | Polar training consistently superior |
| Synthetic data (AoLP) | Angular error | Significantly lower | Higher | AoLP shows the largest improvement |
| Real data | Aligned PSNR | Better | Worse | Confirms real-scene generalization |
| All methods | Overall ranking | p-BSRT / p-Burstormer top | — | Each of the 5 methods has different strengths |

*Note: The core conclusion of the paper is the qualitative advantage of polarization-specific training over generic RGB training, rather than achieving state-of-the-art absolute numbers. Qualitative results on the project page show that AoLP maps exhibit substantially reduced noise and improved color consistency after polarization-specific training.*

### Ablation Study
- **Noise Model Validation**: Noise distribution fitting experiments on PolarNS confirm the accuracy of the heteroscedastic Gaussian model.
- **Noise Propagation Analysis**: Quantification of noise amplification factors from raw pixels to each Stokes parameter validates the theoretical prediction that s1/s2 exhibit higher noise than s0.
- **Synthetic vs. Real**: The two-stage strategy (synthetic pre-training + real fine-tuning) outperforms training solely on real data.
- **Polarization vs. RGB Training**: All five adapted models consistently outperform their RGB-trained counterparts, with the largest improvements observed on the AoLP metric.

## Highlights & Insights
- **Pioneering polarimetric burst super-resolution benchmark**: Fills the gap in the polarization imaging community for a systematic burst super-resolution dataset and evaluation protocol; the 244-scene noise statistics cover diverse real-world conditions.
- **Physical insight into noise propagation**: Through rigorous noise propagation analysis, the work reveals the physical mechanism underlying polarimetric sensitivity to noise, providing theoretical guidance for future polarimetric denoising and super-resolution method design.
- **Comprehensive adaptation and comparison of five methods**: Rather than modifying a single model, this work systematically adapts five recent burst super-resolution methods to the polarization domain, providing a fair and comprehensive benchmark comparison.
- **Fully open-sourced dataset and code**: The complete pipeline from data collection to model training is open-sourced, substantially lowering the barrier for future research.

## Limitations & Future Work
- **Only 2× super-resolution evaluated**: The current benchmark covers only a 2× upscaling factor; 4× or higher polarimetric super-resolution remains unexplored.
- **No polarization-aware architectural innovations**: The five adapted models only modify input/output channels without designing polarization-specific network modules (e.g., leveraging the physical constraints among s0, s1, s2).
- **GT quality limited by frame averaging**: Ground truth in real data is obtained by averaging multiple frames and still contains residual noise, making it less accurate than synthetic ground truth.
- **Dynamic scenes not considered**: Burst capture assumes an approximately static scene; moving objects or highly dynamic scenes are not addressed.
- **Noise model limited to shot + read noise**: Other noise sources (e.g., crosstalk noise, thermal drift, and other sensor-specific systematic noise in polarization cameras) are not incorporated.

## Related Work & Insights
- **vs. BurstSR (Bhat et al., ICCV 2021)**: BurstSR is the canonical burst super-resolution benchmark in the RGB domain. PolarBurstSR adopts a similar dataset structure but is specifically designed for polarimetric 16-channel mosaic data and Stokes parameter ground truth.
- **vs. Sony RSP Dataset**: RSP provides ground truth references for polarimetric synthetic data. This work uses it to generate synthetic training data, while PolarBurstSR supplements RSP with real-scene burst data that RSP lacks.
- **vs. BSRT / Burstormer and other burst SR models**: This work does not propose new architectures but adapts existing methods to the polarization domain and provides a unified, fair comparison benchmark.
- **vs. Polarimetric denoising methods**: Several polarimetric denoising works exist, but most are single-frame methods and lack systematic noise statistics research. This work provides the first large-scale noise statistics dataset for polarization imaging.

---

- **Polarization-aware network design**: Current adaptations simply modify channel counts; future work could design dedicated modules that exploit the physical constraints of Stokes parameters (e.g., $DoLP = \sqrt{s_1^2 + s_2^2}/s_0$), representing a clear direction for improvement.
- **Cross-domain transfer**: The noise propagation analysis framework can be generalized to other computational photography scenarios (e.g., spectral cameras, event cameras), providing a template for noise modeling across different sensor types.

## Rating
- Novelty: ⭐⭐⭐⭐ (Core contributions lie in dataset and benchmark construction rather than methodology, but fill an important gap)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (244-scene noise statistics + 5-method comparison + dual-track synthetic/real evaluation — highly comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Physical analysis is clear, though the presentation of experimental results could better highlight the key findings)
- Value: ⭐⭐⭐⭐ (Provides much-needed benchmarks and tools for the polarization imaging community, though the target audience is relatively niche)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)

</div>

<!-- RELATED:END -->
