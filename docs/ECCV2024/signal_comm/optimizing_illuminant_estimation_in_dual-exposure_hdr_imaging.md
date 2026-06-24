---
title: >-
  [Paper Note] Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging
description: >-
  [ECCV 2024][Signal & Communication][Illuminant Estimation] This paper proposes extracting a compact Dual-Exposure Feature (DEF) from dual-exposure HDR image pairs, based on which two ultra-lightweight illuminant estimators, EMLP and ECCC, are constructed. They achieve or exceed the performance of prior methods requiring hundreds of thousands of parameters, while using only a few hundred to a few thousand parameters.
tags:
  - "ECCV 2024"
  - "Signal & Communication"
  - "Illuminant Estimation"
  - "HDR Imaging"
  - "Dual-Exposure"
  - "White Balance"
  - "Color Constancy"
date: 2026-05-08
content_hash: 5f8ec218860c090f
---

# Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Image Signal Processing / HDR Imaging  
**Keywords**: Illuminant Estimation, HDR Imaging, Dual-Exposure, White Balance, Color Constancy

## TL;DR
This paper proposes extracting a compact Dual-Exposure Feature (DEF) from dual-exposure HDR image pairs, based on which two ultra-lightweight illuminant estimators, EMLP and ECCC, are constructed. They achieve or exceed the performance of prior methods requiring hundreds of thousands of parameters, while using only a few hundred to a few thousand parameters.

## Background & Motivation

**Background**: High Dynamic Range (HDR) imaging extends the dynamic range of light by capturing multiple frames of the same scene with different exposures. In camera ISP (Image Signal Processor) pipelines, illuminant estimation is a critical step aimed at estimating the color of the global illuminant in the scene, which is used by the white balance module to remove undesired color casts from images.

**Limitations of Prior Work**: Although multiple frames with different exposures are captured in HDR pipelines, conventional illuminant estimation methods typically use only a single frame for estimation. This means the additional information acquired by dual-exposure or multi-exposure HDR sensors is wasted. Single-frame methods have limited accuracy under complex lighting conditions, whereas large model methods, despite higher accuracy, require a massive number of parameters and are unsuitable for embedded ISP deployment.

**Key Challenge**: The actual deployment scenario of illuminant estimation is embedded camera ISPs, which impose strict constraints on model size and inference speed, yet highly accurate illuminant estimation often requires complex model architectures. Meanwhile, HDR sensors already provide multi-frame information under different exposures, which contains valuable clues about the illuminant but remains underutilized.

**Goal**: How to effectively leverage the complementary information in dual-exposure HDR images to improve illuminant estimation accuracy, while maintaining an extremely low parameter count for embedded deployment?

**Key Insight**: The authors observe that the variation patterns in pixel values across different exposures of the same scene are closely related to the illuminant color—color variations in overexposed and underexposed regions contain additional illuminant information. By designing a compact feature to capture this cross-exposure discrepancy, estimation accuracy can be significantly improved with almost no added computational overhead.

**Core Idea**: Extract a compact feature named DEF from dual-exposure HDR image pairs, enabling highly accurate illuminant estimation using lightweight models with very few parameters.

## Method

### Overall Architecture
The input consists of a pair of dual-exposure images (long-exposure and short-exposure) in the HDR pipeline. First, the dual-exposure feature (DEF) is computed from this image pair, and then DEF is fed into the illuminant estimator as an auxiliary feature. The authors design two DEF-based estimators: EMLP (based on Multi-Layer Perceptron) and ECCC (based on modified Convolutional Color Constancy), both of which achieve high-precision estimation with extremely small parameter sizes.

### Key Designs

1. **Dual-Exposure Feature (DEF)**:

    - **Function**: Extracts a compact feature vector encoding illuminant information from dual-exposure image pairs.
    - **Mechanism**: DEF extracts illuminant clues by analyzing pixel value differences in the same scene under different exposure times. Specifically, statistical analysis is performed on the long-exposure and short-exposure images respectively (such as mean ratio of color channels, histogram differences, etc.), and then these statistics are combined into a low-dimensional feature vector. The underlying principle of DEF is that, given a known exposure ratio, scenes under different illuminant colors exhibit distinct color variation patterns across the two exposures, and this discrepancy pattern can serve as an auxiliary cue for illuminant estimation.
    - **Design Motivation**: Traditional illuminant estimation only considers the color distribution of a single frame, which offers limited information. Dual-exposure provides "two observations of the same physical scene under different exposure conditions," where this extra dimension of information is naturally complementary for illuminant inference.

2. **EMLP (Exposure-based MLP)**:

    - **Function**: A lightweight Multi-Layer Perceptron model for illuminant estimation based on DEF.
    - **Mechanism**: Takes the DEF feature as input and directly regresses the illuminant color values via a shallow MLP containing only a few hundred parameters. The input dimension of CPU/GPU-friendly MLP equals the dimension of the DEF, and after 1-2 hidden layers, it outputs the RGB illuminant color estimate. The network is extremely compact and inference is ultra-fast.
    - **Design Motivation**: To prove that the DEF feature itself already encodes rich illuminant information, enabling excellent results even with the simplest model architectures. Meanwhile, the extremely low parameter count allows it to be directly deployed on camera ISP chips.

3. **ECCC (Exposure-based Convolutional Color Constancy)**:

    - **Function**: An improved version of Convolutional Color Constancy based on DEF.
    - **Mechanism**: Modifies the classical CCC method by integrating DEF features into CCC's input. While CCC originally uses log-chromaticity histograms as input, ECCC additionally introduces cross-exposure information from DEF, enabling the histogram features and exposure-difference features to complement each other. The parameter count of ECCC is only a few thousand, yet it significantly improves accuracy compared to the original CCC.
    - **Design Motivation**: CCC is a well-established classical method, and integrating DEF within its framework leverages dual-exposure information while maintaining the simplicity of the methodology.

### Loss & Training
Angular error is used as the evaluation metric. During training, the L2 regression loss or angular loss is used to optimize the illuminant color estimation. The dataset contains paired dual-exposure images and corresponding ground-truth illuminant color annotations.

## Key Experimental Results

### Main Results

| Method | Params | Median Angular Error (°) | Mean Angular Error (°) | Notes |
|------|--------|-----------------|-----------------|------|
| Statistics-based | 0 | ~3-5° | ~4-6° | Non-parametric statistical methods |
| FC4 | Hundreds of thousands | ~2-3° | ~3-4° | Deep learning method |
| C5 | Millions | ~2° | ~2.5-3° | Large-model method |
| EMLP (Ours) | ~Hundreds | Near SOTA | Near SOTA | 3 orders of magnitude fewer parameters |
| ECCC (Ours) | ~Thousands | Achieves/Surpasses SOTA | Achieves/Surpasses SOTA | 2 orders of magnitude fewer parameters |

### Ablation Study

| Configuration | Median Angular Error | Notes |
|------|-------------|------|
| Long-exposure only (No DEF) | High | Traditional single-frame scheme |
| Short-exposure only (No DEF) | High | Traditional single-frame scheme |
| Long-exposure + DEF | **Lowest** | DEF brings significant improvement |
| Varying DEF dimensions | Small variation | DEF is insensitive to dimensions |
| Different exposure ratios | Stable | Method is robust to exposure ratios |

### Key Findings
- DEF is the core source of accuracy improvement—with DEF, even the simplest MLP can reach the accuracy of large models.
- EMLP surpasses deep learning methods requiring hundreds of thousands of parameters with only a few hundred parameters, demonstrating high information density of the DEF feature.
- The method performs stably across different HDR sensors and exposure ratio settings.
- Compared to existing large-model methods, the parameter footprint is reduced by 2-3 orders of magnitude, with inference speeds several orders of magnitude faster.

## Highlights & Insights
- **Extreme Parameter Efficiency**: Achieving accuracy on par with million-parameter methods using only a few hundred parameters highlights a deep understanding of the task's physical nature—proper feature design is far more effective than scaling parameters. This design philosophy is highly inspirational for embedded vision tasks.
- **Leveraging Existing Hardware Signals**: HDR sensors naturally capture multi-exposure images. DEF performs simple statistical extraction on this existing data without requiring additional hardware or acquisition steps. This paradigm of "squeezing more information from the existing pipeline" is highly practical.
- **Simple yet Effective Feature Engineering**: In an era dominated by deep learning, this work demonstrates that carefully handcrafted features can still play a substantial role in specific tasks.

## Limitations & Future Work
- The definition of DEF is based on the dual-exposure assumption, which requires adaptation for single-exposure or multi-frame ($>2$) exposure scenarios.
- Experiments are mainly evaluated on color constancy datasets annotated in laboratory environments; noise and non-linear responses of real HDR sensors may impact the effectiveness of DEF.
- The method only estimates global illuminant color, which may be inaccurate for multi-illuminant scenes (e.g., indoor lighting + natural light from Windows).
- It has not been compared with the latest self-supervised or pre-trained large model methods.

## Related Work & Insights
- **vs CCC (Convolutional Color Constancy)**: CCC uses log-chromaticity histograms as input, and ECCC incorporates DEF information on top of it. ECCC is essentially an enhanced version of CCC, and its contribution lies in proving the value of dual-exposure information.
- **vs Deep Learning Methods such as FC4/C5**: These methods use deep CNNs to learn the illuminant end-to-end from the whole image, which yields high accuracy but requires many parameters. This work achieves similar accuracy with extremely few parameters, solving the problem from a different perspective.
- **vs Classical Statistical Methods (Grey-World, White-Patch, etc.)**: Classical methods are parameter-free but have low accuracy. The proposed method finds a superior trade-off between parameter size and accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ The proposed DEF feature is simple yet effective, offering a novel perspective by leveraging dual-exposure information for illuminant estimation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons with multiple methods, parameter analysis, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, concise method descriptions, and intuitive results presentation.
- Value: ⭐⭐⭐⭐ Direct engineering value for practical deployment in HDR camera ISPs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unsupervised Exposure Correction](unsupervised_exposure_correction.md)
- [\[ICCV 2025\] Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors](../../ICCV2025/signal_comm/generalizable_non-line-of-sight_imaging_with_learnable_physical_priors.md)
- [\[ECCV 2024\] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)
- [\[ECCV 2024\] Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)
- [\[ECCV 2024\] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)

</div>

<!-- RELATED:END -->
