---
title: >-
  [Paper Note] UniCAC: Towards Universal Computational Aberration Correction in Photographic Cameras
description: >-
  [CVPR 2026][Image Restoration][Aberration correction] This work constructs UniCAC, the first large-scale universal benchmark for computational aberration correction (CAC) in photographic lenses covering both spherical and aspherical designs. It proposes an Optical Degradation Evaluator (ODE) to replace the traditional RMS radius metric, and derives three key factors governing CAC performance—prior utilization, network architecture, and training strategy—through a comprehensive evaluation of 24 models.
tags:
  - CVPR 2026
  - Image Restoration
  - Aberration correction
  - optical degradation evaluation
  - automatic optical design
  - universal benchmark
  - PSF prior
date: 2026-05-08
content_hash: 27ddc87f388b4cbb
---

# UniCAC: Towards Universal Computational Aberration Correction in Photographic Cameras

**Conference**: CVPR 2026
**arXiv**: [2603.12083](https://arxiv.org/abs/2603.12083)
**Code**: [https://github.com/XiaolongQian/UniCAC](https://github.com/XiaolongQian/UniCAC)
**Area**: Image Restoration / Computational Aberration Correction
**Keywords**: Aberration correction, optical degradation evaluation, automatic optical design, universal benchmark, PSF prior

## TL;DR

This work constructs UniCAC, the first large-scale universal benchmark for computational aberration correction (CAC) in photographic lenses covering both spherical and aspherical designs. It proposes an Optical Degradation Evaluator (ODE) to replace the traditional RMS radius metric, and derives three key factors governing CAC performance—prior utilization, network architecture, and training strategy—through a comprehensive evaluation of 24 models.

## Background & Motivation

1. **Background**: Computational aberration correction (CAC) serves as an image post-processing technique to compensate for residual optical aberrations. Existing methods are typically tailored to specific lenses and generalize poorly.
2. **Limitations of Prior Work**: (a) No comprehensive benchmark covering diverse lens designs exists, as commercial lens configurations are not publicly available; (b) the factors influencing CAC performance and their relative importance remain unclear.
3. **Key Challenge**: Universal CAC requires training data spanning diverse lenses, yet lens prescription files (e.g., Zemax) are difficult to obtain. Moreover, the traditional RMS radius metric correlates poorly with actual CAC difficulty.
4. **Goal**: (a) Construct a large-scale universal CAC benchmark; (b) propose a reliable aberration quantification framework; (c) systematically evaluate 24 models and summarize key findings.
5. **Key Insight**: Leverage automatic optical design methods to generate large numbers of physically valid lens prescriptions.
6. **Core Idea**: Extend automatic optical design (OptiFusion) to cover aspherical parameters, replace RMS with ODE, and conduct systematic evaluation.

## Method

### Overall Architecture

The work consists of three components: (1) benchmark construction—extending OptiFusion automatic optical design to generate diverse lenses; (2) ODE framework—comprehensively evaluating optical degradation difficulty; (3) systematic evaluation—a full comparison of 24 models on UniCAC.

### Key Designs

1. **Extended Automatic Optical Design**:

    - **Function**: Automatically generate large numbers of physically feasible lens prescriptions.
    - **Mechanism**: Building upon OptiFusion, the method redefines spherical parameters and extends to aspherical parameters, enabling the design space to cover both spherical and aspherical lenses. Automatic search generates lens specifications satisfying image quality constraints, yielding Zemax files for simulation.
    - **Design Motivation**: Existing benchmarks cover only a small number of manually designed lenses, which cannot represent the diversity of real optical systems. Automatic design makes large-scale lens generation tractable.

2. **Optical Degradation Evaluator (ODE)**:

    - **Function**: Reliably quantify CAC task difficulty, replacing the traditional RMS radius.
    - **Mechanism**: Integrates image fidelity metrics (PSNR, SSIM) with MTF-based optical image quality evaluation (OIQE). The linear correlation ($R^2$) between ODE and final CAC performance significantly exceeds that of RMS radius—lenses with small RMS values may in fact be harder to correct due to severe loss of fine structure.
    - **Design Motivation**: Experiments reveal that RMS radius correlates poorly with CAC outcomes—lenses with small RMS but severe detail loss exhibit worse correction results. A multi-dimensional evaluation is therefore necessary.

3. **Nine Key Findings**:

    - **Function**: Provide systematic guidance for CAC research.
    - **Mechanism**: Through evaluation of 24 models, findings are organized along three dimensions—prior utilization (both FoV information and PSF cues matter; clean image priors are highly beneficial); architecture (CNNs achieve the best efficiency–performance trade-off); training strategy (regression training improves fidelity, while GAN/diffusion improves perceptual quality).
    - **Design Motivation**: The CAC field lacks systematic method comparison and guiding principles.

### Loss & Training

This work focuses on benchmark construction and evaluation; no novel training methodology is proposed.

## Key Experimental Results

### Main Results

| Model Category | Representative Method | UniCAC Overall Score | Inference Time | Notes |
|---|---|---|---|---|
| CAC-specific | NIPC | High | Fast | PSF prior provides large gains |
| General IR (CNN) | Restormer | High | Moderate | Best efficiency–performance trade-off |
| Diffusion-based | DiffBIR | Moderate | Slow | Good perception but low fidelity |
| Codebook-based | FeMaSR | High | Moderate | Codebook prior effective |

### Ablation Study

| Configuration | Description |
|---|---|
| FoV information | Incorporating field-of-view information significantly improves handling of spatially varying aberrations |
| PSF cues | PSF priors help the model understand aberration patterns |
| Clean image prior | Clean image priors from codebooks/diffusion are highly beneficial for CAC |

### Key Findings

- ODE achieves substantially higher $R^2$ correlation with CAC performance than RMS radius (0.85+ vs. 0.45), confirming ODE as a more reliable difficulty indicator.
- CNN-based architectures achieve the best efficiency–performance trade-off for CAC—convolution is naturally suited to the local characteristics of aberration degradation.
- Clean image priors (e.g., FeMaSR codebooks, DiffBIR diffusion priors) provide substantial benefits for CAC.
- The impact of training paradigm on optical quality has been widely overlooked.

## Highlights & Insights

- Replacing RMS radius with ODE represents an important contribution to optical evaluation—traditional metrics may mislead lens selection and method assessment.
- The systematic evaluation of 24 models provides the field with a much-needed comprehensive comparison benchmark.
- The finding that "convolution is naturally suited to aberration degradation" offers practical guidance for architecture selection.

## Limitations & Future Work

- A domain gap remains between simulated aberration images and real lens imagery.
- The current benchmark covers only photographic lenses; specialized optical systems such as microscopes and telescopes are not included.
- The evaluation of 24 models does not encompass the latest Mamba-based architectures.

## Related Work & Insights

- **vs. lens-specific CAC**: Lens-specific CAC achieves high accuracy but lacks generalization; UniCAC pursues cross-lens universality.
- **vs. general IR methods (Restormer)**: General IR methods do not account for the spatially varying nature of aberrations; incorporating FoV/PSF priors yields significant improvements.
- **vs. optical design optimization**: Traditional optical design aims to minimize aberrations; CAC is a complementary software post-processing solution.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale universal CAC benchmark + ODE evaluation framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24 models, systematic evaluation, nine key findings
- Writing Quality: ⭐⭐⭐⭐ Clear organization, concise summary of findings
- Value: ⭐⭐⭐⭐ Significant benchmark and guidance value for the computational imaging community

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](unicac_universal_computational_aberration_correction_benchmark.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)

<!-- RELATED:END -->
