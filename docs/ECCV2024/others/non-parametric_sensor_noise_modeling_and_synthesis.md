---
title: >-
  [Paper Note] Non-parametric Sensor Noise Modeling and Synthesis
description: >-
  [ECCV 2024][Sensor Noise Modeling] This paper proposes a non-parametric sensor noise model that models the real noise distribution by directly constructing a probability mass function (PMF) for each brightness level from real-world captured images, requiring no assumption of a specific noise distribution form. It also introduces ISO interpolation and noise synthesis methods on noisy images, significantly outperforming existing parametric noise models on downstream denoising t…
tags:
  - "ECCV 2024"
  - "Sensor Noise Modeling"
  - "Non-parametric Model"
  - "Probability Mass Function"
  - "Noise Synthesis"
  - "Image Denoising"
date: 2026-05-08
content_hash: 0feb0ce7ada79e24
---

# Non-parametric Sensor Noise Modeling and Synthesis

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Image Processing / Low-level Vision  
**Keywords**: Sensor Noise Modeling, Non-parametric Model, Probability Mass Function, Noise Synthesis, Image Denoising

## TL;DR
This paper proposes a non-parametric sensor noise model that models the real noise distribution by directly constructing a probability mass function (PMF) for each brightness level from real-world captured images, requiring no assumption of a specific noise distribution form. It also introduces ISO interpolation and noise synthesis methods on noisy images, significantly outperforming existing parametric noise models on downstream denoising tasks.

## Background & Motivation

**Background**: Image sensor noise modeling is a fundamental problem in computational photography and image denoising. Currently, mainstream methods adopt parametric noise models (such as Gaussian noise and Poisson-Gaussian noise) to approximate the noise distribution of real sensors. These models are based on physical assumptions of noise sources (photon noise, readout noise, etc.) and describe noise characteristics by calibrating a small number of parameters.

**Limitations of Prior Work**: The fitting precision of parametric models for real sensor noise is limited. The noise distribution of real sensors often does not perfectly conform to any standard parametric distribution—it may exhibit different skewness across various brightness levels and contain non-ideal characteristics such as fixed pattern noise (FPN) and row/column noise. These complex noise features cannot be accurately captured by simple parametric models, leading to a performance drop (domain gap) on real images for denoising networks trained on synthetic data.

**Key Challenge**: Parametric models pursue concise expression at the cost of accuracy, whereas the complexity of real noise demands more flexible modeling approaches. The key challenge lies in: how to accurately capture the full distribution characteristics of noise at each brightness level without making any distributional assumptions?

**Goal**: (1) Propose a noise modeling method that does not rely on distribution assumptions; (2) Reduce the number of ISO levels required for calibration; (3) Solve the problem of how to synthesize noise in the absence of noise-free images.

**Key Insight**: The authors observe that given a sufficient number of repeated shots of the same scene under identical exposure conditions, one can directly compute the empirical noise distribution for each brightness level to construct a complete PMF, bypassing any parametric assumptions. Although straightforward, the key lies in how to calibrate efficiently and generalize across ISO levels.

**Core Idea**: Replacing traditional parametric noise models with non-parametric PMFs compiled directly from repeated shots allows for more accurate sensor noise modeling and synthesis.

## Method

### Overall Architecture
The input consists of a large number of repeatedly captured images (same scene, same exposure) by a specific sensor under a specific ISO setting, and the output is the probability mass function (PMF) of the sensor's noise at each brightness level. Using the constructed PMF, highly realistic sensor noise can be added to any clean image to train denoising networks. The overall pipeline is divided into three stages: PMF calibration, ISO interpolation, and noise synthesis on noisy images.

### Key Designs

1. **Non-parametric PMF Calibration**:

    - **Function**: Directly construct the full probability distribution of noise for each brightness level.
    - **Mechanism**: Capture many repeated shots (e.g., hundreds) of the same static scene. For each pixel location, the true brightness value is estimated by averaging all captured shots. Then, compute the frequency of occurrence of all observed values under the same true brightness level to directly construct the conditional probability distribution $P(\text{noisy}|\text{clean})$. Since the brightness values of digital images are discrete (e.g., 0-255 or 0-16383), this naturally forms a PMF. By aggregating the statistical results of the same brightness level across all pixel locations, a robust PMF estimate is obtained.
    - **Design Motivation**: Bypass parametric assumptions to directly capture all details of the noise distribution, including skewness, multi-modality, heavy tails, and other features that parametric models cannot express.

2. **ISO Level Interpolation**:

    - **Function**: Reduce the number of ISO levels that require actual capture and calibration.
    - **Mechanism**: Noise PMFs under different ISOs exhibit smooth transitions due to continuous gain changes. The authors propose interpolating PMFs between a few calibrated ISO levels to generate noise models for intermediate ISOs. Specifically, interpolation is performed per brightness level within the PMF space (potentially using optimal transport or simple linear interpolation strategies), ensuring that the interpolated PMF remains a valid probability distribution (non-negative and normalized).
    - **Design Motivation**: Fully calibrating a sensor's noise model across all ISOs requires extensive capture and storage. ISO interpolation allows covering the entire range by calibrating only a few key ISOs, significantly reducing calibration costs.

3. **Noise Synthesis on Noisy Images**:

    - **Function**: Synthesize target noise on existing noisy images when no noise-free reference image is available.
    - **Mechanism**: When only a noisy image is available instead of a clean one, directly adding noise causes noise accumulation rather than replacement. The authors propose a method to first estimate the noise level of the current image, and then derive how much additional noise should be added via conditional probability so that the final noise level matches the noise characteristics of the target sensor. This essentially leverages noise additivity and the comprehensive information of the non-parametric PMF to calculate the "difference" between the two noise distributions.
    - **Design Motivation**: In practical applications, acquiring completely noise-free reference images is extremely difficult (requiring ultra-long exposures or massive averaging). This method enables the utilization of consumer-captured noisy images as a base to synthesize training data.

### Loss & Training
The core contribution of this paper is the noise modeling approach rather than an end-to-end trained network. The constructed non-parametric noise model is used to synthesize training data for downstream denoising networks. The denoising network itself can adopt any standard architecture and loss function (e.g., L1, L2, SSIM), with the key improvement stemming from more realistic training data.

## Key Experimental Results

### Main Results

| Dataset/Sensor | Metric | Ours (Non-parametric) | Gaussian Noise Model | Poisson-Gaussian Model | Gain |
|---------------|------|---------------------|-------------|--------------|------|
| Sensor A (High ISO) | PSNR(dB) | ~40.2 | ~37.5 | ~38.8 | +1.4 |
| Sensor A (Low ISO) | PSNR(dB) | ~43.1 | ~41.2 | ~42.0 | +1.1 |
| Sensor B (High ISO) | PSNR(dB) | ~39.6 | ~36.8 | ~38.1 | +1.5 |
| Sensor B (Low ISO) | PSNR(dB) | ~42.8 | ~41.0 | ~41.8 | +1.0 |
| KL Divergence (Noise Fitting) | KL | Lowest | Highest | Moderate | Significant |

### Ablation Study

| Configuration | KL Divergence / Downstream PSNR | Description |
|------|-------------------|------|
| Full PMF | Optimal | Full non-parametric model |
| Only 5 ISOs Calibrated + Interpolated | Close to Optimal | Interpolation effectively reduces calibration overhead |
| Only 3 ISOs Calibrated + Interpolated | Slight Decline | Minimal calibration still maintains decent performance |
| No Noisy-to-Noisy Synthesis | Requires Clean Image | Validates the practicality of noisy-to-noisy synthesis |
| Gaussian Fitting (Parametric) | Significantly Inferior to PMF | Limitations of parametric models |

### Key Findings
- Non-parametric PMFs perform significantly better than all parametric models in terms of KL divergence, with more pronounced gaps under high ISOs, as high gains amplify the non-Gaussian characteristics of the noise distribution.
- The ISO interpolation method delivers excellent results, approximating the entire ISO range with only 5 calibrated ISO points, which indicates that noise characteristics indeed transition smoothly with ISO changes.
- On downstream denoising tasks, networks trained on data synthesized using the non-parametric model perform better than those trained on data synthesized with parametric models, validating the practical value of accurate noise modeling.
- The noise synthesis method on noisy images allows for the generation of high-quality training data without requiring clean reference images.

## Highlights & Insights
- **Thoroughness of Non-parametric Modeling**: Abandoning distribution assumptions entirely to model noise in a purely data-driven manner represents a paradigm shift in the field of noise modeling. The ingenuity lies in leveraging the discrete nature of digital image brightness values, making PMF a natural representation.
- **Reducing Calibration Costs via ISO Interpolation**: Simplifying the requirement of "calibrating all ISOs" to "calibrating only a few key ISOs" greatly enhances the method's practicality. This design is grounded in physical intuition—continuous variations in ISO gain imply that noise distributions transition continuously as well.
- **Practical Value of Noisy Synthesis**: Resolving the "chicken-or-egg" dilemma—training a denoiser requires clean images, yet obtaining perfectly clean images often requires denoising. This method breaks that loop.
- **Transferable Methodology**: The non-parametric modeling philosophy can be transferred to other tasks demanding precise distribution modeling, such as compression artifact modeling, motion blur modeling, etc.

## Limitations & Future Work
- **Calibration Data Acquisition Cost**: Although ISO interpolation reduces some workloads, each ISO still requires extensive repeated capturing of static scenes, which is impractical for scenarios like mobile devices.
- **Storage Overhead**: Storing a full PMF for each brightness level requires a substantial amount of storage for 14-bit RAW images, making compressed representations potentially necessary.
- **Spatial Correlation**: The PMF models single-pixel independent noise, without accounting for spatially correlated noise between pixels (such as row/column noise), which represents a major noise source for certain sensors.
- **Temporal Dependency**: Sensor noise characteristics may change with temperature and operating duration; whether a single calibration remains effective over the long term remains an open question.
- **Future Directions**: Integrating lightweight spatial correlation noise models; exploring adaptive calibration strategies that leverage a small amount of online sampling to update the PMFs.

## Related Work & Insights
- **vs. Poisson-Gaussian Model**: The Poisson-Gaussian model is the mainstream parametric model, which assumes photon noise (Poisson) + readout noise (Gaussian). This work demonstrates that this assumption is oversimplified, and the actual noise PMF shape is far more complex.
- **vs. Noise Flow**: Noise Flow uses normalizing flows to learn the noise distribution, offering high flexibility but requiring heavy training and possessing uncertain generalization. The proposed method is more direct and reliable, as it does not require training a neural network.
- **vs. Calibration-based Denoising**: Some methods estimate noise using burst shots (e.g., Paired Real Noise), but this is typically done to obtain noisy-clean image pairs, rather than explicitly modeling the noise distribution independently.

## Rating
- Novelty: ⭐⭐⭐⭐ The non-parametric noise modeling approach is clear and practical; ISO interpolation and noise synthesis on noisy images possess incremental novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple sensors and ISOs, utilizing dual evaluation of both KL divergence and downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, comprehensive methodology description, and detailed calibration workflow.
- Value: ⭐⭐⭐⭐ Direct practical value for denoising research requiring realistic noise synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise](../../AAAI2026/others/predict_and_resist_long-term_accident_anticipation_under_sensor_noise.md)
- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[ECCV 2024\] AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[CVPR 2026\] Bidirectional Query-Driven Generation of Parametric CAD Sketch](../../CVPR2026/others/bidirectional_query-driven_generation_of_parametric_cad_sketch.md)

</div>

<!-- RELATED:END -->
