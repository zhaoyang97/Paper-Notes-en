---
title: >-
  [Paper Note] T-FAKE: Synthesizing Thermal Images for Facial Landmarking
description: >-
  [CVPR 2025][Medical Imaging][Thermal Image Synthesis] Proposes the T-FAKE dataset and the RGB2Thermal loss function, which utilize semi-supervised thermal image synthesis to generate the first large-scale synthetic thermal facial landmark dataset (200k images), achieving SOTA sparse/dense facial landmark detection in the thermal domain.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Thermal Image Synthesis"
  - "Facial Landmark Detection"
  - "Synthetic Dataset"
  - "Domain Adaptation"
  - "Wasserstein Distance"
date: 2026-05-08
content_hash: c0da4d1a4d5fde57
---

# T-FAKE: Synthesizing Thermal Images for Facial Landmarking

**Conference**: CVPR 2025  
**arXiv**: [2408.15127](https://arxiv.org/abs/2408.15127)  
**Code**: [github.com/phflot/tfake](https://github.com/phflot/tfake)  
**Area**: Medical Imaging / Facial Analysis  
**Keywords**: Thermal Image Synthesis, Facial Landmark Detection, Synthetic Dataset, Domain Adaptation, Wasserstein Distance

## TL;DR

Proposes the T-FAKE dataset and the RGB2Thermal loss function, which utilize semi-supervised thermal image synthesis to generate the first large-scale synthetic thermal facial landmark dataset (200k images), achieving SOTA sparse/dense facial landmark detection in the thermal domain.

## Background & Motivation

Thermal imaging holds significant application value in fields such as infectious disease screening, emotional state analysis, and biometrics, where facial landmark detection serves as a core foundational task. However, the thermal domain faces severe data bottlenecks:

1. **Data Scarcity**: The largest existing thermal facial landmark datasets only contain dozens to hundreds of individuals (AACHEN with 90 subjects, CHARLOTTE with 10 subjects), whereas MegaFace in the RGB domain reached 1 million faces as early as 2016.
2. **Limited Annotation Types**: Existing thermal datasets provide mostly sparse landmarks (5-72 points) and lack dense landmark annotations (>300 points), restricting the potential for fine-grained facial analysis.
3. **Laboratory Constraints**: Most thermal datasets are collected under controlled laboratory conditions with monotonous facial poses (predominantly frontal), failing to cover complex real-world scenarios (shadows, diverse skin tones, varied expressions).
4. **Substantial Domain Gap**: Directly applying RGB landmark detectors to thermal images yields poor performance, maintaining high failure rates especially in challenging scenarios (contrast inversion due to nose cooling, sweating, outdoor rain).

Core Idea: Drawing on the success of using synthetic data (the FAKE dataset) in the RGB domain to train landmark detectors, this work designs a specialized RGB2Thermal loss function to "thermalize" synthetic RGB images, generating a large-scale synthetic thermal dataset to break through the data bottleneck.

## Method

### Overall Architecture

The method comprises a three-stage training pipeline: (1) Thermalization network $\mathcal{T}_\theta$: a U-Net-based RGB-to-thermal image translation model trained with a three-part semi-supervised loss function; (2) Landmark prediction network $\mathcal{T}_\psi$: a MobileNet V2-based probabilistic landmark regression model jointly trained on the FAKE and T-FAKE datasets, outputting both landmark locations and uncertainties; (3) Label adaptation network $\mathcal{T}_\zeta$: a 5-layer fully connected network that handles conversion between different landmark annotation standards.

### Key Designs

1. **RGB2Thermal Loss Function**:
    - Function: Translates synthetic RGB images into realistic thermal-style images, ensuring that high-quality thermal images can be generated even in "in-the-wild" scenarios without paired training data.
    - Mechanism: A combination of three loss terms—(a) Supervised reconstruction loss (Eq. 1): minimizes the L2 distance on paired RGB-Thermal data (SEJONG dataset, lab conditions) to provide a baseline mapping signal; (b) Wasserstein distance term (Eq. 2): uses entropy-regularized Wasserstein-2 distance to align the patch distribution of synthetic thermal images with real thermal images, efficiently computed via the Sinkhorn algorithm and applied across multiple image scales to ensure the generated images exhibit realistic thermal texture statistics; (c) Temperature prior regularization (Eq. 3): utilizes medical clinical temperature statistics of 18 facial semantic regions (nose, eyes, forehead, etc.) to constrain the mean temperature of each region to comply with medical priors (e.g., warmer periorbital area, cooler scalp).
    - Design Motivation: Purely supervised approaches can only learn the thermal mapping under frontal lab conditions, failing to generalize to the complex poses, shadows, and expressions found in synthetic data. Wasserstein distance provides unsupervised distribution matching, and the temperature prior injects domain knowledge; their synergy enables domain generalization.

2. **Probabilistic Landmark Prediction (GLL+SW)**:
    - Function: Simultaneously predicts landmark locations and estimation uncertainties, while integrating face detection capabilities.
    - Mechanism: Models landmark prediction as $L$ independent 2D Gaussian distributions $\mathcal{N}(\mu_l, \sigma_l I_2)$. It optimizes the negative log-likelihood loss (including a localization accuracy term and an uncertainty prediction term). During inference, a multi-scale sliding window (GLL+SW) approach is used, selecting the window with the lowest average standard deviation. Face detection is achieved by thresholding the average uncertainty $\bar{\sigma}$ (detecting a face when $\bar{\sigma} < t$).
    - Design Motivation: Facial appearance in thermal images varies heavily (due to temperature changes, sweat, etc.), and probabilistic modeling naturally quantifies prediction confidence; uncertainty info can be directly reused as a face detection criterion, avoiding the need to train an extra detector.

3. **Label Adaptation Network**:
    - Function: Converts between different landmark annotation standards (e.g., 3DA-2D 70 points $\rightarrow$ 2D 68 points).
    - Mechanism: Trains a 5-layer fully connected network that takes predicted coordinates from one standard as input and outputs coordinates in the target standard. It is trained only on low-dimensional landmark coordinates and does not depend on images.
    - Design Motivation: Different datasets use different annotation standards (2D vs. 3DA-2D, 68 points vs. 72 points, etc.); the adaptation network allows a single model to be evaluated fairly across multiple evaluation criteria.

### Loss & Training

Thermalization network loss: $\mathcal{L}(\theta) = \text{MSE}_{paired} + \lambda_W \mathcal{W}_{2,E}^2(\mu_{FAKE}, \mu_T) + \lambda_R R(T_\theta(X_{FAKE}))$

Training Details: Two models are trained corresponding to "cold" and "warm" ambient temperature conditions. The thermalization network is trained at 256×256 resolution, and generates at 512×512 during inference (as U-Net is resolution-agnostic). The landmark network uses MobileNet V2 with an image size of 224×224, trained on FAKE+T-FAKE simultaneously. Texture images (with randomized landmark annotations) are introduced as negative samples to learn meaningful uncertainty estimation.

## Key Experimental Results

### Main Results (NME W/H $\downarrow$ on the CHARLOTTE Thermal Dataset)

| Method | Training Data | High | Low | Side | Front | Full |
|------|---------|------|-----|------|-------|------|
| Star | 300W(RGB) | 0.071 | 0.102 | 0.067 | 0.106 | 0.087 |
| YOLO5Face | TFW(Thermal) | 0.079 | 0.125 | 0.070 | 0.131 | 0.101 |
| DAN | AACHEN(Thermal) | 0.168 | 0.241 | 0.294 | 0.173 | 0.205 |
| GLL+SW(RGB only) | FAKE | 0.106 | 0.268 | 0.124 | 0.253 | 0.189 |
| **GLL+SW Sparse** | **FAKE+T-FAKE** | **0.068** | **0.124** | **0.065** | **0.129** | **0.097** |
| **GLL+SW (σ̄<6e-4)** | **FAKE+T-FAKE** | **0.067** | **0.112** | **0.059** | **0.118** | **0.089** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Supervised only (no regularization) | High artifacts, poor generalization | Overfitting to laboratory conditions |
| + Wasserstein patch distance | More natural texture statistics | Effective distribution matching |
| + Temperature prior | More pronounced regional temperature differences | Prior knowledge injection |
| Full semi-supervised | Optimal performance | Synergistic combination of three terms |
| Dense 478-point | NME=0.113 | First dense thermal detection |
| Sparse 70-point | NME=0.097 | More accurate sparse detection |

### Key Findings

- The multi-modal model trained on T-FAKE ($NME=0.097$) significantly outperforms all existing thermal landmark methods.
- Joint multi-modal training is far superior to direct RGB-to-thermal transfer ($NME=0.097$ vs. $0.189$).
- The probabilistic uncertainty thresholding mechanism significantly improves prediction accuracy at the cost of slight coverage loss (NME $0.097 \rightarrow 0.089$).
- The domain generalization capability of the semi-supervised loss is far superior to purely supervised methods, especially for profile views, shadows, and different skin tones.
- The T-FAKE dataset enables 478-point dense thermal landmark detection for the first time.

## Highlights & Insights

- **Significant Dataset Contribution**: T-FAKE is the first large-scale synthetic thermal landmark dataset (200k images, $\ge 2000$ individuals, sparse + dense annotations), filling a critical gap in the thermal domain.
- **Exquisite Semi-Supervised Loss Design**: Wasserstein patch distribution matching + clinical temperature prior + paired supervision, with each component performing a clear and complementary role.
- **Uncertainty as Detection**: Cleverly reuses landmark prediction uncertainty as a face detection signal, simplifying system design.
- **Resolution-Agnostic Training**: The 256² training $\rightarrow$ 512² generation scheme offers valuable insights for future work.

## Limitations & Future Work

- The thermalization model trains its paired component on the SEJONG laboratory dataset, which may limit its generalization to extreme outdoor environments.
- The temperature prior hard-codes the reference temperatures of 18 regions; individual-specific or dynamic temperature modeling could further improve quality.
- The ground-truth annotations for dense landmarks are transferred from Mediapipe predictions, which introduces some noise.
- Future research could explore diffusion-model-based thermalization methods and video-level thermal landmark tracking.

## Related Work & Insights

- **vs. Mallat et al.**: Previous work only used perceptual loss for thermalization; the semi-supervised design of T-FAKE (Wasserstein + temperature prior) significantly enhances generalization capability.
- **vs. AACHEN Dataset (DAN)**: AACHEN only contains laboratory data from 90 subjects, and DAN yields an NME of 0.205 on CHARLOTTE; the scale advantage of T-FAKE ($\ge 2000$ subjects) reduces the NME to 0.097.
- **vs. FAKE / Wood et al.**: Successfully transfers synthetic data methodology from the RGB domain to the thermal modality.

## Rating

- Novelty: ⭐⭐⭐⭐ RGB2Thermal loss function (Wasserstein + temperature prior) and the first large-scale thermal landmark dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Structured evaluation across multiple benchmarks covering temperature prediction, perceptual quality, and landmark accuracy.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, detailed dataset comparisons, and comprehensive methodological descriptions.
- Value: ⭐⭐⭐⭐⭐ The dataset and method hold long-term value for the thermal community, and the approach offers insights for cross-modal synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reanimating Images using Neural Representations of Dynamic Stimuli](reanimating_images_using_neural_representations_of_dynamic_stimuli.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)
- [\[ICLR 2026\] MedVLSynther: Synthesizing High-Quality Medical Visual Question Answering from Biomedical Literature with Generator-Verifier LMMs](../../ICLR2026/medical_imaging/synthesizing_high-quality_visual_question_answering_from_medical_documents_with_.md)
- [\[ECCV 2024\] Topology-Preserving Downsampling of Binary Images](../../ECCV2024/medical_imaging/topology-preserving_downsampling_of_binary_images.md)
- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](../../ICCV2025/medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)

</div>

<!-- RELATED:END -->
