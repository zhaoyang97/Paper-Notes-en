---
title: >-
  [Paper Note] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights
description: >-
  [ICCV 2025][Object Detection][Adversarial Patch Defense] This paper systematically revisits 11 adversarial patch defense methods, establishes the first patch defense benchmark covering 13 attacks, 11 detectors…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Adversarial Patch Defense"
  - "Benchmark Evaluation"
  - "Large-Scale Dataset"
  - "Adaptive Attack"
date: 2026-05-08
content_hash: b1c19ec2aabe08dd
---

# Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights

**Conference**: ICCV 2025
**arXiv**: [2508.00649](https://arxiv.org/abs/2508.00649)
**Code**: [https://github.com/Gandolfczjh/APDE](https://github.com/Gandolfczjh/APDE)
**Area**: Adversarial Robustness / Object Detection
**Keywords**: Adversarial Patch Defense, Object Detection, Benchmark Evaluation, Large-Scale Dataset, Adaptive Attack

## TL;DR

This paper systematically revisits 11 adversarial patch defense methods, establishes the first patch defense benchmark covering 13 attacks, 11 detectors, and 4 metrics, constructs a large-scale APDE dataset of 94,000 images, and reveals three key insights: the difficulty of defending against natural adversarial patches stems from data distribution rather than high-frequency components; patch detection accuracy is inconsistent with defense performance; and adaptive attacks can circumvent most existing defenses.

## Background & Motivation

Adversarial patch attacks pose a significant security threat to DNNs in the physical world, particularly in scenarios such as pedestrian detection and autonomous driving. Numerous defense methods have been proposed in recent years; however, existing evaluations suffer from four major issues:

**Lack of a unified framework**: Different papers employ inconsistent hyperparameters, attack methods, and patch placement strategies, preventing fair comparison.

**Inappropriate metrics**: Some works evaluate defense performance solely via patch detection accuracy, yet high detection accuracy does not imply effective defense.

**Incomplete analysis**: Key factors such as inference latency, the impact of varying patch sizes and types, and physical-world applicability are often neglected.

**Insufficient attacks**: Existing patch datasets are small-scale and lack coverage categorized by detector.

## Method

### Overall Architecture

A complete evaluation pipeline is established: attack generation → dataset construction → unified evaluation → comprehensive analysis. The core contributions are the APDE dataset and the benchmark evaluation framework.

### Key Designs

1. **APDE Dataset Construction**:

    - White-box attacks are performed using 13 attack methods against 11 detectors, generating 94 distinct patches.
    - Patches are applied to the INRIA-Person and MS COCO test sets, yielding 94,000 images in total.
    - The dataset is split into 56,400 training images and 37,600 test images (6:4 ratio).
    - Compared to existing datasets: Apricot (60 patch types, 1,011 images) and GAP (25 types, 9,266 images).
    - Advantages: large scale, diverse patch distribution, and white-box setting (worst-case evaluation).

2. **Evaluation Metric System**:

    - **AP@0.5**: Average precision on attacked targets, directly reflecting defense effectiveness.
    - **ASR (Attack Success Rate)**: Measures the residual effect of the attack.
    - **mIoU (SmIoU / NmIoU)**: A substitute for patch AP@0.5, applicable to irregularly shaped patches.
    - **Inference Time**: Evaluates computational practicality.
    - Core finding: AP on attacked targets reflects true defense capability more faithfully than patch detection accuracy.

3. **Defense Method Taxonomy and Evaluation**:

    - Patch detection/segmentation-based: SAC, PAD, Adyolo, NAPGuard.
    - Patch prior knowledge-based: LGS, Zmask, Jedi.
    - Generative model-based: DIFFender, NutNet.
    - Certified defenses: DetectorGuard, ObjectSeeker.
    - Both hiding attacks and appearance attacks are covered.

### Loss & Training

Adversarial patch generation follows the general objective: $\delta^* = \arg\min_\delta \mathbb{E}_{x \sim X}[\mathcal{L}(f_i(\mathcal{A}(x, \delta, t)), y)] + \lambda L_{tv}(\delta)$, where $L_{tv}$ denotes total variation loss to promote smoother patches.

## Key Experimental Results

### Main Results

Defense performance against hiding attacks across 11 detectors (Person AP@0.5):

| Defense Method | Type | Overall Mean | Overall Min | Inference Time (ms) |
|---|---|---|---|---|
| w/o defense | - | 30.74 | - | - |
| SAC | Patch Segmentation | 60.88 | 20.62 | 44 |
| PAD | Patch Segmentation | 76.12 | 40.99 | 32,100 |
| Adyolo | Patch Detection | 63.12 | 26.59 | 62 |
| NAPGuard | Patch Detection | 75.94 | 46.93 | 59 |
| DIFFender | Generative Model | 56.23 | 11.98 | 1,240 |
| **NutNet** | **Generative Model** | **76.53** | **55.79** | **71** |
| LGS | Prior Knowledge | 71.58 | 29.55 | 82 |
| Zmask | Prior Knowledge | 56.71 | 6.43 | 417 |
| Jedi | Prior Knowledge | 58.85 | 18.96 | 349 |

### Ablation Study

Comparison of defense performance before and after retraining with the APDE dataset (average AP@0.5 on YOLOv3 + FRCNN):

| Attack Method | SAC (Original) | SAC (Retrained) | NAPGuard (Original) | NAPGuard (Retrained) |
|---|---|---|---|---|
| T-SEA | 51.82 | 71.61 | 83.61 | 86.31 |
| TC-EGA | 58.16 | 71.36 | 68.51 | 85.30 |
| AdvPatch | 56.53 | 73.29 | 78.45 | 85.10 |
| GNAP (Natural Patch) | 70.03 | 76.86 | 78.96 | 85.42 |
| AdvCloak (Out-of-domain) | 4.17 | 71.29 | 52.21 | 73.16 |
| AdvTshirt (Out-of-domain) | 34.27 | 64.47 | 50.21 | 70.89 |

An average gain of **15.09% AP@0.5** is observed, with particularly significant improvements on out-of-domain patches.

### Key Findings

1. **The difficulty of defending against natural patches stems from data distribution, not high-frequency components**: The high-frequency components of NAP (natural adversarial patches) and non-NAP patches differ marginally, whereas their FID distances are substantially larger. Defense methods fundamentally rely on data distribution to determine whether a pixel belongs to a patch region.
2. **Patch detection accuracy ≠ defense effectiveness**: NAPGuard achieves the highest detection accuracy yet performs worse than NutNet in terms of defense; AP reflects defense performance more faithfully than mIoU.
3. **Adaptive attacks can bypass most defenses**: PAD (utilizing the complex SAM model) and DIFFender (leveraging stochastic diffusion) exhibit greater robustness; Zmask and Jedi, which exploit universal patch properties (feature over-activation and high entropy), also demonstrate relative robustness.
4. **Physical-world defense is effective**: Methods that perform well in the digital domain generally transfer to the physical world; increased distance and enhanced illumination tend to benefit defense.
5. **Multi-patch scenarios**: The performance of certified defenses degrades more gradually as the number of patches increases, but computational cost grows exponentially.

## Highlights & Insights

- **First systematic patch defense benchmark**: Unifies the evaluation paradigm, resolving the long-standing issue of incomparable results across papers.
- **Novel data-distribution perspective**: Challenges the prevailing belief that high-frequency features are the primary reason natural adversarial patches are difficult to defend against.
- **Practicality-oriented**: Beyond evaluation, the APDE dataset can be directly leveraged to improve the performance of existing defenses.
- **NutNet achieves the best overall performance**: It offers the best combination of defense effectiveness, inference speed, and robustness.

## Limitations & Future Work

- The study primarily focuses on pedestrian detection; generalizability to other object categories remains to be verified.
- Certified defenses are constrained by strict threat model assumptions (patch count and size), limiting practical applicability.
- Physical-world experiments were conducted exclusively with an iPhone 16 Pro, offering limited sensor diversity.
- More complex scenarios such as 3D adversarial attacks and inter-frame consistency in video are not covered.

## Related Work & Insights

- The data distribution perspective can inspire the design of novel defense methods, such as using distributional distance to identify patch regions.
- The success of generative model-based defenses (NutNet, DIFFender) suggests that diffusion model-based image inpainting and denoising may constitute a promising defense paradigm.
- The dataset construction methodology of APDE can be generalized to other adversarial robustness research domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First unified benchmark; the data distribution findings are of substantial value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 defenses × 13 attacks × 11 detectors — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, in-depth analysis, and convincing findings.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and benchmark represent a significant contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)

</div>

<!-- RELATED:END -->
