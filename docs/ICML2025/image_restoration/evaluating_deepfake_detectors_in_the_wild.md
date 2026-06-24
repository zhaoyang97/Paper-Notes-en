---
title: >-
  [Paper Note] Evaluating Deepfake Detectors in the Wild
description: >-
  [ICML 2025][Image Restoration][Deepfake Detection] A new dataset containing over 500k high-quality deepfake images is constructed. By introducing in-the-wild enhancements such as JPEG compression, resolution reduction, and image restoration, six open-source deepfake detectors are systematically evaluated, revealing that fewer than half achieved an AUC > 60%, with the lowest performance around 50% (random-guess level).
tags:
  - "ICML 2025"
  - "Image Restoration"
  - "Deepfake Detection"
  - "Robustness"
  - "Face Swapping"
  - "In-the-wild Evaluation"
date: 2026-05-08
content_hash: 09e2263ba3b65b0b
---

# Evaluating Deepfake Detectors in the Wild

**Conference**: ICML 2025  
**arXiv**: [2507.21905](https://arxiv.org/abs/2507.21905)  
**Code**: [github.com/SumSubstance/Deepfake-Detectors-in-the-Wild](https://github.com/SumSubstance/Deepfake-Detectors-in-the-Wild)  
**Area**: Image Restoration  
**Keywords**: Deepfake Detection, Robustness, Face Swapping, In-the-wild Evaluation  

## TL;DR
A new dataset containing over 500k high-quality deepfake images is constructed. By introducing in-the-wild enhancements such as JPEG compression, resolution reduction, and image restoration, six open-source deepfake detectors are systematically evaluated, revealing that fewer than half achieved an AUC > 60%, with the lowest performance around 50% (random-guess level).

## Background & Motivation
**Background**: With the development of generative models (GANs/diffusion models), high-quality facial forgery has become highly accessible. Numerous deepfake detectors have been proposed, but they are primarily evaluated on legacy datasets such as DFDC and FaceForensics. These datasets rely on pre-2019 generation methods, which feature low image quality and obvious artifacts.

**Limitations of Prior Work**: (1) Existing benchmarks do not reflect in-the-wild deepfake quality—the quality of deepfakes in the DFDC/FaceForensics datasets is far inferior to currently generated ones; (2) detectors perform almost perfectly on legacy benchmarks, but their actual performance when deployed remains unknown; (3) fraudsters often use post-processing such as JPEG compression, downscaling, and GPEN/CodeFormer enhancement to bypass detection.

**Goal**: Systematically evaluate the practical capability of open-source deepfake detectors under in-the-wild conditions, revealing the huge gap between current detection techniques and actual demands.

**Key Insight**: Generate a new dataset using the most popular current zero-shot face-swapping models (SimSwap, Inswapper/roop), combined with post-processing techniques commonly used by fraudsters for evaluation.

## Method

### Overall Architecture
A testing pipeline mimicking in-the-wild scenarios is designed: (1) Over 500k high-quality deepfake images are generated using two popular face-swapping models, SimSwap and Inswapper, across three face datasets (CelebA-HQ, LFW, FairFace); (2) four types of in-the-wild enhancements are applied to the generated images: JPEG compression (factors of 75/50/30/10), resolution reduction (128 pixels), and GPEN image enhancement; (3) six open-source detectors are evaluated: FaceForensics++ (XceptionNet), MAT, M2TR, RECCE, CADDM, and SBI.

### Key Designs
1. **In-the-Wild Dataset Construction**:
    - Function: Generate a high-quality deepfake dataset using state-of-the-art face-swapping models.
    - Mechanism: Perform face-swapping on three datasets (CelebA-HQ, LFW, and FairFace) using SimSwap (resolutions of $224\times224$ and $512\times512$) and Inswapper (resolution of $128$) to ensure gender, age, and race match. A total of over 500k deepfake images are generated.
    - Design Motivation: Pre-existing DFDC and FaceForensics datasets use outdated generation methods with prominent artifacts, which fail to reflect the competency of current deepfake technologies.

2. **Real Attack Simulation Enhancement**:
    - Function: Simulate common strategies used by fraudsters to bypass detection.
    - Mechanism: Grouped into artificial degradation (JPEG compression, downscaling to 128 pixels) and artificial enhancement (GPEN face restoration). JPEG compression alters pixel values leading to compression artifacts; downscaling simulates low-quality recording; GPEN enhancement elevates low-quality generation results to a level visually indistinguishable from real images.
    - Design Motivation: Testing only on pristine images is insufficient—in practical fraud scenarios, attackers post-process generated images to bypass security measures.

3. **Multi-Dimensional Systematic Evaluation**:
    - Function: Comprehensively assess detector performance across multiple metrics and dimensions.
    - Mechanism: Six detectors are cross-evaluated across three datasets, two generators, and four enhancement conditions using five metrics: ROC-AUC, F1 (threshold 0.5), PR-AUC, LogLoss, and Accuracy.
    - Design Motivation: Evaluation under a single metric or condition tends to yield misleading conclusions.

### Loss & Training
This paper is an evaluation study and does not introduce new training strategies. All detectors perform inference using their original publicly available weights.

## Key Experimental Results

### Main Results (Pristine Data, No Enhancement)

| Detector | SimSwap AUC | SimSwap Acc | Inswapper AUC | Inswapper Acc |
|--------|-------------|-------------|---------------|---------------|
| FF (XceptionNet) | 51.7 | 50.1 | 56.7 | 54.8 |
| MAT | 79.7 | 53.7 | 80.3 | 53.7 |
| M2TR | 55.3 | 53.4 | 53.9 | 52.6 |
| RECCE | 56.7 | 53.5 | 56.1 | 53.6 |
| CADDM | 78.2 | 68.5 | 59.8 | 56.9 |
| SBI | **95.5** | **69.7** | 75.9 | 64.2 |

### Performance Changes After Enhancement (Overall AUC)

| Detector | Original (SimSwap) | JPEG (75) | Resolution Reduction | GPEN Enhancement |
|--------|--------------|----------|---------|---------|
| FF | 51.7 | 54.9 | 62.9 | 20.2 |
| MAT | 79.7 | 88.1 | 83.0 | 76.3 |
| SBI | 95.5 | 95.0 | 72.3 | 67.1 |
| CADDM | 78.2 | 78.3 | 72.6 | 58.2 |

### Key Findings
- SBI achieves the highest AUC on SimSwap (95.5), but its performance plummets to 67.1 after GPEN enhancement—showing that enhancement dramatically impairs detection capability.
- FF (XceptionNet) yields an AUC of only 51.7, which is close to a random guess, indicating that models trained on FaceForensics++ generalize poorly.
- GPEN enhancement severely affects all detectors, improving low-quality generation results to a level where detectors struggle to differentiate.
- Fewer than half of the detectors achieve an AUC > 60% in the overall evaluation.

## Highlights & Insights
- Systematically reveals the vulnerability of current deepfake detectors in real-world conditions—almost perfect performance on legacy benchmarks does not translate to practical utility.
- Image enhancement tools such as GPEN/CodeFormer pose severe threats to deepfake detection, restoring low-quality generated results to look highly realistic.
- Publicly releases the dataset of over 500k high-quality deepfakes and the complete evaluation codebase, providing essential infrastructure for future research.
- Evaluations on FairFace partitioned by race, gender, and age group uncover fairness issues in current detectors.

## Limitations & Future Work
- Evaluates only face-swapping deepfakes, leaving entire face synthesis (e.g., Stable Diffusion-generated faces) out of scope.
- Employs only two generators (SimSwap and Inswapper), without covering faceset-based methods like DeepFaceLab.
- Does not investigate defense strategies or adversarial training schemes against enhancement attacks.
- Only evaluates open-source detectors; commercial detectors (such as Microsoft Video Authenticator) are not included in the evaluation.

## Related Work & Insights
- **vs FaceForensics++ (ICCV19)**: Provides datasets and the XceptionNet baseline. This work shows that models trained on it generalize extremely poorly (AUC $\approx$ 50%).
- **vs SBI (CVPR22)**: Trained on self-blended images, it achieves a SimSwap AUC of 95.5 on the new dataset, making it the best detector, but it lacks robustness against GPEN enhancement.
- **vs CADDM (CVPR23)**: Addresses implicit identity leakage; achieves a SimSwap AUC of 78.2 but only 59.8 on Inswapper, indicating poor generalization across generators.
- Insight: Future detectors should incorporate image enhancement and compression into their training data pipelines to enhance robustness in real-world scenarios.

## Rating
- Novelty: ⭐⭐⭐ Primarily an evaluation work with limited methodological innovation, but the evaluation design and dataset contribution are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive cross-evaluation spanning 6 detectors $\times$ 2 generators $\times$ 3 datasets $\times$ 4 enhancement types.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and well-designed evaluation protocol.
- Value: ⭐⭐⭐⭐ Highly valuable alert to the deepfake detection community, with public access to the dataset and code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration](harmonica_harmonizing_training_and_inference_for_better_feature_caching_in_diffu.md)
- [\[ICML 2025\] Adaptive Estimation and Learning under Temporal Distribution Shift](adaptive_estimation_and_learning_under_temporal_distribution_shift.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](epsilon-vae_denoising_as_visual_decoding.md)
- [\[ICML 2025\] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](../../NeurIPS2025/image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)

</div>

<!-- RELATED:END -->
