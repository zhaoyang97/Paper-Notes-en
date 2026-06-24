---
title: >-
  [Paper Note] Beyond One-Hot Labels: Semantic Mixing for Model Calibration
description: >-
  [ICML 2025][Image Generation][Model Calibration] Proposes CSM (Calibration-aware Semantic Mixing), which leverages pre-trained diffusion models to generate high-fidelity semantically mixed samples (e.g., cat-dog hybrids) and accurately re-annotates soft label confidence using CLIP. Training with an $L_2$ loss achieves superior model confidence calibration compared to existing calibration methods.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Model Calibration"
  - "Data Augmentation"
  - "Diffusion Models"
  - "Semantic Mixing"
  - "Confidence Annotation"
date: 2026-05-08
content_hash: 080d60bcdb0a766a
---

# Beyond One-Hot Labels: Semantic Mixing for Model Calibration

**Conference**: ICML 2025  
**arXiv**: [2504.13548](https://arxiv.org/abs/2504.13548)  
**Code**: Yes (GitHub)  
**Area**: Image Generation / Model Calibration  
**Keywords**: Model Calibration, Data Augmentation, Diffusion Models, Semantic Mixing, Confidence Annotation

## TL;DR
Proposes CSM (Calibration-aware Semantic Mixing), which leverages pre-trained diffusion models to generate high-fidelity semantically mixed samples (e.g., cat-dog hybrids) and accurately re-annotates soft label confidence using CLIP. Training with an $L_2$ loss achieves superior model confidence calibration compared to existing calibration methods.

## Background & Motivation

**Background**: Deep neural networks are usually overconfident, meaning that their predicted confidence does not accurately reflect the actual accuracy. Model calibration aims to align confidence with accuracy. Existing methods are categorized into post-processing calibration (Temperature Scaling), training-time regularization (Label Smoothing, Focal Loss), and data augmentation (Mixup).

**Limitations of Prior Work**:
   - All existing methods rely on one-hot labels, implicitly assuming that all annotations possess 100% certainty; however, a clear cat and a blurry cat-dog hybrid should have different confidence annotations.
   - Mixup-like methods generate low-fidelity samples (pixel overlap/concatenation), exhibiting a large discrepancy from the real data distribution.
   - There is a lack of training data annotated with realistic "uncertainty," which is extremely rare in nature.

**Key Challenge**: Models need to learn "when to be uncertain," yet training data lacks uncertain samples and corresponding annotations.

**Goal**: Generate high-fidelity mixed samples with realistic uncertainty annotations to train model calibration.

**Key Insight**: Leverage the conditional generation capability of diffusion models to generate a series of semantically continuous samples using the same initial noise but under different class mixing ratios.

**Core Idea**: Generate semantically mixed samples via diffusion models + accurate confidence re-annotation with CLIP + balanced learning via $L_2$ loss.

## Method

### Overall Architecture
The CSM workflow is as follows:
1. **Data Generation**: Uses a pre-trained diffusion model to generate a sequence of semantically mixed images (e.g., cat $\to$ cat-dog hybrid $\to$ dog) under the same noise and varying class mixing ratios.
2. **Confidence Re-annotation**: Employs class prototype projection in the CLIP feature space to accurately estimate the class posterior probability for each mixed image.
3. **Calibration Training**: Jointly trains the model on both real and mixed data using an $L_2$ loss (instead of cross-entropy).

### Key Designs

1. **Diffusion Model Semantic Mixing**:

    - Function: Generate high-fidelity samples that represent continuous transitions between classes.
    - Mechanism: Fix the initial noise $z_T$, and use $\alpha \cdot c_{\text{cat}} + (1-\alpha) \cdot c_{\text{dog}}$ as the condition during reverse diffusion, where $\alpha$ varies from 0 to 1.
    - Difference from Mixup: Mixup overlaps two images in pixel space, producing "ghosting" artifacts, whereas CSM mixes concepts in semantic space to generate complete, coherent transitional objects.
    - Design Motivation: The conditional generation of diffusion models ensures semantic coherence and image fidelity.

2. **CLIP Re-annotation (Calibrated Reannotation)**:

    - Function: Correct the discrepancy between the diffusion model's mixing ratio and the actual visual class posterior.
    - Mechanism:
        - Encode the mixed image using CLIP to obtain the feature $f$.
        - Calculate class prototypes $\{p_k\}$ (the average CLIP features of samples from the same class).
        - Project $f$ onto the class prototype frame to obtain accurate soft labels.
    - Design Motivation: The mixing ratio $\alpha$ in diffusion models does not always accurately reflect the class posterior of the final generated image. CLIP provides a more objective measure of semantic distance.
    - Formulation: $\hat{y}_k = \text{sim}(f, p_k) / \sum_{k'} \text{sim}(f, p_{k'})$

3. **Theoretical Advantages of $L_2$ Loss**:

    - Function: Prove that the $L_2$ loss is more suitable for soft-label training than cross-entropy.
    - Mechanism: Cross-entropy loss has a zero gradient for classes with a label of 0, which leads to "imbalanced fitting" in soft-label scenarios (where the model over-focuses on high-confidence classes). Conversely, $L_2$ loss yields non-zero gradients for all classes, leading to balanced fitting.
    - Theoretical Conclusion: The optimal solution for $L_2$ is $p^* = y$ (prediction matches annotation), while the optimal solution $p^*$ for cross-entropy is biased toward 0 for non-target classes under soft labels.
    - Design Motivation: Calibration requires the model to learn to "output intermediate values at intermediate confidence levels," which the $L_2$ loss naturally facilitates.

### Loss & Training
- Total Loss = $L_{\text{CE}}$(real data) + $\lambda L_2$(mixed data)
- Alternate training between mixed data and real data.
- No modifications required to the model architecture—improvements are strictly at the data and loss levels.

## Key Experimental Results

### Main Results
ECE (Expected Calibration Error) ↓ on CIFAR-100 / ImageNet:

| Method | ECE (CIFAR-100) ↓ | ECE (ImageNet) ↓ | Acc ↑ |
|------|------------------|-----------------|-------|
| Cross-Entropy (Baseline) | 8.74 | 5.12 | 78.2 / 76.5 |
| Temperature Scaling | 3.21 | 2.85 | 78.2 / 76.5 |
| Mixup | 5.43 | 3.92 | 79.1 / 76.8 |
| Label Smoothing | 4.15 | 3.45 | 78.5 / 76.6 |
| RegMixup | 4.02 | 3.31 | 79.0 / 76.9 |
| **CSM (Ours)** | **2.51** | **2.12** | **79.3 / 77.2** |

### Ablation Study

| Configuration | ECE (CIFAR-100) | Description |
|------|----------------|------|
| Mixup Augmentation + CE Loss | 5.43 | Low-fidelity mixing |
| Diffusion Mixing + Mixing Ratio Labels + CE | 3.85 | High-fidelity but imprecise labels |
| Diffusion Mixing + CLIP Re-annotation + CE | 3.12 | Accurate labels but biased CE |
| **Diffusion Mixing + CLIP Re-annotation + $L_2$** | **2.51** | Full method |

### Key Findings
- CSM simultaneously improves both calibration (ECE) and accuracy, unlike other methods that exhibit an accuracy-calibration trade-off.
- Mixed samples generated by diffusion models are indeed closer to the natural data distribution compared to Mixup.
- CLIP re-annotation is more accurate than directly using the mixing ratios (ECE drops from 3.85 to 3.12).
- The theoretical advantages of the $L_2$ loss under soft-label scenarios are validated experimentally.

## Highlights & Insights
- **"A model needs to see uncertain samples to learn how to be uncertain"**—a simple yet profound insight.
- The cross-domain combination of diffusion models and calibration is highly innovative—solving a discriminative model problem using generative models.
- CLIP re-annotation compensates for the imprecision of diffusion mixing ratios—an elegant synergy of two powerful tools.
- Theoretical analysis of the $L_2$ loss reveals why cross-entropy is unsuitable for soft labels—a finding with independent value.
- General methodology: any classifier can benefit without modifying the model architecture.

## Limitations & Future Work
- Requires pre-trained diffusion models and CLIP, thus depending on external models.
- The computational cost of data generation is non-trivial (diffusion sampling).
- Validated only on image classification; other tasks (detection, segmentation) remain to be explored.
- Mixing is restricted to two classes; mixing three or more classes is more challenging.

## Related Work & Insights
- **vs Mixup/CutMix**: Pixel-level mixing has low fidelity, whereas CSM offers high-fidelity semantic-level mixing.
- **vs Temperature Scaling**: Post-processing methods do not improve the model itself, whereas CSM improves the model from the training data.
- **vs Label Smoothing**: Uniformly smooths the confidence of all samples, whereas CSM generates precise, personalized confidence for each sample.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Cross-field innovation combining diffusion models and calibration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across multiple datasets, comparison with multiple methods, and thorough ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, excellent integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Calibration is a key issue in trustworthy AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment](beyond_bradley-terry_models_a_general_preference_model_for_language_model_alignm.md)
- [\[NeurIPS 2025\] Enhancing Diffusion Model Guidance through Calibration and Regularization](../../NeurIPS2025/image_generation/enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](../../CVPR2026/image_generation/emf_meanflow_text_to_image.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](../../CVPR2025/image_generation/osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICLR 2026\] VMDiff: Visual Mixing Diffusion for Limitless Cross-Object Synthesis](../../ICLR2026/image_generation/vmdiff_visual_mixing_diffusion_for_limitless_cross-object_synthesis.md)

</div>

<!-- RELATED:END -->
