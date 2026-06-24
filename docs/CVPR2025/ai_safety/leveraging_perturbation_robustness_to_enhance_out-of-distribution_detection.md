---
title: >-
  [Paper Note] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection
description: >-
  [CVPR 2025][AI Safety][Out-of-Distribution Detection] The authors find that the detection scores of OOD samples are more vulnerable to adversarial perturbations than those of IND samples. They propose the PRO method, which searches for the minimum OOD score within the $\epsilon$-ball using gradient descent during inference to enhance IND/OOD separability, reducing FPR@95 on CIFAR-10 from 44.35% to 19.95%.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Out-of-Distribution Detection"
  - "Perturbation Robustness"
  - "Adversarial Training"
  - "Post-Processing Method"
  - "Score Function"
date: 2026-05-08
content_hash: 35b9f2ccb54be895
---

# Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.18784](https://arxiv.org/abs/2503.18784)  
**Code**: [https://github.com/wenxichen2746/Perturbation-Rectified-OOD-Detection](https://github.com/wenxichen2746/Perturbation-Rectified-OOD-Detection)  
**Area**: Optimization / OOD Detection  
**Keywords**: Out-of-Distribution Detection, Perturbation Robustness, Adversarial Training, Post-Processing Method, Score Function

## TL;DR

The authors find that the detection scores of OOD samples are more vulnerable to adversarial perturbations than those of IND samples. They propose the PRO method, which searches for the minimum OOD score within the $\epsilon$-ball using gradient descent during inference to enhance IND/OOD separability, reducing FPR@95 on CIFAR-10 from 44.35% to 19.95%.

## Background & Motivation

**Background**: OOD detection determines whether an input belongs to the training distribution of the model. Post-processing methods (e.g., MSP, Energy, GEN) compute OOD scores during inference without modifying the model training process, offering high practicality.

**Limitations of Prior Work**: The core assumption of post-processing methods is that IND and OOD samples have a clear boundary in the score space. However, in practice, their score distributions often overlap significantly, which is particularly severe in near-distribution OOD tasks (e.g., CIFAR-10 vs. CIFAR-100).

**Key Challenge**: IND and OOD samples overlap in the original score space, but they exhibit different levels of "vulnerability" to perturbations. The scores of OOD samples are more easily degraded by minor perturbations because the model's predictions on them are inherently unstable.

**Key Insight**: Leveraging this difference in perturbation robustness: by searching for the minimum OOD score within the $\epsilon$-ball via gradient descent, the minimum scores of OOD samples are significantly suppressed, while those of IND samples remain relatively unchanged, thereby widening the gap.

**Core Idea**: Searching for the perturbed minimum OOD score during inference $\rightarrow$ OOD scores are suppressed $\rightarrow$ IND and OOD become more separable.

## Method

### Key Designs

1. **Perturbation-Rectified OOD (PRO)**:

    - **Function**: Enhances the discriminative capability of any OOD score function
    - **Mechanism**: $g^*(\mathbf{x}) = \min_{\|\delta\|_\infty \leq \epsilon} g(\mathbf{x}+\delta)$, using an iterative PGD-style descent search: $\mathbf{x}_t = \mathbf{x}_{t-1} - \epsilon \cdot \text{sign}(\nabla g(\mathbf{x}_{t-1}))$. Due to unstable model predictions, the scores of OOD samples decrease significantly after minimization, whereas IND samples maintain stable scores due to prediction robustness.
    - **Design Motivation**: Opposite to the direction of adversarial attacks—while adversarial attacks maximize loss, PRO minimizes the OOD score. Both leverage the same "non-robust region".

2. **Synergy with Adversarial Training**:

    - **Function**: Provides a clearer boundary for adversarially trained models
    - **Mechanism**: Adversarial training makes the IND distribution more compact, leading to smaller score variations after perturbation. Comparison: robust model FPR@95 = 26.36% vs. non-robust model 31.38%.
    - **Design Motivation**: PRO is best suited for deployment with adversarially trained models.

### Loss & Training

PRO is a post-processing method that does not modify training. During inference, $K$-step gradient descent is performed on each test sample to search for the minimum score. It can be integrated with any OOD score function (e.g., MSP, Entropy, Temperature, GEN).

## Key Experimental Results

### Main Results

CIFAR-10 FPR@95↓:

| Method | Enhanced by PRO | Original |
|------|----------|------|
| Scale (GEN) | **19.95%** | 44.35% |
| Temperature | **31.38%** | 37.21% |
| MSP | Significant Gain | - |

### Key Findings
- **Consistently effective across all OOD score functions**: MSP, Entropy, Temperature, and GEN are all significantly improved by PRO.
- **Better performance with adversarially trained models**: Robust models achieve an additional 5% FPR reduction compared to standard models.
- **Effective for near-distribution OOD**: The FPR for CIFAR-100/TIN (near-distribution) drops from 37.21% to 31.38%.

## Highlights & Insights
- **First explicit connection between adversarial robustness and OOD detection**: The "vulnerability" of OOD samples is precisely the cue to detect them.
- **Universal post-processing enhancer**: Can improve any existing OOD scoring method.

## Limitations & Future Work
- Requires gradient descent during inference, increasing computational overhead by $K$ times.
- Requires an adversarially trained model to achieve optimal performance.
- Limited experiments on ImageNet scale.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel insight connecting perturbation robustness to OOD detection
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple score functions, comparison between robust/non-robust models
- Writing Quality: ⭐⭐⭐⭐ Clear motivational reasoning
- Value: ⭐⭐⭐⭐ Universal enhancement method for OOD detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[ICLR 2026\] GradPCA: Leveraging NTK Alignment for Reliable Out-of-Distribution Detection](../../ICLR2026/ai_safety/gradpca_leveraging_ntk_alignment_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[NeurIPS 2025\] SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
