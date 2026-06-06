---
title: >-
  [Paper Note] Epistemic Uncertainty for Generated Image Detection
description: >-
  [NeurIPS 2025][Image Generation][AI-generated image detection] This paper proposes WePe (Weight Perturbation), which estimates epistemic uncertainty by applying weight perturbations to a pretrained vision foundation mode…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "AI-generated image detection"
  - "epistemic uncertainty"
  - "weight perturbation"
  - "DINOv2"
  - "out-of-distribution detection"
date: 2026-05-08
content_hash: f667cf7fffd62718
---

# Epistemic Uncertainty for Generated Image Detection

**Conference**: NeurIPS 2025
**arXiv**: [2412.05897](https://arxiv.org/abs/2412.05897)  
**Code**: [tmlr-group/WePe](https://github.com/tmlr-group/WePe)  
**Area**: Image Generation
**Keywords**: AI-generated image detection, epistemic uncertainty, weight perturbation, DINOv2, out-of-distribution detection

## TL;DR

This paper proposes WePe (Weight Perturbation), which estimates epistemic uncertainty by applying weight perturbations to a pretrained vision foundation model (DINOv2). The method exploits the divergence between natural and AI-generated images in uncertainty space for detection, requiring no training.

## Background & Motivation

With the advancement of generative models such as Stable Diffusion, DALL-E, and Midjourney, highly photorealistic AI-generated images pose security threats including deepfakes. Existing detection methods face several core challenges:

- **Poor generalization**: Binary classification-based methods (e.g., CNNspot) generalize poorly to unseen generators.
- **Data dependency**: Continuous collection of images from the latest generators is required to update training sets.
- **Inaccessibility**: When the latest generative models are not open-sourced, obtaining sufficient generated images for training is difficult.

A key observation is that even on a relatively simple network such as Inception v3, state-of-the-art generative models like ADM achieve an FID of 11.84 (far from 0), indicating a significant distributional gap between natural and generated images in feature space. This gap is even more pronounced on the more powerful DINOv2.

The core insight of this paper is that pretrained vision models, trained on natural images, exhibit low epistemic uncertainty for natural images and high epistemic uncertainty for generated images — this uncertainty gap can be directly exploited for detection without exposure to any generated images.

## Method

### Overall Architecture

WePe reformulates AI-generated image detection as an uncertainty estimation problem. The framework proceeds in three steps:

1. Extract image features using the pretrained DINOv2 model.
2. Apply random perturbations to model weights to obtain multiple feature predictions.
3. Estimate epistemic uncertainty via the variance of feature similarities; high uncertainty is classified as AI-generated.

The entire pipeline requires no generated images as training data.

### Key Designs

**Uncertainty analysis under the Bayesian framework**: Epistemic uncertainty reflects the model's "knowledge deficiency" regarding a data distribution and can be quantified through the posterior distribution over parameters. According to the Bernstein–von Mises theorem, as sample size $N \to \infty$, the posterior approximates a Gaussian centered at the MLE with covariance equal to the inverse Fisher information matrix. For in-distribution (natural image) data, posterior variance decreases as training data increases. For out-of-distribution (generated image) data, the Fisher information matrix mismatches the test distribution, causing epistemic uncertainty to remain persistently high.

**Uncertainty estimation via weight perturbation**: Conventional approaches such as MC Dropout and Deep Ensemble are unsuitable for large models (DINOv2 does not use dropout during training; multi-model ensembles are computationally infeasible). WePe proposes replacing these with weight perturbation:

The student model parameters $\theta$ of DINOv2 are perturbed $n$ times to obtain multiple sets of perturbed parameters, and the variance of student–teacher feature similarities is used as the uncertainty measure. Since the teacher model is not always available, an upper bound on uncertainty is derived: $u(x) \leq 2 - \frac{2}{n}\sum_k \cos\text{sim}(f(x;\theta_k), f(x;\theta))$.

This upper bound requires only the original and perturbed model parameters, without the teacher model. The core intuition is that if the cosine similarity between pre- and post-perturbation features is high (close to 1), uncertainty is low, and the image is more likely to be natural.

**Theoretical guarantee for perturbation sensitivity (Theorem 3.2)**: Perturbation sensitivity is defined as the squared Frobenius norm of the Jacobian of the feature map with respect to the parameters. It is proven that the expected sensitivity of natural images is lower than that of generated images — i.e., the feature representations of natural images are more robust to parameter perturbations, while generated images are more sensitive.

**Perturbation strategy**: Using DINOv2 ViT-L/14 (24 transformer blocks), only the first 19 blocks are perturbed (perturbations at higher layers excessively disrupt natural image features). The variance of Gaussian perturbations is proportional to the mean of each block's parameters, with a ratio of 0.1.

**WePe* (training-augmented variant)**: When training data is available, fine-tuning is used to amplify the uncertainty gap. The loss function encourages high pre/post-perturbation feature similarity for natural images and low similarity for generated images.

### Loss & Training

- **Training-free WePe**: Directly uses pretrained DINOv2 without any additional training.
- **Trained WePe***: Fine-tunes the DINOv2 student model with a contrastive loss to amplify the uncertainty gap between natural and generated images.

## Key Experimental Results

### Main Results

**Detection performance on ImageNet (9 generators, AUROC/AP in %):**

| Method | Type | ADM | BigGAN | GigaGAN | StyleGAN-XL | Avg. AUROC | Avg. AP |
|--------|------|-----|--------|---------|-------------|------------|---------|
| CNNspot | Trained | 62.25 | 85.71 | 74.85 | 68.41 | 67.04 | 66.78 |
| FatFormer | Trained | 91.77 | 98.76 | 97.65 | 97.64 | 93.68 | 93.11 |
| DRCT | Trained | 90.26 | 95.87 | 86.89 | 89.11 | 90.36 | 89.92 |
| **WePe*** | **Trained** | **93.89** | **99.85** | **99.03** | **99.52** | **95.57** | **94.33** |
| RIGID | Training-free | 87.16 | 90.08 | 86.39 | 86.32 | 83.58 | 81.58 |
| **WePe** | **Training-free** | **89.79** | **94.24** | **92.15** | **93.86** | **87.99** | **85.04** |

**Detection performance across backbone models:**

| Model | AUROC | AP |
|-------|-------|-----|
| DINOv2: ViT-S/14 | 72.83 | 71.63 |
| DINOv2: ViT-B/14 | 81.82 | 80.64 |
| DINOv2: ViT-L/14 | **87.99** | **85.04** |
| DINOv2: ViT-g/14 | 84.92 | 81.83 |
| CLIP: ViT-L/14 | 84.82 | 84.20 |

### Ablation Study

**Comparison of perturbation types:**

| Perturbation Type | AUROC | AP |
|-------------------|-------|-----|
| Gaussian noise | 87.99 | 85.04 |
| Uniform noise | **89.06** | **86.32** |
| Laplace noise | 87.13 | 84.22 |
| MC Dropout | 81.63 | 79.71 |

- All three weight perturbation strategies outperform MC Dropout.
- Uniform noise slightly outperforms Gaussian noise.

**Number of perturbed layers**: Good performance is achieved by perturbing the first 9–20 blocks, indicating robustness to layer selection.

**Perturbation magnitude**: The method is robust to noise levels, with performance degrading only at extreme magnitudes.

### Key Findings

1. Training-free WePe achieves an average AUROC of 87.99%, surpassing all training-free baselines (RIGID: 83.58%).
2. Trained WePe* achieves an average AUROC of 95.57% across 9 generators, comprehensively outperforming prior SOTA.
3. DINOv2 outperforms CLIP, as DINOv2's purely image-based self-supervised training focuses more on visual details.
4. WePe is robust to adversarial image perturbations (JPEG compression, Gaussian noise, Gaussian blur); such perturbations actually widen the distributional gap.
5. ViT-g/14 underperforms ViT-L/14, possibly due to excessive redundancy in the feature space of overly large models.

## Highlights & Insights

- **Paradigm shift**: Recasts detection as uncertainty estimation, transitioning from "learning to discriminate" to "sensing the unknown."
- **Solid theoretical grounding**: Provides a Bayesian-motivated analysis with a formal proof of perturbation sensitivity (Theorem 3.2).
- **FID correlation**: WePe's detection performance strongly correlates with the FID scores of generators, validating the distributional gap hypothesis.
- **Training-free deployment**: Leverages intrinsic properties of pretrained models without requiring any generated image collection.
- **Practical utility**: Code is open-sourced; the method is simple and efficient.

## Limitations & Future Work

- Training-free detection performance on diffusion model outputs (LDM, DiT) is relatively weaker (78.47 and 77.13 AUROC, respectively).
- The method depends on a specific pretrained model (DINOv2) and may fail if generators learn to "mimic" the DINOv2 feature space.
- The reason for ViT-g/14's performance degradation is not deeply analyzed.
- Only cosine similarity is used as the feature distance metric, potentially missing finer-grained distributional differences.
- Computational efficiency is not discussed: the time overhead of multiple weight-perturbed forward passes is unaddressed.

## Related Work & Insights

- **RIGID (He et al., 2024)**: Demonstrates that natural images are more robust to input noise perturbations, inspiring WePe's approach from the weight perturbation perspective.
- **AEROBLADE**: A training-free method based on autoencoder reconstruction error, but with overly strong assumptions.
- **NPR (Tan et al., 2024)**: Exploits differences in neighboring pixel relationships for detection, but is not robust to adversarial attacks.
- Inspiration: The uncertainty characteristics of pretrained models may serve as an effective signal for general OOD detection.

## Rating

- **Novelty**: 4/5 — The uncertainty-based perspective is original, though the underlying idea is related to OOD detection.
- **Value**: 5/5 — Training-free, open-source, and plug-and-play.
- **Experimental Thoroughness**: 5/5 — Evaluated on 4 benchmarks, 9 generators, with extensive ablations and adversarial robustness tests.
- **Writing Quality**: 4/5 — Motivation and theoretical derivations are clear; experiments are thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Is Artificial Intelligence Generated Image Detection a Solved Problem?](is_artificial_intelligence_generated_image_detection_a_solved_problem.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](../../CVPR2026/image_generation/diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)

</div>

<!-- RELATED:END -->
