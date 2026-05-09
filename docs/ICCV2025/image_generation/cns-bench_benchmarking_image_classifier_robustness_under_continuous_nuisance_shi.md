---
title: >-
  [Paper Note] CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts
description: >-
  [ICCV2025][Image Generation][OOD Robustness] This paper proposes CNS-Bench, the first benchmark that leverages LoRA adapters to impose **continuous** and **photorealistic** nuisance shifts on diffusion models for systematically evaluating the OOD robustness of image classifiers, covering 14 shift types, 5 severity levels, and 40+ classifiers.
tags:
  - ICCV2025
  - Image Generation
  - OOD Robustness
  - Continuous Nuisance Shifts
  - LoRA Adapters
  - Diffusion Models
  - Image Classifier Benchmarking
date: 2026-05-08
content_hash: c564ae168f07364e
---

# CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts

**Conference**: ICCV2025  
**arXiv**: [2507.17651](https://arxiv.org/abs/2507.17651)  
**Code**: [odunkel/CNS-Bench](https://github.com/odunkel/CNS-Bench)  
**Area**: Image Generation  
**Keywords**: OOD Robustness, Continuous Nuisance Shifts, LoRA Adapters, Diffusion Models, Image Classifier Benchmarking

## TL;DR

This paper proposes CNS-Bench, the first benchmark that leverages LoRA adapters to impose **continuous** and **photorealistic** nuisance shifts on diffusion models for systematically evaluating the OOD robustness of image classifiers, covering 14 shift types, 5 severity levels, and 40+ classifiers.

## Background & Motivation

Evaluating visual models under out-of-distribution (OOD) conditions is critical for real-world deployment. Existing robustness evaluation methods fall into four categories:

**Manually collected data** (e.g., OOD-CV): labor-intensive, difficult to control nuisance factors, and nuisance types are heavily entangled.

**Synthetic corruptions** (e.g., ImageNet-C): support continuous severity levels, but are limited to simple pixel-level perturbations and fail to reflect real-world distribution shifts.

**Rendering pipelines** (e.g., 3D asset rendering): require large amounts of 3D models and do not scale to large-scale categories.

**Diffusion model-based generation** (e.g., Dataset Interfaces): can produce realistic images, but only support **binary** shifts (present/absent), failing to capture continuous variation.

**Key gap**: Nuisance shifts in the real world (e.g., snowfall, fog, style changes) are inherently **continuous**. For instance, in autonomous driving, snow accumulation is a gradual process; different models may fail at different severity levels. No existing benchmark simultaneously satisfies the properties of realism, continuity, and scalability. CNS-Bench addresses this gap.

## Method

### 3.1 Replicating the ImageNet Distribution (IN*)

The image distribution $p(X_{SD}|c)$ generated directly by Stable Diffusion differs substantially from the ImageNet distribution $p(X_{IN}|c)$, leading to significant drops in classification accuracy. To address this, the authors employ **Textual Inversion** to learn class-specific text embeddings for each ImageNet category, bringing the generated images closer to the ImageNet distribution. The optimization objective minimizes the diffusion model's noise prediction error:

$$\|\epsilon - \epsilon_\psi(\cdot, f_\psi(c))\|^2$$

The learned distribution is referred to as IN*: $p(X|c) = p(X_{IN^*}|c)$. Experiments show that IN* reduces FID from 33.8 to 27.1 and improves ResNet-50 classification accuracy from 0.68 to 0.74 compared to standard SD generation.

### 3.2 LoRA-Based Continuous Nuisance Shifts

The core idea is to leverage **LoRA adapters** to learn the "direction" of a specific nuisance shift, enabling continuous control via a scaling factor $s$. Specifically:

- An independent LoRA adapter is trained for each ImageNet class and shift type.
- LoRA parameters modify the original model weights: $\theta^* = \theta + s \cdot \theta_{LoRA}$
- Training follows the Concept Sliders framework, directing the adapter to capture the semantic direction from "`<class>`" to "`<class> in <shift>`".
- The training loss uses an MSE objective combined with the Tweedie formula:

$$\text{MSE}(\epsilon_{\theta^*}(X, c, t); \epsilon_\theta(X, c, t) + \epsilon_\theta(X, c^+, t))$$

**Key design**: The LoRA adapter is activated only during the last 75% of the diffusion denoising steps (i.e., disabled for the first 25%), preserving the semantic structure of the image while modifying only its appearance. This avoids the spatial layout disruptions caused by binary text prompt methods.

A total of **14 shift types** are considered:

- **Style shifts** (8 types): cartoon, plush toy, pencil sketch, painting, sculpture, graffiti, video game, tattoo
- **Weather shifts** (6 types): heavy snow, heavy rain, dense fog, haze, dust, sandstorm

### 3.3 Failure Point Concept

Continuous shifts also enable **failure point** analysis — the minimum shift scale at which a model first misclassifies an image:

$$s^* = \min\{S \in \mathbb{R} \mid f(X(S)) \neq c\}$$

By analyzing the distribution of failure points across all samples, one can obtain a fine-grained understanding of how different models degrade under different shifts: some models degrade gradually (e.g., under weather shifts), while others collapse abruptly at a specific scale (e.g., cartoon style at $s=1.5$).

### 3.4 Out-of-Class (OOC) Filtering

Generated images may drift from the target class (out-of-class, OOC) and must be filtered. The proposed filtering strategy combines **four filters** using a majority voting scheme (≥2 out of 4):

1. **CLIP text alignment**: cosine similarity between the image and "`A picture of a <class>`".
2. **CLIP text alignment (with shift)**: cosine similarity between the image and "`A picture of a <class> in <shift>`".
3. **CLIP image similarity**: cosine similarity between CLIP features of the shifted and original images.
4. **DINOv2 CLS token similarity**: cosine similarity between DINOv2 features of the shifted and original images.

$$\mathcal{A}_{text} = \cos(\text{CLIP}_{img}(I_k), \text{CLIP}_{text}(p))$$
$$\mathcal{A}_{feat} = \cos(\mathcal{F}_0, \mathcal{F}_k)$$

An image is filtered when ≥2 of the 4 filters are triggered. Each filter's threshold is set to remove >90% of OOC samples. Crucially, none of the filters are trained on ImageNet data, avoiding evaluation bias.

## Key Experimental Results

### Distribution Gap and Filtering Effectiveness

| Metric | SD | IN* |
|------|-----|------|
| FID (↓) | 33.8 | **27.1** |
| ResNet-50 Accuracy | 0.68 | **0.74** |

| Filtering Method | TPR | FPR (↓) | Filtering Precision (↑) |
|----------|------|---------|-------------|
| CLIP-only | 0.90 | 0.36 | 0.65 |
| **Ours** | **0.88** | **0.12** | **0.88** |

### Large-Scale Robustness Evaluation (40+ Classifiers)

The benchmark dataset contains **192,168 images** covering 100 ImageNet classes, 14 shift types, and 6 scales (0, 0.5, 1, 1.5, 2, 2.5).

**Architecture comparison** (similar parameter counts, same training data; lower rCE indicates better robustness):

| Model | rCE (↓) |
|------|---------|
| ViT | 0.926 |
| RN152 | 0.790 |
| ConvNeXt | 0.686 |
| DeiT3 | 0.610 |
| **VMamba** | **0.574** |

**Model scale** (DeiT3 family):

| Model | rCE (↓) |
|------|---------|
| DeiT3-S | 0.747 |
| DeiT3-M | 0.758 |
| DeiT3-B | 0.610 |
| DeiT3-L | **0.574** |
| DeiT3-H | 0.582 |

**Pre-training paradigm** (all using ViT-B/16):

| Pre-training | rCE (↓) |
|--------|---------|
| SUP-IN1k | 0.926 |
| SUP-IN21k-1k | 0.722 |
| MAE-IN1k | 0.732 |
| MoCov3-IN1k | 0.669 |
| **DINOv1-IN1k** | **0.636** |

### Comparison with OOD-CV Real Data

After training ResNet-50 on 10 OOD-CV classes with weather shifts: images generated by CNS-Bench consistently yield higher accuracy than real OOD-CV images, indicating that OOD-CV data is confounded by additional nuisance factors (image quality, cropping, occlusion, etc.), whereas CNS-Bench better isolates individual shifts.

### Fine-Tuning Gains with Synthetic Data

Fine-tuning ResNet-50 with CNS-Bench data improves ImageNet-R accuracy from 27.34% to **37.57%** (+10.23%), with only a marginal drop on the ImageNet validation set (80.15% → 78.11%).

## Highlights & Insights

1. **Model rankings vary across shift types and severity levels**: For example, ViT outperforms other models under low-scale painting style shifts but degrades more severely at high scales — a phenomenon that binary shift benchmarks cannot capture.
2. **VMamba (visual state space model) is the most robust**: Under comparable parameter counts, VMamba achieves lower rCE than both Transformers and CNNs.
3. **Self-supervised pre-training outperforms more supervised data**: DINOv1 with self-supervised pre-training on IN1k alone surpasses supervised pre-training on IN21k, suggesting that representation quality matters more than data volume.
4. **Diffusion classifiers are surprisingly non-robust**: The average accuracy drop of DiT classifiers under snow and cartoon shifts (0.106) substantially exceeds that of discriminative models (ViT: 0.07, MAE: 0.05).
5. **Failure point analysis reveals distinct degradation patterns**: Weather shifts lead to gradual failure accumulation, while style shifts (e.g., cartoon) concentrate failures at specific scales, likely due to confusion with ImageNet categories such as "comic book."
6. **User study validation**: Only 1% of samples in the final dataset are OOC, with a margin of error of ±0.5%.

## Limitations & Future Work

1. **CLIP training data bias**: Biases inherent in the training data of CLIP and Stable Diffusion cannot be fully eliminated during shift generation; failures cannot always be fully attributed to the target nuisance concept.
2. **Synthetic vs. real distribution shift**: A domain gap exists between generated and real images, potentially introducing additional bias.
3. **Limited category coverage**: Only 100 of the 1,000 ImageNet classes are currently evaluated, though ablation studies show consistent accuracy degradation trends.
4. **LoRA slider consistency**: Shift intensity increases monotonically in only 73% of cases when the slider weight is increased, indicating some inconsistency.
5. **High computational cost**: Training 1,400 LoRA adapters requires approximately 2,000 GPU hours, and image generation requires approximately 350 GPU hours.
6. **Extensibility to other tasks**: The current benchmark evaluates classification only; future work could extend to segmentation, detection, domain adaptation, and beyond.

## Related Work & Insights

- **Concept Sliders** (Gandikota et al., ECCV 2024): The foundational framework for LoRA slider training used in this work.
- **Dataset Interfaces** (Vendrow et al., 2023): A pioneering approach to generating benchmark images with diffusion models, but limited to binary shifts.
- **ImageNet-C** (Hendrycks & Dietterich, ICLR 2018): The canonical synthetic corruption benchmark; this work fills its gap of continuous, realistic shifts.
- **OOD-CV** (Zhao et al., ECCV 2022): A real-world OOD dataset used for direct comparison to validate the authenticity of the generated shifts.
- **DINOv2** (Oquab et al., 2023): Vision-only self-supervised features used for OOC filtering.
- **Inspiration**: The continuous shift evaluation paradigm can be extended to video understanding (temporal continuous change), 3D vision (continuous viewpoint variation), and related areas.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[NeurIPS 2025\] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](../../NeurIPS2025/image_generation/t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)
- [\[ICCV 2025\] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](semantic_watermarking_reinvented_enhancing_robustness_and_generation_quality_wit.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)

<!-- RELATED:END -->
