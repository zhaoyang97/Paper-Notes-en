---
title: >-
  [Paper Note] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models
description: >-
  [AAAI 2026][Image Generation][Flow Matching] This paper proposes Symmetrical Flow Matching (SymmFlow), which unifies semantic segmentation, classification, and image generation into a single model. By jointly modeling forward and reverse flow transformations through a symmetric learning objective, SymmFlow achieves state-of-the-art performance in semantic image synthesis with only 25 inference steps (CelebAMask-HQ FID 11.9, COCO-Stuff FID 7.0)…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Semantic Segmentation"
  - "Classification"
  - "Unified Framework"
date: 2026-05-08
content_hash: 988935b78e2844a9
---

# Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models

**Conference**: AAAI 2026
**arXiv**: [2506.10634](https://arxiv.org/abs/2506.10634)  
**Code**: [github.com/caetas/SymmetricFlow](https://github.com/caetas/SymmetricFlow)  
**Area**: Segmentation
**Keywords**: Flow Matching, Semantic Segmentation, Image Generation, Classification, Unified Framework

## TL;DR

This paper proposes Symmetrical Flow Matching (SymmFlow), which unifies semantic segmentation, classification, and image generation into a single model. By jointly modeling forward and reverse flow transformations through a symmetric learning objective, SymmFlow achieves state-of-the-art performance in semantic image synthesis with only 25 inference steps (CelebAMask-HQ FID 11.9, COCO-Stuff FID 7.0), while obtaining competitive results on segmentation and classification.

## Background & Motivation

In computer vision, classification, segmentation, and generation are typically handled by separate specialized models. Ideally, a unified framework should simultaneously understand and generate images: accurate understanding of visual structure facilitates semantically consistent generation, while strong generative capacity yields more expressive image representations.

**Limitations of Prior Work**:

**Diffusion models for classification**: Require iterative sampling over all classes, leading to extremely slow inference (e.g., Diffusion Classifier requires 2,750 steps).

**Diffusion models for segmentation**: Existing frameworks can only generate masks and cannot map back to real images.

**Unified methods such as SemFlow** still exhibit three key limitations:
   - Classification is not supported
   - Image generation quality is inferior to dedicated generative models
   - The segmentation mask is required to share the same number of channels as the image, limiting flexibility

**Core Motivation**: The Flow Matching framework is inherently bidirectional — the forward flow generates images from noise, while the reverse flow recovers semantic information from images. SymmFlow exploits this symmetry by maintaining sufficient entropy during generation to ensure diversity, while enforcing semantic consistency during segmentation and classification.

## Method

### Overall Architecture

SymmFlow models semantic segmentation and semantic image synthesis as opposing flow processes. Given data distribution $X$ (images) and semantic representation $Y$ (masks or class labels), SymmFlow models bidirectional flows between them:
- **Forward process**: $X$ is transformed from noise, while $Y$ evolves toward noise
- **Reverse process**: These transformations are inverted to generate $Y$ from $X$

Key innovation: $Y$ is not required to share the same dimensionality as $X$, enabling flexible conditioning such as global class labels (classification) and pixel-level masks (segmentation).

### Key Designs

#### 1. **Symmetric Training Objective**

For each sample, $t \sim \mathcal{U}(0,1)$ is drawn, and perturbed samples are constructed via convex combination:

$$x_t = (1-t)\xi_x + tx, \quad y_t = (1-t)y + t\xi_y$$

where $\xi_x, \xi_y$ are independent Gaussian noise. The optimal transport velocity fields are:

$$v_x = x - \xi_x, \quad v_y = \xi_y - y$$

The model $v_\theta(x_t, y_t, t)$ jointly approximates both flows by minimizing mean squared error:

$$\mathcal{L} = \mathbb{E}_{x,y,t}[\|v_\theta(x_t, y_t, t) - v\|^2]$$

Design Motivation: The symmetric formulation ensures that the image generation branch retains sufficient entropy for diversity, while the segmentation branch preserves semantic structure. Unlike SemFlow's unidirectional flow, SymmFlow simultaneously learns transformations in both directions.

#### 2. **Inference for Classification and Segmentation**

Unlike conventional generative classifiers that evaluate noise prediction errors for each class individually, SymmFlow directly obtains results by integrating the predicted velocity field:

$$y_0 = y_1 + \int_1^0 v_\theta(x_t, y_t, t)_y dt$$

Inference is performed using an ODE solver without iterating over all classes, substantially reducing inference time. For segmentation, each pixel's class is determined by nearest-neighbor matching to predefined RGB class encodings; for classification, the predicted class corresponds to the nearest label to the model's predicted mean.

#### 3. **Label Dequantization**

Uniform noise is added to discrete labels $Y$ to prevent training instability:

$$Y' = Y + \epsilon, \quad \epsilon \sim U(-\beta, +\beta)$$

Dequantization is critical for preventing the model from assigning excessively high likelihood to a small number of specific values, which would lead to collapse. For classification models, labels are further normalized to the $[-1, +1]$ interval.

Design Motivation: Without dequantization, distributions with excessively low entropy impede modeling quality and lead to degenerate solutions. This is a classical technique from Normalizing Flows, cleverly applied here to discrete semantic labels.

### Loss & Training

- A single symmetric mean squared error loss serves as the training objective
- The U-Net from Stable Diffusion 2.1 and a pretrained VAE are used, with input/output channels doubled to accommodate SymmFlow
- The pixel-level implementation (classification) uses the Guided Diffusion U-Net
- An Euler ODE solver is used for sampling, with 25 inference steps by default

## Key Experimental Results

### Main Results

**Semantic Image Synthesis** (forward: mask→image):

| Dataset | Metric | SymmFlow | SemFlow | SDM | SCDM | SC-GAN |
|--------|------|----------|---------|-----|------|--------|
| CelebAMask-HQ | FID↓ | **11.9** | 32.6 | 18.8 | 17.4 | 19.2 |
| CelebAMask-HQ | LPIPS↑ | 0.464 | 0.393 | 0.422 | 0.418 | 0.395 |
| COCO-Stuff | FID↓ | **7.0** | *90.0 | 15.9 | 15.3 | 18.1 |
| COCO-Stuff | LPIPS↑ | 0.609 | *0.685 | 0.518 | 0.519 | — |

**Semantic Segmentation** (reverse: image→mask):

| Dataset | Metric | SymmFlow | SemFlow | SegFormer | MaskFormer |
|--------|------|----------|---------|-----------|------------|
| CelebAMask-HQ | mIoU↑ | 69.3 | *69.4 | — | — |
| COCO-Stuff | mIoU↑ | **39.6** | *35.7 | 46.7 | 37.1 |

**Classification**:

| Dataset | Steps | SymmFlow | Diffusion Classifier |
|--------|------|----------|---------------------|
| MNIST | 1 / 25 | 99.3 / **99.6** | — |
| CIFAR-10 | 1 / 25 | 88.2 / **90.6** | 88.5 (2,750 steps) |

SymmFlow surpasses Diffusion Classifier on CIFAR-10 using only 25 steps, whereas the latter requires 2,750 steps.

### Ablation Study

**Effect of inference steps on generation quality**:

| Steps | CelebA FID↓ | CelebA LPIPS↓ | COCO FID↓ | COCO LPIPS↓ |
|------|------------|---------------|----------|-------------|
| 1 | 88.5 | 0.598 | 102.6 | 0.777 |
| 5 | 49.5 | 0.522 | 44.3 | 0.704 |
| 10 | 28.2 | 0.486 | 18.2 | 0.652 |
| 25 | **11.9** | **0.464** | **7.0** | **0.609** |

**Effect of inference steps on segmentation performance**:

| Steps | CelebA mIoU↑ | COCO mIoU↑ |
|------|-------------|-----------|
| 1 | 65.3 | 29.3 |
| 2 | **70.3** | 33.8 |
| 5 | 70.3 | 38.1 |
| 20 | 69.4 | **40.1** |

**Classification accuracy on toy experiment (spiral dataset)**:

| Steps | 1 | 2 | 5 | 10 | 20 | 50 |
|------|---|---|---|----|----|-----|
| Acc(%) | **100.0** | 92.0 | 87.0 | 83.6 | 82.6 | 82.0 |

### Key Findings

1. **Image generation benefits substantially from more steps**, with FID decreasing from 88.5 to 11.9 on CelebA; segmentation, however, approaches optimum at just 2 steps.
2. **Classification achieves peak accuracy with only 1 step**, as class boundaries become blurred when $X$ evolves toward a Gaussian distribution.
3. **Generation quality surpasses all prior methods**, including diffusion models requiring 200–1,000 steps.
4. **LPIPS should be interpreted alongside FID**: at low step counts, high LPIPS reflects poor quality rather than high diversity.

## Highlights & Insights

1. **Elegant bidirectional modeling**: Treating segmentation and generation as symmetric flow processes yields a conceptually clean and mathematically unified formulation.
2. **Breaking the channel constraint**: $Y$ is not required to share the same dimensionality as $X$, enabling a unified treatment of classification (global labels) and segmentation (pixel-wise masks).
3. **Computational efficiency**: 25-step inference substantially outperforms diffusion models requiring hundreds of steps; classification requires only 1 step.
4. **Pedagogical value of the toy experiment**: The spiral dataset clearly illustrates the semantic separation capability of forward and reverse flows.

## Limitations & Future Work

1. **Segmentation resolution bottleneck**: Operating in a 64×64×4 latent space limits segmentation quality for small-area classes (e.g., earrings, eyebrows).
2. **Large model footprint**: Reliance on the Stable Diffusion U-Net entails high per-step computation, despite the reduced number of steps.
3. **Classification validated only on MNIST/CIFAR-10**: Evaluation on large-scale datasets such as ImageNet is absent.
4. **Segmentation accuracy still lags behind specialized models** (e.g., SegFormer achieves 46.7 vs. 39.6 on COCO-Stuff).
5. Future directions include distillation into a one-step model, fine-tuning the VAE decoder to improve segmentation, and exploring stronger architectures such as MMDiT.

## Related Work & Insights

- **SemFlow** (2024): The direct predecessor, which also employs flow matching to unify segmentation and generation; SymmFlow substantially surpasses it in generation quality.
- **Diffusion Classifier** (2023): A pioneering approach to classification with diffusion models, but with prohibitively high inference cost.
- **Flow Matching** (Lipman 2022): The theoretical foundation of SymmFlow, which simplifies CNF training from requiring simulation to direct matching.
- The symmetry concept is potentially extensible to tasks such as depth estimation and image editing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The unified modeling via symmetric flow matching is highly elegant; the design that removes the channel constraint is insightful.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Generation and segmentation experiments are thorough, but classification validation is relatively weak.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Logic is clear, toy experiments aid understanding, and figures are well-crafted.)
- Value: ⭐⭐⭐⭐ (The unified modeling direction is promising, but segmentation accuracy remains to be improved.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[AAAI 2026\] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)
- [\[ICLR 2026\] FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference](../../ICLR2026/image_generation/fastflow_accelerating_the_generative_flow_matching_models_with_bandit_inference.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] BiGain: Unified Token Compression for Joint Generation and Classification](../../CVPR2026/image_generation/bigain_unified_token_compression_for_joint_generation_and_classification.md)

</div>

<!-- RELATED:END -->
