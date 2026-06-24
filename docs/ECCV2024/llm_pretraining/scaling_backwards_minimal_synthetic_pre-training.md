---
title: >-
  [Paper Note] Scaling Backwards: Minimal Synthetic Pre-training?
description: >-
  [ECCV 2024][LLM Pretraining][synthetic pre-training] Proposes 1p-frac—achieving pre-training performance comparable to the ImageNet-1k level using minute perturbations of a single fractal image. This challenges the conventional wisdom that "pre-training requires large-scale datasets" and reveals that the essence of pre-training might be closer to weight initialization than visual concept learning.
tags:
  - "ECCV 2024"
  - "LLM Pretraining"
  - "synthetic pre-training"
  - "fractal images"
  - "minimal datasets"
  - "visual representation learning"
  - "ViT"
date: 2026-05-08
content_hash: 46c54412673bff45
---

# Scaling Backwards: Minimal Synthetic Pre-training?

**Conference**: ECCV 2024  
**arXiv**: [2408.00677](https://arxiv.org/abs/2408.00677)  
**Code**: [GitHub](https://github.com/SUPER-TADORY/1p-frac)  
**Area**: LLM Pre-training  
**Keywords**: synthetic pre-training, fractal images, minimal datasets, visual representation learning, ViT

## TL;DR

Proposes 1p-frac—achieving pre-training performance comparable to the ImageNet-1k level using minute perturbations of a single fractal image. This challenges the conventional wisdom that "pre-training requires large-scale datasets" and reveals that the essence of pre-training might be closer to weight initialization than visual concept learning.

## Background & Motivation

**Background**: Pre-training is a fundamental technology for current vision systems. Mainstream methods use large-scale real image datasets (1.28 million images in ImageNet-1k, 14 million images in ImageNet-21k) for supervised or self-supervised pre-training. Synthetic pre-training directions such as FractalDB (1 million fractal images) and OFDB (1,000 fractal images) have demonstrated that effective representations can be obtained without real images.

**Limitations of Prior Work**: The scale of foundation models continues to expand (from millions to billions of images), yet the essence of pre-training remains unclear—does it discover general visual concepts, or merely provide better weight initialization? Furthermore, large-scale real datasets suffer from privacy, copyright, and fairness issues.

**Key Challenge**: OFDB has reduced fractal images to 1,000, but further reducing the number of classes leads to performance degradation. The key question is: "How small can the minimal effective pre-training dataset actually be?"

**Goal**: To find the minimal purely synthetic pre-training dataset and investigate the minimum requirements for successful pre-training.

**Key Insight**: Instead of adding more images, "scaling backwards" is proposed—constructing "classes" using subtle parameter perturbations of a single fractal image, and training the model to distinguish these perturbations which are indistinguishable to the human eye.

**Core Idea**: The key to pre-training lies not in the volume of data, but in the structured diversity during the data generation process—recursive self-similar structures (fractals) paired with tiny affine transformation perturbations can provide sufficient pre-training signals.

## Method

### Overall Architecture

1p-frac consists of three core components:
- A single fractal image (defined by an Iterated Function System IFS)
- A local integration empirical distribution (LIEP distribution) to generate perturbed images
- A local perturbation cross-entropy loss (LPCE loss) for pre-training

Pre-training workflow: Starting from a single IFS $\Omega$, a minute perturbation $\epsilon$ is applied to the affine transformation parameters, generating $L$ perturbed images as different "classes" to train the ViT to classify these perturbations.

### Key Designs

1. **Local Integration Empirical Distribution (LIEP Distribution)**: Key mathematical tool.

    - For a single fractal image $I$, the empirical distribution degenerates to $p_{\text{data}}(x,y) = \delta(x-I)\delta(y)$. Direct training with cross-entropy would result in a trivial solution.
    - The LIEP distribution introduces a perturbation parameter $\Delta$, integrating within the range $\boldsymbol{\epsilon} \in \mathcal{R}_\Delta = [-\Delta/2, \Delta/2]^{6j}$:
    $$p_\Delta(x,y) = \frac{1}{|\mathcal{R}_\Delta|}\int_{\mathcal{R}_\Delta}\delta(x - I_{\boldsymbol{\epsilon}})\delta(y - \boldsymbol{\epsilon})d\boldsymbol{\epsilon}$$
    - As $\Delta \to 0$, the LIEP distribution converges to the original single-image empirical distribution.
    - **Design Motivation**: To provide a continuously controllable way to shrink or expand the support of the data distribution, thereby precisely studying the minimal distribution range required for pre-training.

2. **Local Perturbation Cross-Entropy Loss (LPCE Loss)**: Pre-training objective function.

    - $\mathcal{L}_\Delta = -\mathbb{E}_{x,y \sim p_\Delta}[\log p_\theta(y|x)]$
    - In practice, the objective is approximated via numerical integration by uniformly sampling $L=1000$ perturbation points.
    - Perturbation is applied to the affine transformation parameters of the IFS:
    $$w_j(\boldsymbol{v}; \boldsymbol{\epsilon}_j) = \left(\begin{bmatrix}a_j & b_j & e_j \\ c_j & d_j & f_j\end{bmatrix} + \boldsymbol{\epsilon}_j\right)\begin{bmatrix}\boldsymbol{v} \\ 1\end{bmatrix}$$
    - **Design Motivation**: To enable the model to learn to distinguish minute shape differences that are indistinguishable to the human eye, forcing the network to focus on structural patterns rather than surface features.

3. **σ-factor Controlling Fractal Complexity**: Using Anderson's $\sigma$-factor to evaluate the complexity of IFS.

    - The smaller the $\sigma$, the more complex the fractal (resembling the recursive structures of natural objects).
    - An overly large $\sigma$ (e.g., 6.0) causes the fractal to degenerate into something resembling Gaussian noise, but it still has a positive pre-training effect.
    - The optimal value is $\sigma = 3.5$.
    - **Design Motivation**: To explore what kind of image structures are most critical for pre-training—concluding that recursive self-similar structures are more important than mere complexity.

### Loss & Training

- Pre-training uses the LPCE loss, with hyperparameters following DeiT standard settings.
- Data augmentation adopts DeiT's settings (RandomCrop, RandAug, Mixup, CutMix, etc.).
- Ablation studies find that RandomCrop and Mixup/CutMix have the greatest impact on pre-training performance.
- Exploration studies use ViT-Tiny, and scaling studies use ViT-Base.
- Fine-tuning datasets include CIFAR-10/100, ImageNet-100/1k, Cars, Flowers, etc.

## Key Experimental Results

### Main Results

Comparison with pre-training datasets of different scales (ViT-Tiny, CIFAR-100 fine-tuning accuracy):

| Dataset | No. of Images | Type | CIFAR-100 | ImageNet-100 |
|--------|--------|------|-----------|--------------|
| Scratch | - | - | 64.2 | 74.9 |
| FractalDB | 1M | FDSL | 81.6 | 88.5 |
| OFDB | 1k | FDSL | 84.0 | 88.6 |
| **1p-frac** | **1** | **FDSL** | **84.2** | **89.0** |
| ImageNet-1k | 1.28M | SL | 85.5 | - |

ViT-Base fine-tuned on ImageNet-1k: 1p-frac (1 image) achieves 82.1%, surpassing the 81.8% of ImageNet-21k pre-training.

### Ablation Study

| Configuration | CIFAR-100 | Explanation |
|------|-----------|------|
| Δ=0.001 | 1.2 | Perturbation too small, pre-training collapses |
| Δ=0.01 | 19.9 | Positive effects begin to emerge |
| Δ=0.05 | 83.0 | Near-optimal performance |
| Δ=0.1 | **84.2** | Optimal perturbation magnitude |
| σ=3.5 (Most complex) | **84.2** | Optimal fractal complexity |
| σ=6.0 (Noise-like) | 81.3 | IFS structure still provides positive effect |
| Gaussian Noise | 1.1 | Complete failure, requires structured images |
| Uniform Noise| 2.0 | Similarly fails |
| L=16 sample points | 78.7 | Small sample size still yields positive effects |
| L=1000 sample points | **84.2** | More samples are better |

### Key Findings

- **"Scaling backwards" holds true**: As the synthetic pre-training images scale from 1M → 1k → 1, performance unexpectedly improves from 81.6 → 84.0 → 84.2.
- **Perturbation threshold exists**: Pre-training collapses when $\Delta < 0.01$, indicating that the distribution support requires a minimum scale.
- **Structure > Randomness**: Gaussian/uniform noise completely fails, proving that the recursive self-similar structure of fractals is crucial.
- **Real images can also scale backwards**: Applying LPCE loss to gray-scaled + Canny-edged + affine-transformed real images also yields positive pre-training effects (C100: 82.2%), which is virtually equivalent to the 1p-frac configuration.
- **Early layers benefit more**: Linear probing experiments show that the representation quality of the first three layers of ViT trained with 1p-frac even exceeds that of ImageNet-1k pre-training.
- **Extremely fast dataset construction**: 1p-frac only takes 0.04 hours (~2 minutes), compared to the 19 hours of FractalDB.

## Highlights & Insights

- **Subversive Discovery**: Pre-training with a single synthetic image can match or even surpass pre-training on millions of real images—strongly implying that the essence of pre-training is closer to a "better weight initialization" than "visual concept learning."
- **Elegant Mathematical Framework**: The LIEP distribution and LPCE loss provide a continuously controllable experimental tool to precisely investigate the minimal requirements of pre-training.
- **Profound Counter-intuitive Conclusion**: Minute shape differences indistinguishable to the human eye are crucial for model pre-training—suggesting a fundamental discrepancy between the "concepts" learned by networks and human-perceived concepts.
- **Practical Value**: Compresses dataset construction time from hours to 2 minutes, completely bypassing privacy and copyright issues associated with real-world data.

## Limitations & Future Work

- The study only verifies the ViT architecture, leaving it unexplored whether CNNs (like ResNet) exhibit a similar "scaling backwards" effect.
- The current optimal $\sigma$ and $\Delta$ are determined empirically via grid search, lacking theoretical guidance.
- Although full fine-tuning performance is comparable to ImageNet, a gap still exists in deeper layers under linear probing—indicating that learning deep semantic representations still requires real-world data.
- Self-supervised pre-training (e.g., MAE) has not been explored on extremely sparse synthetic images.
- A few tasks on VTAB (such as CLEVR-Count) still lag behind ImageNet supervised pre-training.

## Related Work & Insights

- **Asano et al. (2020)**: Pioneers of single-image self-supervised learning, but only effective for shallow layers and without using modern architectures.
- **OFDB** (Nakamura et al.): Compressed FractalDB to 1,000 images; this work further compresses it to a single image.
- **Visual Atoms**: Another FDSL dataset that generates images using parametric wave functions; while superior to FractalDB at 1 million images, it underperforms compared to 1p-frac with only 1 image.
- Insight: Pre-training may not require "learning the visual structure of the world" at all; instead, it optimizes the geometric configuration of network weights via classification signals. This insight has profound implications for understanding the inner workings of foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Disruptive conclusion, pushing minimal pre-training to the limit.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively verified across five dimensions: exploration, hyperparameters, scaling, analysis, and applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, elegant mathematical formulations, and progressively structured experimental design.
- Value: ⭐⭐⭐⭐⭐ Deep insight into the essence of pre-training with direct practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PreLAR: World Model Pre-training with Learnable Action Representation](prelar_world_model_pre-training_with_learnable_action_representation.md)
- [\[ICLR 2026\] Synthetic Bootstrapped Pretraining](../../ICLR2026/llm_pretraining/synthetic_bootstrapped_pretraining.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Pre-training under Infinite Compute](../../ICLR2026/llm_pretraining/pre-training_under_infinite_compute.md)
- [\[ICLR 2026\] OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation](../../ICLR2026/llm_pretraining/optimsyn_influence-guided_rubrics_optimization_for_synthetic_data_generation.md)

</div>

<!-- RELATED:END -->
