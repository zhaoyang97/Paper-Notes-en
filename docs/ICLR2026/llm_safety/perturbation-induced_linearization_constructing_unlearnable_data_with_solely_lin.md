---
title: >-
  [Paper Note] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers
description: >-
  [ICLR 2026][LLM Safety][Unlearnable Examples] This paper proposes PIL, a method that generates unlearnable perturbations using only a bias-free linear classifier as the surrogate model. By inducing linearization in deep…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Unlearnable Examples"
  - "Data Protection"
  - "Linearization"
  - "Shortcut Learning"
  - "Adversarial Perturbation"
date: 2026-05-08
content_hash: 699ff503e663b859
---

# Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers

**Conference**: ICLR 2026
**arXiv**: [2601.19967](https://arxiv.org/abs/2601.19967)  
**Code**: [GitHub](https://github.com/jinlinll/pil)  
**Area**: LLM Safety
**Keywords**: Unlearnable Examples, Data Protection, Linearization, Shortcut Learning, Adversarial Perturbation

## TL;DR

This paper proposes PIL, a method that generates unlearnable perturbations using only a bias-free linear classifier as the surrogate model. By inducing linearization in deep models, PIL prevents them from learning semantic features, achieving over 100× speedup compared to existing methods (under 1 minute of GPU time on CIFAR-10).

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: The use of web-crawled data for training deep learning models has become increasingly prevalent, yet much of this data is collected without the consent of its creators. **Unlearnable Examples** protect data from unauthorized use by adding imperceptible perturbations such that models trained on the perturbed data fail to generalize to clean test data.

Existing methods (e.g., EM, REM) typically rely on deep networks as surrogate models to generate perturbations, incurring prohibitive computational costs (REM requires 15+ hours of GPU time on CIFAR-10). A natural question arises: **can simpler models generate equally effective perturbations?**

At a deeper level, the paper asks: **what is the underlying mechanism behind the effectiveness of unlearnable examples?** The answer identified is **linearization induction**—perturbations force deep models to behave like linear models, thereby stripping them of the ability to learn complex semantic features.

## Method

### Overall Architecture

PIL follows a two-step pipeline: (1) train a bias-free linear classifier on clean data; (2) optimize perturbations using the linear classifier to simultaneously satisfy two objectives—semantic obfuscation and shortcut learning. The final unlearnable dataset is constructed by subtracting the perturbations from the original images.

### Key Designs

1. **Semantic Obfuscation**:

    - Function: Renders the semantic information of the original image useless after perturbation.
    - Mechanism: Optimizes $\delta_1$ such that $f_{lin}(x - \delta_1)$ approximates a uniform distribution (minimizing KL divergence).
    - Design Motivation: Once the deep model is linearized, the $x - \delta$ component no longer carries useful classification information.

2. **Shortcut Learning**:

    - Function: Makes the perturbation itself a strong class-discriminative signal.
    - Mechanism: Optimizes $\delta_2$ such that the linear model can directly predict the label from $\delta_2$ with high accuracy (minimizing cross-entropy).
    - Design Motivation: Deep models tend to take shortcuts by learning perturbation-based cues rather than the semantic features of the images.

3. **Joint Optimization**:

    - Function: Combines both objectives into a single optimization.
    - Mechanism: $L_{total} = \lambda L_{CE}(f_{lin}(\delta), y) + (1-\lambda) L_{KL}(f_{lin}(x-\delta), \text{uniform})$, with $\lambda=0.9$ emphasizing shortcut learning.
    - Design Motivation: In practice, a single perturbation $\delta$ is optimized rather than separate $\delta_1$ and $\delta_2$.
    - Key Detail: PGD-style updates are used with step size $\alpha = 8/2550$; the linear model is pre-trained before perturbation optimization begins.

### Loss & Training

- The linear model is first trained for $M$ epochs on clean data using SGD to capture the semantic structure of the data.
- Perturbations for each sample are then updated over $N$ PGD steps, constrained to $L_\infty \leq 8/255$.
- Perturbations are initialized from a uniform distribution $[-\varepsilon, \varepsilon]$.

## Key Experimental Results

### Main Results: Test Accuracy Under Different Datasets and Architectures (Lower is Better)

| Model | SVHN-Clean | SVHN-PIL | CIFAR-10-Clean | CIFAR-10-PIL | ImageNet100-Clean | ImageNet100-PIL |
|-------|-----------|----------|----------------|-------------|-------------------|----------------|
| ResNet-18 | 95.64 | **15.94** | 92.11 | **12.77** | 66.00 | **2.26** |
| VGG-19 | 95.22 | **9.12** | 90.61 | **15.22** | 36.04 | **1.36** |
| MobileNet-V2 | 95.95 | **28.48** | 91.94 | **14.05** | 71.26 | **2.20** |

### Ablation Study: Robustness Under Data Augmentation (CIFAR-10 Test Accuracy ↓)

| Method | No Aug. | Basic | Rotation | Cutout | CutMix |
|--------|---------|-------|----------|--------|--------|
| PIL | **14.70** | **12.87** | **18.15** | 14.62 | **11.05** |
| SEP | 28.43 | 8.94 | 19.68 | 9.74 | 10.48 |
| TAP | 35.90 | 19.11 | 21.18 | 15.09 | 20.30 |

### Key Findings

- PIL requires less than 1 minute of GPU time on CIFAR-10, compared to 15+ hours for REM, representing a speedup of over 100×.
- Perturbations generated by a linear surrogate effectively degrade the generalization of multiple deep architectures, demonstrating architecture-agnostic effectiveness.
- All unlearnable example methods—including those using nonlinear surrogates such as EM and REM—increase the linearity of trained models; PIL simply pushes this mechanism to its logical extreme.
- On the higher-resolution ImageNet-100 benchmark, test accuracy drops to 1–3%, with even stronger protection than on smaller datasets.
- PIL maintains substantial robustness against JPEG compression defenses.

## Highlights & Insights

- **The core insight is remarkably elegant**: the fundamental mechanism of unlearnable examples is linearization induction—and given this, a linear surrogate model is sufficient.
- The method reduces the complex problem of generating unlearnable examples to linear model training combined with PGD optimization, significantly lowering both implementation and computational barriers.
- The decomposition into dual objectives—semantic obfuscation and shortcut learning—is intuitive and effective.
- The paper also reveals a fundamental limitation of partial perturbation: unlearnable examples fail to significantly reduce test accuracy when only a subset of the data is perturbed.

## Limitations & Future Work

- Adversarial training as a defense may still weaken the effectiveness of PIL.
- Protection degrades sharply in partial perturbation scenarios, where only a fraction of the data is protected.
- Non-image modalities such as text and audio have not been evaluated.
- The theoretical explanation of the linearization mechanism remains empirical.

## Related Work & Insights

PIL is directly compared against unlearnable example methods including EM, REM, TAP, and NTGA. The work is closely connected to the shortcut learning literature, highlighting the susceptibility of deep models to simple spurious features. A key takeaway is that the simplest surrogate model can sometimes be the most effective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The finding that a linear model suffices is both surprising and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, architectures, and defense strategies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the method is presented concisely.
- Value: ⭐⭐⭐⭐⭐ Offers both practical impact (100× speedup) and theoretical insight (linearization mechanism).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](../../ICCV2025/llm_safety/temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)
- [\[AAAI 2026\] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](../../AAAI2026/llm_safety/from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](../../AAAI2026/llm_safety/perturb_your_data_paraphrase-guided_training_data_watermarking.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](../../ACL2026/llm_safety/synthia_scalable_grounded_persona_generation_from_social_media_data.md)

</div>

<!-- RELATED:END -->
