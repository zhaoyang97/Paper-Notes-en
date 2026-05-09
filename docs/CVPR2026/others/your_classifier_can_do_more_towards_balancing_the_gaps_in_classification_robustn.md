---
title: >-
  [Paper Note] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation
description: >-
  [CVPR 2026][adversarial robustness] This paper proposes EB-JDAT, a framework that models the joint energy distribution $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$ over clean, adversarial, and generated samples, achieving — for the first time in a single model — high classification accuracy, strong adversarial robustness, and competitive generative capability. On CIFAR-10, it attains 66.12% AutoAttack robustness, surpassing state-of-the-art adversarial training methods by over 10 percentage points.
tags:
  - CVPR 2026
  - adversarial robustness
  - energy-based models
  - joint generative-discriminative modeling
  - adversarial training
  - JEM
date: 2026-05-08
content_hash: 80c3693fc93794cb
---

# Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation

**Conference**: CVPR 2026
**arXiv**: [2505.19459](https://arxiv.org/abs/2505.19459)
**Code**: [https://github.com/yujkc/EB-JDAT](https://github.com/yujkc/EB-JDAT)
**Area**: Other
**Keywords**: adversarial robustness, energy-based models, joint generative-discriminative modeling, adversarial training, JEM

## TL;DR

This paper proposes EB-JDAT, a framework that models the joint energy distribution $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$ over clean, adversarial, and generated samples, achieving — for the first time in a single model — high classification accuracy, strong adversarial robustness, and competitive generative capability. On CIFAR-10, it attains 66.12% AutoAttack robustness, surpassing state-of-the-art adversarial training methods by over 10 percentage points.

## Background & Motivation

**Background**: Joint Energy-based Models (JEM) unify classification and generation within a single framework and exhibit some inherent robustness; adversarial training (AT) is the most effective approach for improving robustness but sacrifices clean accuracy and lacks generative capability.

**Limitations of Prior Work**: JEM's robustness falls far short of AT; AT substantially degrades clean accuracy (typically by 5–10%); no existing method simultaneously achieves high classification accuracy, adversarial robustness, and generation quality — a fundamental trilemma.

**Key Challenge**: AT introduces adversarial examples during training, causing the model to deviate from the true data manifold, while JEM does not explicitly model the adversarial distribution; each approach is thus inherently limited.

**Goal**: Can a single model simultaneously achieve high accuracy, strong robustness, and good generative quality?

**Key Insight**: The authors conduct a systematic **energy landscape analysis**, revealing that AT reduces the energy gap between clean and adversarial samples (the source of robustness), while JEM reduces the energy gap between clean and generated samples (the source of generative capability). Aligning the energy distributions of all three data types would unify the advantages of both approaches.

**Core Idea**: By maximizing the joint probability $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$ of clean and adversarial distributions, a min-max energy optimization explicitly aligns the energy distributions of all three data types.

## Method

### Overall Architecture

EB-JDAT extends the JEM framework by expanding the original joint distribution $p_\theta(\mathbf{x}, y)$ to a ternary joint distribution $p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y)$. This is decomposed via Bayes' rule into three optimizable components:

$$p_\theta(\mathbf{x}, \tilde{\mathbf{x}}, y) = p_\theta(y|\tilde{\mathbf{x}}, \mathbf{x}) \cdot p_\theta(\tilde{\mathbf{x}}|\mathbf{x}) \cdot p_\theta(\mathbf{x})$$

- $p_\theta(\mathbf{x})$: clean data distribution, sampled via SGLD
- $p_\theta(y|\tilde{\mathbf{x}}, \mathbf{x})$: robust classifier (cross-entropy)
- $p_\theta(\tilde{\mathbf{x}}|\mathbf{x})$: conditional EBM over adversarial distribution — the core contribution of this work

### Key Designs

1. **Energy Distribution Analysis and Alignment Insight**:

    - Function: Systematically analyzes the energy distributions of clean, adversarial, and generated samples in AT and JEM
    - Core Finding: AT causes the energy distributions of clean and adversarial samples to overlap (source of robustness); JEM causes the energy distributions of clean and generated samples to overlap (source of generative capability)
    - Design Motivation: Aligning all three distributions simultaneously unifies both advantages — this serves as the theoretical foundation of the entire method

2. **Conditional Adversarial Energy Modeling $p_\theta(\tilde{\mathbf{x}}|\mathbf{x})$**:

    - Function: Explicitly models the adversarial distribution using a conditional EBM
    - Mechanism: Adversarial examples reside in low-density (high-energy) regions; min-max optimization pulls them back to high-density regions:
    $\min_\theta \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}}\left[\max_{\|\tilde{\mathbf{x}}-\mathbf{x}\|\in\Omega}\left(E_\theta(\tilde{\mathbf{x}}|\mathbf{x}) - E_\theta(\mathbf{x})\right)\right]$
    - Inner maximization: samples high-energy adversarial examples along the $-\nabla_{\mathbf{x}}\log p_\theta((\tilde{\mathbf{x}}|\mathbf{x}), y)$ direction
    - Outer minimization: minimizes the energy gap between adversarial and clean samples, pulling adversarial examples back to low-energy regions
    - Distinction from JEAT: JEAT models only $p_\theta(\tilde{\mathbf{x}}, y)$, ignoring the intrinsic relationship between clean and adversarial data; EB-JDAT models the complete joint distribution

3. **SGLD Sampling and Adversarial Sampling**:

    - Function: Provides samples for the generative branch and the adversarial branch, respectively
    - Generative sampling: $\mathbf{x}_{t+1}^- = \mathbf{x}_t^- + \frac{c^2}{2}\frac{\partial \log p_\theta(\mathbf{x}_t^-)}{\partial \mathbf{x}_t^-} + c\epsilon$, used to approximate $p_\theta(\mathbf{x})$
    - Adversarial sampling: $\tilde{\mathbf{x}}_{t+1} = \tilde{\mathbf{x}}_t - \frac{c^2}{2}\frac{\partial \log p_\theta((\tilde{\mathbf{x}}|\mathbf{x}), y)}{\partial \tilde{\mathbf{x}}_t}$ (note the negative sign; the objective is to find high-energy samples)

### Loss & Training

The total gradient is a weighted sum of three components: $h_\theta = w_1 h_1 + w_2 h_2 + w_3 h_3$

- $h_1 = \frac{\partial \log p_\theta(\mathbf{x})}{\partial \theta}$: generative gradient
- $h_2 = \frac{\partial \log p_\theta(\tilde{\mathbf{x}}|\mathbf{x})}{\partial \theta}$: adversarial energy alignment gradient (core)
- $h_3 = \frac{\partial \log p_\theta(y|\mathbf{x}, \tilde{\mathbf{x}})}{\partial \theta}$: robust classification gradient

The WRN28-10 architecture is used, with a learning rate of 0.01, perturbation budget of 8/255, and 5 adversarial sampling steps.

## Key Experimental Results

### Main Results

| Dataset | Method | Clean Acc (%) | PGD-20 (%) | AutoAttack (%) |
|--------|------|-------------|-----------|---------------|
| CIFAR-10 | DHAT-CFA (Prev. SOTA) | 84.49 | 62.38 | 54.05 |
| CIFAR-10 | LAS-AWP | 87.74 | 60.16 | 55.52 |
| CIFAR-10 | **EB-JDAT-SADAJEM** | **90.37** | **68.76** | **66.12** |
| CIFAR-100 | DHAT-CFA | 61.54 | 37.15 | 30.93 |
| CIFAR-100 | **EB-JDAT-SADAJEM** | **68.32** | **38.42** | **35.57** |
| ImageNet subset | LAS-AT | 50.66 | 27.34 | 21.78 |
| ImageNet subset | **EB-JDAT-JEM++** | **63.02** | **34.50** | **32.40** |

### Ablation Study

| Configuration ($w_1, w_2, w_3$) | Clean (%) | AA (%) | FID | Notes |
|------------------------|----------|-------|-----|------|
| (0, 0, 1) — Standard AT | 88.95 | 62.96 | 173.53 | No generative capability; collapses at epoch 41 |
| (0, 1, 1) — No generation | 89.84 | 64.69 | 42.57 | $h_2$ is critical |
| (1, 0.5, 1) | 90.39 | 64.09 | 40.12 | Reducing $w_2$ improves accuracy/FID, slightly reduces robustness |
| **(1, 1, 1)** — Full model | **90.37** | **64.61** | **39.67** | Best overall balance |

### Key Findings
- $h_2$ (adversarial energy alignment gradient) is the most critical component: removing it not only degrades robustness but also causes training collapse (at epoch 41)
- $h_1$ (generative gradient) simultaneously improves both classification and generation quality
- Five adversarial sampling steps yield the best trade-off; too many steps lead to model collapse
- EB-JDAT requires no additional generated data; training takes only 31–66 GPU hours, far less than data-augmentation-based AT methods (1,438+ hours)

## Highlights & Insights
- **Energy landscape analysis drives method design**: Rather than designing modules heuristically, the authors first analyze the root cause of the problem (energy distribution discrepancy) and then devise corresponding solutions. This "analysis → insight → method" paradigm is highly instructive.
- **Conditional EBM for adversarial distribution modeling**: This is the first work to incorporate the adversarial sample distribution into the probabilistic graph of a joint energy-based model, elevating adversarial training from "regularization" to "distribution modeling."
- **Elegant min-max energy optimization**: During adversarial sampling, examples are pushed toward higher energy; during training, they are pulled toward lower energy — structurally analogous to AT's min-max formulation but unified from an energy perspective.
- **Substantial resolution of the trilemma**: Without additional data, the method surpasses the state of the art on CIFAR-10 AutoAttack robustness by over 10 percentage points while maintaining clean accuracy above 90% and a competitive FID.

## Limitations & Future Work
- Excessive adversarial sampling steps can cause EBM collapse; stability on high-resolution, large-scale datasets (full ImageNet) remains to be validated
- Generation quality still lags behind the leading JEM variant (SADAJEM), reflecting an inherent tension between adversarial training and generative quality
- Evaluation is currently limited to the $\ell_\infty$ norm; other perturbation types ($\ell_2$, spatial) have not been tested
- Training requires both SGLD and adversarial sampling, making it slower than pure AT methods

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to unify classification, robustness, and generation from a joint energy distribution perspective; theoretically elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of CIFAR-10/100 and an ImageNet subset, though full ImageNet experiments are absent
- Writing Quality: ⭐⭐⭐⭐ — Clear logical structure; energy analysis figures are highly illustrative
- Value: ⭐⭐⭐⭐⭐ — Substantially advances the performance frontier in adversarial robustness with a broadly applicable methodology

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)
- [\[CVPR 2026\] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)

<!-- RELATED:END -->
