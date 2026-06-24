---
title: >-
  [Paper Note] Gradient-Guided Annealing for Domain Generalization
description: >-
  [CVPR 2025][Domain Generalization] This paper proposes the GGA method, which uses simulated annealing in the early stages of training to search for parameter space points where gradients across domains are aligned (by maximizing the minimum cosine similarity of gradients between domains). This guides the model to find starting points for domain-invariant features at the beginning of optimization, improving domain generalization without data augmentation. It can be combined wi…
tags:
  - "CVPR 2025"
  - "Domain Generalization"
  - "Gradient Alignment"
  - "Simulated Annealing"
  - "Domain Shift"
  - "Early Training"
date: 2026-05-08
content_hash: f2b1b58dc52a846c
---

# Gradient-Guided Annealing for Domain Generalization

**Conference**: CVPR 2025  
**arXiv**: [2502.20162](https://arxiv.org/abs/2502.20162)  
**Code**: [https://github.com/aristotelisballas/GGA](https://github.com/aristotelisballas/GGA)  
**Area**: LLM Evaluation  
**Keywords**: Domain Generalization, Gradient Alignment, Simulated Annealing, Domain Shift, Early Training

## TL;DR
This paper proposes the GGA method, which uses simulated annealing in the early stages of training to search for parameter space points where gradients across domains are aligned (by maximizing the minimum cosine similarity of gradients between domains). This guides the model to find starting points for domain-invariant features at the beginning of optimization, improving domain generalization without data augmentation. It can be combined with existing DG methods to obtain significant improvements.

## Background & Motivation

**Background**: Domain Generalization (DG) aims to train models that can generalize to unseen domains. Existing methods include data augmentation, meta-learning, and domain alignment. However, extensive experiments show that a simple ERM baseline equipped with strong training strategies can outperform many DG methods.

**Limitations of Prior Work**: The gradient update steps during the early phases of training have a decisive impact on the model's final generalization ability. Gradient conflicts between domains (where gradient directions of loss from different domains disagree) push the optimization toward local optima that capture domain-specific features rather than class-specific features. Once trapped in such local optima, it is difficult for subsequent training to escape.

**Key Challenge**: Standard SGD optimization averages the loss gradients of all domains. When there are gradient direction conflicts between domains, the average gradient direction may not be optimal for any domain, causing the model to learn domain-specific rather than domain-invariant features.

**Goal**: To find a starting point in the parameter space during early training where "gradient directions of all domains are consistent", so that subsequent SGD optimization naturally tends towards learning domain-invariant features.

**Key Insight**: Inspired by the stochastic search strategy of simulated annealing, uniform random perturbations are applied to parameters during the early training phase. Perturbations that increase cross-domain gradient alignment are accepted, while those that decrease alignment are rejected.

**Core Idea**: Use simulated annealing in the early stage of training to search for parameter points where gradients are aligned across domains. This point serves as the starting state for subsequent standard SGD training, guiding the model to learn domain-invariant features from the beginning.

## Method

### Overall Architecture
Three phases: (1) Warmup phase: Standard SGD warmup for a few steps to provide the model with basic gradient signals; (2) Annealing phase: Calculate gradient pairs for each source domain, apply a random perturbation to the parameters $\theta' \leftarrow \theta + \mathcal{U}(-\rho, \rho)$. If the minimum inter-domain gradient similarity of the new parameters is improved and the loss increment is controlled ($< 0.1$), the perturbation is accepted. Iterate multiple times to find gradient-aligned parameter points; (3) Standard training: Resume normal SGD training from the aligned point.

### Key Designs

1. **Gradient Alignment Metric**:

    - **Function**: Measures the consistency of cross-domain gradients at the current parameter point.
    - **Mechanism**: Calculate the cosine similarity of gradients between all source domain pairs, taking the minimum value as the overall alignment: $\text{grad\_sim} = \min_{i \neq j} \frac{g_i^T \cdot g_j}{\|g_i\| \cdot \|g_j\|}$. Using the minimum rather than the average ensures that even the worst-performing domain pair is aligned, preventing any domain from being neglected.
    - **Design Motivation**: If the minimum gradient similarity between domain pairs is high, it indicates that the optimization directions of all domains are fundamentally consistent, meaning the average gradient of SGD is effective for every domain.

2. **Annealing Search Strategy**:

    - **Function**: Searches for gradient-aligned points in the parameter space.
    - **Mechanism**: Add a uniform random perturbation $\mathcal{U}(-\rho, \rho)$ to the current parameters $\theta$ to obtain $\theta'$. The acceptance conditions are: (a) $\text{grad\_sim}(\theta') > \text{grad\_sim}(\theta)$ (increased alignment) and (b) $\mathcal{L}(\theta') - \mathcal{L}(\theta) < 0.1$ (controlled loss increment). It does not require the loss to decrease, allowing a temporary increase in loss in exchange for better gradient alignment.
    - **Design Motivation**: Similar to simulated annealing allowing temporary "uphill" steps to find the global optimum, GGA allows temporary increases in loss to find parameter regions with aligned gradients.

3. **GGA-L Lightweight Variant**:

    - **Function**: Reduces the computational overhead of the annealing search.
    - **Mechanism**: Draws inspiration from SGLD (Stochastic Gradient Langevin Dynamics) to inject noise during gradient updates to implicitly search for alignment points, bypassing explicit perturbation-evaluation loops.
    - **Design Motivation**: Since the full GGA requires calculating gradients for all domains multiple times, GGA-L approximates the annealing effect through noise injection.

### Loss & Training
The annealing phase does not alter the loss function but only applies random perturbations to the parameters. The annealing window $[A_s, A_e]$ is typically during the first 10-20% of training iterations. Hyperparameters include $\rho$ (perturbation magnitude) and $n_a$ (number of annealing iterations per step). After annealing, the training switches back to standard ERM or any DG algorithm.

## Key Experimental Results

### Main Results

| Dataset | GGA (+ERM) | ERM | Other SOTA | Notes |
|--------|-----------|-----|---------|------|
| PACS | Significant Gain | Baseline | Competitive | No data augmentation required |
| VLCS | Significant Gain | Baseline | Competitive | |
| OfficeHome | Significant Gain | Baseline | Competitive | |
| TerraInc | Significant Gain | Baseline | Competitive | |

The core value of GGA: It can be seamlessly combined with any DG method to provide a better initialization.

### Key Findings
- Significant conflicts in cross-domain gradient directions exist during early training, and GGA effectively reduces this conflict.
- GGA does not change the training objective but only the starting point—making it orthogonally compatible with any DG method.
- The loss tolerance mechanism is crucial—if the loss is also required to decrease, the effectiveness of annealing is severely reduced.
- GGA-L achieves performance close to GGA with much lower computational cost.

## Highlights & Insights
- **Deep insight of "the beginning determines the end"**: The gradient directions in the first few steps of training determine whether the model ultimately learns domain-invariant or domain-specific features. This finding itself holds significant theoretical value.
- **Orthogonality of the method**: GGA only modifies the starting point of training without changing the loss, architecture, or augmentation strategies. Thus, it can be superimposed on any DG method—making this "preprocessing-like" approach highly practical.
- **Elegant application of simulated annealing in DL**: Applying classical optimization methods to solve modern gradient conflict issues is simple and effective.

## Limitations & Future Work
- The annealing phase requires calculating gradients for each source domain separately, which increases overhead when the number of source domains is large.
- The perturbation magnitude $\rho$ and annealing window $[A_s, A_e]$ require tuning.
- Currently only validated on classification DG; regression, detection, and other DG tasks are not addressed.
- Theoretical analysis is insufficient—there is a lack of rigorous proof showing why a gradient-aligned starting point guarantees convergence to a domain-invariant solution.

## Related Work & Insights
- **vs FISH/AND-Mask and other gradient methods**: These methods modify gradient directions throughout the entire training process; GGA only seeks a good starting point in the early stages and then trains normally, which is much simpler.
- **vs SAM (Sharpness-Aware Minimization)**: SAM searches for flat optima; GGA searches for gradient-aligned regions. While the objectives differ, their search strategies share similarities.
- **vs Stochastic Weight Averaging (SWA)**: SWA averages parameters in the late stages of training; GGA optimizes starting parameters in the early stages, making their time windows complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of searching for gradient alignment is novel, and the application of simulated annealing in DG is unique.
- Experimental Thoroughness: ⭐⭐⭐ Five standard DG benchmarks, though more detailed reporting of specific numbers is needed.
- Writing Quality: ⭐⭐⭐⭐ In-depth analysis of theoretical motivation with clear problem definition.
- Value: ⭐⭐⭐⭐ A plug-and-play DG preprocessing method with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](../../ICML2025/others/fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[CVPR 2025\] On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ICML 2025\] Set-Valued Predictions for Robust Domain Generalization](../../ICML2025/others/set_valued_predictions_for_robust_domain_generalization.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](../../ICCV2025/others/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)

</div>

<!-- RELATED:END -->
