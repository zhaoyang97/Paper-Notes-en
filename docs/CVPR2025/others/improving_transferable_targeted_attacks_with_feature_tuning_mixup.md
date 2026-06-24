---
title: >-
  [Paper Note] Improving Transferable Targeted Attacks with Feature Tuning Mixup
description: >-
  [CVPR 2025][Adversarial Attack] Proposed FTM (Feature Tuning Mixup) to improve the transferability of targeted adversarial attacks by mixing optimized attack-specific perturbations and random clean perturbations in the feature space of the surrogate model. Using a momentum-based stochastic update strategy to maintain computational efficiency, the average success rate across 14 black-box models is improved from 74.6% to 77.4%.
tags:
  - "CVPR 2025"
  - "Adversarial Attack"
  - "Transferable Attack"
  - "Feature Perturbation"
  - "Mixup"
  - "Black-box Attack"
date: 2026-05-08
content_hash: 9259ee45db9ee4c9
---

# Improving Transferable Targeted Attacks with Feature Tuning Mixup

**Conference**: CVPR 2025  
**arXiv**: [2411.15553](https://arxiv.org/abs/2411.15553)  
**Code**: [https://github.com/uhiu/feature-tuning-mixup](https://github.com/uhiu/feature-tuning-mixup)  
**Area**: Others  
**Keywords**: Adversarial Attack, Transferable Attack, Feature Perturbation, Mixup, Black-box Attack

## TL;DR
Proposed FTM (Feature Tuning Mixup) to improve the transferability of targeted adversarial attacks by mixing optimized attack-specific perturbations and random clean perturbations in the feature space of the surrogate model. Using a momentum-based stochastic update strategy to maintain computational efficiency, the average success rate across 14 black-box models is improved from 74.6% to 77.4%.

## Background & Motivation

**Background**: Targeted transferable attacks require adversarial examples generated on a surrogate model to successfully attack a designated target class on unseen black-box models. Existing methods improve transferability through input augmentation (DI, SI) or feature mixing (CFM).

**Limitations of Prior Work**: Clean Feature Mixup (CFM) mixes clean image features in the feature space to enhance diversity, but it only employs random clean features—which are not optimized for the target of the attack, thus limiting the improvement of perturbation diversity.

**Key Challenge**: Feature perturbations need to be sufficiently diverse to improve transferability, but also must be relevant to the attack target—random clean features and attack-optimized features both have their own merits.

**Goal**: To introduce attack-optimized learnable perturbations in the feature space, mixed with clean features, to further enhance transferability.

**Key Insight**: Designing learnable feature perturbations (added element-wise to middle-layer outputs) optimized via a min-max objective—where the perturbation maximizes adversarial loss and the adversarial image minimizes it. A momentum-based stochastic update prevents extra forward/backward computation.

**Core Idea**: Mix attack-optimized learnable perturbations with random clean perturbations in the surrogate model's feature space, achieving transferability improvement with zero extra overhead via min-max optimization and momentum-based stochastic updates.

## Method

### Overall Architecture
In each iteration: the current adversarial image undergoes forward propagation $\rightarrow$ the middle-layer feature is augmented with a learnable perturbation $\delta$ $\rightarrow$ $\delta$ is optimized via a min-max objective (the adversarial image minimizes the loss, while $\delta$ maximizes the loss) $\rightarrow$ momentum-based stochastic update (only a randomly selected subset of layers is updated, initializing with $\delta$ from previous iterations as momentum) $\rightarrow$ finally, FTM and CFM are mixed and utilized.

### Key Designs

1. **Learnable Attack Perturbation**:

    - Function: Generates feature diversity relevant to the attack target.
    - Mechanism: Element-wise addition of learnable perturbation $\delta$ to middle-layer outputs. $\delta$ is optimized via a min-max objective—the inner loop maximizes adversarial loss (making the perturbation highly "destructive"), while the outer loop minimizes adversarial loss (adapting the adversarial image to this destructive perturbation).
    - Design Motivation: The random clean perturbations in CFM are independent of the attack target, whereas the attack-optimized perturbations increase diversity in a more targeted manner.

2. **Momentum-based Stochastic Update**:

    - Function: Updates multi-layer perturbations with zero extra forward/backward overhead.
    - Mechanism: In each iteration, only a randomly selected subset of layers (with probability $p$) is chosen to update $\delta$, while the remaining layers use $\delta$ from the previous iteration (momentum). The gradients of the perturbation and the adversarial image are jointly computed in the same forward/backward pass.
    - Design Motivation: Independently updating $\delta$ for every layer at each step would require multiple forward/backward passes—random selection + momentum approximates full updates with zero extra overhead. Ablation studies show that updating all layers simultaneously actually performs worse.

3. **FTM-E (Ensemble Variant)**:

    - Function: Further enhances transferability.
    - Mechanism: Multiple copies of the surrogate model are used, each independently applying FTM, followed by ensemble attacks. Using 2 copies achieves the best performance-cost trade-off (79.5% vs. 77.4% for a single copy).
    - Design Motivation: Different copies exhibit distinct random perturbation paths, and ensembling enhances the robustness of the attack.

### Loss & Training
The adversarial image is iteratively optimized using PGD. Min-max objective: $\min_x \max_\delta \mathcal{L}(f_\theta(x + \delta), y_{target})$. No extra forward/backward overhead is incurred.

## Key Experimental Results

### Main Results

| Method | Time | RN-50$\rightarrow$14 Models Avg. Success Rate |
|------|------|----------------------|
| RDI | 1.23s | 49.4% |
| ODI | 4.38s | 69.2% |
| RDI-CFM | 1.39s | 74.6% |
| **RDI-FTM** | **1.54s** | **77.4%** |
| **RDI-FTM-E** | **2.92s** | **79.5%** |

### Ablation Study

| Configuration | Success Rate |
|------|-------|
| CFM alone | 74.6% |
| FTM alone | Better than CFM |
| **FTM + CFM** | **77.4%** |
| Update all layers simultaneously | Decreased (overfitting surrogate) |
| Random layer selection + momentum | **Optimal** |

### Key Findings
- **Attack-optimized perturbation > Random clean perturbation**: FTM alone outperforms CFM, and their combination is even better.
- **Random layer selection is crucial**: Updating all layers leads to overfitting the surrogate model, while random selection introduces implicit regularization.
- **Zero extra computation**: 1.54s vs. CFM 1.39s, featuring an overhead increase of only 11%.

## Highlights & Insights
- **Min-max feature perturbation** performs adversarial training in the feature space to improve attack robustness—essentially making the adversarial image more "immune" to perturbations in the feature space.
- **Momentum-based stochastic update** is an elegant efficiency solution—avoiding the computational explosion of multi-layer updates.

## Limitations & Future Work
- Only validated on image classification attacks; target detection/segmentation attacks remain untested.
- Transferability remains limited when there is a significant discrepancy between the surrogate model and target models.
- Targeted attacks are more challenging than untargeted ones, leaving room for further improvement in success rates.

## Related Work & Insights
- **vs. CFM**: CFM only uses random clean perturbations. FTM introduces attack-optimized perturbations to further improve performance by 2.8%.
- **vs. ODI**: ODI requires multiple forward passes (4.38s). FTM takes only 1.54s and achieves better performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of min-max perturbation in the feature space is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 14 black-box models, multiple surrogate models, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of methodology and ablation analysis.
- Value: ⭐⭐⭐ Contributes to adversarial attack research, though application scenarios are somewhat limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](../../ACL2025/others/targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)

</div>

<!-- RELATED:END -->
