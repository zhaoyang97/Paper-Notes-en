---
title: >-
  [Paper Note] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation
description: >-
  [CVPR 2026][AI Safety][adversarial training] This paper analyzes the energy landscape to reveal the complementarity between adversarial training (AT) and JEM—AT aligns the clean-adversarial energy distribution (→ robustn…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "adversarial training"
  - "energy-based model"
  - "JEM"
  - "robustness"
  - "generation"
date: 2026-05-08
content_hash: 3e0d225699d391d1
---

# Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation

**Conference**: CVPR 2026
**arXiv**: [2505.19459](https://arxiv.org/abs/2505.19459)  
**Code**: [GitHub](https://github.com/yujkc/EB-JDAT)  
**Area**: AI Security / Adversarial Robustness / Energy-Based Models
**Keywords**: adversarial training, energy-based model, JEM, robustness, generation

## TL;DR

This paper analyzes the energy landscape to reveal the complementarity between adversarial training (AT) and JEM—AT aligns the clean-adversarial energy distribution (→ robustness); JEM aligns the clean-generated energy distribution (→ accuracy + generation). The proposed EB-JDAT models the joint distribution $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$ and employs min-max energy optimization to align the energy distributions of all three data types. On CIFAR-10, AutoAttack robustness reaches 68.76% (surpassing SOTA AT by +10.78%), while maintaining 90.39% clean accuracy and competitive generation quality with FID=27.42.

## Background & Motivation

**Background**: Classifiers face a trilemma among accuracy, robustness, and generative capability. Adversarial training (AT) methods such as PGD/TRADES are the most effective for robustness but sacrifice clean accuracy and offer no generative ability. Joint Energy-based Models (JEM) reinterpret the softmax logits as an EBM to unify classification and generation, yet fall far short of AT in adversarial robustness.

**Limitations of Prior Work**: (1) AT methods are robust but suffer a 5–10% drop in clean accuracy and possess no generative capability; (2) JEM achieves classification and generation jointly but achieves far lower adversarial robustness than AT; (3) augmenting AT with additional generated data (e.g., 1M diffusion images) can improve robustness but incurs prohibitive computational cost (1000+ GPU hours) and still provides no generative capability.

**Key Challenge**: AT and JEM each address only two dimensions of the trilemma and cannot be unified. The fundamental issue is that both model the data distribution incompletely—AT focuses solely on $p(y|\tilde{x})$, while JEM focuses solely on $p(x,y)$.

**Goal**: To achieve high classification accuracy, adversarial robustness, and generative capability simultaneously within a single model, thereby breaking the trilemma.

**Key Insight**: Diagnosis from an energy distribution perspective—AT causes the clean and adversarial energy distributions to overlap (Tab. 1: AT mean difference 1.46 vs. 10.18 for a standard model); JEM causes the clean and generated energy distributions to overlap. Aligning all three energy distributions would unify all three capabilities.

**Core Idea**: Model the joint distribution of clean and adversarial examples, $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$, and use min-max energy optimization to pull adversarial examples from high-energy regions back to low-energy regions, while simultaneously maintaining generative sampling and classification training.

## Method

### Overall Architecture

The joint distribution $p(\mathbf{x}, y)$ from JEM is extended to a three-way joint distribution $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$, which is decomposed via Bayes' rule into three terms: $p(y|\tilde{\mathbf{x}}, \mathbf{x})$ (robust classification via CE), $p(\tilde{\mathbf{x}}|\mathbf{x})$ (adversarial distribution modeling via min-max energy optimization), and $p(\mathbf{x})$ (data distribution modeling via SGLD sampling and energy maximum likelihood). The total gradient $h_\theta = h_1 + h_2 + h_3$ drives generation, energy alignment, and robust classification, respectively.

### Key Designs

1. **Min-Max Energy Optimization for Modeling $p(\tilde{\mathbf{x}}|\mathbf{x})$**

    - **Function**: Learns to pull adversarial examples back to low-energy regions through energy maximization-minimization, without requiring prior knowledge of the adversarial distribution.
    - **Mechanism**: The key observation is that adversarial perturbations almost always push samples away from the high-density data manifold into low-density (high-energy) regions. **Inner max**: Reverse SGLD samples adversarial examples along the energy ascent direction, pushing them toward high-energy regions. **Outer min**: Minimizes the clean-adversarial energy gap $\min_\theta \mathbb{E}[\max_{\|\tilde{\mathbf{x}}-\mathbf{x}\| \in \Omega}(E_\theta(\tilde{\mathbf{x}}|\mathbf{x}) - E_\theta(\mathbf{x}))]$, pulling adversarial examples back to low-energy regions. The gradient is approximated as $h_2 \approx \frac{\partial}{\partial\theta}[\frac{1}{L_1}\sum E_\theta(\mathbf{x}_i^+) - \frac{1}{L_2}\sum E_\theta(\tilde{\mathbf{x}}_i|\mathbf{x}_i^+)]$.
    - **Design Motivation**: Unlike conventional AT, which performs min-max over cross-entropy (finding the most misleading samples), EB-JDAT performs min-max over the energy gap (finding the highest-energy samples and then pulling them back), directly operating on the energy landscape rather than the cross-entropy loss.

2. **Joint Optimization of Three Gradient Terms**

    - **Function**: Simultaneously drives generation, energy alignment, and robust classification.
    - **Mechanism**: $h_1 = \partial \log p(\mathbf{x})/\partial\theta$ (drives generation via SGLD positive/negative sample energy difference); $h_2 = \partial \log p(\tilde{\mathbf{x}}|\mathbf{x})/\partial\theta$ (clean-adversarial energy alignment); $h_3 = \partial \log p(y|\mathbf{x}, \tilde{\mathbf{x}})/\partial\theta$ (standard CE for robust classification). Default weights are $w_1=w_2=w_3=1$.
    - **Design Motivation**: Ablations show that $h_2$ (energy alignment) is critical to preventing training collapse—removing $h_2$ causes collapse at epoch 41 (ECO=41), while retaining it stabilizes training throughout. $h_1$ contributes generative capability and additional classification accuracy.

3. **Plug-and-Play Compatibility**

    - **Function**: EB-JDAT serves as a general framework that can be directly grafted onto existing JEM variants.
    - **Mechanism**: Seamlessly integrates with JEM++ (faster SGLD sampling) or SADAJEM (more stable training), leveraging their improved sampling strategies without modifying the main framework. EB-JDAT-SADAJEM achieves the best robustness on CIFAR-10 at 68.76%/66.12% (PGD-20/AA); EB-JDAT-JEM++ trains faster (31.66h vs. 66.64h).
    - **Design Motivation**: The modular design enhances practicality and allows the community's accumulated improvements to JEM variants to directly benefit from the framework.

### Loss & Training

WRN28-10 backbone; lr=0.01; 5-step adversarial sampling; $\ell_\infty$ constraint $\epsilon=8/255$; 100 epochs; 3090 GPU.

## Key Experimental Results

### Main Results — Comparison with SOTA AT Methods

| Method | Clean (%) | PGD-20 (%) | AA (%) |
|--------|-----------|------------|--------|
| MART | 82.99 | 55.48 | 50.67 |
| AWP | 82.67 | 57.21 | 51.90 |
| LAS-AWP | 87.74 | 60.16 | 55.52 |
| DHAT-CFA | 84.49 | 62.38 | 54.05 |
| **EB-JDAT-JEM++** | **90.30** | **64.88** | **64.78** |
| **EB-JDAT-SADAJEM** | **90.37** | **68.76** | **66.12** |

### Comparison with AT Using Additional Generated Data

| Method | Extra Data | Clean (%) | AA (%) | GPU Time |
|--------|-----------|-----------|--------|----------|
| SCORE | 1M | 88.10 | 61.51 | ~1438h |
| Better DM | 1M | 91.12 | 63.35 | ~1438h |
| [Gowal] | 100M | 87.50 | 63.38 | ~719460h |
| **EB-JDAT-SADAJEM** | **None** | **90.39** | **66.30** | **66.64h** |

### Three-Dimensional Comparison with JEM / Energy-Based AT Methods

| Method | Clean (%) | AA (%) | FID↓ | IS↑ |
|--------|-----------|--------|------|-----|
| JEM | 92.90 | 4.28 | 38.40 | 8.76 |
| JEM++ | 93.73 | 41.06 | 37.12 | 8.29 |
| SADAJEM | 96.03 | 29.63 | 17.38 | 8.07 |
| JEAT | 85.16 | 28.43 | 38.24 | 8.80 |
| WEAT | 83.36 | 49.02 | 30.74 | 8.97 |
| **EB-JDAT-SADAJEM** | **90.39** | **66.30** | **27.42** | **8.05** |

### Ablation Study (EB-JDAT-JEM++)

| $w_1$ | $w_2$ | $w_3$ | Clean | AA | FID | Collapse Epoch |
|-------|-------|-------|-------|----|-----|----------------|
| 0 | 0 | 1 | 88.95 | 62.96 | 173.53 | 41 |
| 0 | 1 | 1 | 89.84 | 64.69 | 42.57 | n/a |
| 1 | 0.5 | 1 | 90.39 | 64.09 | 40.12 | n/a |
| 1 | 1 | 1 | 90.37 | 64.61 | 39.67 | n/a |

### Key Findings

- **$h_2$ (energy alignment) is critical to preventing collapse**: Setting $w_2=0$ causes collapse at epoch 41; retaining $h_2$ stabilizes training to completion.
- **No additional data required**: Without any extra data and within 100 epochs, EB-JDAT surpasses SCORE—which uses 1M generated images—by +4.79% AA, with training time of only 66h vs. 1438h.
- **Breaking the trilemma**: Simultaneously achieves 90.39% clean accuracy (only 5.71% below the standard model's 96.10%), 66.30% AA robustness (SOTA), and competitive generation with FID=27.42.
- Effectiveness is also validated on an ImageNet subset: Clean 63.02%, AA 32.40%, surpassing WEAT by +7.88%.

## Highlights & Insights

1. **Energy landscape analysis as a diagnostic methodology**: By visualizing the energy distributions of clean, adversarial, and generated samples, the paper intuitively reveals the respective mechanisms of AT and JEM—AT compresses the clean-adversarial gap, JEM compresses the clean-generated gap. This analytical approach is transferable to other scenarios requiring understanding of model behavior.
2. **Min-max energy optimization as an alternative to max-CE**: Conventional AT performs min-max in cross-entropy space; EB-JDAT performs min-max in energy space—semantically more intuitive (high energy = low density = adversarial region) and capable of additionally capturing the structure of the data distribution.
3. **Computational efficiency vastly superior to data augmentation approaches**: The method surpasses data-augmented AT counterparts without generating 1M images, as directly modeling the energy distribution is more fundamental than using generated data for indirect regularization.

## Limitations & Future Work

1. Experiments are conducted only on CIFAR-10/100 and an ImageNet subset; validation on full ImageNet is absent (attributed by the authors to resource constraints).
2. The method is sensitive to the number of adversarial sampling steps (5 steps optimal); increasing steps leads to EBM collapse—a longstanding instability issue with energy-based model training.
3. Compared to the strongest JEM baseline (SADAJEM at 96.03%), clean accuracy drops to 90.39%, indicating that the trilemma is substantially mitigated but not fully resolved.
4. Generation quality (FID=27.42) still lags behind diffusion models significantly; SGLD inherently limits the sampling quality of EBMs.

## Related Work & Insights

- **vs. JEAT**: JEAT models $p(\tilde{\mathbf{x}}, y)$ by directly incorporating adversarial examples into JEM, neglecting the clean-adversarial relationship. EB-JDAT models the complete $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$ and explicitly aligns energy distributions.
- **vs. WEAT**: WEAT reinterprets TRADES as an EBM modeling $p(y|\tilde{\mathbf{x}}, \mathbf{x})$, remaining essentially a discriminative model. EB-JDAT achieves a genuine generative-discriminative unification.
- **vs. TRADES**: TRADES constrains clean-adversarial output consistency via KL divergence; EB-JDAT performs a more fundamental alignment at the energy distribution level.

## Rating

⭐⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐⭐: The energy landscape diagnostic approach combined with min-max energy optimization is highly natural and insightful; the three-way joint distribution modeling is the first of its kind.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Comprehensive comparisons against four categories of methods (AT, JEM, energy-based AT, data-augmented AT) with clear ablations, though full ImageNet evaluation is absent.
- **Writing Quality** ⭐⭐⭐⭐⭐: The method follows naturally from the energy landscape analysis, with an exceptionally clear logical chain.
- **Value** ⭐⭐⭐⭐⭐: The first work to simultaneously achieve top-tier performance across all three dimensions, providing a compelling answer to what classifiers can do.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](../../ICML2026/ai_safety/demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](../../NeurIPS2025/ai_safety/enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](../../AAAI2026/ai_safety/breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)
- [\[CVPR 2026\] All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference](all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co.md)

</div>

<!-- RELATED:END -->
