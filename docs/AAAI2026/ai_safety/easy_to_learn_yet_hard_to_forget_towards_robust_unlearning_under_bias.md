---
title: >-
  [Paper Note] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias
description: >-
  [AAAI2026][AI Safety][machine unlearning] Proposes the CUPID framework, which partitions the forget set into causal/bias subsets through loss landscape sharpness analysis, and identifies and disentangles the causal/bias pathways in the model. This achieves precise class unlearning on biased models, effectively resolving the "shortcut unlearning" issue.
tags:
  - "AAAI2026"
  - "AI Safety"
  - "machine unlearning"
  - "shortcut learning"
  - "data bias"
  - "loss landscape"
  - "causal pathway"
date: 2026-05-08
content_hash: 40d368ddbb158937
---

# Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias

**Conference**: AAAI2026  
**arXiv**: [2602.21773](https://arxiv.org/abs/2602.21773)  
**Code**: TBD  
**Area**: AI Safety  
**Keywords**: machine unlearning, shortcut learning, data bias, loss landscape, causal pathway

## TL;DR

Proposes the CUPID framework, which partitions the forget set into causal/bias subsets through loss landscape sharpness analysis, and identifies and disentangles the causal/bias pathways in the model. This achieves precise class unlearning on biased models, effectively resolving the "shortcut unlearning" issue.

## Background & Motivation

The goal of machine unlearning is to efficiently remove the influence of specific data from pretrained models to comply with privacy regulations such as the "right to be forgotten." Existing methods generally assume that target information is separable within the model parameters. However, real-world data often contains spurious correlations—for example, the "waterbird" class is highly correlated with "water surface" backgrounds. Models learn such shortcuts, leading to deep entanglement between class features and bias features.

The authors systematically investigate the behavior of unlearning algorithms on biased models for the first time, revealing two key phenomena:

1. **"Easy-to-learn, hard-to-forget" asymmetry**: Bias-aligned samples (i.e., samples where spurious features match class labels) are learned fastest but are the hardest to forget; conversely, bias-conflicting samples are easy to forget.
2. **Counter-intuitive debiasing effect**: The unlearning process abnormally improves the accuracy of bias-conflicting samples in the target class.

These two phenomena collectively constitute what the authors define as **shortcut unlearning**—when requested to forget the target class, the model primarily erases the spurious shortcut features rather than the actual causal class features.

## Core Problem

How can an unlearning algorithm precisely erase causal class information, rather than taking a shortcut and erasing only bias features, when internal representations of the model are highly entangled? The key challenges include:

- Needing to distinguish parameters in the model that rely on causal features vs. shortcut features.
- Needing to apply different update strategies to different parameter subsets.
- The entire process should not rely on a retain set to accommodate privacy-constrained scenarios.

## Method

CUPID (Causal Unlearning via Pathway Identification and Disentanglement) consists of three stages:

### Phase 1: Sharpness-Aware Partitioning (Sharpness-Based Partitioning)

The core intuition comes from generalization theory: the model converges to a flat minimum (low curvature) for "easy-to-learn" bias-aligned samples, while remaining in sharp regions (high curvature) for "hard-to-learn" bias-conflicting samples.

Calculate local sharpness for each sample in the forget set:

- First, apply an adversarial perturbation of step size $\eta$ along the gradient direction: $\theta_{adv} = \theta_o + \eta \frac{\nabla L(\theta_o, x_i)}{\|\nabla L(\theta_o, x_i)\|}$
- Sharpness is defined as the difference in loss before and after perturbation: $\omega_{sharpness}(x_i) = L(\theta_{adv}, x_i) - L(\theta_o, x_i)$

Based on a top-$k$% threshold of sharpness values, partition the forget set into:

- $\mathcal{D}_f^{bias}$ (low sharpness, approximating bias-aligned samples)
- $\mathcal{D}_f^{causal}$ (high sharpness, approximating bias-conflicting/causal samples)

Experiments show that $k=5\%$ yields the best results—it is not "the purer, the better," as moderately including some bias-aligned samples can regularize the causal gradient direction.

### Phase 2: Causal Pathway Identification (Causal Pathway Identification)

The goal is to separate the model parameters $\theta_o$ into causal pathways and bias pathways. For each parameter $\theta_{o,i}$, a causal mask is defined by combining its magnitude and Hessian diagonal elements:

$$m_c(\theta_{o,i}) = \mathbb{1}\left(\frac{1}{2}\theta_{o,i}^2 \cdot \mathbb{E}_{x \sim \mathcal{D}_f^{causal}}[H(\theta_o, x)_{ii}] \geq \tau_p\right)$$

where $\tau_p$ is set to select the top 50% most influential parameters. This design draws on classical network pruning concepts (LeCun et al. 1989), measuring parameter significance using parameter magnitude $\times$ second derivative. Parameters with $m_c=1$ constitute the causal pathways, while the rest form the bias pathways.

### Phase 3: Targeted Pathway Update (Targeted Pathway Update)

Apply different gradient updates to the two pathways:

1. Compute the causal gradient direction $g_{causal}$ (average gradient on $\mathcal{D}_f^{causal}$).
2. Project the full forget set gradient $g_f$ onto the causal direction: $g_{proj} = \frac{g_f \cdot g_{causal}}{\|g_{causal}\|^2} g_{causal}$
3. The orthogonal component serves as the bias gradient: $g_{bias} = g_f - g_{proj}$

Final update rule:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \cdot [(\omega_{sharpness} \cdot g_{proj} \odot m_c) + (g_{bias} \odot (1 - m_c))]$$

- Causal pathway ($m_c=1$): Updated using the projected causal gradient and weighted by sample sharpness, applying stronger unlearning force to "hard samples".
- Bias pathway ($m_c=0$): Updated only with the bias gradient to avoid mistakenly deleting causal information.

## Key Experimental Results

Evaluated on three biased datasets (training set bias ratio 99.5:0.5, test set 50:50):

**Unlearning performance on biased training sets (Table 1)**:

| Method | Waterbirds FA↓ | BAR FA↓ | NICO++ FA↓ |
|------|----------------|---------|------------|
| Retrain (Upper Bound) | 0.00 | 0.00 | 0.00 |
| NegGrad | 34.96 | 58.59 | 22.33 |
| DELETE | 18.42 | 34.86 | 27.84 |
| **CUPID** | **6.91** | **7.70** | **7.71** |

**Generalized unlearning on unbiased test sets (Table 2)**:

| Method | Waterbirds FA↓ | BAR FA↓ | NICO++ FA↓ |
|------|----------------|---------|------------|
| Retrain | 0.00 | 0.00 | 0.00 |
| DELETE | 8.73 | 34.38 | 22.95 |
| **CUPID** | **6.02** | **3.75** | **8.34** |

CUPID achieves the lowest FA across all datasets, and $\triangle_{gap}$ and WGA are also the lowest, indicating that the unlearning effect is most balanced between the bias-aligned and bias-conflicting sample groups.

**Ablation Study (Table 3, Waterbirds)**:

| Sharpness Partitioning | Pathway Identification | Targeted Update | FA↓ |
|----------|---------|---------|-----|
| ✗ | ✗ | ✗ | 34.96 |
| ✓ | ✗ | ✗ | 20.38 |
| ✓ | ✓ | ✗ | 14.56 |
| ✓ | ✓ | ✓ | **6.91** |

All three components contribute incrementally, and none can be omitted.

## Highlights & Insights

- **Novel Problem Definition**: Formally introduces the shortcut unlearning problem for the first time, exposing the asymmetry of "easy-to-learn, hard-to-forget" and pointing out a fundamental failure mode of existing unlearning methods on biased data.
- **Elegant Method Design**: Leverages loss landscape geometric properties (flat vs. sharp regions) as an unsupervised signal to distinguish sample types without requiring bias labels.
- **No Retain Set Required**: CUPID operates using only the forget set, making it highly practical in privacy-restricted scenarios.
- **Significant Performance**: Achieves an FA of only 3.75% on the BAR dataset, compared to 30.26% for the second-best method, showing a substantial performance gap.
- **Clear Grad-CAM Visualization** clearly shows CUPID's attention shifting away from spurious features, validating the effectiveness of the method.

## Limitations & Future Work

- Only validated on image classification tasks, leaving NLP or generative models unexplored.
- The computation cost of the Hessian diagonal might be a bottleneck on large-scale models.
- The sharpness threshold $k$ and pathway ratio $\tau_p$ require tuning and may need different settings for different bias intensities.
- Only considered scenarios with a single bias attribute; behavior under co-existing multiple biases remains unexplored.
- The applicability to concept unlearning in LLMs (such as RLHF unlearning) was not discussed.

## Related Work & Insights

- **NegGrad** (direct gradient reversal): Performs poorly on biased data, with FA as high as 34.96%, because gradient reversal preferentially erases the most salient shortcut features.
- **SALUN** (saliency-based unlearning): Attempts to select key parameters to update but does not distinguish between causal and bias pathways, remaining limited on biased data.
- **Bad Teaching** (no retain set needed): Uses an incompetent teacher for distillation but exhibits extremely high FA in biased scenarios (88.35% on Waterbirds).
- **DELETE** (state-of-the-art distillation method): Represents the strongest baseline but still suffers from a notable $\triangle_{gap}$, indicating imbalanced unlearning.

The core advantage of CUPID lies in elevating "parameter selection" from mere saliency analysis to "causal vs. bias pathway" separation, which is absent in other methods.

## Inspirations & Connections

- The concept of utilizing loss landscape sharpness as an unsupervised signal to distinguish sample types has broad applicability and could be extended to areas such as data cleaning and anomaly detection.
- The separation framework of "causal pathway vs. bias pathway" could inspire subsequent work in the direction of model debiasing.
- It shares common ground with parameter localization concepts in the field of model editing, offering opportunities for cross-disciplinary insights.
- The shortcut unlearning problem serves as a reminder: when discussing the "unlearning" capabilities of LLM safety alignment, we must remain vigilant about whether the model is merely forgetting superficial patterns.

## Rating
- **Novelty**: 9/10 (First to formalize shortcut unlearning, providing deep insights into the problem)
- **Experimental Thoroughness**: 8/10 (Evaluated on three datasets with ablations and visualizations, though lacking NLP experiments)
- **Writing Quality**: 9/10 (Analysis-driven narrative structure with clear and highly persuasive charts)
- **Value**: 8/10 (Establishes a solid foundation for the unlearning problem in biased scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Machine Unlearning under Retain–Forget Entanglement](../../ICLR2026/ai_safety/machine_unlearning_under_retainforget_entanglement.md)
- [\[ICLR 2026\] Mitigating Privacy Risk via Forget Set-Free Unlearning](../../ICLR2026/ai_safety/mitigating_privacy_risk_via_forget_set-free_unlearning.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](../../ICML2026/ai_safety/how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[ICLR 2026\] Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning](../../ICLR2026/ai_safety/dont_shift_the_trigger_robust_gradient_ascent_for_backdoor_unlearning.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)

</div>

<!-- RELATED:END -->
