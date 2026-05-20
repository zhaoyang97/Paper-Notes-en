---
title: >-
  [Paper Note] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias
description: >-
  [AAAI2026][AI Safety][machine unlearning] This paper proposes the CUPID framework, which partitions the forget set into causal and bias subsets via loss landscape sharpness analysis…
tags:
  - "AAAI2026"
  - "AI Safety"
  - "machine unlearning"
  - "shortcut learning"
  - "data bias"
  - "loss landscape"
  - "causal pathway"
date: 2026-05-08
content_hash: e9f345fd49137096
---

# Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias

**Conference**: AAAI2026
**arXiv**: [2602.21773](https://arxiv.org/abs/2602.21773)  
**Code**: To be confirmed  
**Area**: AI Safety
**Keywords**: machine unlearning, shortcut learning, data bias, loss landscape, causal pathway

## TL;DR

This paper proposes the CUPID framework, which partitions the forget set into causal and bias subsets via loss landscape sharpness analysis, identifies and disentangles causal and bias pathways within the model, and achieves precise class-level unlearning on biased models — effectively addressing the shortcut unlearning problem.

## Background & Motivation

Machine unlearning aims to efficiently remove the influence of specific data from a pretrained model, in order to comply with privacy regulations such as the right to be forgotten. Existing methods generally assume that target information is separable in the model's parameter space; however, real-world data often contains spurious correlations — for instance, the class "waterbird" is highly correlated with "water" backgrounds. Models tend to exploit such shortcuts, causing class features and bias features to become deeply entangled.

The authors are the first to systematically study the behavior of unlearning algorithms applied to biased models, and identify two key phenomena:

1. **"Easy to learn, hard to forget" asymmetry**: Bias-aligned samples (where spurious features are consistent with class labels) are learned fastest yet are the most difficult to unlearn; bias-conflicting samples, by contrast, are relatively easy to unlearn.
2. **Counter-intuitive debiasing effect**: The unlearning process paradoxically improves accuracy on bias-conflicting samples of the target class.

These two phenomena jointly constitute what the authors define as **shortcut unlearning** — when asked to forget a target class, the model primarily erases spurious shortcut features rather than genuine causal class features.

## Core Problem

How can an unlearning algorithm precisely erase causal class information when internal model representations are highly entangled, without taking the shortcut of merely erasing bias features? The core challenges include:

- Distinguishing parameters that rely on causal features versus shortcut features within the model
- Applying different update strategies to different parameter subsets
- Accomplishing all of this without relying on a retain set, to accommodate privacy-constrained scenarios

## Method

CUPID (Causal Unlearning via Pathway Identification and Disentanglement) consists of three stages:

### Stage 1: Sharpness-Aware Partitioning

The core intuition stems from generalization theory: the model converges to flat minima (low curvature) for easy-to-learn bias-aligned samples, and resides in sharp regions (high curvature) for hard-to-learn bias-conflicting samples.

For each sample in the forget set, local sharpness is computed as follows:

- An adversarial perturbation of step size $\eta$ is applied along the gradient direction: $\theta_{adv} = \theta_o + \eta \frac{\nabla L(\theta_o, x_i)}{\|\nabla L(\theta_o, x_i)\|}$
- Sharpness is defined as the loss difference before and after perturbation: $\omega_{sharpness}(x_i) = L(\theta_{adv}, x_i) - L(\theta_o, x_i)$

Using the top-$k$% sharpness threshold, the forget set is partitioned into:

- $\mathcal{D}_f^{bias}$ (low sharpness, approximating bias-aligned samples)
- $\mathcal{D}_f^{causal}$ (high sharpness, approximating bias-conflicting/causal samples)

Experiments show that $k=5\%$ yields the best performance — purity is not monotonically beneficial; moderately including some bias-aligned samples regularizes the causal gradient direction.

### Stage 2: Causal Pathway Identification

The goal is to decompose the model parameters $\theta_o$ into causal and bias pathways. For each parameter $\theta_{o,i}$, a causal mask is defined by combining its magnitude with the diagonal of the Hessian:

$$m_c(\theta_{o,i}) = \mathbb{1}\left(\frac{1}{2}\theta_{o,i}^2 \cdot \mathbb{E}_{x \sim \mathcal{D}_f^{causal}}[H(\theta_o, x)_{ii}] \geq \tau_p\right)$$

where $\tau_p$ is set to select the top 50% most influential parameters. This design draws on classical network pruning (LeCun et al. 1989), using parameter magnitude × second-order derivatives to measure parameter saliency. Parameters with $m_c=1$ constitute the causal pathway; the remainder form the bias pathway.

### Stage 3: Targeted Pathway Update

Different gradient updates are applied to the two pathways:

1. Compute the causal gradient direction $g_{causal}$ (mean gradient over $\mathcal{D}_f^{causal}$)
2. Project the full forget-set gradient $g_f$ onto the causal direction: $g_{proj} = \frac{g_f \cdot g_{causal}}{\|g_{causal}\|^2} g_{causal}$
3. The orthogonal component serves as the bias gradient: $g_{bias} = g_f - g_{proj}$

The final update rule is:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \cdot [(\omega_{sharpness} \cdot g_{proj} \odot m_c) + (g_{bias} \odot (1 - m_c))]$$

- **Causal pathway** ($m_c=1$): Updated using the projected causal gradient, weighted by sample sharpness to impose stronger forgetting on harder samples.
- **Bias pathway** ($m_c=0$): Updated using only the bias gradient, preventing inadvertent erasure of causal information.

## Key Experimental Results

Evaluated on three biased datasets (training set bias ratio 99.5:0.5; test set 50:50):

**Unlearning performance on the biased training set (Table 1)**:

| Method | Waterbirds FA↓ | BAR FA↓ | NICO++ FA↓ |
|--------|----------------|---------|------------|
| Retrain (upper bound) | 0.00 | 0.00 | 0.00 |
| NegGrad | 34.96 | 58.59 | 22.33 |
| DELETE | 18.42 | 34.86 | 27.84 |
| **CUPID** | **6.91** | **7.70** | **7.71** |

**Generalized unlearning on the unbiased test set (Table 2)**:

| Method | Waterbirds FA↓ | BAR FA↓ | NICO++ FA↓ |
|--------|----------------|---------|------------|
| Retrain | 0.00 | 0.00 | 0.00 |
| DELETE | 8.73 | 34.38 | 22.95 |
| **CUPID** | **6.02** | **3.75** | **8.34** |

CUPID achieves the lowest FA across all datasets, with the lowest $\triangle_{gap}$ and WGA, indicating the most balanced unlearning effect between bias-aligned and bias-conflicting sample groups.

**Ablation Study (Table 3, Waterbirds)**:

| Sharpness Partitioning | Pathway Identification | Targeted Update | FA↓ |
|------------------------|----------------------|-----------------|-----|
| ✗ | ✗ | ✗ | 34.96 |
| ✓ | ✗ | ✗ | 20.38 |
| ✓ | ✓ | ✗ | 14.56 |
| ✓ | ✓ | ✓ | **6.91** |

Each component contributes incrementally and is indispensable.

## Highlights & Insights

- **Novel problem formulation**: The paper is the first to formally define the shortcut unlearning problem, revealing the "easy to learn, hard to forget" asymmetry and exposing fundamental failure modes of existing unlearning methods on biased data.
- **Elegant method design**: Loss landscape geometry (flat vs. sharp regions) is leveraged as an unsupervised signal to differentiate sample types without requiring bias labels.
- **No retain set required**: CUPID operates solely on the forget set, making it more practical in privacy-constrained scenarios.
- **Substantial performance gains**: On the BAR dataset, CUPID achieves an FA of only 3.75%, compared to 30.26% for the next-best method — a remarkable margin.
- **Grad-CAM visualizations** clearly demonstrate that CUPID shifts model attention away from spurious features, providing interpretable evidence of its effectiveness.

## Limitations & Future Work

- Validated only on image classification tasks; NLP and generative model settings are not explored.
- Computing the Hessian diagonal may become a computational bottleneck for large-scale models.
- The sharpness threshold $k$ and pathway ratio $\tau_p$ require tuning and may need adjustment for different bias intensities.
- Only single-attribute bias scenarios are considered; behavior under multiple co-existing biases remains unexplored.
- Applicability to concept unlearning in LLMs (e.g., RLHF unlearning) is not discussed.

## Related Work & Insights

- **NegGrad** (gradient negation): Performs poorly on biased data with FA as high as 34.96%, as gradient negation preferentially erases the most salient shortcut features.
- **SALUN** (saliency-based unlearning): Attempts to selectively update key parameters but does not distinguish causal from bias pathways, and remains limited under biased data.
- **Bad Teaching** (retain-set-free): Uses an incompetent teacher for distillation, but yields extremely high FA in biased settings (88.35% on Waterbirds).
- **DELETE** (latest distillation-based method): The strongest baseline, yet still exhibits a notable $\triangle_{gap}$, indicating unbalanced forgetting.

CUPID's core advantage lies in elevating parameter selection from simple saliency analysis to the disentanglement of causal versus bias pathways — a distinction absent from all competing methods.

The sharpness-based approach to unsupervised sample differentiation has broad applicability and may extend to data cleaning and anomaly detection. The causal/bias pathway disentanglement framework may inspire future work in model debiasing. Conceptual overlap with parameter localization in model editing suggests opportunities for cross-domain inspiration. Finally, the shortcut unlearning problem serves as a cautionary note for discussions of unlearning capabilities in LLM safety alignment, where models may similarly erase only surface-level patterns.

## Rating
- Novelty: 9/10 (first formalization of shortcut unlearning; profound problem insights)
- Experimental Thoroughness: 8/10 (three datasets + ablation + visualization, but lacking NLP experiments)
- Writing Quality: 9/10 (analysis-driven narrative structure is clear; figures and tables are persuasive)
- Value: 8/10 (provides a solid foundation for unlearning under biased settings)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](../../NeurIPS2025/ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [\[AAAI 2026\] An Information Theoretic Evaluation Metric for Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlearning.md)
- [\[AAAI 2026\] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks](improving_the_convergence_rate_of_ray_search_optimization_for_query-efficient_ha.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)

</div>

<!-- RELATED:END -->
