---
title: >-
  [Paper Note] 机器遗忘的两个盲点：过度遗忘与原型重学习攻击
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper reveals two critical blind spots in machine unlearning—over-unlearning (collateral damage to samples near the decision boundary) and prototype relearning attacks (recovering forgotten knowledge using a few samples). It proposes the Spotter framework to simultaneously mitigate these issues via boundary mask d
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: 3a41520949e5a6ba
---
# Two Blind Spots in Machine Unlearning: Over-Unlearning and Prototype Relearning Attacks

**Conference**: ICML 2026  
**arXiv**: [2506.01318](https://arxiv.org/abs/2506.01318)  
**Code**: To be confirmed  
**Area**: AI Security / Privacy Protection / Machine Unlearning  
**Keywords**: Machine Unlearning, Over-unlearning, Relearning Attack, Privacy, Classifier

## TL;DR
This paper reveals two critical blind spots in machine unlearning—over-unlearning (collateral damage to samples near the decision boundary) and prototype relearning attacks (recovering forgotten knowledge using a few samples). It proposes the Spotter framework to simultaneously mitigate these issues via boundary mask distillation and intra-class dispersion loss.

## Background & Motivation

**Background**: Machine Unlearning (MU) aims to rapidly remove the influence of specific data from a model to avoid expensive full retraining. Existing methods include parameter resetting, decision boundary shifting, data partitioning, and knowledge distillation.

**Limitations of Prior Work**: Existing unlearning methods suffer from two serious but long-neglected issues: **Over-unlearning** (performance degradation of nearby retained samples when deleting a forgotten class) and **vulnerability after unlearning** (adversaries can rapidly relearn deleted knowledge with only a few samples).

**Key Challenge**: How to simultaneously achieve thorough unlearning and maintain retain integrity? Existing methods typically focus only on unlearning quality (forget accuracy $\to 0$) or retain accuracy, ignoring hidden damage in boundary regions and the threat of subsequent relearning attacks.

**Goal**: For class-level unlearning—quantify over-unlearning, expose relearning risks, and design a defense scheme.

**Key Insight**: The focus is shifted from global retain accuracy to the **boundary neighborhood**, as proximal retained samples are most susceptible to damage when the decision boundary moves. Furthermore, it is observed that features of the forgotten class remain highly clustered in the embedding space, providing an opportunity for prototype relearning.

**Core Idea**: Define the boundary neighborhood using invertible perturbations, design a retain-data-independent over-unlearning metric $OU@\varepsilon$, and construct an unlearning framework that resists both attacks by combining boundary mask distillation and intra-class feature dispersion.

## Method

### Overall Architecture
Spotter aims to resolve two side effects in class-level unlearning: the collateral damage to retained samples near the decision boundary (over-unlearning) and the ease with which the model can be recovered with few samples (relearning attack). It splits these problems into two tracks—"quantification" and "defense." First, it generates perturbed samples along the decision boundary and uses a retain-data-independent metric to quantify boundary damage, which is corrected by mask distillation. Simultaneously, it forcibly scatters the forgotten class in the feature space, preventing attackers from obtaining a usable prototype structure.

### Key Designs

**1. Over-Unlearning Metric $OU@\varepsilon$: Quantifying Invisible Boundary Damage**

Global retain accuracy hides a pain point: when unlearning pushes the decision boundary outward, retained samples close to the boundary are easily misclassified, yet average accuracy remains nearly unchanged. This paper instantiates the "boundary neighborhood" by adding an invertible perturbation $\delta$ around each forgotten sample $\boldsymbol{x}$ to obtain the perturbation set $\mathcal{A}_{\varepsilon}(\mathcal{D}_f) = \{\boldsymbol{x} + \delta \mid \boldsymbol{x} \in \mathcal{D}_f, \delta \in \Delta_{\varepsilon}\}$. Using a masked softmax $\tilde{\sigma}$ to zero out the forgotten class probability (avoiding contamination of the metric by the deletion itself), the predictive distribution shift between the original model $\theta$ and the unlearned model $\theta_u$ is compared:

$$OU@\varepsilon := \mathbb{E}_{\boldsymbol{x}_p \sim \mathcal{A}_\varepsilon}\left[D\big(\tilde{\sigma}(\boldsymbol{z}(\boldsymbol{x}_p;\theta)) \,\|\, \sigma(\boldsymbol{z}(\boldsymbol{x}_p;\theta_u))\big)\right]$$

A larger KL divergence indicates more significant alterations near the boundary. Crucially, the metric uses only the forgotten samples and does not require the original retain set, making it practical for real-world scenarios.

**2. Prototype Relearning Attack (PRA): Exposing Feature Space Vulnerabilities**

Many unlearning methods only modify the classification head. While the forget accuracy reaches zero, the feature extractor $\phi_{\theta_u}$ still clusters forgotten samples together. Attackers can exploit this: with $k$ forgotten samples, they compute the mean of their features as a class prototype $\mathbf{p}^{(c)} = \frac{1}{k}\sum_{i=1}^k \phi_{\theta_u}(\boldsymbol{x}_i^{(c)})$ and insert this prototype back into the classifier head. In experiments, only 1–10 images are needed to recover near-original accuracy. This attack demonstrates that unlearning is often incomplete.

**3. Joint Optimization Objective: Simultaneous Pressure on Decision and Feature Spaces**

Spotter combines standard unlearning, boundary protection, and anti-relearning into a single objective:

$$\mathcal{L} = \lambda_1 \mathcal{L}_u + (1-\lambda_1) \mathcal{L}_o + \lambda_2 \mathcal{L}_{sim}$$

Where $\mathcal{L}_u$ is the standard unlearning loss; $\mathcal{L}_o$ is the mask distillation loss on the boundary neighborhood that constrains the unlearned model's predictions on $\mathcal{A}_\varepsilon$ to match the original model's distribution; and $\mathcal{L}_{sim}$ is the sum of intra-class cosine similarities, which minimizes feature clustering for the forgotten class.

### Function
Taking the unlearning of the "plane" class in CIFAR-10 as an example: boundary points are generated from plane samples, revealing $OU@\varepsilon \approx 0.16$ after basic unlearning. Enabling $\mathcal{L}_o$ reduces $OU@\varepsilon$ to the $0.03$ level. Simultaneously, while a PRA with 10 images could recover 60%+ accuracy without $\mathcal{L}_{sim}$, enabling $\lambda_2=1$ for $\mathcal{L}_{sim}$ scatters the features, suppressing recovery accuracy to 0.24%—making unlearning truly irreversible.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 Forget Acc↓ | CIFAR-10 Retain Acc↑ | Over-Unlearning↓ | PRA Acc↓ | CIFAR-100 Retain↑ |
|------|----------|----------|---------|---------|----------|
| Original Model | 100.00 | 100.00 | - | 100.00 | 99.99 |
| Retrain Baseline | 0.00 | 100.00 | 0.2384 | 58.70 | 99.78 |
| NegGrad | 0.18 | 87.73 | 0.3269 | 2.54 | 15.61 |
| Boundary Shrink | 3.82 | 93.79 | 0.1435 | 72.96 | 11.90 |
| UNSC | 0.00 | 99.98 | 0.1575 | 71.10 | 99.09 |
| **Spotter (λ₂=0.1)** | **0.00** | **100.00** | **0.0139** | **62.12** | **99.79** |
| **Spotter (λ₂=1)** | **0.00** | **99.98** | **0.0228** | **0.24** | **99.69** |

### Ablation Study

| Base Method | PRA (Before) | PRA (After Spotter) | Gain |
|--------|---------|---------|------|
| SalUn | 11.70% | 4.44% | ↓62% |
| DELETE | 31.72% | 3.34% | ↓89% |
| UNSC | 73.62% | 18.54% | ↓75% |

### Key Findings
- Spotter serves as a plug-and-play module for any base unlearning method.
- Combined with SalUn, $OU@\varepsilon$ decreased from 0.1664 to 0.0345 (79% reduction).
- Combined with DELETE, over-unlearning decreased from 0.1216 to 0.0232.
- At $\lambda_2=1$, PRA is almost completely defeated, though over-unlearning increases slightly.

## Highlights & Insights
- **Quantification of Boundary Damage**: The first to propose the retain-data-independent $OU@\varepsilon$ metric.
- **Empirical Threat of PRA**: Recovering 90%+ accuracy with just 1-10 images poses a real security risk for identity-sensitive applications like face recognition.
- **Dual Defense Design**: Mask distillation and intra-class dispersion loss exert pressure from different dimensions (decision space vs. feature space).
- **Plug-and-play Framework**: Simple addition of two terms in the loss function, validated across heterogeneous methods like DELETE, UNSC, and SalUn.

## Limitations & Future Work
- Parameter sensitivity of the boundary definition—the choice of $\varepsilon$ affects $OU@\varepsilon$ calculations.
- Sample size assumptions—PRA experiments assume the attacker possesses $k$ forgotten samples.
- Extension to other unlearning scenarios—currently focuses on class-level unlearning; applicability to sample-level or concept unlearning is yet to be clarified.
- Improvement: Incorporating sample difficulty weighting; exploring adaptive $\lambda_1, \lambda_2$ scheduling strategies.

## Related Work & Insights
- **vs. Over-unlearning Research (Hu et al., 2024a)**: Prior work reported the existence qualitatively; this paper provides the first quantification ($OU@\varepsilon$).
- **vs. Relearning Defense (Lynch et al., 2024)**: LLM scenarios use SAM for robust optimization; this work proposes feature dispersion for visual classification.
- **vs. Knowledge Distillation Unlearning**: Reuses distillation concepts but introduces unlearning constraints and boundary neighborhood regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Systematically exposes two long-ignored but serious practical blind spots.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Evaluated across CIFAR-10/100, TinyImageNet, Face Recognition datasets with 8+ base methods.
- Writing Quality: ⭐⭐⭐⭐  Clear problem formulation and rigorous methodological derivation.
- Value: ⭐⭐⭐⭐⭐  Spotter is plug-and-play, enhancing existing unlearning methods with direct industrial value for GDPR compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)
- [\[ICML 2026\] COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)
- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](how_does_bayesian_sampling_help_membership_inference_attacks.md)
- [\[ICML 2026\] Rethinking Evaluation Paradigms in IBP-based Certified Training](rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)

</div>

<!-- RELATED:END -->
