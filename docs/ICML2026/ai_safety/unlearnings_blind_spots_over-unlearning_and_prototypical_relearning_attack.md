---
title: >-
  [Paper Note] Two Blind Spots in Machine Unlearning: Over-Unlearning and Prototype Re-learning Attacks
description: >-
  [ICML 2026][AI Safety][Machine Unlearning] This paper reveals two critical blind spots in machine unlearning—over-unlearning (collateral damage to samples near the decision boundary) and prototype re-learning attacks (recovery of forgotten knowledge using few samples)—and proposes the Spotter framework to simultaneously mitigate both issues through boundary mask distillation and intra-class dispersion loss.
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Machine Unlearning"
  - "Over-unlearning"
  - "Re-learning Attacks"
  - "Privacy"
  - "Classifiers"
date: 2026-05-08
content_hash: b9f91074a82c1835
---

# Two Blind Spots in Machine Unlearning: Over-Unlearning and Prototype Re-learning Attacks

**Conference**: ICML 2026  
**arXiv**: [2506.01318](https://arxiv.org/abs/2506.01318)  
**Code**: To be confirmed  
**Area**: AI Security / Privacy Protection / Machine Unlearning  
**Keywords**: Machine Unlearning, Over-unlearning, Re-learning Attacks, Privacy, Classifiers

## TL;DR
This paper reveals two critical blind spots in machine unlearning—over-unlearning (collateral damage to samples near the decision boundary) and prototype re-learning attacks (recovery of forgotten knowledge using few samples)—and proposes the Spotter framework to simultaneously mitigate both issues through boundary mask distillation and intra-class dispersion loss.

## Background & Motivation

**Background**: Machine unlearning (MU) aims to rapidly remove the influence of specified data from a model to avoid expensive full retraining. Existing methods include parameter resetting, decision boundary shifting, data partitioning, and knowledge distillation.

**Limitations of Prior Work**: Existing unlearning methods suffer from two severe but long-neglected issues—**over-unlearning** (where retained samples close to the forgotten class suffer performance degradation during its removal) and **post-unlearning vulnerability** (adversaries can quickly relearn deleted knowledge using only a few samples).

**Key Challenge**: How to simultaneously achieve thorough unlearning and maintain retention integrity? Existing methods typically focus only on unlearning quality (forget class accuracy $\to$ 0) or retention accuracy, ignoring hidden damage in boundary regions and the threat of subsequent re-learning attacks.

**Goal**: Focus on class-level unlearning—quantitatively measuring over-unlearning, exposing re-learning risks, and designing defensive solutions.

**Key Insight**: The focus is shifted from global retention accuracy to **boundary neighborhoods**, as proximal retained samples are most susceptible to collateral damage when decision boundaries shift. It is also observed that features of forgotten classes remain highly clustered in the embedding space, providing an opportunity for prototype re-learning.

**Core Idea**: Define boundary neighborhoods via invertible perturbations; design a retention-data-independent over-unlearning metric $OU@\varepsilon$; and construct an unlearning framework that simultaneously resists both attacks by combining boundary mask distillation and intra-class feature dispersion.

## Method

### Overall Architecture
Spotter addresses two side effects in class-level unlearning that have not been seriously addressed: the collateral damage to retained samples near the decision boundary (over-unlearning) and the ability to easily recover the model with few samples after unlearning (re-learning attacks). It deconstructs these issues into two tracks: "quantification first, then defense." First, it generates perturbed samples along the decision boundary to quantify damage using a retention-data-independent metric and corrects it via mask distillation. Simultaneously, it forcibly scatters forgotten class features in the feature space to prevent attackers from obtaining usable prototype structures.

### Key Designs

**1. Over-unlearning Metric $OU@\varepsilon$: Quantifying Invisible Boundary Damage**

Global retention accuracy only reflects overall correctness, masking a hidden pain point: when unlearning pushes the decision boundary outward, retained samples close to the boundary are easily misclassified, though average accuracy barely detects this. This work concretizes the "boundary neighborhood" by adding invertible perturbations $\delta$ around each forgotten sample $\boldsymbol{x}$ to obtain a perturbation set $\mathcal{A}_{\varepsilon}(\mathcal{D}_f) = \{\boldsymbol{x} + \delta \mid \boldsymbol{x} \in \mathcal{D}_f, \delta \in \Delta_{\varepsilon}\}$, which falls precisely near the boundary. Using a masked softmax $\tilde{\sigma}$, the probability of the forgotten class is zeroed out (to prevent the unlearning itself from contaminating the metric), and the prediction distribution shift between the original model $\theta$ and the unlearned model $\theta_u$ is compared:

$$OU@\varepsilon := \mathbb{E}_{\boldsymbol{x}_p \sim \mathcal{A}_\varepsilon}\left[D\big(\tilde{\sigma}(\boldsymbol{z}(\boldsymbol{x}_p;\theta)) \,\|\, \sigma(\boldsymbol{z}(\boldsymbol{x}_p;\theta_u))\big)\right]$$

A larger KL divergence indicates more significant alterations near the boundary. Crucially, this metric only uses forgotten samples to generate perturbations and does not require the original retained set, making it calculable in realistic scenarios where training data is unavailable.

**2. Prototype Re-learning Attack (PRA): Exposing Feature Space Vulnerabilities**

Many unlearning methods only modify the decision head. Once the forgotten class accuracy reaches zero, they assume success, failing to notice that the feature extractor $\phi_{\theta_u}$ still clusters samples of the forgotten class. Attackers exploit this: by obtaining $k$ forgotten samples, they compute the mean of their features as a class prototype $\mathbf{p}^{(c)} = \frac{1}{k}\sum_{i=1}^k \phi_{\theta_u}(\boldsymbol{x}_i^{(c)})$ and plug this prototype directly back into the classifier head as weights. The deleted category is then resurrected—experiments show that 1–10 images are sufficient to recover near-original accuracy. This attack turns the suspicion that "unlearning is incomplete" into a reproducible empirical threat.

**3. Joint Optimization Objective: Simultaneous Pressure in Decision and Feature Spaces**

To address the two pain points, Spotter incorporates basic unlearning, boundary protection, and anti-re-learning into a single objective:

$$\mathcal{L} = \lambda_1 \mathcal{L}_u + (1-\lambda_1) \mathcal{L}_o + \lambda_2 \mathcal{L}_{sim}$$

Here, $\mathcal{L}_u$ is the standard unlearning loss; $\mathcal{L}_o$ is the masked distillation loss on the boundary neighborhood, constraining the unlearned model's predictions on $\mathcal{A}_\varepsilon$ to match the original model's distribution; and $\mathcal{L}_{sim}$ is the sum of intra-class feature cosine similarities. Minimizing $\mathcal{L}_{sim}$ effectively scatters forgotten class features in all directions, dismantling the prototype structure required for PRA. The first two terms ensure "clean unlearning without boundary damage" in the decision space, while the third term actively defends against re-learning in the feature space.

### A Complete Example
Taking the unlearning of the "airplane" class in CIFAR-10 as an example: invertible perturbations are added to airplane samples to obtain boundary points. After basic unlearning, $OU@\varepsilon$ is approximately 0.16, indicating that proximal retained samples (e.g., birds with similar shapes) have been quietly affected. After enabling $\mathcal{L}_o$, the model's predictions on these points are pulled back to the original model's judgment, reducing $OU@\varepsilon$ to the 0.03 magnitude. Meanwhile, an attacker using 10 airplane images to form a prototype could recover airplane accuracy to over 60% without $\mathcal{L}_{sim}$. With $\lambda_2=1$ for $\mathcal{L}_{sim}$, features are scattered, the prototype no longer represents a cluster center, and recovery accuracy is suppressed to 0.24%—making the unlearning truly irreversible.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 Forget Acc.↓ | CIFAR-10 Retain Acc.↑ | Over-unlearning↓ | PRA Acc.↓ | CIFAR-100 Retain↑ |
|------|----------|----------|---------|---------|----------|
| Original Model | 100.00 | 100.00 | - | 100.00 | 99.99 |
| Retrain Baseline | 0.00 | 100.00 | 0.2384 | 58.70 | 99.78 |
| NegGrad | 0.18 | 87.73 | 0.3269 | 2.54 | 15.61 |
| Boundary Shrink | 3.82 | 93.79 | 0.1435 | 72.96 | 11.90 |
| UNSC | 0.00 | 99.98 | 0.1575 | 71.10 | 99.09 |
| **Ours (λ₂=0.1)** | **0.00** | **100.00** | **0.0139** | **62.12** | **99.79** |
| **Ours (λ₂=1)** | **0.00** | **99.98** | **0.0228** | **0.24** | **99.69** |

### Ablation Study

| Base Method | Pre-PRA | Post-Spotter | Gain |
|--------|---------|---------|------|
| SalUn | 11.70% | 4.44% | ↓62% |
| DELETE | 31.72% | 3.34% | ↓89% |
| UNSC | 73.62% | 18.54% | ↓75% |

### Key Findings
- Spotter serves as a plug-and-play module that can be stacked onto any base unlearning method.
- When combined with SalUn, $OU@\varepsilon$ drops from 0.1664 to 0.0345 (a 79% reduction).
- When combined with DELETE, over-unlearning drops from 0.1216 to 0.0232.
- At $\lambda_2=1$, re-learning is completely defeated, though over-unlearning increases slightly.

## Highlights & Insights
- **Quantifying Boundary Damage**: The first to propose the retention-data-independent $OU@\varepsilon$ metric.
- **Empirical Threat of PRA**: Only 1-10 images are needed to recover over 90% accuracy, posing a real security risk to identity-sensitive applications like face recognition.
- **Dual Defense Design**: Masked distillation and intra-class dispersion loss exert pressure from different dimensions (decision space vs. feature space).
- **Plug-and-play Generality**: Requires adding only two terms to the loss function, validated effectively across heterogeneous methods like DELETE, UNSC, and SalUn.

## Limitations & Future Work
- Parameter sensitivity of boundary definitions—the choice of $\varepsilon$ impacts $OU@\varepsilon$ calculations.
- Sample size assumptions—PRA experiments assume the attacker possesses $k$ forgotten samples.
- Extension to other unlearning scenarios—currently focused on class-level unlearning; applicability to sample-level or concept unlearning is not yet explicit.
- Improvements: Incorporating sample difficulty weighting; exploring adaptive scheduling for $\lambda_1, \lambda_2$.

## Related Work & Insights
- **vs. Over-unlearning studies (Hu et al., 2024a)**: Prior work reported its qualitative existence; this work provides the first quantification ($OU@\varepsilon$).
- **vs. Re-learning defense (Lynch et al., 2024)**: Focused on LLMs using robust optimization like SAM; this work proposes feature dispersion for visual classification.
- **vs. Knowledge Distillation Unlearning**: Reuses distillation concepts but incorporates unlearning constraints and boundary neighborhood regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First to systematically expose two long-ignored but practically severe blind spots.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Multiple datasets (CIFAR-10/100, TinyImageNet, Face Recognition) + comparisons with over 8 base methods.
- Writing Quality: ⭐⭐⭐⭐  Problem statements are clear and method derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐  The plug-and-play nature of Spotter enhances existing unlearning methods, offering direct industrial value for GDPR compliance and privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Forgetting is Not Deletion: An Investigation of Reversibility in LLM Machine Unlearning](unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)
- [\[ICLR 2026\] ReTrace: Reinforcement Learning-Guided Reconstruction Attacks on Machine Unlearning](../../ICLR2026/ai_safety/retrace_reinforcement_learning-guided_reconstruction_attacks_on_machine_unlearni.md)
- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ICLR 2026\] Label Smoothing Improves Machine Unlearning](../../ICLR2026/ai_safety/label_smoothing_improves_machine_unlearning.md)
- [\[ICLR 2026\] Machine Unlearning under Retain–Forget Entanglement](../../ICLR2026/ai_safety/machine_unlearning_under_retainforget_entanglement.md)

</div>

<!-- RELATED:END -->
