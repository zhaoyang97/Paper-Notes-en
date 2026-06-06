---
title: >-
  [Paper Note] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks
description: >-
  [ICML 2026][AI Safety][Machine Unlearning] This paper reveals two critical blind spots in machine unlearning—over-unlearning (inadvertent damage to samples near the decision boundary) and prototype relearning attacks (re…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Machine Unlearning"
  - "Over-unlearning"
  - "Relearning Attack"
  - "Privacy"
  - "Classifier"
date: 2026-05-08
content_hash: cbf48b952fad1ae9
---

# Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks

**Conference**: ICML 2026  
**arXiv**: [2506.01318](https://arxiv.org/abs/2506.01318)  
**Code**: TBD  
**Area**: AI Security / Privacy Protection / Machine Unlearning  
**Keywords**: Machine Unlearning, Over-unlearning, Relearning Attack, Privacy, Classifier

## TL;DR
This paper reveals two critical blind spots in machine unlearning—over-unlearning (inadvertent damage to samples near the decision boundary) and prototype relearning attacks (restoration of forgotten knowledge using a few samples). It proposes the Spotter framework to simultaneously mitigate these issues through boundary masked distillation and intra-class dispersion loss.

## Background & Motivation

**Background**: Machine Unlearning (MU) aims to rapidly remove the influence of specific data from a model to avoid the high cost of full retraining. Existing methods include parameter resetting, decision boundary shifting, data partitioning, and knowledge distillation.

**Limitations of Prior Work**: Existing unlearning methods suffer from two severe but long-neglected issues—**over-unlearning** (where remaining samples close to the forgotten class suffer performance degradation when the class is removed) and **post-unlearning vulnerability** (where an adversary can quickly relearn deleted knowledge using only a few samples).

**Key Challenge**: How to achieve thorough unlearning while maintaining preservation integrity? Current methods typically focus only on unlearning quality (forget class accuracy $\rightarrow$ 0) or global retain accuracy, ignoring hidden damage in boundary regions and the threat of subsequent relearning attacks.

**Goal**: Target class-level unlearning—quantifying over-unlearning, exposing relearning risks, and designing defense solutions.

**Key Insight**: The focus is shifted from global retain accuracy to the **boundary neighborhood**, as remaining samples near the boundary are most vulnerable when decision boundaries move. Additionally, it is observed that features of the forgotten class remain highly clustered in the embedding space, providing an opportunity for prototype relearning.

**Core Idea**: Define boundary neighborhoods using reversible perturbations and design a remaining-data-independent over-unlearning metric $OU@\varepsilon$; combine boundary masked distillation and intra-class feature dispersion to build an unlearning framework resistant to both attacks.

## Method

### Overall Architecture
The Spotter framework for class-level unlearning consists of two main components—(1) **Over-unlearning Quantification and Mitigation**: Perturbed samples are generated along decision boundaries, distribution drift between the original and unlearned models in the boundary neighborhood is measured via KL divergence, and boundary predictions are corrected through masked distillation loss; (2) **Defense Against Relearning Attacks**: Intra-class dispersion loss is used to scatter features of the forgotten class as much as possible, destroying the prototype structure.

### Key Designs

1.  **Over-unlearning Metric $OU@\varepsilon$**:
    - **Function**: Quantifies the extent of inadvertent damage caused by class-level unlearning to remaining samples in the boundary neighborhood.
    - **Mechanism**: Defines a perturbation set $\mathcal{A}_{\varepsilon}(\mathcal{D}_f) = \{\boldsymbol{x} + \delta \mid \boldsymbol{x} \in \mathcal{D}_f, \delta \in \Delta_{\varepsilon}\}$. Using a masked softmax to zero out probabilities of forgotten classes, it calculates the KL divergence between the original and unlearned models over the perturbation set: $OU@\varepsilon := \mathbb{E}_{\boldsymbol{x}_p \sim \mathcal{A}_\varepsilon}[D(\tilde{\sigma}(\boldsymbol{z}(\boldsymbol{x}_p;\theta)) \| \sigma(\boldsymbol{z}(\boldsymbol{x}_p;\theta_u)))]$.
    - **Design Motivation**: Global retain accuracy fails to expose damage near the boundary; being independent of the remaining data makes it feasible in scenarios where the original training set is inaccessible.

2.  **Prototype Relearning Attack (PRA)**:
    - **Function**: Rapidly restores the performance of deleted classes using a small number of forgotten samples.
    - **Mechanism**: Observes that the feature extractor $\phi_{\theta_u}$ of the unlearned model still exhibits high clustering for the forgotten class. The attacker computes a class prototype from $k$ samples $\mathbf{p}^{(c)} = \frac{1}{k}\sum_{i=1}^k \phi_{\theta_u}(\boldsymbol{x}_i^{(c)})$ and uses this prototype as the classifier head weights—restoring near-original performance with only 1-10 samples.
    - **Design Motivation**: Reveals a deep vulnerability in current unlearning methods—while decision heads are removed, the structure of the feature space remains intact.

3.  **Combined Optimization Objective**:
    - **Function**: Simultaneously achieves thorough unlearning and relearning resistance through three loss terms (basic unlearning, boundary masked distillation, intra-class dispersion).
    - **Mechanism**: The loss function is $\mathcal{L} = \lambda_1 \mathcal{L}_u + (1-\lambda_1) \mathcal{L}_o + \lambda_2 \mathcal{L}_{sim}$, where $\mathcal{L}_u$ is the standard unlearning loss, $\mathcal{L}_o$ is the boundary masked distillation, and $\mathcal{L}_{sim}$ is the sum of intra-class cosine similarities (to drive feature dispersion).
    - **Design Motivation**: Each loss performs a specific role—the first two ensure unlearning and boundary protection, while the third actively defends against relearning.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 Forget Acc↓ | CIFAR-10 Retain Acc↑ | Over-unlearning↓ | Prototype Attack Acc↓ | CIFAR-100 Retain↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original Model | 100.00 | 100.00 | - | 100.00 | 99.99 |
| Retrain Baseline | 0.00 | 100.00 | 0.2384 | 58.70 | 99.78 |
| NegGrad | 0.18 | 87.73 | 0.3269 | 2.54 | 15.61 |
| Boundary Shrink | 3.82 | 93.79 | 0.1435 | 72.96 | 11.90 |
| UNSC | 0.00 | 99.98 | 0.1575 | 71.10 | 99.09 |
| **Spotter (λ₂=0.1)** | **0.00** | **100.00** | **0.0139** | **62.12** | **99.79** |
| **Spotter (λ₂=1)** | **0.00** | **99.98** | **0.0228** | **0.24** | **99.69** |

### Ablation Study

| Base Method | Before PRA | After Spotter | Gain |
| :--- | :--- | :--- | :--- |
| SalUn | 11.70% | 4.44% | ↓62% |
| DELETE | 31.72% | 3.34% | ↓89% |
| UNSC | 73.62% | 18.54% | ↓75% |

### Key Findings
- Spotter serves as a plug-and-play module that can be stacked onto any base unlearning method.
- When combined with SalUn, $OU@\varepsilon$ decreased from 0.1664 to 0.0345 (79% reduction).
- When combined with DELETE, over-unlearning decreased from 0.1216 to 0.0232.
- At $\lambda_2=1$, relearning is completely defeated, though over-unlearning increases slightly.

## Highlights & Insights
- **Quantification of Boundary Damage**: The first to propose the remaining-data-independent metric $OU@\varepsilon$.
- **Empirical Threat of PRA**: Only 1-10 images are needed to recover over 90% accuracy, posing a real security risk to identity-aware applications like face recognition.
- **Ingenious Dual Defense**: Masked distillation and intra-class dispersion loss exert pressure from different dimensions (decision space vs. feature space).
- **Versatility of Plug-and-play Framework**: Requires adding only two terms to the loss function; effectiveness is verified across heterogeneous methods such as DELETE, UNSC, and SalUn.

## Limitations & Future Work
- Parameter sensitivity of the boundary definition—the choice of $\varepsilon$ impacts the $OU@\varepsilon$ calculation.
- Sample size assumptions—PRA experiments are based on the premise that "the attacker possesses $k$ forgotten samples."
- Extension to other unlearning scenarios—focus is on class-level unlearning; applicability to sample-level or concept unlearning is not yet explicitly defined.
- Future work: Incorporating sample difficulty weighting; exploring adaptive $\lambda_1, \lambda_2$ scheduling strategies.

## Related Work & Insights
- **vs. Over-unlearning Research (Hu et al., 2024a)**: Prior work reported the existence qualitatively; this paper quantifies it ($OU@\varepsilon$) for the first time.
- **vs. Relearning Defense (Lynch et al., 2024)**: LLM scenarios use robust optimization like SAM; this paper proposes feature dispersion strategies for visual classification.
- **vs. Knowledge Distillation Unlearning**: Masked distillation reuses distillation concepts but adds unlearning constraints and boundary neighborhood regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first to systematically expose two long-neglected but practically severe blind spots.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparisons across multiple datasets (CIFAR-10/100, TinyImageNet, Face Recognition) and over 8 base methods.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, and method derivation is rigorous.
- Value: ⭐⭐⭐⭐⭐ Spotter’s plug-and-play nature can enhance existing unlearning methods, offering direct industrial value for GDPR compliance and privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](../../NeurIPS2025/ai_safety/efficient_verified_machine_unlearning_for_distillation.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](../../NeurIPS2025/ai_safety/rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](../../NeurIPS2025/ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](../../NeurIPS2025/ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)

</div>

<!-- RELATED:END -->
