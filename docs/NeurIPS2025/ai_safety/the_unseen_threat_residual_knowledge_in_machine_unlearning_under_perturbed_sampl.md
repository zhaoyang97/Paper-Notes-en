---
title: >-
  [Paper Note] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples
description: >-
  [NeurIPS 2025][AI Safety][machine unlearning] This paper identifies a critical security vulnerability in machine unlearning: even when an unlearned model is statistically indistinguishable from a retrained model…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "machine unlearning"
  - "residual knowledge"
  - "perturbed samples"
  - "adversarial robustness"
  - "RURK"
date: 2026-05-08
content_hash: a629aaecf7d91d91
---

# The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples

**Conference**: NeurIPS 2025
**arXiv**: [2601.22359](https://arxiv.org/abs/2601.22359)
**Code**: To be confirmed
**Area**: AI Safety / Machine Unlearning
**Keywords**: machine unlearning, residual knowledge, perturbed samples, adversarial robustness, RURK

## TL;DR
This paper identifies a critical security vulnerability in machine unlearning: even when an unlearned model is statistically indistinguishable from a retrained model, applying small adversarial perturbations to forgotten samples causes the unlearned model to correctly classify them while the retrained model fails — revealing a novel privacy risk termed "residual knowledge." The authors propose RURK, a fine-tuning strategy that penalizes correct predictions on perturbed forgotten samples, effectively suppressing residual knowledge across 11 unlearning methods on CIFAR-10 and ImageNet-100.

## Background & Motivation

**Background**: Machine unlearning aims to efficiently remove the influence of specific training data from a trained model, serving as an alternative to full retraining. Existing methods (CR/Fisher/NTK/GA/SCRUB/SSD, etc.) are standardly validated by demonstrating $(\epsilon, \delta)$-indistinguishability between the unlearned model and a retrained reference.

**Limitations of Prior Work**: $(\epsilon, \delta)$-indistinguishability only guarantees behavioral consistency on **original samples**, not within their local neighborhoods. Consequently, an unlearned model and a retrained model may produce identical predictions on a given forgotten sample yet diverge on its small perturbations.

**Key Challenge**: A particularly dangerous scenario arises when small perturbations of a forgotten sample are still correctly classified by the unlearned model but not by the retrained model (which has never seen that data). This indicates that the unlearned model retains informational traces near the decision boundary of forgotten samples.

**Goal**: (a) Formally define "residual knowledge" as a novel privacy risk; (b) theoretically prove that such divergence is inevitable in high-dimensional spaces; (c) propose a mitigation strategy.

**Key Insight**: Combining adversarial robustness with machine unlearning — using adversarial examples as probes to detect whether unlearning is thorough.

**Core Idea**: The security criterion for unlearning should not only assess statistical indistinguishability on original samples, but also indistinguishability within their local neighborhoods. Residual knowledge is eliminated by penalizing the unlearned model's correct predictions on perturbed forgotten samples.

## Method

### Overall Architecture
The paper proceeds in three steps: (1) **Problem exposition** — theoretical and empirical demonstration that existing unlearning methods exhibit residual knowledge under perturbed samples; (2) **Formal measurement** — definition of the residual knowledge ratio $r_\tau((x,y))$; (3) **Mitigation** — the RURK fine-tuning strategy to eliminate residual knowledge.

### Key Designs

1. **Residual Knowledge Ratio**:

    - **Function**: Quantifies the degree to which an unlearned model retains information about forgotten samples within their local neighborhoods, relative to a retrained model.
    - **Mechanism**: $r_\tau((x,y)) = \frac{\Pr[m(x')=y]}{\Pr[a(x')=y]}$, where $m$ is the unlearned model, $a$ is the retrained model, and $x' \sim \mathcal{B}_p(x, \tau)$ is sampled from the neighborhood. $r_\tau > 1$ indicates that the unlearned model correctly identifies perturbed samples more often than the retrained model, signaling residual knowledge.
    - **Design Motivation**: Directly computing the adversarial disagreement metric is expensive (requiring enumeration over all output combinations). The residual knowledge ratio is more tractable and provides upper and lower bounds on disagreement.

2. **Theoretical Guarantee (Proposition 2: Inevitable Divergence)**:

    - **Function**: Proves that adversarially perturbed samples inevitably induce divergence even when $(\epsilon, \delta)$-unlearning is satisfied.
    - **Mechanism**: On the unit sphere $\mathbb{S}^{d-1}$, the isoperimetric inequality is used to show that even if models agree on original samples, small perturbations can cause divergence with probability that increases with dimensionality $d$ and perturbation radius $\tau$.
    - **Design Motivation**: Establishes a theoretical basis for the universality of residual knowledge — this is not a defect of any particular method but an inherent property of high-dimensional spaces.

3. **RURK Fine-tuning Strategy**:

    - **Function**: Fine-tunes an already-unlearned model to maintain retain-set performance while eliminating residual knowledge.
    - **Mechanism**: The loss function is defined as
      $$L_{RURK} = \underbrace{\frac{1}{|S_r|}\sum_{(x,y) \in S_r} \ell(w,(x,y))}_{\text{retain set}} - \lambda \underbrace{\frac{1}{|S_f|}\sum_{(x,y) \in S_f} \kappa(w,(x,y))}_{\text{residual knowledge penalty}}$$
      where $\kappa$ computes the loss over the set of "vulnerable perturbations" of forgotten samples — neighborhood samples that are still correctly classified. A PGD-style procedure is used to identify these vulnerable perturbations, after which the model is penalized for correctly predicting them.
    - **Design Motivation**: Directly minimizes the numerator of the residual knowledge ratio, $\Pr[m(x')=y]$, without requiring access to the retrained model. Term (ii) effectively performs "neighborhood-level unlearning" around the forgotten samples.

### Theoretical Contributions
- **Proposition 1**: Indistinguishability degrades under adversarial examples — an $(\epsilon, \delta)$ guarantee weakens to $(2\epsilon, 2\delta/(1-e^{-\epsilon}))$.
- **Proposition 2**: In high-dimensional spaces, models satisfying $(\epsilon, \delta)$-unlearning inevitably exhibit divergence on perturbed samples.
- **Lemma A.4**: The residual knowledge ratio provides upper and lower bounds on adversarial disagreement.

## Key Experimental Results

### Main Results (Table 1, CIFAR-10, ResNet-18)

| Method | Retain Acc | Unlearn Acc | Test Acc | MIA Acc | Avg Gap↓ | Re-learn Time |
|--------|-----------|-------------|---------|---------|----------|--------------|
| Re-train | 100.0 | 9.47 | 93.30 | 22.50 | 0.00 | 17.33 |
| GD | 99.98 | 0.00 | 94.29 | 0.10 | 8.22 | 0.20 |
| NegGrad+ | 99.28 | 14.00 | 92.02 | 18.18 | **2.71** | 1.00 |
| SCRUB | 99.61 | 12.45 | 92.70 | 7.10 | 4.84 | >30 |
| GA | 95.41 | 61.37 | 85.98 | 0.00 | 21.25 | 1.00 |
| **RURK** | 99.55 | 14.63 | 92.60 | 18.20 | **2.65** | **>30** |

### Ablation Study (Residual Knowledge Analysis)

| Method | $r_\tau$ at $\tau=0.03$ | Residual Knowledge Status |
|--------|------------------------|--------------------------|
| Original | >>1 | Severe residual knowledge |
| GD | >>1 | Severe (equivalent to Original) |
| CF-k | >>1 | Severe residual knowledge |
| NTK | >1 | Residual present (linearization ignores higher-order terms) |
| NegGrad+ | Slightly >1 | Minor residual knowledge |
| GA/SSD | <1 | Excessive forgetting |
| **RURK** | ≈1 | Effectively suppressed |

### Key Findings
- **Residual knowledge is a universal problem**: On CIFAR-10 at $\epsilon \approx 0.03$, over 7% of forgotten samples exhibit residual knowledge. Among 11 existing methods, all except the over-forgetting GA/SSD are affected.
- **GD and CF-k are nearly equivalent to the original model**: Fine-tuning only on the retain set or updating only the final few layers has negligible effect on the neighborhood of forgotten samples, demonstrating that evaluations based solely on original sample behavior are wholly insufficient.
- **NTK achieves small Avg Gap but high residual knowledge**: Because NTK linearization ignores higher-order terms, the decision boundary in the local neighborhood closely resembles that of the original model, validating that standard evaluation metrics cannot detect residual knowledge.
- **RURK achieves the smallest Avg Gap while effectively suppressing residual knowledge**: At $\tau < 0.01$, $r_\tau \approx 1$; at larger $\tau$, residual knowledge is suppressed to below 1. Re-learn Time >30 further confirms thorough unlearning.
- **A fundamental indistinguishability–robustness trade-off exists**: Increasing $\tau$ makes the adversarial loss in RURK less smooth, causing $\epsilon$ to grow — better suppression of residual knowledge may come at the cost of statistical indistinguishability.
- **Effectiveness on ImageNet-100**: RURK achieves the smallest Avg Gap with effective residual knowledge suppression on ResNet-50, demonstrating scalability.

## Highlights & Insights
- **Important security finding**: The paper reveals a critical blind spot — "statistical indistinguishability ≠ secure unlearning." Even models that pass all standard evaluations (MIA, Unlearn Accuracy, Re-learn Time) may still leak information within local neighborhoods. This has direct implications for GDPR compliance.
- **Cross-disciplinary connection between adversarial robustness and unlearning security**: This work is the first to systematically bridge these two fields — adversarial examples serve as probes for detecting unlearning completeness, a genuinely novel perspective.
- **Simplicity of RURK**: The method requires only an additional regularization term that extends unlearning to perturbed samples, with time complexity comparable to GD/NGD.

## Limitations & Future Work
- The theoretical analysis assumes samples lie on the unit sphere; while this has been extended to the unit hypercube, a gap remains regarding applicability to real image data.
- Searching for vulnerable perturbations requires PGD-style iteration, increasing computational overhead.
- Validation is limited to visual classification tasks (CIFAR-5/10/ImageNet-100); applicability to LLM unlearning and generative model unlearning remains unexplored.
- A fundamental trade-off exists: perfectly eliminating residual knowledge for all $\tau$ is equivalent to achieving perfect adversarial robustness — which is computationally intractable.
- The practical threat posed by residual knowledge requires more careful assessment — an adversary must know which samples were forgotten in order to construct perturbations.

## Related Work & Insights
- **vs. standard unlearning evaluation**: Standard methods assess model behavior only on original samples (MIA, $(\epsilon,\delta)$-indistinguishability); this paper reveals residual information within local neighborhoods.
- **vs. Zhao et al. (2024)**: That work studies how malicious unlearning requests can degrade a model's adversarial robustness; this paper takes the opposite direction — using adversarial examples to detect unlearning completeness.
- **vs. differential privacy**: $(\epsilon,\delta)$-indistinguishability is definitionally equivalent to DP, but DP is typically applied during training rather than after unlearning. Residual knowledge exposes the limitations of DP-style guarantees within local neighborhoods.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First paper to reveal "residual knowledge under perturbed samples" as a novel privacy risk, with solid theoretical and empirical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 11 unlearning methods × 3 datasets × multiple architectures, with comprehensive residual knowledge analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem definition is precise, theoretical derivations are clear, and Figure 1 intuitively illustrates the issue.
- **Value**: ⭐⭐⭐⭐⭐ Likely to reshape security standards in machine unlearning and has direct implications for GDPR compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)

</div>

<!-- RELATED:END -->
