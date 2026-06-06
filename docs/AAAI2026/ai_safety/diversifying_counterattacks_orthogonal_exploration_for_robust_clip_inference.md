---
title: >-
  [Paper Note] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference
description: >-
  [AAAI 2026][AI Safety][Adversarial Robustness] This paper proposes Directional Orthogonal Counterattack (DOC), a method that expands the search space during counterattack optimization by introducing orthogonal gradient c…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Adversarial Robustness"
  - "CLIP Defense"
  - "Test-Time Defense"
  - "Orthogonal Counterattack"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: df3e87d75acfab7e
---

# Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference

**Conference**: AAAI 2026
**arXiv**: [2511.09064](https://arxiv.org/abs/2511.09064)  
**Code**: [Available](https://github.com/bookman233/DOC)  
**Area**: AI Security
**Keywords**: Adversarial Robustness, CLIP Defense, Test-Time Defense, Orthogonal Counterattack, Vision-Language Models

## TL;DR

This paper proposes Directional Orthogonal Counterattack (DOC), a method that expands the search space during counterattack optimization by introducing orthogonal gradient components and momentum updates, and adaptively modulates counterattack intensity via a cosine-similarity-based Directional Sensitivity Score (DSS). DOC significantly improves the test-time adversarial robustness of CLIP across 16 datasets.

## Background & Motivation

Vision-language pre-trained models such as CLIP exhibit strong zero-shot generalization but are highly vulnerable to adversarial examples. Existing defenses fall into three categories:

**Adversarial Fine-tuning** (e.g., TeCoA, PMG-AFT, FARE): Fine-tunes CLIP on adversarial examples, but incurs high computational cost and may degrade generalization.

**Adversarial Prompt Tuning**: Adjusts prompts in the embedding space, but sacrifices semantic interpretability.

**Test-Time Counterattack (TTC)**: A recent parameter-free defense that generates counterattack perturbations to maximize the embedding distance between adversarial inputs and their variants.

**Core issue with TTC**: A **fundamental objective mismatch** exists between adversarial attacks and counterattacks:

- Adversarial attack objective: maximize classification loss
- Counterattack objective: maximize embedding distance

TTC uses PGD to generate counterattacks along the gradient direction, but due to this mismatch, the search space is confined to a narrow region, causing the counterattack to overfit to a limited set of adversarial patterns and lack the diversity needed to neutralize a broad distribution of perturbations.

## Method

### Overall Architecture

DOC (Directional Orthogonal Counterattack) comprises two core components:

1. **Orthogonal Gradient Augmentation (OGA)**: Adds a random component orthogonal to the primary gradient direction at each counterattack optimization step, combined with momentum updates.
2. **Directional Sensitivity Score (DSS)**: Assesses whether an input is adversarial based on cosine similarity, and adaptively modulates counterattack intensity.

### Key Designs

**Orthogonal Gradient Augmentation (OGA)**:

1. Compute the normalized gradient $g$ (gradient of the counterattack loss w.r.t. the adversarial input, then normalized).
2. Sample a random vector $r$ from the standard normal distribution; apply Gram-Schmidt orthogonalization to obtain a component orthogonal to the gradient: $r_\perp = (r - \langle r, g \rangle g) / \|r - \langle r, g \rangle g\|$.
3. Combine the update direction: $d = g + \lambda \cdot r_\perp$ (where $\lambda$ controls orthogonal injection strength).
4. Momentum update: $m_t = \mu \cdot m_{t-1} + (1 - \mu) \cdot d$.
5. Counterattack perturbation iteration: $\delta_{t+1} = \mathrm{Proj}(\delta_t + \alpha \cdot \mathrm{sign}(m_t))$.

Design intuition: The orthogonal component enables the counterattack to explore regions beyond the gradient direction, while momentum helps escape narrow local optima, yielding more diverse counterattack perturbations. t-SNE visualizations confirm that DOC produces a more dispersed counterattack distribution than TTC.

**Directional Sensitivity Score (DSS)**:

TTC uses $\ell_2$ distance to detect adversarial inputs, which suffers from two issues: (a) embeddings with similar directions but different scales produce spuriously large $\ell_2$ distances; (b) single noise samples introduce instability.

DOC replaces this with cosine similarity averaged over multiple samples:

$$\hat{\tau}(x) = 1 - \frac{1}{M} \sum \cos\!\left(I_\theta(x_m),\, I_\theta(x)\right)$$

- Low $\hat{\tau}$: perturbed embeddings maintain consistent directions, indicating a clean sample.
- High $\hat{\tau}$: directional inconsistency, indicating a likely adversarial sample.

A soft gating function adaptively modulates counterattack intensity:

$$w = \mathrm{sigmoid}\!\left(\gamma \cdot (\tau - \hat{\tau}(x))\right)$$

$$\delta_{ca} = w \cdot \delta_{ca} + (1 - w) \cdot \delta_{ca}^0$$

For clean samples, $w \approx 0$ (counterattack is nearly suppressed); for adversarial samples, $w \approx 1$ (full counterattack is applied).

### Loss & Training

DOC is a **training-free** test-time defense:

- No model parameters are modified; no training data or label supervision is required.
- Counterattack budget: $\epsilon_{ca} = 4/255$.
- Default: 4 counterattack steps, step size $\alpha = 3/255$.
- Batch size 256; requires only a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results

**Average results across 16 datasets under PGD-10 attack** ($\epsilon_{atk} = 4/255$):

| Method | Type | Avg. Robust Acc. | Avg. Clean Acc. |
|---|---|---|---|
| CLIP (original) | — | 0.06% | 61.51% |
| HD | Test-time defense | 0.56% | 54.85% |
| TeCoA4 | Adversarial fine-tuning | 10.95% | 37.58% |
| FARE4 | Adversarial fine-tuning | 1.38% | 56.62% |
| TTC | Test-time defense | 21.22% | 55.63% |
| **DOC** | **Test-time defense** | **31.02%** | **58.26%** |

DOC improves robust accuracy over TTC by **9.80%**, while also achieving higher clean accuracy (+2.63%).

**Per-dataset key results** (robust accuracy under PGD-10):

| Dataset | CLIP | TTC | DOC | Gain |
|---|---|---|---|---|
| CIFAR-10 | 0.00% | 30.25% | 38.14% | +7.89% |
| STL-10 | 0.04% | 51.89% | 69.16% | +17.27% |
| ImageNet | 0.00% | 13.07% | 24.64% | +11.57% |
| OxfordPets | 0.00% | 25.89% | 46.52% | +20.63% |
| Caltech-256 | 0.13% | 26.38% | 43.08% | +16.70% |

### Ablation Study

| DSS | OGA | Clean Acc. | PGD Robust | CW Robust | AutoAttack |
|---|---|---|---|---|---|
| ✗ | ✗ | 55.66% | 21.43% | 20.70% | 21.97% |
| ✓ | ✗ | 58.23% | 23.37% | 22.27% | 22.66% |
| ✗ | ✓ | 55.38% | 31.83% | 29.02% | 26.07% |
| ✓ | ✓ | **58.27%** | **31.04%** | **28.15%** | **25.89%** |

- **DSS alone**: Primarily improves clean accuracy (+2.57%) by suppressing unnecessary perturbations on clean samples.
- **OGA alone**: Substantially boosts robust accuracy (+10.4%), validating the effectiveness of diversified counterattacks.
- **Combined**: Achieves favorable trade-offs between robustness and clean accuracy.

Average robust accuracy under CW attack: DOC 28.18% vs. TTC 20.61% (+7.58%). Under AutoAttack, DOC outperforms TTC by approximately 4.1%.

### Key Findings

- DOC outperforms TTC on nearly all 16 datasets, with EuroSAT as the only exception.
- DOC functions as a **plug-and-play module** compatible with adversarial fine-tuning: combining with FARE yields average robust accuracy exceeding vanilla CLIP by 18%.
- Counterattack performance saturates at as few as $N = 3$–$4$ steps, incurring minimal computational overhead.
- Clean accuracy remains stable as the number of steps increases; robustness gains do not come at the expense of clean performance.

## Highlights & Insights

1. **Precise problem identification**: Reveals the fundamental objective mismatch between adversarial attacks and counterattacks.
2. **Clear design intuition for OGA**: Introducing exploration noise via orthogonalization is both mathematically elegant and practically effective.
3. **Cosine similarity replaces $\ell_2$ distance** for adversarial sample detection, which is more principled in high-dimensional spaces due to scale invariance.
4. **Completely training-free**: Requires no data, no parameter modification, and runs on a single GPU, resulting in an extremely low deployment barrier.
5. t-SNE visualizations intuitively demonstrate DOC's ability to push adversarial samples toward the clean distribution.

## Limitations & Future Work

1. The counterattack budget is set equal to the attack budget; in practice, the attack budget is unknown.
2. The orthogonal component is randomly sampled, potentially causing inference results to vary across runs (though empirical variance is small).
3. Clean accuracy decreases on ImageNet (−3.25%) and fluctuates on some fine-grained classification datasets.
4. Validation is limited to CLIP; the approach has not been extended to other VLPs (e.g., BLIP-2, LLaVA).
5. Robustness against adaptive attacks is not sufficiently discussed.

## Related Work & Insights

- **TTC** (Xing et al. 2025): The pioneering test-time counterattack work that DOC directly improves upon.
- **TeCoA** (Mao et al.): A representative adversarial fine-tuning method.
- **PMG-AFT** (Wang et al. 2024): Adversarial fine-tuning augmented with CLIP-guided regularization.
- **FARE** (Schlarmann et al. 2024): Adversarial fine-tuning under large perturbation budgets.
- **Hedge Defense** (Wu et al. 2021): A test-time defense that maximizes loss across all classes.
- Insight: **In unsupervised test-time defense, diversity matters more than precision.** The orthogonal exploration paradigm is generalizable to other robust optimization scenarios.

## Rating

- Novelty: 4/5 — OGA and DSS represent meaningful and original contributions.
- Technical Depth: 4/5 — The method is grounded in clear theoretical motivation and mathematical derivation.
- Experimental Thoroughness: 5/5 — 16 datasets × 3 attack types × ablations × combination experiments + visualizations.
- Writing Quality: 4/5 — Problem motivation is clearly articulated with rich figures and tables.
- Overall: 4.0/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[AAAI 2026\] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute](secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)

</div>

<!-- RELATED:END -->
