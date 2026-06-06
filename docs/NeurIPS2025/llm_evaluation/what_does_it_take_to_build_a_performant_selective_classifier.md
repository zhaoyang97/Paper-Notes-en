---
title: >-
  [Paper Note] What Does It Take to Build a Performant Selective Classifier?
description: >-
  [NeurIPS 2025][LLM Evaluation][selective classification] This paper presents the first finite-sample decomposition of the selective classification gap, attributing it to five sources—Bayes noise, approximation error…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "selective classification"
  - "confidence calibration"
  - "oracle bound"
  - "error decomposition"
  - "uncertainty estimation"
date: 2026-05-08
content_hash: 49f9895c218c141f
---

# What Does It Take to Build a Performant Selective Classifier?

**Conference**: NeurIPS 2025
**arXiv**: [2510.20242](https://arxiv.org/abs/2510.20242)  
**Code**: Not available  
**Area**: Reliable Machine Learning / Selective Classification
**Keywords**: selective classification, confidence calibration, oracle bound, error decomposition, uncertainty estimation

## TL;DR

This paper presents the first finite-sample decomposition of the selective classification gap, attributing it to five sources—Bayes noise, approximation error, ranking error, statistical noise, and implementation bias—and demonstrates that monotone calibration methods have limited effect on closing this gap.

## Background & Motivation

In high-stakes domains such as healthcare, finance, and autonomous driving, models must be able to abstain under uncertainty—a paradigm known as selective classification. The core evaluation metric is the **accuracy–coverage trade-off**: how accuracy changes as the model accepts more inputs. In theory, a "perfect-ranking oracle" that orders all samples by their true correctness probability provides an upper bound.

**Limitations of Prior Work**:

- The classical realizable setting assumes noiseless data and a true predictor within the hypothesis class, which is overly idealistic.
- In the agnostic setting, the reference point itself may fall far below the oracle, and the sources of the gap are not distinguished.
- In practice, model capacity is limited, data is finite, and distribution shift exists; asymptotic guarantees offer little operational guidance.

**Core Problem**: For a finite model trained on finite data, which aspects of the learning setup truly determine the distance between the accuracy–coverage curve and the oracle upper bound?

**Key Insight**: The paper transforms this qualitative question into a quantitative diagnostic—defining a coverage-uniform selective classification gap $\Delta(c)$ and decomposing it into five measurable, improvable error terms.

## Method

### Overall Architecture

A selective classifier is a pair $(h, g)$, where $h$ is the classifier and $g$ is the selection function (outputting a confidence score). Given threshold $\tau$, the model predicts when $g(x,h) \geq \tau$ and abstains otherwise. The core metric is the selective classification gap:

$$\Delta(c) = \overline{\mathrm{acc}}(a_{\text{full}}, c) - \mathrm{acc}_c(h, g)$$

where $\overline{\mathrm{acc}}$ is the accuracy upper bound of the perfect-ranking oracle (Definition 3).

### Key Designs

1. **Finite-Sample Gap Decomposition (Theorem 1)**: With probability $1-\delta$:

$$\hat{\Delta}(c) \leq \underbrace{\varepsilon_{\text{Bayes}}(c)}_{\text{irreducible}} + \underbrace{\varepsilon_{\text{approx}}(c)}_{\text{capacity}} + \underbrace{\varepsilon_{\text{rank}}(c)}_{\text{ranking}} + \underbrace{\varepsilon_{\text{stat}}(c)}_{\text{statistical}} + \underbrace{\varepsilon_{\text{misc}}(c)}_{\text{optimization \& shift}}$$

   Definitions of each term:
   - $\varepsilon_{\text{Bayes}}(c) = \mathbb{E}[1-\max\{\eta(X), 1-\eta(X)\} \mid X \in A_c]$: inherent label uncertainty of data in the acceptance region.
   - $\varepsilon_{\text{approx}}(c) = \mathbb{E}[|\eta_h(X) - \eta(X)| \mid X \in A_c]$: the degree to which the model's hypothesis class fails to approximate the Bayes optimum.
   - $\varepsilon_{\text{rank}}(c) = \mathbb{E}[\eta_h \mid A_c^*] - \mathbb{E}[\eta_h \mid A_c]$: discrepancy between the confidence score ranking and the true correctness ranking.
   - $\varepsilon_{\text{stat}}(c) = C\sqrt{\log(1/\delta)/n}$: sampling fluctuations due to finite validation set size.
   - $\varepsilon_{\text{misc}}(c)$: optimization error and distribution shift.

2. **Limited Effectiveness of Monotone Calibration (Section 3.4)**: A key insight—monotone post-hoc calibration methods (e.g., isotonic regression, the monotone component of temperature scaling) preserve score rankings and thus leave $A_c$ unchanged, leaving $\Delta(c)$ unchanged. Although temperature scaling may produce **weak** non-monotone re-ranking effects through the nonlinearity of softmax, this effect is fundamentally limited. Truly reducing the gap requires methods that alter rankings:

    - Deep Ensembles: alter rankings through multi-model aggregation.
    - SAT: alter rankings through relabeling.
    - Feature-aware calibration heads: directly predict correctness using hidden-layer features.

3. **Characterization of Ranking Distance (Remark)**: The paper defines mis-ordered mass:

$$D_{\text{rank}}(c) = \Pr(X \in A_c^* \setminus A_c) + \Pr(X \in A_c \setminus A_c^*)$$

   This is the total probability mass of samples that need to be swapped between $A_c$ and $A_c^*$. When $D_{\text{rank}} = 0$, $\varepsilon_{\text{rank}} = 0$.

### Actionable Design Guidelines

- Reduce $\varepsilon_{\text{Bayes}}$: additional annotation, noise-robust loss functions.
- Reduce $\varepsilon_{\text{approx}}$: increase model capacity, distill from stronger models.
- Reduce $\varepsilon_{\text{rank}}$: Deep Ensembles, learned correctness prediction heads.
- Reduce $\varepsilon_{\text{stat}}$: enlarge the validation set.
- Reduce $\varepsilon_{\text{misc}}$: domain adaptation, importance weighting.

## Key Experimental Results

### Main Results: Calibration and Selective Classification on CIFAR-100

| Architecture | Method | E-AURC↓ | ECE↓ | Notes |
|---|---|---|---|---|
| CNN | MSP | 0.086 | 0.142 | Baseline |
| CNN | TEMP | 0.085 | **0.008** | ECE greatly improved; E-AURC nearly unchanged |
| CNN | SAT | 0.081 | 0.116 | Relabeling improves both |
| CNN | **DE** | **0.065** | 0.019 | Ensemble method closes gap most significantly |
| ResNet-18 | MSP | 0.033 | 0.052 | Greater capacity reduces approximation error |
| ResNet-18 | **DE** | **0.026** | 0.034 | Best |
| WRN-50 | MSP | 0.031 | 0.066 | |
| WRN-50 | **DE** | **0.026** | **0.030** | |

**Core Finding**: Temperature scaling reduces ECE from 0.142 to 0.008 (a 17× improvement), but E-AURC only drops from 0.086 to 0.085—**nearly ineffective**.

### Ablation Study: Isolating Error Sources

| Experimental Setting | Key Observation | Corresponding Error Term |
|---|---|---|
| Two moons noise σ=0.1→1.5 | Accuracy–coverage curve shifts down systematically | $\varepsilon_{\text{Bayes}}$ |
| Two moons: logistic regression→MLP | MLP significantly narrows the gap | $\varepsilon_{\text{approx}}$ |
| CIFAR-10N/100N noisy labels | Noisiest 50% of samples show the largest gap | $\varepsilon_{\text{Bayes}}$ |
| CNN→ResNet→WRN | Larger capacity yields smaller gap | $\varepsilon_{\text{approx}}$ |
| CIFAR-10C corruption severity 1→5 | Gap grows with shift severity | $\varepsilon_{\text{misc}}$ |
| Camelyon17-WILDS real-world shift | Gap increases substantially | $\varepsilon_{\text{misc}}$ |

### Key Findings

- Bayes noise and approximation error are the primary drivers of the gap (validated across both two moons and CIFAR benchmarks).
- Temperature scaling improves calibration but **does not improve ranking**, offering almost no benefit for selective classification.
- Only methods that **alter rankings** (SAT, DE) can substantively close the gap.
- Distribution shift introduces an independent slack term requiring dedicated robust training to address.

## Highlights & Insights

- **Perfect bridge between theory and practice**: the decomposition is not merely a theoretical tool—each error term directly corresponds to a measurable experiment and an actionable improvement direction.
- **"Calibration ≠ good selective classification"** is an important practical insight that challenges a common misconception.
- The **error budget** perspective enables practitioners to quantitatively diagnose bottlenecks and allocate improvement resources accordingly.
- Connections to multicalibration and loss prediction (Section 3.4) provide a self-evaluation mechanism.

## Limitations & Future Work

- Interactions among the five error terms exist (e.g., increasing capacity simultaneously affects approximation and ranking errors), making fully independent attribution difficult.
- Training-time calibration methods such as SAT, mixup, and focal loss simultaneously affect ranking and full-coverage accuracy, complicating budget separation.
- Validation is primarily conducted on synthetic and visual benchmarks; evaluation on large language models remains preliminary (Appendix F.2).
- The oracle bound and gap definition are based on 0-1 loss; extension to asymmetric or class-dependent cost functions requires additional work.
- A unified framework for out-of-distribution rejection and selective classification is not discussed.

## Related Work & Insights

- Relationship to the oracle bound of Geifman et al. (2019): the E-AURC in this paper is equivalent to their definition, but provides a finer-grained decomposition.
- Relationship to AUGRC (Traub et al., 2024): the latter avoids bias toward low-coverage regions through coverage weighting; the decomposition proposed here is complementary.
- The advantage of Deep Ensembles in selective classification can be explained as follows: averaging across multiple models provides a better posterior estimate of correctness, directly improving ranking.
- The equivalence of "loss prediction = multicalibration" (Gollakota et al., 2025) offers a new perspective on model self-assessment.

## Rating

- Novelty: ⭐⭐⭐⭐ The gap decomposition idea is not entirely new (analogous to bias–variance decomposition), but the five-term finite-sample decomposition and calibration analysis are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Each error term is individually validated across settings from two moons to CIFAR-10C/100N to Camelyon17.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical flow is clear, Figure 1 is intuitive, and the Takeaway boxes at the end of each section are highly useful.
- Value: ⭐⭐⭐⭐⭐ Provides actionable design guidelines with direct implications for any real-world deployment scenario requiring reliable predictions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[ICLR 2026\] LCA: Local Classifier Alignment for Continual Learning](../../ICLR2026/llm_evaluation/lca_local_classifier_alignment_for_continual_learning.md)
- [\[ACL 2026\] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](../../ACL2026/llm_evaluation/language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)

</div>

<!-- RELATED:END -->
