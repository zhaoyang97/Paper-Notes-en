---
title: >-
  [Paper Note] One Sample is Enough to Make Conformal Prediction Robust
description: >-
  [NeurIPS 2025][conformal prediction] This paper proposes RCP1 (Robust Conformal Prediction with One sample), which certifies the conformal procedure itself rather than individual conformity scores. Requiring only a singl…
tags:
  - "NeurIPS 2025"
  - "conformal prediction"
  - "robustness"
  - "randomized smoothing"
  - "prediction sets"
  - "conformal risk control"
date: 2026-05-08
content_hash: 1794a406d5e01e88
---

# One Sample is Enough to Make Conformal Prediction Robust

**Conference**: NeurIPS 2025
**arXiv**: [2506.16553](https://arxiv.org/abs/2506.16553)
**Code**: None
**Area**: Machine Learning / Uncertainty Quantification
**Keywords**: conformal prediction, robustness, randomized smoothing, prediction sets, conformal risk control

## TL;DR
This paper proposes RCP1 (Robust Conformal Prediction with One sample), which certifies the conformal procedure itself rather than individual conformity scores. Requiring only a single randomly perturbed forward pass at inference, RCP1 yields smaller robust prediction sets than state-of-the-art methods that require 100 forward passes.

## Background & Motivation
**Background**: Conformal Prediction (CP) provides prediction sets with tunable probabilistic coverage guarantees for arbitrary black-box models. Robust CP (RCP) extends these guarantees to worst-case perturbations within a predefined radius.

**Limitations of Prior Work**: Randomized smoothing–based RCP methods require multiple forward passes per input (e.g., 100) to estimate smoothed conformity scores, incurring prohibitively high computational cost.

**Key Challenge**: A fundamental tension exists between robustness and computational efficiency — deterministic methods (e.g., RSCP+) produce overly large prediction sets, while smoothing-based methods (e.g., RSCP/SmoothFull) yield smaller sets at substantial computational expense.

**Core Insight**: Even with a single randomly perturbed forward pass, the conformal prediction procedure itself already possesses a certain degree of robustness.

**Core Idea**: Certify the conformal **procedure** rather than individual conformity scores, shifting the computational burden of smoothing from inference time to the calibration phase.

## Method

### Overall Architecture
Given a black-box model $f$ and noise radius $\epsilon$:
1. **Calibration phase**: Compute the threshold $\hat{q}$ from conformity scores on calibration data.
2. **Inference phase**: Apply a single random perturbation $\delta \sim \mathcal{N}(0, \sigma^2 I)$ to input $x$ and compute $f(x+\delta)$.
3. **Certification**: Use a binary certificate to determine whether the perturbed score is suitable for robust prediction.

### Key Designs

1. **Procedure-Level Certification**

   - Conventional approach: robustly certify the smoothed conformity score $\bar{s}(x)$ for each sample.
   - Proposed approach: directly certify the coverage guarantee of the conformal procedure, i.e., $\Pr[Y \in C_\epsilon(X)] \geq 1-\alpha$.
   - Core inequality: exploits the distributional properties of the random perturbation $\delta$ to establish a probabilistic relationship between $s(x+\delta, y)$ and $s(x', y)$, where $x'$ denotes the adversarially perturbed sample.

2. **Binary Certificate**

   - For any binary certificate $\phi(x, \delta)$, if $\Pr_\delta[\phi=1] \geq p$, then $x$ is certified robust within radius $\epsilon$.
   - Concrete realization: derives the optimal binary certificate via the Neyman–Pearson lemma.
   - RCP1 requires only a single sample to check whether $\phi=1$; a conservative prediction set is returned upon failure.

3. **Extension to Robust Conformal Risk Control**

   - Generalizes the framework to the broader conformal risk control setting.
   - Applicable to both classification and regression tasks.

### Theoretical Guarantees
- **Theorem 1**: The coverage of RCP1 satisfies $\Pr[Y \in C_\epsilon^{RCP1}(X)] \geq 1-\alpha$ for any adversarial perturbation within radius $\epsilon$.
- **Theorem 2**: The expected prediction set size of RCP1 is no larger than that of smoothed RCP with $N$ samples; the two converge as $N \to \infty$.

## Key Experimental Results

### Main Results — CIFAR-10 Classification ($\epsilon = 0.25$, $\alpha = 0.1$)

| Method | Forward Passes | Avg. Set Size↓ | Coverage |
|--------|---------------|---------------|----------|
| RSCP+ (deterministic) | 1 | 4.82 | 0.912 |
| RSCP (N=100) | 100 | 2.31 | 0.903 |
| SmoothFull (N=100) | 100 | 2.15 | 0.907 |
| **RCP1 (Ours)** | **1** | **1.98** | **0.905** |

### CIFAR-100 Classification ($\epsilon = 0.25$, $\alpha = 0.1$)

| Method | Forward Passes | Avg. Set Size↓ | Coverage |
|--------|---------------|---------------|----------|
| RSCP+ | 1 | 28.7 | 0.914 |
| RSCP (N=100) | 100 | 14.2 | 0.906 |
| SmoothFull (N=100) | 100 | 12.8 | 0.909 |
| **RCP1** | **1** | **11.3** | **0.903** |

### Ablation Study — Effect of Noise Radius $\epsilon$ (CIFAR-10)

| $\epsilon$ | RSCP+ Set Size | SmoothFull Set Size | RCP1 Set Size |
|------------|---------------|---------------------|---------------|
| 0.125 | 2.41 | 1.52 | **1.38** |
| 0.25 | 4.82 | 2.15 | **1.98** |
| 0.5 | 7.93 | 3.87 | **3.52** |
| 1.0 | 9.85 | 6.14 | **5.71** |

### Key Findings
- RCP1 achieves smaller prediction sets than the 100-sample SOTA baseline using only a single forward pass.
- The deterministic method RSCP+ produces sets 2–3× larger than RCP1.
- Coverage guarantees are satisfied in all cases, consistent with theoretical requirements.
- The advantage of RCP1 over baselines becomes more pronounced as $\epsilon$ increases.
- The method is also effective on regression tasks.

## Highlights & Insights
- **Elegant conceptual shift**: Transitioning from "certifying each score" to "certifying the entire procedure" elegantly bypasses the computational bottleneck of smoothing.
- **100× speedup**: Reducing inference-time forward passes from 100 to 1 has significant practical deployment value.
- **Theory–experiment consistency**: Theoretical analysis of prediction set sizes aligns perfectly with empirical results.
- **Task-agnostic**: Applicable to both classification and regression settings.

## Limitations & Future Work
- Robustness guarantees are restricted to $\ell_2$-norm ball perturbations; other threat models ($\ell_\infty$, semantic perturbations) require separate analysis.
- Single-sample estimation introduces randomness, which may yield conservative sets in extreme cases.
- The choice of certificate affects performance; the optimal certificate requires knowledge of the noise distribution.

## Related Work & Insights
- **RSCP (Gendler et al. 2022)**: Pioneering work on smoothing-based robust conformal prediction.
- **RSCP+ (Yan et al. 2024)**: Deterministic robust conformal prediction.
- **Randomized Smoothing (Cohen et al. 2019)**: Foundational framework for adversarial robustness certification.
- Inspiration: The procedure-level certification paradigm may generalize to other statistical inference tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The procedure-level certification idea is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and tasks.
- Writing Quality: ⭐⭐⭐⭐ Motivation and theoretical exposition are clearly presented.
- Value: ⭐⭐⭐⭐⭐ The 100× speedup carries substantial significance for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](evolutionary_prediction_games.md)
- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)
- [\[NeurIPS 2025\] FlashMD: Long-Stride, Universal Prediction of Molecular Dynamics](flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
