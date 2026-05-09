---
title: >-
  [Paper Note] Bounds on Agreement between Subjective and Objective Measurements
description: >-
  [CVPR2026][Subjective quality assessment] This paper derives closed-form expressions for the upper bound on PCC and the lower bound on MSE between subjective MOS values and any objective quality estimator, and proposes BinoVotes — a binomial distribution-based voting model — to estimate these bounds when per-vote variance information is unavailable.
tags:
  - CVPR2026
  - Subjective quality assessment
  - MOS
  - Pearson correlation coefficient
  - mean squared error
  - binomial voting model
  - upper bound on objective estimator performance
date: 2026-05-08
content_hash: a43e87213444d5b3
---

# Bounds on Agreement between Subjective and Objective Measurements

**Conference**: CVPR2026
**arXiv**: [2603.13204](https://arxiv.org/abs/2603.13204)
**Code**: [NTIA/its-mos-agreement](https://github.com/NTIA/its-mos-agreement)
**Area**: Others (Multimedia Quality Assessment / Subjective Test Theory)
**Keywords**: Subjective quality assessment, MOS, Pearson correlation coefficient, mean squared error, binomial voting model, upper bound on objective estimator performance

## TL;DR

This paper derives closed-form expressions for the upper bound on PCC and the lower bound on MSE between subjective MOS values and any objective quality estimator, and proposes BinoVotes — a binomial distribution-based voting model — to estimate these bounds when per-vote variance information is unavailable.

## Background & Motivation

1. **Subjective tests are the gold standard but are noisy**: Multimedia quality assessment relies on subjective tests to obtain MOS scores, yet discrete rating scales, limited numbers of raters, and individual biases mean that MOS values do not equal ground-truth quality. Targeting PCC = 1.0 or MSE = 0.0 is neither realistic nor reproducible.
2. **Objective estimator evaluation lacks a principled baseline**: Existing practice directly compares PCC/MSE against MOS while ignoring the noise floor inherent in MOS itself — even a perfect oracle estimator cannot perfectly agree with noisy MOS scores.
3. **Prior methods introduce additional metrics**: Previous work proposed specialized measures such as classification error rate, resolving power, and ε-insensitive RMSE to address MOS noise, but these deviate from the PCC/MSE framework most familiar to researchers.
4. **Continuous Gaussian approximation of MOS is problematic**: Approximating MOS with a Gaussian distribution violates its discrete and bounded nature; clipping operations introduce additional bias and lack rigorous mathematical grounding.
5. **Per-vote variance is often unavailable**: Large-scale crowdsourced subjective tests (e.g., commonly used speech quality datasets) typically release only MOS values without individual vote variance, making data-driven bound computation infeasible.
6. **Dataset-specific performance targets are needed**: Researchers need to know the maximum achievable PCC and minimum achievable MSE on a given test set in order to assess whether an objective estimator still has room for improvement.

## Method

### Overall Architecture

The core idea is that the optimal objective estimator is an oracle possessing the ground-truth quality $Y$. Therefore, the PCC/MSE between $Y$ and the MOS $X$ constitutes the upper bound on PCC and the lower bound on MSE for any objective estimator compared against MOS.

**Sole assumption (well-behaved)**: $\mathbb{E}(R_j|Y)=Y$, i.e., given true quality, the expected rating equals true quality. This is the fundamental axiom of all subjective testing.

### MSE Lower Bound Derivation

Applying the law of total expectation and conditional variance decomposition:

$$\mathbb{E}(D^2) = \frac{\mathbb{E}(v_r(Y))}{n_v}$$

where $v_r(Y) = \text{Var}(R_j|Y)$ is the conditional vote variance function and $n_v$ is the number of votes per stimulus. The MSE lower bound depends solely on the expected vote variance and the number of raters.

### PCC Upper Bound Derivation

$$\rho(X,Y) = \sqrt{\frac{\text{Var}(X) - \mathbb{E}(D^2)}{\text{Var}(X)}}$$

The PCC upper bound depends on the variance of the MOS distribution and the MSE lower bound; as MSE → 0, PCC → 1.

### BinoVotes Voting Model

Individual votes are modeled with a binomial distribution: $R_j = \frac{s_H-s_L}{n_s-1}B_j + s_L$, where $B_j \sim \text{Binomial}(n_s-1, \frac{Y-s_L}{s_H-s_L})$.

**Key properties**:
- Satisfies the well-behaved condition $\mathbb{E}(R_j|Y)=Y$
- Vote variance follows a parabolic function of true quality: $v_r(Y)=\frac{(Y-s_L)(s_H-Y)}{n_s-1}$, which is zero at both scale endpoints and maximal in the middle — closely matching empirical data
- Naturally respects the discrete rating scale and bounded range

### BinoMOS and Bound Estimation Without Variance Information

BinoMOS is obtained by averaging BinoVotes; closed-form expressions exist for its variance and PMF. When a dataset does not provide per-vote variance, the BinoVotes predicted variance can be estimated from the sample mean and variance of MOS:

$$\hat{\sigma}_{BV}^2 = \frac{n_v}{n_m-1}\left((\hat{\mu}_X-s_L)(s_H-\hat{\mu}_X)-\hat{\sigma}_X^2\right)$$

Alternatively, the globally averaged observed variance across 18 tests, $\hat{\sigma}_{GV}^2=0.64$, may be used as a substitute.

## Key Experimental Results

### Dataset Scale

Validation is performed on 22 subjective tests (18 with variance information, 4 without), covering speech (14), image (2), and video (2) domains, comprising **86,450 stimuli and over 493,000 votes**. The number of votes per stimulus $n_v$ ranges from 3.52 to 28.33.

### Main Results

| Comparison Dimension | BinoVotes Bound vs. Data-Driven Bound | Global Average Bound vs. Data-Driven Bound |
|---|---|---|
| Max RMSE difference | 0.05 (ITS1997) | 0.09 (TMHINT-QI) |
| Mean RMSE difference | +0.02 | −0.004 |
| Max PCC difference | 0.021 (ITS1997) | 0.05 (TMHINT-QI) |
| Mean PCC difference | −0.006 | +0.005 |

- RMSE bound range: 0.12–0.51; PCC bound range: 0.86–0.99
- BinoVotes tends to slightly overestimate vote variance (in 17 of 18 tests), yielding conservatively higher RMSE bounds and lower PCC bounds
- Mean difference between BinoVotes predicted variance and observed variance is 0.13, with a maximum of 0.28

### Ablation & Analysis

- **Effect of vote count**: Increasing $n_v$ decreases the RMSE bound and increases the PCC bound (Figure 3), consistent with the central limit theorem
- **Effect of quality distribution**: Different ground-truth quality distributions (uniform/triangular/Beta) produce notable differences in PCC bounds but negligible differences in RMSE bounds
- **Sample size convergence**: For $n_f \geq 50$, the difference between sample PCC and population PCC bounds is negligible (Figure 4)
- **Comparison of two substitutes when variance is unavailable**: BinoVotes is more conservative but stable; the global average variance is closer but relies on external data

## Highlights & Insights

1. **Pure mathematical derivation requiring no assumption on MOS distribution**: Relying solely on the well-behaved axiom, the derived PCC/MSE bounds apply to any voting process
2. **BinoVotes is concise and effective**: The single-parameter binomial distribution naturally respects discreteness, boundedness, and the parabolic variance shape, closely matching the median variance across 28 experimental datasets
3. **Highly practical**: Researchers can directly apply the derived formulas to compute principled performance ceilings for specific datasets and assess whether a model has reached its limit
4. **Open-source implementation**: A complete implementation is available on GitHub

## Limitations & Future Work

1. Bounds for the Spearman rank correlation coefficient (SRCC), which is equally common in quality assessment, are not derived
2. Coverage of image and video test sets is limited (only 2 each), with speech tests dominating
3. BinoVotes tends to overestimate variance (it does not capture rater behavior such as avoiding extreme scores); incorporating individual bias terms would further widen the gap
4. A fixed number of votes per stimulus $n_v$ is assumed, whereas crowdsourced tests often have unequal vote counts
5. Detailed numerical results focus on the 1–5 MOS scale; other rating scales (7-point, continuous sliders, etc.) are not discussed

## Related Work & Insights

| Method | Characteristics | Advantage of This Work |
|---|---|---|
| ε-insensitive RMSE [13] | New metric tolerating MOS noise | Retains standard PCC/MSE and directly provides bounds |
| Classification error rate [41,24,33] | Treats estimator as equivalent to $k$ raters | More precise bounds on continuous values |
| Resolving power [3] | Measures discriminative ability | Directly linked to commonly used metrics |
| Gaussian MOS approximation [6] | Continuous model | BinoVotes is naturally discrete and bounded |
| SOS variance parabola [9] | Empirical fitting | BinoVotes derives the same parabolic shape mathematically |

## Rating

- Novelty: ⭐⭐⭐⭐ — First derivation of PCC/MSE bounds from the intrinsic mathematical properties of voting; BinoVotes model is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ — 22 tests and 86K stimuli, though coverage of image/video domains is limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear, with coherent logical flow from basic assumptions to closed-form solutions
- Value: ⭐⭐⭐⭐ — Provides quality assessment researchers with a practical tool for determining performance ceilings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[CVPR 2026\] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[CVPR 2026\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)

</div>

<!-- RELATED:END -->
