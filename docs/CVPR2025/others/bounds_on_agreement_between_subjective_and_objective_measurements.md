---
title: >-
  [Paper Note] Bounds on Agreement between Subjective and Objective Measurements
description: >-
  [CVPR 2025][Subjective tests] By assuming only that the voting mean converges to the true quality, mathematical bounds on PCC (upper bound) and MSE (lower bound) between subjective tests (MOS) and objective estimators are derived. A Binomial-based voting model, BinoVotes, is proposed to enable the calculation of these bounds even when voting variance is unavailable. Validation on 18 subjective test datasets demonstrates that BinoVotes bounds align closely with full-data-drive…
tags:
  - "CVPR 2025"
  - "Subjective tests"
  - "Objective estimators"
  - "MOS"
  - "Pearson correlation coefficient"
  - "Mean squared error"
  - "Binomial voting model"
date: 2026-05-08
content_hash: e71ec88c18e91679
---

# Bounds on Agreement between Subjective and Objective Measurements

**Conference**: CVPR 2025  
**arXiv**: [2603.13204](https://arxiv.org/abs/2603.13204)  
**Code**: TBD  
**Area**: Others / Quality Assessment / Statistical Theory  
**Keywords**: Subjective tests, Objective estimators, MOS, Pearson correlation coefficient, Mean squared error, Binomial voting model

## TL;DR

By assuming only that the voting mean converges to the true quality, mathematical bounds on PCC (upper bound) and MSE (lower bound) between subjective tests (MOS) and objective estimators are derived. A Binomial-based voting model, BinoVotes, is proposed to enable the calculation of these bounds even when voting variance is unavailable. Validation on 18 subjective test datasets demonstrates that BinoVotes bounds align closely with full-data-driven bounds.

## Background & Motivation

### Background
- The **gold standard** for multimedia signal (audio, video) quality assessment is subjective testing: asking subjects to rate quality on a 1-5 scale and calculating the Mean Opinion Score (MOS).
- The performance of objective estimators is typically evaluated using the **Pearson correlation coefficient (PCC)** and **Mean Squared Error (MSE)** against MOS.
- Subjective tests are expensive and time-consuming, continuously driving the development of objective estimators that can approximate MOS.

### Limitations of Prior Work
- MOS itself is a **noisy measurement**: noise originates from discrete rating scales, a limited number of subjects, subject bias, etc.
- Pursuing PCC = 1.0 or MSE = 0.0 is both unrealistic and irreproducible.
- Existing methods either propose completely new evaluation frameworks and metrics (increasing complexity) or model MOS uncertainty but potentially violate the discrete and bounded nature of MOS.

### Key Challenge
A method is needed to determine whether an objective estimator has reached the optimal performance allowed by subjective test data—i.e., where the "ceiling" is—without introducing additional assumptions or new metrics.

### Key Insight
Starting from the mathematical properties of MOS, theoretical bounds for PCC and MSE are directly derived under the sole assumption that "the mean converges to the true quality as the number of votes becomes sufficiently large." These bounds are driven by **voting variance**.

## Method

### Basic Assumptions and Notations
- Rating scale S: ns discrete values, range [sL, sH] (commonly 5-point scale: sL=1, sH=5, ns=5)
- nv independent subjects rate each file, Rj ∈ S
- MOS is defined as X = (1/nv) Σ Rj
- **Core assumption (Well-behaved)**: E(Rj|Y) = Y, meaning the voting expectation equals the true quality
- Definition of the conditional voting variance function: vr(Y) = Var(Rj|Y)

### MSE Lower Bound Derivation (Sec. III-A)
- The optimal objective estimator is an oracle that knows the true quality Y.
- Derivation yields: **E(D²) = E(vr(Y)) / nv**
- Meaning: MSE lower bound = average voting variance / number of votes per file.
- Equivalent interpretation: E(D²) = Var(X) - Var(Y), which means MSE equals the additional variance introduced by MOS onto the true quality distribution.

### PCC Upper Bound Derivation (Sec. III-B)
- Derivation using the law of total expectation and the law of total variance:
- **ρ(X,Y) = √(Var(Y) / Var(X))** = √(Var(X) - E(D²)) / Var(X))
- Meaning: The PCC upper bound is jointly determined by the MOS variance and the voting variance.
- As MSE approaches 0, PCC approaches 1.

### BinoVotes Voting Model (Sec. IV)
Core innovation—modeling subjective voting using a **Binomial distribution**:
- Bj ~ Binomial(ns-1, (Y-sL)/(sH-sL))
- A single parameter Y simultaneously determines both the mean and the variance of the votes.
- **Voting variance function**: vr(Y) = (Y-sL)(sH-Y) / (ns-1)
    - Variance is 0 at both ends of the scale (as it must be) and max in the middle.
    - For a 1-5 scale: vr(Y) = (Y-1)(5-Y)/4
- This parabolic variance function is close to BinoVotes' 0.25, with a median scaling factor of 0.24 across 28 real subjective tests.

### BinoMOS (Sec. V)
- Averaging BinoVotes yields BinoMOS: X ~ BinoMOS(nv, Y)
- Deriving the PMF of BinoMOS by leveraging the property that the sum of independent binomial random variables remains binomial.
- In special cases, it degenerates to a Beta-Binomial distribution.

### Three Calculation Schemes for Bounds (Sec. VII)
1. **Full-data-driven**: When voting variance information is available, calculate the bounds directly from the data.
2. **Borrowed variance**: Use voting variance information from other tests.
3. **BinoVotes model**: Requires only the mean and variance of MOS to calculate the bounds.

## Key Experimental Results

### Validation Scale
- Validated using the results of **22 subjective tests**, of which 18 contain voting variance information.
- Covers domains such as speech, audio, video, and multimedia.

### Key Figures and Tables Observations
- **Figure 1**: Comparison between BinoVotes PMF and the actual voting distributions of 10 tests shows highly consistent trends (the shapes of the Bad/Poor/Fair/Good/Excellent distributions varying with MOS are identical).
- **Figure 3**: PCC upper bound and RMSE lower bound as functions of the number of votes per file, nv—more votes allow higher PCC and lower RMSE.
- Different true quality distributions (uniform, triangular, Beta(2,2), Beta(2,2.5)) have little impact on the bounds, with RMSE bounds being almost indistinguishable.
- **Figure 4**: BinoVotes' population correlation bounds converge rapidly with sample correlation, with the difference being already very small for sample sizes > 20.

### Core Conclusions
- For the 18 tests: the bounds calculated by the BinoVotes model are highly consistent with the full-data-driven bounds.
- Even when voting variance is unavailable (4 tests), BinoVotes still provides a reasonable estimation of the bounds.

### Typical Numerical Examples (1-5 scale)
- 1 voter: minimum error can reach 0.30 (at a true quality of 3.3).
- 2 voters: minimum error drops to 0.20.
- 3 voters: minimum error is about 0.03.
- The peak of the voting variance function is at the center of the scale (vr = 1.0 when Y = 3).

## Highlights & Insights

1. **Strong conclusion under minimal assumptions**: Only a single assumption that "the voting mean converges to the true quality" is required, avoiding other complex MOS uncertainty models.
2. **Constraining existing metrics rather than proposing new ones**: Keeps the widely used PCC and MSE metrics, providing a theoretical anchor.
3. **Elegance of the BinoVotes model**: The single-parameter model naturally satisfies properties such as discrete ratings, bounded ranges, and zero variance at the endpoints.
4. **Clear practical value**: When the PCC/MSE of an objective estimator approaches the bounds, it indicates that there is virtually no room left for true improvement.
5. **Self-consistent mathematical derivation**: A complete logical chain is presented, from the properties of the rating scale to the properties of MOS, and finally to the bound formulas.
6. **Compatible with subject bias**: Proves that a well-behaved MOS is equivalent to well-behaved voting, and bias only increases voting variance.

## Limitations & Future Work

1. **Limited to PCC and MSE**: Not extended to other commonly used consistency statistics such as Spearman's rank correlation coefficient (SRCC).
2. **BinoVotes is a simplified model**: For certain test scenarios (such as image quality assessment), the actual variance scaling factor (0.15) is lower than BinoVotes' 0.25.
3. **Assumption of independent and identically distributed (i.i.d.) votes**: In reality, votes are affected by sequential effects, fatigue, and other factors.
4. **Impact of non-linear mapping on bounds is not discussed**: Although it is mentioned that the oracle estimator does not require mapping, practical estimators frequently employ non-linear mapping.
5. **Focus on "overall quality" rating**: Its applicability to other rating attributes (e.g., noisiness) remains to be empirically verified.
6. **Discrete scale assumption**: Applicability to continuous scales (such as slider ratings) needs adjustment.

## Related Work & Insights

- Complementary to Hossfeld2011's classification error rate method: the latter equates an objective estimator to 1/2/3/6/9 subjects, whereas this work directly provides PCC/MSE bounds.
- The variance function of BinoVotes is highly consistent with the parabolic variance-MOS relationship discovered in prior studies (median scaling factor of 0.24 across 28 tests vs 0.25).
- Directly instructive for the multimedia quality assessment community: helping to judge whether objective metrics have reached the "ceiling" of subjective data.
- Inspiring for learning with noisy labels: subjective ratings are inherently noisy labels.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unique perspective of deriving bounds from basic mathematical properties, with a naturally elegant BinoVotes model)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated with 22 real subjective tests covering multiple domains)
- Writing Quality: ⭐⭐⭐⭐⭐ (Concise and clear mathematical derivations, logical progression, easy to follow)
- Value: ⭐⭐⭐⭐ (Provides a theoretical benchmark for the quality assessment field, carrying strong practical significance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Entailed Between the Lines: Incorporating Implication into NLI](../../ACL2025/others/entailed_between_the_lines_incorporating_implication_into_nli.md)
- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](../../ACL2025/others/the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](../../NeurIPS2025/others/depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[AAAI 2026\] Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](../../AAAI2026/others/improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)
- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/others/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)

</div>

<!-- RELATED:END -->
