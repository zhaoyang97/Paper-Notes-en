---
title: >-
  [Paper Note] Bounds on Agreement between Subjective and Objective Measurements
description: >-
  [CVPR 2026][quality assessment] Starting from the mathematical properties of MOS, this paper derives theoretical formulas for the upper bound on PCC and the lower bound on MSE between subjective test results and any objective estimator. It further proposes the BinoVotes/BinoMOS voting model and validates both the bounds and the model on 18 subjective test datasets.
tags:
  - CVPR 2026
  - quality assessment
  - MOS
  - subjective test
  - PCC bound
  - MSE bound
  - BinoVotes
date: 2026-05-08
content_hash: fe8098c8aeac0ab6
---

# Bounds on Agreement between Subjective and Objective Measurements

**Conference**: CVPR 2026
**arXiv**: [2603.13204](https://arxiv.org/abs/2603.13204)
**Code**: None
**Area**: Other
**Keywords**: quality assessment, MOS, subjective test, PCC bound, MSE bound, BinoVotes

## TL;DR
Starting from the mathematical properties of MOS, this paper derives theoretical formulas for the upper bound on PCC and the lower bound on MSE between subjective test results and any objective estimator. It further proposes the BinoVotes/BinoMOS voting model and validates both the bounds and the model on 18 subjective test datasets.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Objective estimators of multimedia quality are typically evaluated by their PCC and MSE against a subjective "ground truth" (MOS). However, targeting PCC = 1.0 or MSE = 0.0 is neither realistic nor reproducible — MOS inherently contains noise due to the discrete nature of rating scales, limited number of voters, and individual biases.

**Shortcomings of existing approaches**: (1) Some works propose new evaluation metrics (e.g., classification error rate, discriminability, ε-insensitive RMSE), but these lack a unified theoretical foundation. (2) MOS uncertainty models (e.g., Gaussian modeling) may violate the discrete and bounded nature of MOS; post-hoc fixes such as clipping introduce additional bias.

**Uniqueness of the proposed approach**: Rather than introducing new metrics, this paper derives theoretical bounds for the two most widely used existing metrics (PCC and MSE). The derivation relies solely on the most fundamental assumption — that the expected vote equals the true quality — without requiring any model of MOS uncertainty. The bounds naturally reflect the discrete nature of MOS and its dependence on the number of votes.

## Method

### Overall Architecture
The paper proceeds in three steps: (1) derive theoretical bounds on PCC and MSE based on vote variance; (2) propose the BinoVotes voting model to estimate vote variance when empirical variance is unavailable; (3) validate the bounds on 18 subjective test datasets.

### Key Designs

1. **Derivation of the PCC upper bound and MSE lower bound**:
    - Function: Derive the best achievable PCC and lowest achievable MSE for any objective estimator under given subjective test conditions.
    - Mechanism: Consider the best possible objective estimator — an oracle estimator with direct access to the true quality $Y$. The PCC/MSE between $Y$ and MOS $X$ then defines the limit for any estimator. Key result: $\mathbb{E}(D^2) = \frac{\mathbb{E}(v_r(Y))}{n_v}$, meaning the MSE lower bound depends only on the expected vote variance $\mathbb{E}(v_r(Y))$ and the number of voters $n_v$. The PCC upper bound follows similarly, using $\text{Var}(X) = \frac{\mathbb{E}(v_r(Y))}{n_v} + \text{Var}(Y)$.
    - Design Motivation: The derivation is grounded in probability theory (the law of total expectation, the law of total variance, and the i.i.d. votes assumption), making it rigorous and minimally assumptive. The "well-behaved" condition $\mathbb{E}(R_j|Y)=Y$ is the foundational assumption underlying all subjective tests.

2. **BinoVotes voting model**:
    - Function: Provide theoretical estimates of the vote distribution and variance when empirical vote variance is unavailable.
    - Mechanism: Votes are modeled as a scaled binomial distribution — the true quality $Y \in [s_L, s_H]$ is mapped to $p = (Y-s_L)/(s_H-s_L) \in [0,1]$, and the vote distribution is $R = \frac{s_H-s_L}{n_s-1}\text{Binom}(n_s-1, p) + s_L$. The BinoVotes variance is $v_r(Y) = \frac{(s_H-s_L)^2}{(n_s-1)} p(1-p)$ — zero at the extremes of the quality range and maximal in the middle.
    - Design Motivation: The binomial distribution naturally satisfies the discrete and bounded constraints of rating scales. The BinoVotes variance is a parabolic function of true quality — voter agreement is high (low variance) at extreme quality levels (e.g., clearly good or clearly bad) and disagreement is greatest (high variance) at intermediate quality levels, which aligns with intuition.

3. **BinoMOS model**:
    - Function: Model the theoretical distribution of MOS by averaging BinoVotes.
    - Mechanism: The average of $n_v$ BinoVotes naturally inherits the discrete value range of MOS — $|M| = n_v(n_s-1)+1$ possible values — and its dependence on the number of voters. As $n_v$ increases, BinoMOS converges to the true quality (by the central limit theorem).
    - Design Motivation: Direct MOS modeling approaches (e.g., Gaussian distributions) frequently violate the discrete nature of MOS. BinoMOS is derived naturally from the voting model and preserves all mathematical properties.

### Three Methods for Computing the Bounds
1. **Fully data-driven**: The subjective test directly provides vote variance $v_r(Y_i)$ → substitute directly into the bound formulas.
2. **Borrowed variance**: Empirical variance functions are borrowed from other subjective tests that provide variance information.
3. **BinoVotes model**: The theoretical variance formula from BinoVotes is used as an estimate → only the rating scale and number of voters need to be known.

## Key Experimental Results

### Main Results (PCC/MSE bound validation across 18 subjective tests)

| Validation Method | PCC Upper Bound Agreement | MSE Lower Bound Agreement |
|---------|-------------|-------------|
| Data-driven vs. BinoVotes | Very close | Very close |
| Tests with many votes (large $n_v$) | Bound approaches 1.0 | Bound is small |
| Tests with few votes (small $n_v$) | Bound noticeably < 1.0 | Bound is relatively large |

Across 18 subjective tests covering speech, audio, and video quality assessment, bounds derived from the BinoVotes model closely match those from the fully data-driven approach, validating BinoVotes as an effective approximation for vote variance.

### Ablation Study

| Factor | Effect on MSE Lower Bound | Effect on PCC Upper Bound |
|---------|---------------|---------------|
| Increasing number of voters $n_v$ | Bound decreases → tighter | Bound increases → closer to 1 |
| Concentrated vs. uniform quality distribution | Tighter when uniform | Depends on $\text{Var}(Y)$ |
| Rating scale $n_s$ | Larger $n_s$ → larger vote variance | Bound becomes looser |

### Key Findings
- When the number of voters $n_v$ is small (e.g., 4–10), the MSE lower bound can reach 0.1–0.3 (on a 5-point scale), meaning an objective estimator achieving MSE = 0.1 is already near the theoretical limit.
- The BinoVotes assumption of a parabolic vote variance as a function of quality closely matches the empirical distribution of actual vote variance.
- For 4 subjective tests without empirical vote variance, the BinoVotes model still yields reasonable bounds.
- Computing the bounds requires only the number of voters and the rating scale — no additional complex assumptions are needed.

## Highlights & Insights
- **Theoretical contribution**: This is the first work to derive rigorous bounds on PCC and MSE based solely on the intrinsic mathematical properties of MOS, entirely avoiding the need to model MOS uncertainty.
- The simplicity and effectiveness of the BinoVotes model are impressive — the binomial distribution naturally matches the discrete nature of rating scales.
- **Practical value**: Developers of any objective quality estimator can use these bounds to assess "how much room for improvement remains" or "whether the theoretical limit has been reached."
- The paper shows that per-subject bias is mathematically equivalent to increased vote variance — explicit modeling of bias is therefore unnecessary.

## Limitations & Future Work
- The i.i.d. votes assumption may be too strong in certain scenarios (e.g., fatigue effects, order effects).
- The BinoVotes model assumes a symmetric parabolic variance as a function of quality; the true distribution may be asymmetric.
- Validation covers only overall quality ratings; bounds for other attributes (e.g., noisiness, sharpness) may differ.
- The effect of nonlinear mapping on the bounds is not discussed — in practice, PLCC is often computed after such mapping.

## Related Work & Insights
- **MOS uncertainty research**: A large body of prior work models MOS noise (alpha-stable, Gaussian, Gaussian mixture), and this paper provides a more fundamental alternative.
- **ε-insensitive RMSE**: This metric attempts to tolerate subjective noise; the bounds proposed here offer a more precise tolerance baseline.
- **Objective quality assessment**: Developers of estimators such as POLQA, VISQOL, and VMAF can directly apply these bounds.
- Inspiration: In any scenario where models are evaluated against noisy ground truth (e.g., LLM scoring, crowd-sourced annotation), analogous bound derivations could provide valuable performance expectations.

## Rating
- Novelty: ⭐⭐⭐⭐ First derivation of PCC/MSE bounds from the mathematical properties of MOS; BinoVotes model is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale validation across 18 cross-domain subjective tests covering speech, audio, and video.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, notation is consistent, and the logical flow is clear.
- Value: ⭐⭐⭐⭐ Provides a foundational theoretical tool for the multimedia quality assessment research community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [\[CVPR 2026\] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)
- [\[CVPR 2026\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_a.md)
- [\[CVPR 2026\] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)

<!-- RELATED:END -->
