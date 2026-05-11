---
title: >-
  [Paper Note] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems
description: >-
  [NeurIPS 2025][LLM Pretraining][Neural ODE] This paper presents a systematic empirical study of the extrapolation capabilities of Neural ODEs (NODEs) and the equation recovery ability of Symbolic Regression (SR) for dyna…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Neural ODE"
  - "symbolic regression"
  - "dynamical systems"
  - "extrapolation"
  - "scientific discovery"
date: 2026-05-08
content_hash: 5ae2a93e83eb76c8
---

# An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems

**Conference**: NeurIPS 2025
**arXiv**: [2601.20637](https://arxiv.org/abs/2601.20637)
**Code**: Available (JAX/Diffrax + PySR)
**Area**: LLM Pretraining
**Keywords**: Neural ODE, symbolic regression, dynamical systems, extrapolation, scientific discovery

## TL;DR

This paper presents a systematic empirical study of the extrapolation capabilities of Neural ODEs (NODEs) and the equation recovery ability of Symbolic Regression (SR) for dynamical systems. It finds that NODEs can extrapolate to new boundary conditions under dynamically similar settings, and proposes a NODE→SR pipeline: training a NODE on only 10% of the original data to generate augmented trajectories, from which SR recovers 2/3 of the governing equations exactly and provides good approximations for an additional 1/3.

## Background & Motivation

**Data-driven paradigm for scientific discovery.** Automatically discovering governing equations of dynamical systems from experimental data is a central challenge in accelerating scientific discovery. Neural ODEs, by virtue of their continuous-time formulation, are naturally suited to modeling systems described by differential equations, and have been successfully applied in fluid mechanics, pharmacokinetics, and related fields.

**Blind spots in NODE extrapolation.** Existing research has focused primarily on architectural improvements and robustness evaluations of NODEs, but has insufficiently examined their critical capability in practical settings—namely, extrapolation to unseen boundary conditions (new initial conditions) and unseen time horizons, especially under noisy synthetic or real-world data. This is precisely the most critical requirement in practice: exhaustively enumerating all initial conditions for training is infeasible.

**Data hunger in SR.** Symbolic Regression can discover interpretable symbolic equations but typically requires large amounts of high-quality data, which is often unavailable in experimental science. The root cause is a fundamental tension: NODEs can learn dynamics from limited data but produce black-box models, whereas SR produces interpretable equations but demands abundant data. The core idea of this paper is to combine the two—using the NODE as a data augmentation tool to generate large numbers of trajectories from sparse real observations, which are then fed to SR for equation recovery.

## Method

### Overall Architecture

The paper proposes a NODE→SR scientific discovery pipeline consisting of three stages: (1) train a NODE on noisy, sparse real data; (2) use the trained NODE to generate a large set of trajectories covering diverse conditions (data augmentation); (3) apply PySR symbolic regression to the augmented data to recover governing equations. The paper also systematically evaluates NODE extrapolation and interpolation capabilities and SR equation-recovery success rates under varying data conditions.

### Key Designs

1. **Systematic evaluation of NODE extrapolation capability**:

    - Function: Investigate the conditions under which NODEs generalize beyond the training distribution.
    - Mechanism: On two damped oscillation systems (cart-pole and a biological model), NODEs are trained on data covering only a subset of initial conditions or time horizons, then tested outside the training domain. Cart-pole Model B, trained on a small range of initial conditions, nonetheless performs well in regions of phase space that share trajectories with the training data.
    - Design Motivation: To identify the key condition for successful NODE extrapolation—*dynamic similarity*: extrapolation succeeds when the trajectories of new initial conditions share the same dynamical characteristics in phase space as the training data, and fails otherwise.

2. **NODE as a denoiser and data augmentor**:

    - Function: Generate large quantities of clean, augmented trajectories from noisy, sparse data.
    - Mechanism: The NODE is trained on only 10% of the original simulation data (2 hours × 12 nutrient-shift conditions × 10 observations per hour), then generates complete 8-hour trajectories for SR. Experiments reveal that the NODE acts as a denoising filter—SR applied to NODE-generated data from 5%-noise observations recovers more equations than SR applied directly to the noisy data.
    - Design Motivation: Experimental data are invariably noisy and limited; the smoothness and continuity of NODEs naturally confer denoising capability.

3. **Sensitivity analysis of SR to input variable selection**:

    - Function: Reveal the prerequisite conditions for successful equation discovery by SR.
    - Mechanism: On the biological model, providing the intermediate variable $\lambda$ as input allows SR to recover all 3 equations; providing only the 3 primary state variables yields recovery of only 1. The reason is that the rational term $\frac{\psi_A \phi_R}{\psi_A + k_\alpha}$ is approximated as $\phi_R$ within the data range (since $\psi_A \gg k_\alpha$), masking the true functional structure.
    - Design Motivation: To demonstrate that SR requires not only sufficient data but also appropriate feature engineering—the choice of input variables can fundamentally determine whether equation discovery succeeds or fails.

### Loss & Training

NODEs are trained with a standard MSE loss, implemented using the JAX-based Diffrax library. For cart-pole Model A, training data consist of 35 initial conditions × the first 1 second × 25 Hz sampling; Model B uses a smaller training window. For the biological model, training data consist of 12 nutrient-shift conditions × the first 2 hours × 10 Hz sampling. SR is performed with the PySR toolkit under default search settings.

## Key Experimental Results

### Main Results

| System | Data Source | Eq. 2 | Eq. 3 | Eq. 4 | Notes |
|--------|-------------|:-----:|:-----:|:-----:|-------|
| Bio-model (real data, no noise, with λ) | Ground Truth | ✓ | ✓ | ✓ | SR recovers all equations |
| Bio-model (real data, 5% noise, with λ) | Ground Truth | ✗ | ✓ | ✗ | Noise severely hinders discovery |
| Bio-model (NODE data, no noise, with λ) | NODE augmented | ~✓ (approx.) | ✓ | ✓ | NODE→SR recovers 2/3 + approximates 1/3 |
| Bio-model (NODE data, 5% noise, with λ) | NODE augmented | ~✓ (approx.) | ✓ | ✓ | NODE denoising effect is significant |
| Cart-pole (Model A) | NODE | — | — | — | Train on 1 s → successful extrapolation to 5 s |
| Cart-pole (Model B) | NODE | — | — | — | Train on small range → successful extrapolation to dynamically similar regions |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| NODE training sampling rate (5–100 Hz) | No significant difference in 8-hour MSE | Long-horizon prediction is insensitive to sampling rate |
| NODE training sampling rate (5–100 Hz) | 1-hour MSE elevated at <10 Hz | Extremely sparse data (<6 points/variable) degrades interpolation accuracy |
| SR with intermediate variable λ | 3/3 equations recovered (no noise) | Correct input variable selection is critical |
| SR with primary state variables only | 1/3 equations recovered | Rational terms masked by data range |
| SR on NODE (no noise) vs. Ground Truth (5% noise) | NODE: 2/3 + approx. vs. GT: 1/3 | NODE serves as a denoiser |

### Key Findings
- **The key condition for NODE extrapolation is dynamic similarity**: what matters is not numerical proximity of initial conditions to the training data, but rather sharing the same dynamical trajectories in phase space. This implies that training sets should prioritize dynamical diversity over dense sampling of initial conditions.
- **NODEs act as denoising filters**: augmented data generated by a NODE trained on 5%-noise observations enables SR to recover more equations (2/3) than SR applied directly to the noisy data (1/3).
- **SR exhibits an "occlusion" problem for equation structure**: when the data range causes certain mathematical structures (e.g., rational fractions) to degenerate into simpler forms, SR cannot recover the true underlying structure.
- **Sampling frequency has little impact on long-horizon prediction**: NODEs trained at rates from 5 Hz to 100 Hz show no significant difference in 8-hour MSE—a small amount of data suffices to train a viable NODE.

## Highlights & Insights
- The **NODE→SR pipeline** is a practical scientific discovery approach for data-scarce settings: train a NODE on 10% of the data, generate augmented trajectories, and feed them to SR for equation discovery.
- The concept of *dynamic similarity* provides actionable guidance for NODE training set design: prioritize trajectory diversity over dense coverage of initial conditions.
- The denoising effect of NODEs is an emergent finding: the smooth inductive bias of continuous-time models naturally filters out measurement noise.
- The failure mode analysis of SR (sensitivity to variable selection, occlusion by data range) offers practical reference value.

## Limitations & Future Work
- Validation is limited to 2 relatively simple damped oscillation systems (cart-pole and bacterial nutrient adaptation); chaotic systems and high-dimensional systems remain unexplored.
- SR analysis relies on data from a single nutrient-shift condition (single initial condition to final state); joint analysis across multiple conditions may improve equation discovery.
- The NODE architecture used is the basic formulation; augmented NODEs or Neural CDEs may improve extrapolation capability.
- The SR framework is restricted to PySR; methods that incorporate physical priors, such as SINDy, may be more appropriate in certain settings.
- Equation (2) of the biological model is never perfectly recovered (even from NODE-generated data), indicating that the pipeline still has limitations for certain equation structures.

## Related Work & Insights
- **Relation to SINDy**: SINDy uses sparse regression to discover equations from a predefined function library; PySR searches a larger space but requires more data. The two approaches are complementary.
- **Relation to augmented NODEs**: This paper uses basic NODEs; augmented NODEs, which extend the state space, may improve extrapolation.
- **Broader implication**: The NODE→SR pipeline is generalizable to other experimental science domains—wherever sparse time-series measurements are available, it may be possible to automatically discover governing equations.

## Rating
- Novelty: ⭐⭐⭐ Combines existing methods (NODE + SR), but the identification of dynamic similarity and the NODE denoising effect are valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐ Only two systems, but ablation analyses (sampling rate, noise, variable selection) are relatively systematic.
- Writing Quality: ⭐⭐⭐⭐ Experimental design is clear; failure mode analysis is candid and thorough.
- Value: ⭐⭐⭐ Offers a useful reference for scientific discovery pipelines in data-scarce settings, but requires validation on more complex systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/llm_pretraining/stochastic_self-organization_in_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
