---
title: >-
  [Paper Note] Certified Human Trajectory Prediction
description: >-
  [CVPR 2025][Autonomous Driving][Certified Robustness] This work introduces randomized smoothing certification to human trajectory prediction for the first time. By leveraging mean/median aggregation functions and a diffusion denoiser, it provides certified robustness for trajectory prediction models—ensuring that the output remains within a certified boundary regardless of how the input noise is perturbed (within a radius $R$).
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Certified Robustness"
  - "Trajectory Prediction"
  - "Randomized Smoothing"
  - "Diffusion Denoiser"
  - "Adversarial Attack Defense"
date: 2026-05-08
content_hash: 531b743513481536
---

# Certified Human Trajectory Prediction

**Conference**: CVPR 2025  
**arXiv**: [2403.13778](https://arxiv.org/abs/2403.13778)  
**Code**: [https://s-attack.github.io/](https://s-attack.github.io/)  
**Area**: Autonomous Driving / Trajectory Prediction  
**Keywords**: Certified Robustness, Trajectory Prediction, Randomized Smoothing, Diffusion Denoiser, Adversarial Attack Defense

## TL;DR

This work introduces randomized smoothing certification to human trajectory prediction for the first time. By leveraging mean/median aggregation functions and a diffusion denoiser, it provides certified robustness for trajectory prediction models—ensuring that the output remains within a certified boundary regardless of how the input noise is perturbed (within a radius $R$).

## Background & Motivation

**Background**: Human trajectory prediction is a critical task in autonomous driving. Existing data-driven methods (such as Social-LSTM, EqMotion, etc.) perform excellently on clean inputs but have been proven to be extremely vulnerable to adversarial attacks and perception noise.

**Limitations of Prior Work**: Previous robustness defense methods (such as data augmentation, adversarial training, etc.) are heuristic and lack guarantees—they ultimately fail when facing a sufficiently strong adversary. The output of trajectory prediction consists of unbounded continuous values (unlike classification which has finite categories), making it fundamentally difficult to directly apply certification methods designed for image classification.

**Key Challenge**: Three specific challenges of trajectory prediction: (1) Unbounded outputs—continuous coordinates have no natural upper or lower bounds; (2) Multimodality—how to certify multiple reasonable predicted trajectories; (3) Performance degradation—the smoothing operation itself reduces prediction accuracy.

**Core Idea**: Generalize randomized smoothing to multi-output regression tasks, propose adaptive clamping to solve the unboundedness issue, introduce a diffusion denoiser to alleviate accuracy degradation, and design new certification metrics to evaluate the actual reliability of the model under noise.

## Method

### Overall Architecture

Input observation trajectory $X$ → add $n$ Gaussian perturbations $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ → denoiser $h$ preprocessing → predictor $g$ yields $n$ predictions → aggregation function $\mathcal{A}$ (mean or median) yields final smoothed prediction $Y$ and certified bounds.

### Key Designs

1. **Mean vs Median Smoothing**:

    - Mean smoothing: Requires bounded output $f: \mathbb{R}^d \to [l, u]$, certified bounds are calculated through the normal CDF $\Phi$; requires knowing the upper and lower bounds of the output.
    - Median smoothing: Does not require bounded output, bounds are calculated using quantile functions; more robust as it is unaffected by outliers.
    - Key finding: **median aggregation significantly outperforms mean**—because trajectory predictors can generate extreme outlier predictions for noisy inputs, which can easily skew the mean, while the median is more robust.

2. **Adaptive Clamping**:

    - **Function**: Imposes output upper and lower bounds on unbounded trajectory predictors.
    - **Mechanism**: Traverses the predictor on the training set, calculates the maximum/minimum value of each coordinate dimension as $l_j, u_j$, and then clamps using $\min(u_j, \max(l_j, \cdot))$.
    - Only mean smoothing requires this step, whereas median smoothing does not.

3. **Diffusion Denoiser**:

    - **Function**: Preprocesses noisy inputs before the smoothing operation to alleviate performance degradation.
    - **Mechanism**: Trains an unconditional diffusion model to learn the trajectory data distribution. During inference, multi-step denoising is used to recover the clean trajectory $h(X + \epsilon) \approx X$.
    - **Effects**: Significantly tightens certified bounds (FBD decreases from 0.96 to 0.65 at the same FDE=1.3) while maintaining prediction accuracy.
    - **Comparison**: Compared to Wiener filters, moving averages, and polynomial fitting, the diffusion denoiser has the smallest residual noise.

4. **Certification Metrics Design**:

    - **ABD/FBD** (Average/Final Bound half-Diameter): Measures the size of the certified bounds.
    - **Certified-ADE/FDE**: The worst-case displacement error within the certified boundary, reflecting the actual performance under noise.
    - **Certified-Collision Rate**: Whether neighboring agents fall within the predicted certified boundary.
    - Key finding: **The most accurate model is not necessarily the most robust**—EqMotion has the lowest FDE (1.12) but its Certified-FDE is not the lowest, whereas D-Pool's Certified-FDE (2.0) is the best.

### Multimodal / Multi-agent Extensions

- Multimodal: Certify $k=20$ modes separately and select the mode with the smallest Certified-FDE. Multimodal FBD (0.64) is much smaller than unimodal FBD (0.99).
- Multi-agent: Noise is simultaneously added to all agents' trajectories; mutual dependency expands the certified boundaries (FBD 1.21 vs. single-agent 0.99).

## Key Experimental Results

### Main Results: Comparison of Smoothed vs. Non-smoothed Predictors

| Model | FDE | Certified-FDE | Col(%) | Certified-Col(%) |
|------|-----|---------------|--------|-------------------|
| Social-Force | 1.25 | N/A | 7.4 | N/A |
| Smoothed Social-Force | 1.26 | 2.27 | 8.0 | 46 |
| D-Pool | 1.14 | N/A | 9.4 | N/A |
| Smoothed D-Pool | 1.23 | **2.00** | 9.0 | 49 |
| EqMotion | **1.12** | N/A | 10.1 | N/A |
| Smoothed EqMotion | 1.14 | 2.07 | 10.6 | 57 |

### Ablation Study: Effect of the Denoiser

| Configuration | FBD at FDE=1.2 | FBD at FDE=1.3 | FBD at FDE=1.4 |
|------|:-:|:-:|:-:|
| Without Denoiser | 1.20 | 0.96 | 0.80 |
| **With Diffusion Denoiser** | **0.78** | **0.65** | **0.57** |

### Key Findings

- **Minimal accuracy cost**: The smoothing operation only increases FDE by 1-6% (EqMotion: 1.12 $\to$ 1.14), but achieves guaranteed robustness.
- **Most accurate $\neq$ Most robust**: EqMotion has the highest accuracy, but D-Pool has the strongest certified robustness, revealing that looking only at FDE can overlook safety hazards.
- **Huge gap in collision rates**: The gap between Col and Certified-Col is as high as 40%+ (10.1% vs. 57%), showing that the risk of collision under noise is severely underestimated.
- **Acceptable computational overhead**: $n=100$ Monte Carlo runs is only 42% slower than a single inference (0.07s $\to$ 0.1s) and can be parallelized.
- **Downstream tasks benefit**: In robotic navigation tasks, the robust model with the denoiser reduces the collision rate from 21% to 15.1%.

## Highlights & Insights

- **First trajectory prediction certification**: Generalizes randomized smoothing from classification to multi-output sequence regression, solving the two major technical challenges of unbounded outputs and multimodality. This provides a new paradigm for AI reliability in safety-critical systems.
- **"Most accurate $\neq$ Safest" warning**: This finding has significant implications for the autonomous driving community—solely pursuing FDE/ADE can introduce safety hazards, and robustness metrics should be evaluated simultaneously.
- **Diffusion model for trajectory denoising**: Applying diffusion models to denoise sequential data (rather than generation) is a novel application direction with significant performance.
- **The $\sigma$ accuracy-robustness trade-off**: Adjusting $\sigma$ allows flexible control over the balance between accuracy and certified boundaries, providing an adjustable safety margin for practical deployment.

## Limitations & Future Work

- Randomized smoothing requires $n$ forward passes, which limits real-time capability (although it can be parallelized).
- The upper and lower bounds of adaptive clamping depend on the training set distribution, and new scenarios may require recalibration.
- Currently only validated on Trajnet++, and has not been tested on larger-scale real-world driving datasets (such as nuScenes full).
- The certified boundaries of multi-agents increase significantly, making the practicality in dense scenes still to be verified.

## Related Work & Insights

- **vs Trajpac**: Trajpac uses PAC strategies for probabilistic verification, but it relies on noise distribution, requires over 30,000 samples, and offers no guarantees. The proposed method only requires 100 samples, is independent of the noise distribution, and provides guarantees.
- **vs Adversarial Training**: Adversarial training only provides empirical robustness and fails against stronger adversaries. Randomized smoothing provides mathematical guarantees.
- **vs Conformal Prediction**: CP guarantees ground truth coverage, whereas this work guarantees the output region—the two are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introduces certified robustness to trajectory prediction for the first time, solving multiple non-trivial technical challenges.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple models, multimodalities, multi-agents, and downstream tasks, but lacks large-scale datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation, rigorous theoretical derivation, and well-designed new metrics.
- Value: ⭐⭐⭐⭐⭐ Provides a new tool for reliability assurance in safety-critical autonomous driving systems, carrying significant practical and academic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [\[ECCV 2024\] Adaptive Human Trajectory Prediction via Latent Corridors](../../ECCV2024/autonomous_driving/adaptive_human_trajectory_prediction_via_latent_corridors.md)
- [\[ECCV 2024\] Progressive Pretext Task Learning for Human Trajectory Prediction](../../ECCV2024/autonomous_driving/progressive_pretext_task_learning_for_human_trajectory_prediction.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](../../NeurIPS2025/autonomous_driving/towards_predicting_any_human_trajectory_in_context.md)

</div>

<!-- RELATED:END -->
