---
title: >-
  [Paper Note] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization
description: >-
  [NeurIPS 2025][Optimization][Conformal Prediction] A Feedback-Based Conformal Prediction (Fb-CP) framework is proposed, which feeds back information from the executed trajectory to CP to dynamically adjust prediction region sizes, simultaneously guaranteeing coverage and significantly improving trajectory performance in receding-horizon trajectory optimization.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Conformal Prediction"
  - "Trajectory Optimization"
  - "Uncertainty Quantization"
  - "Closed-Loop Feedback"
  - "Risk Allocation"
date: 2026-05-08
content_hash: 0a80206951ca697b
---

# Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization

**Conference**: NeurIPS 2025  
**arXiv**: [2510.16376](https://arxiv.org/abs/2510.16376)  
**Code**: [github.com/DOCU-Lab/Feedback-based_Conformal_Prediction](https://github.com/DOCU-Lab/Feedback-based_Conformal_Prediction)  
**Area**: LLM Evaluation  
**Keywords**: Conformal Prediction, Trajectory Optimization, Uncertainty Quantization, Closed-Loop Feedback, Risk Allocation

## TL;DR
A Feedback-Based Conformal Prediction (Fb-CP) framework is proposed, which feeds back information from the executed trajectory to CP to dynamically adjust prediction region sizes, simultaneously guaranteeing coverage and significantly improving trajectory performance in receding-horizon trajectory optimization.

## Background & Motivation

**Background**: Conformal Prediction (CP) is a powerful tool for constructing prediction regions with finite-sample coverage guarantees. It is widely used in trajectory optimization (TO) under uncertain environments to generate prediction regions for obstacle locations, enabling probabilistic safety collision avoidance.

**Limitations of Prior Work**: Existing methods adopt a **sequential** pipeline—generating prediction regions using CP first, where decision-making unidirectionally depends on these prediction regions. Information from the decision end is never fed back to CP, leading to overly conservative prediction regions.

**Key Challenge**: In receding-horizon TO, the executed trajectory $x_{0:t}^*$ contains rich posterior collision information (the actual collision probability is much lower than the prior allocated $\alpha_\tau$), but this information is wasted and cannot be used to shrink the prediction regions for subsequent time steps.

**Key Insight**: Establish a closed-loop information channel from decision to CP, replacing the prior $\alpha_\tau$ with the posterior collision probability $\beta_\tau$ of the executed trajectory to reallocate the released risk budget to future time steps.

## Method

### Problem Modeling
Consider a discrete-time nonlinear system $x_{t+1} = f(x_t, u_t)$ with $M$ dynamic obstacles with unknown trajectories in the environment. Define the joint probabilistic safety constraint:

$$\mathbb{P}\left\{\bigcap_{\tau=1}^{T}\{c(x_\tau, Y_\tau) \geq 0\}\right\} \geq 1 - \alpha$$

Using Boole's inequality, this is decomposed into individual constraints $\mathbb{P}\{c(x_\tau, Y_\tau) \geq 0\} \geq 1 - \alpha_\tau$ subject to the total risk constraint $\sum_\tau \alpha_\tau \leq \alpha$.

### CP Prediction Region Construction
Partition the calibration set $D_{cal}$ into $D_{cal}^1$ ($K$ trajectories) and $D_{cal}^2$ ($L$ trajectories). Define the non-conformity score $R_{\tau|t}^{(i)} = \|Y_\tau^{(i)} - \hat{Y}_{\tau|t}^{(i)}\|$ to obtain the coverage guarantee:

$$\mathbb{P}\{\|Y_\tau - \hat{Y}_{\tau|t}\| \leq C_{\tau|t}^{1-\alpha_\tau}\} \geq 1 - \alpha_\tau$$

where $C_{\tau|t}^{1-\alpha_\tau} = \text{Quantile}_{1-\alpha_\tau}(R_{\tau|t}^{(1)}, \ldots, R_{\tau|t}^{(K)}, \infty)$.

### Posterior Collision Probability Computation (Core Innovation)
At time step $t$, use $D_{cal}^2$ and the determined system states $x_\tau^*$ to compute the upper bound of the posterior collision probability:

$$\beta_\tau = \frac{1 + \sum_{i=1}^{L} \mathbb{I}(S_\tau^{(K+i)} < 0)}{1 + L}$$

where $S_\tau^{(K+i)} = c(x_\tau^*, \hat{Y}_{\tau|\tau-1} + \omega_\tau^{(K+i)})$. Using an independent $D_{cal}^2$ guarantees coverage.

### Feedback-Based Risk Allocation
Reframing the optimization problem: replacing the past $\alpha_\tau$ with $\beta_\tau$ to release the remaining risk budget to future steps:

$$\sum_{\tau=t+1}^{T} \alpha_\tau \leq \alpha - \sum_{\tau=0}^{t} \beta_\tau$$

Since $\beta_\tau$ is lower than $\alpha_\tau$ with high probability, the available future risk increases $\rightarrow$ prediction regions shrink $\rightarrow$ trajectory performance improves.

### Iterative Risk Allocation Algorithm (IRA)
To avoid high computational complexity from treating $\alpha_{t+1:T}$ as decision variables, a two-stage iteration is proposed:
1. **Tighten Inactive Constraints**: For steps where constraints are not active, reduce their allocated risk $\tilde{\alpha}_\tau^n = (1-\eta)\alpha_\tau^n + \eta \underline{\alpha}_\tau^n$
2. **Relax Active Constraints**: Uniformly distribute the released risk to all time steps with active constraints

Based on the monotonicity lemma $\partial J^*/\partial \alpha_\tau \leq 0$, the cost sequence $\{J^*(\alpha_{t+1:T}^n)\}$ generated by IRA is proved to be monotonically decreasing and convergent.

### Theoretical Guarantees
- **Coverage Guarantee** (Theorem 5.4): The entire trajectory satisfies $\mathbb{P}\{\bigcap_{\tau=1}^{T}\{c(x_\tau^*, Y_\tau) \geq 0\}\} \geq 1 - \alpha$
- **Convergence Guarantee** (Theorem 5.3): Under bounded $\mathcal{X}$, $\mathcal{U}$, and continuous objective functions, IRA converges to a finite limit
- **Performance Improvement Guarantee**: The adjustments in Fb-CP always maintain coverage and provide provable performance improvements

## Key Experimental Results

### 3D Quadrotor Model (1000 Monte Carlo runs, $\alpha=0.05$)

| Method | Avg. Cost↓ | Avg. Compute Time (s) | Collision Avoidance Rate↑ |
|------|----------|----------------|-----------|
| CC ($\eta$=1000) | 59.25 | 0.019 | 97.0% |
| ACI-MP | 17.970 | 0.022 | 98.6% |
| RF-CP | 15.794 | 0.487 | 98.7% |
| S-CP | 17.321 | 0.022 | 98.8% |
| Fb-CP-ARA | 15.356 | 0.027 | 98.2% |
| **Fb-CP-IRA** | **7.189** | 0.038 | 96.3% |

### Performance under Different $\alpha$ (Quadrotor Model)

| $\alpha$ | S-CP Cost | Fb-CP-ARA Cost | Fb-CP-IRA Cost | IRA Reduction |
|---------|----------|---------------|---------------|---------|
| 0.05 | 17.321 | 15.356 | 7.189 | -58.5% |
| 0.10 | 16.170 | 14.228 | 6.798 | -57.9% |
| 0.15 | 14.830 | 12.354 | 6.191 | -58.3% |
| 0.20 | 13.217 | 10.220 | 5.398 | -59.1% |

### Key Findings
- Fb-CP-ARA reduces the average cost by 11.3% compared to S-CP (with almost zero computational overhead)
- Fb-CP-IRA reduces the average cost by **58.5%** compared to S-CP, but increases computation time due to iterative TO solving
- RF-CP's normalized non-conformity score introduces mixed-integer variables, resulting in computation times an order of magnitude higher than Fb-CP-IRA
- All methods satisfy the theoretical coverage guarantee of $1-\alpha$ for collision avoidance rate

## Highlights & Insights
- **The feedback loop from decisions to CP** is a brand new paradigm: breaking the conventional mindset of "CP only makes predictions" and establishing bidirectional information flow.
- **Clever posterior probability computation**: using an independent $D_{cal}^2$ to evaluate "how many calibration trajectories would collide given the current position," which is intuitive and computationally efficient.
- **Elegant theoretical structure of active/inactive constraint analysis in IRA**: the tightening-relaxation iterations monotonically decrease cost while maintaining feasibility.
- **High framework generalization**: applicable to any receding-horizon optimization problem with joint probabilistic constraints.

## Limitations & Future Work
- Requires splitting the calibration set into two parts ($D_{cal}^1$ and $D_{cal}^2$), halving the effective sample size.
- IRA iterations increase computation time, which may be limiting in scenarios with extremely high real-time requirements.
- Extensions to distribution shift (Appendix G) rely on weighting schemes, offering limited adaptability to severe shifts.
- Obstacle trajectory models assume LSTM predictors; stronger predictors (e.g., Transformers) might alter the optimal risk allocation strategy.
- Experiments do not cover high-dimensional state spaces or large-scale obstacle scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose decision-feedback CP, opening a new research direction
- Theory Depth: ⭐⭐⭐⭐⭐ Triple theoretical guarantees of coverage, convergence, and performance
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models and baselines + 1000 MC runs + real datasets
- Writing Quality: ⭐⭐⭐⭐ Clear logic, seamlessly weaving together problem formulation, methodology, and theory
- Overall: ⭐⭐⭐⭐⭐ Elegant theory + practical effectiveness + paradigm innovation

## Related Work & Insights
- **vs S-CP (Lindemann et al. 2023)**: The standard method for sequential CP, where prediction regions are fixed and do not update with decisions. Fb-CP introduces a feedback channel on top of this, reducing cost by 58.5% on average.
- **vs ACI-MP (Dixit et al. 2023)**: Handles distribution shift using ACI, but remains a unidirectional pipeline that does not utilize executed trajectory information. More suitable for testing distribution shifts rather than the setup of this paper.
- **vs RF-CP (Stamouli et al. 2024)**: Proposes normalized non-conformity scores, achieving a cost comparable to Fb-CP-ARA but introducing mixed-integer variables, resulting in computation times an order of magnitude higher.
- **vs CC (Lekeufack et al. 2024)**: Controls collision rates via cost weights but underutilizes the calibration set, resulting in 184% higher cost.
- The paradigm of feedback-based CP can be transferred to any online optimization problem with sequential probabilistic constraints (e.g., robot swarms, UAV path planning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)

</div>

<!-- RELATED:END -->
