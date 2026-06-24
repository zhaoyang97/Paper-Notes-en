---
title: >-
  [Paper Note] Tight Lower Bounds and Improved Convergence in Performative Prediction
description: >-
  [NeurIPS 2025][Performative Prediction] Under the performative prediction framework, this paper provides the first tight convergence rate analysis for Repeated Risk Minimization (RRM) and proposes the Affine Risk Minimizers (ARM) algorithm class, which achieves convergence over a broader problem class by leveraging data from historical training snapshots.
tags:
  - "NeurIPS 2025"
  - "Performative Prediction"
  - "Convergence Rate"
  - "Lower Bounds"
  - "Repeated Risk Minimization"
  - "Decision-Dependent Distribution"
date: 2026-05-08
content_hash: 1d268a110807840c
---

# Tight Lower Bounds and Improved Convergence in Performative Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2412.03671](https://arxiv.org/abs/2412.03671)  
**Code**: Unavailable  
**Area**: Machine Learning Theory / Decision-Dependent Learning
**Keywords**: Performative Prediction, Convergence Rate, Lower Bounds, Repeated Risk Minimization, Decision-Dependent Distribution

## TL;DR

Under the performative prediction framework, this paper provides the first tight convergence rate analysis for Repeated Risk Minimization (RRM) and proposes the Affine Risk Minimizers (ARM) algorithm class, which achieves convergence over a broader problem class by leveraging data from historical training snapshots.

## Background & Motivation

Performative prediction studies settings where deployed models alter the data distribution — model predictions influence the behavior of the predicted subjects, thereby shifting the distribution of future data. Examples include:

- **Credit scoring**: After deployment, applicants strategically adjust behavior to obtain better scores.
- **Environmental regulation**: Firms may manipulate emission data to satisfy regulatory targets.
- **Education**: Schools engage in teaching-to-the-test in response to standardized assessments.

The core objective is to converge to a **performatively stable point** $\theta_{\text{PS}}$: a point at which the model minimizes risk under its own induced distribution, and further retraining does not change the model parameters.

**Limitations of Prior Work**:

- Perdomo et al. (2020) proved that RRM converges linearly at rate $(\frac{\beta\epsilon}{\gamma})^t$ under the condition $\frac{\beta\epsilon}{\gamma} < 1$, but **never analyzed whether this upper bound is tight**.
- Mofakhami et al. (2023) established convergence conditions under Pearson $\chi^2$ divergence, but the upper bound contains a superfluous constant $C$.
- Both frameworks **use only the most recent iteration's data**, discarding historical information.

**Core Problem**: (1) Is the convergence rate of RRM already optimal? (2) Can historical snapshot data be leveraged to break the RRM convergence lower bound?

## Method

### Overall Architecture

The RRM update in performative prediction is $\theta^{t+1} = \arg\min_\theta \mathbb{E}_{z \sim \mathcal{D}(f_{\theta^t})}[\ell(f_\theta(x), y)]$, i.e., retraining on the distribution induced by the current model. This paper extends the framework to ARM: retraining on affine combinations of historical distributions.

### Key Designs

1. **Tight Lower Bounds (Theorems 2 & 3)**: The paper provides the first proofs that RRM convergence rates are tight in both mainstream frameworks.

   **Under the Mofakhami framework (Theorem 2)**: There exists a problem instance such that

   $\|f_{\theta^t} - f_{\theta_{\text{PS}}}\|_{f_{\theta_{\text{PS}}}} = \Omega\left(\left(\frac{\sqrt{\epsilon}M}{\gamma}\right)^t\right)$

   **Under the Perdomo framework (Theorem 3)**: There exists a problem instance such that

   $\|\theta^t - \theta_{\text{PS}}\| = \Omega\left(\left(\frac{\epsilon\beta}{\gamma}\right)^t \|\theta_0 - \theta_{\text{PS}}\|\right)$

   **Proof technique**: Inspired by Nesterov's lower bound constructions for convex optimization, the proof constructs a special matrix $A$ (a lower-triangular all-ones matrix) such that each RRM iteration "discovers" a new dimension. The loss is set to $\ell(f_\theta(x),y) = \frac{\gamma}{2}\|\theta - \frac{\beta}{\gamma}z\|^2$ and the distribution map to $\mathcal{D}(\theta) = \mathcal{N}(\frac{\epsilon}{2}A\theta + e_1, \sigma^2)$. This ensures the RRM update follows $\theta^{t+1} = \frac{\beta}{\gamma}(\frac{\epsilon}{2}A\theta^t + e_1)$, yielding the lower bound via accumulated contributions from undiscovered dimensions.

2. **Improved Upper Bound (Theorem 1)**: Under the Mofakhami framework, the RRM convergence rate is improved from $(\frac{\sqrt{\epsilon}CM}{\gamma})^t$ to $(\frac{\sqrt{\epsilon}M}{\gamma})^t$, eliminating the constant $C$. The improvement stems from modifying the $\epsilon$-sensitivity definition in Assumption 1 to use $\|f_\theta - f_{\theta'}\|_{f_\theta}^2$ rather than $\|f_\theta - f_{\theta'}\|^2$.

3. **Affine Risk Minimizers (ARM) (Section 5)**: The core contribution — retraining on affine combinations of historical snapshot distributions:

   $D_t = \sum_{i=0}^{t-1} \alpha_i^{(t)} D(f_{\theta^i}), \quad \sum_{i=0}^{t-1}\alpha_i^{(t)} = 1$

   **Lemma 1 (2-Snapshots ARM)**: Using only the average of the two most recent snapshots $D_t = \frac{1}{2}D(f_{\theta^t}) + \frac{1}{2}D(f_{\theta^{t-1}})$, the convergence condition is relaxed from $\frac{\sqrt{\epsilon}M}{\gamma} < 1$ to $\frac{\sqrt{3}}{2}\frac{\sqrt{\epsilon}M}{\gamma} < 1$, equivalent to $\frac{\sqrt{\epsilon}M}{\gamma} < \frac{2}{\sqrt{3}} \approx 1.155$.

   **Theorem 4**: Proves that 2-Snapshots ARM generates a Cauchy sequence and converges to a stable point. This breaks the RRM lower bound of Theorem 2, demonstrating that leveraging historical data genuinely enables convergence over a broader problem class.

4. **ARM Lower Bounds (Theorems 5 & 6)**: Lower bounds on the achievable convergence rates for the ARM algorithm class are established.

   Under the Perdomo framework (Theorem 6): $\|\theta^t - \theta_{\text{PS}}\| = \Omega((\frac{\epsilon\beta}{2\gamma})^t)$. This implies that the RRM upper bound $(\frac{\epsilon\beta}{\gamma})^t$ and the ARM lower bound $(\frac{\epsilon\beta}{2\gamma})^t$ differ by only a **constant factor of 2**, precisely quantifying the maximum potential gain from leveraging historical data.

### Loss & Training

In practice, ARM is implemented at each step $t$ by uniformly mixing data from the most recent $\tau$ snapshots: $D_t = \frac{1}{\tau}\sum_{i=t-\tau+1}^{t} D(f_{\theta^i})$. The window size $\tau$ controls the trade-off between convergence speed and computational overhead.

## Key Experimental Results

### Main Results: Credit Scoring RIR Environment

| Window Size $\tau$ | Early Convergence Speed | $\Delta\mathcal{R}_t$ after 50 Steps | Notes |
|---|---|---|---|
| $\tau=1$ (RRM) | Baseline | ~0.02 | Standard RRM |
| $\tau=2$ | ~2× faster | ~0.01 | Loss shift halved |
| $\tau=4$ | ~3× faster | ~0.005 | Further improvement |
| $\tau=t/2$ | Fastest | ~0.003 | Near saturation |
| $\tau=\text{all}$ | ≈ $t/2$ | ~0.003 | Diminishing returns |

### Ablation Study: Lower Bound Verification

| Framework | Theoretical Lower Bound | Empirical Observation | Notes |
|---|---|---|---|
| Perdomo | $(\frac{\epsilon\beta}{\gamma})^t$ | Not broken for any $\tau$ | Validates tightness of Theorem 3 |
| ARM lower bound | $(\frac{\epsilon\beta}{2\gamma})^t$ | $\tau \geq 2$ breaks RRM upper bound but not ARM lower bound | Validates Theorem 6 |
| Improved Mofakhami | $(\frac{\sqrt{\epsilon}M}{\gamma})^t$ | Matches upper bound | Validates tightness of Theorem 1 |

### Key Findings

- Increasing $\tau$ from 1 to 2 yields the largest improvement (loss shift halved); marginal gains diminish thereafter.
- Convergence trajectories in all experiments remain above the theoretical lower bounds, validating the tightness analysis.
- Larger $\tau$ incurs linearly growing computational and storage costs, requiring a trade-off between speed and resources.
- The set of stable points for ARM is identical to that of RRM (Lemma 9); leveraging historical data does not alter the equilibrium, only accelerates convergence.

## Highlights & Insights

- **First tightness analysis**: The paper proves that existing RRM upper bounds cannot be further improved without additional assumptions — a clear impossibility result.
- **Elegant lower bound construction**: The dimension-growing matrix $A$ inspired by Nesterov's lower bounds is a clever construction that simultaneously satisfies all assumptions and witnesses the unimprovability of the convergence rate.
- **Clear conditions for breaking the lower bound**: ARM relaxes the convergence threshold from 1 to 1.155 by using two snapshots; while the improvement is modest in magnitude, its significance lies in breaking the RRM lower bound.
- The lower bound analysis technique generalizes to other iterative decision-dependent optimization settings (e.g., proximal ARM, Theorem 7).

## Limitations & Future Work

- The improvement in ARM's convergence threshold is limited (1 → 1.155); practical gains are primarily in convergence speed rather than an expansion of the problem class.
- A factor-of-2 gap remains between the RRM upper bound and the ARM lower bound; whether this gap can be closed is an open question.
- Experiments are conducted only on two semi-synthetic environments (credit scoring and ride-sharing markets), without large-scale real-world evaluation.
- Weights in the affine combination are currently assigned uniformly; adaptive weighting may yield better convergence.
- Finite-sample convergence guarantees are not discussed (the current analysis assumes access to exact distributional expectations at each step).
- In practical deployments, the cost of storing all historical snapshots may be prohibitive.

## Related Work & Insights

- Relationship to stateful performative prediction (Brown et al., 2022): the latter considers general history-dependent distributions, of which ARM is a special case (affine combinations).
- Analogy to Nesterov acceleration: ARM leverages historical information to accelerate convergence, analogous to momentum methods that exploit historical gradients.
- The lower bound construction technique is applicable to the analysis of other iterative learning frameworks (e.g., distribution shift in federated learning).
- Theorem 8 demonstrates that the revised $\epsilon$-sensitivity definition yields tighter constants under the RIR mechanism ($\delta^{-3/2}$ vs. $\delta^{-2}$).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The ARM idea is natural but formally introduced for the first time; the tightness analysis represents an important theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experimental setups are relatively simple (two semi-synthetic environments), though the theoretical results are sufficient to support the conclusions.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear; Figures 1–3 intuitively illustrate the core ideas.
- **Value**: ⭐⭐⭐⭐ — Makes a significant contribution to the performative prediction theory community; the lower bound technique has methodological value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting](tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)
- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/others/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[ICML 2026\] Optimal Regularization for Performative Learning](../../ICML2026/others/optimal_regularization_for_performative_learning.md)

</div>

<!-- RELATED:END -->
