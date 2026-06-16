---
title: >-
  [Paper Note] Stability Analysis of Sharpness-Aware Minimization
description: >-
  [ICML 2026][Optimization & Theory][SAM] This paper analyzes the convergence instability of SAM near saddle points from a dynamical systems perspective. It proves that under deterministic gradient flow, a saddle point becomes an attractor for SAM if the neighborhood radius $\rho > -1/\lambda_1$. Furthermore, within a stochastic diffusion framework, it demonst
tags:
  - ICML 2026
  - Optimization & Theory
  - SAM
date: 2026-05-08
content_hash: d8fe62df3ce5faaf
---
# Stability Analysis of Sharpness-Aware Minimization

**Conference**: ICML 2026  
**arXiv**: [2301.06308](https://arxiv.org/abs/2301.06308)  
**Code**: None  
**Area**: Optimization / Training Dynamics  
**Keywords**: SAM, Saddle Point Escape, Dynamical Systems, Diffusion Equation, Momentum and Batch Size

## TL;DR
This paper analyzes the convergence instability of SAM near saddle points from a dynamical systems perspective. It proves that under deterministic gradient flow, a saddle point becomes an attractor for SAM if the neighborhood radius $\rho > -1/\lambda_1$. Furthermore, within a stochastic diffusion framework, it demonstrates that SAM's mean square displacement for saddle point escape is smaller than SGD's by $2\eta t^2|\lambda_j|^3\rho/B$. Finally, the SAM diffusion formula is used to explain why momentum and batch size are the true behind-the-scenes contributors to SAM's SOTA generalization performance.

## Background & Motivation
**Background**: "Flat minima" training frameworks, represented by Sharpness-Aware Minimization (SAM) proposed by Foret et al. (2020), have become standard for enhancing generalization in multiple fields including CIFAR, ImageNet, ViT, and NLP. The core of SAM involves taking a small step in the current gradient direction to obtain an adversarial perturbation $\bm{w}^p = \bm{w} + \rho \nabla \ell(\bm{w})$, then updating the original parameters using the gradient at the perturbed point $\bm{w}_{t+1} = \bm{w}_t - \eta \nabla \ell(\bm{w}_t^p)$. This forces the optimizer to find solutions where the worst-case loss in the neighborhood is small, thereby tending towards flat minima.

**Limitations of Prior Work**: The authors compared SAM and vanilla GD on the Beale function (a classic optimization test function); GD converged to the global minimum, while SAM stalled at a saddle point. Kaddour et al. (2022) also reported anomalous SAM behavior in certain settings. This "saddle point trapping" is more costly in highly non-linear deep learning loss landscapes where saddle points far outnumber minima. If SAM is truly captured by them, its continued success in large-scale experiments requires new explanations.

**Key Challenge**: The geometric motivation of sharpness encourages SAM to move toward directions where "loss within the neighborhood is small," which conflicts with the need to "quickly escape saddle points along unstable manifolds." Near a saddle point, the loss variation within a neighborhood is small and the worst-case is not large, leading SAM to perceive it as "flat" and stop. Conversely, GD, which only considers the current gradient, is carried out by tiny perturbations around the saddle point.

**Goal**: (1) Find precise conditions under which SAM treats saddle points as attractors under deterministic dynamics; (2) Quantitatively compare the saddle point escape speeds of SAM and SGD under stochastic diffusion; (3) Explain how momentum and batch size mitigate this instability to become the hidden keys to SAM's success.

**Key Insight**: Qualitative theory of dynamical systems is used to decompose SAM's trajectory near saddle points into Case-I, II, and III. The Lambda Lemma is then used to argue that Case-III inevitably oscillates between two basins of attraction. Subsequently, the analysis switches to the Fokker-Planck framework to approximate the SAM diffusion tensor using the Fisher Information Matrix, framing the escape problem as a mean square displacement problem.

**Core Idea**: The perturbation $\bm{w}^p$ might fall into the attraction basins of adjacent minima, causing SAM's update direction to flip repeatedly near the saddle point. When $\rho|\lambda_j|$ is sufficiently large, the escape force $\lambda_j$ along the negative Hessian eigenvalue direction is overridden by $\rho\lambda_j^2$ in the opposite direction. Geometrically, the saddle point transforms from a "hyperbolic unstable point" into a "stable attractor."

## Method

### Overall Architecture
This paper does not propose a new algorithm but provides a complete "pathological report" on SAM's stability. The paper proceeds in four steps: (a) Geometric analysis using Lambda Lemma and three cases to show why gradient oscillations occur; (b) Analytical conditions for saddle points becoming attractors via Hessian eigenvalues (Theorem 1); (c) Derivation of SAM's diffusion formula and proof of slower escape using Fokker-Planck equations (Theorem 2 + Corollary 1); (d) Incorporation of momentum and batch size into the diffusion formula to provide precise quantification of how they "save" SAM (Theorem 3). Finally, CIFAR-10/100 experiments serve as empirical validation for this theory.

### Key Designs

**1. Lambda Lemma Case Analysis + Saddle Point Attractor Condition (Theorem 1): Closed-form critical value for SAM trapping**

To explain why SAM stalls, one must observe its trajectory. Consider gradient flow $d\bm{w}/dt = -\nabla \ell(\bm{w})$ and an index-1 saddle point $\bm{d}$ between two adjacent minima $\bm{s}_1, \bm{s}_2$. Trajectories are categorized by distance from $\bm{d}$: Case-I (far from $\bm{d}$ and $W^s(\bm{d})$), where $-\nabla\ell(\bm{w}_t^p) \sim -\nabla\ell(\bm{w}_t)$, making SAM and GD behavior consistent; Case-II (near $\bm{d}$ but in $A(\bm{s}_1)$), following $W^u(\bm{d})$ per Lambda Lemma; Case-III is the pathological case—within the $\rho$-neighborhood, $\bm{w}_t^p$ may cross into $A(\bm{s}_2)$, causing $-\nabla\ell(\bm{w}_t^p)$ to point to $\bm{s}_2$, only to be pulled back to $A(\bm{s}_1)$, creating oscillation. Analytically, SAM's perturbation $\rho\nabla\ell$ rewrites the Hessian term from $\Lambda$ to $\Lambda + \rho\Lambda^2$. Since $\rho\lambda^2$ is always positive, it can flip negative eigenvalues. Theorem 1 gives the condition: for an index-1 saddle point with negative eigenvalue $\lambda_1 < 0$, if $\rho > -1/\lambda_1$ (equivalent to $\lambda_1 + \rho\lambda_1^2 > 0$), the saddle point is upgraded from an unstable point to an attractor. Numerical experiments on Beale and $f(x,y)=x^2-y^2$ replicate Case-III and this condition.

**2. Fokker-Planck Derivation of SAM Diffusion and Escape Speed Comparison (Theorem 2 + Corollary 1): SAM escapes slower than SGD**

Deterministic analysis cannot explain why SAM works in noisy SGD training. Writing SAM as an SDE $d\bm{w} = -\nabla\ell(\bm{w}^p)dt + [\eta C(\bm{w}^p)]^{1/2} dW_t$ and approximating noise covariance as $C(\bm{w})\approx \frac{1}{B}[H(\bm{w})]^+$, the variance along each eigen-direction is derived as:

$$\sigma_j^2(t) = \frac{\eta|\lambda_j|}{2B \lambda_j (1+\rho\lambda_j)^2}\Big[1 - \exp\big(-2\lambda_j(1+\rho\lambda_j)^2 t\big)\Big]$$

Subtracting the mean square displacement of SAM from SGD under a small time window $|\lambda_j|t \ll 1$ yields $\Delta_{SGD} - \Delta_{SAM} = 2\eta t^2 |\lambda_j|^3 \rho / B + \mathcal{O}(B^{-1}\eta t^3 \lambda_j^4)$. This difference is strictly positive and grows linearly with $\rho$. The mechanism is clear: the perturbation $\rho$ increases the denominator $(1+\rho\lambda_j)^2$, suppressing stochastic diffusion and weakening the noise's escape capability. The term $\rho|\lambda_j|^3/B$ proves that larger $\rho$, sharper Hessian, and larger batch size widen the gap.

**3. SAM Diffusion Formula with momentum + batch size (Theorem 3): Quantifying the hidden pillars**

Finally, momentum $\gamma$ and batch size $B$ are incorporated into the diffusion formula to investigate how they salvage saddle point escape. With momentum, the mean square displacement becomes:

$$\Delta_{SAM} = C_1 \frac{(1-e^{-C_2(1-\gamma)})^2}{(1-\gamma)^3 B} + C_3 \frac{(1-e^{-C_4/(1-\gamma)})}{(1-\gamma)B}$$

where $C_1=\eta^2|\lambda_j|/2$, $C_2=\eta/t$, $C_3=\eta|\lambda_j|/[2\lambda_j(1+\rho\lambda_j)^2]$, $C_4=2\lambda_j(1+\rho\lambda_j)^2 t$. Since $(1-\gamma)$ appears in the denominator up to the 3rd power, $\gamma\to 1^-$ significantly increases $\Delta_{SAM}$. Larger $B$ in the denominator also impacts escape. Crucially, $\rho$ decreases $C_3$, meaning larger $\rho$ requires larger $\gamma$ to compensate. Experiments on CIFAR-10 confirm this: with $B=512$, SAM fails ($\gamma=0$); $\gamma=0.9$ boosts SAM accuracy by 20%+, whereas it only boosts SGD by 5%.

### Loss & Training
The paper introduces no new loss functions. Experiments use standard Cross-Entropy (CIFAR) or MSE (toy NN). Theoretical derivations rely on (1) second-order Taylor expansion around $\bm{d}$, and (2) Fisher Information Matrix approximation $\frac{1}{N}\sum_i \nabla\ell_i \nabla\ell_i^T \approx [H]^+$.

## Key Experimental Results

### Main Results

| Experiment | Setting | GD/SGD Results | SAM Results | Interpretation |
|------|------|-------------|----------|------|
| Beale Function | $\eta=10^{-4}$, saddle point at $(0,1)$ | Converged to global min | Stalled at $(0,1)$ | Verifies Case-III and Theorem 1 |
| $f(x,y)=x^2-y^2$ | Starting point $(-3,-\epsilon), \epsilon=0.01$ | Escaped saddle point | Attracted to saddle point | $\lambda+\rho\lambda^2 > 0$ for $\{2,-2\}$ |
| Toy NN (Ziyin et al.) | Two-layer, single neuron, $\varphi(x)=x^2$ | Most seeds converged | Most seeds trapped | Verifies Theorem 2 |
| Increasing $\rho$ | Same toy NN | — | Average loss increases | Consistent with Corollary 1 |

### Ablation Study (CIFAR-10/100, ResNet-18, BN and augmentation disabled)

| Configuration | CIFAR-10 Test Results | Description |
|------|-------------------|------|
| SAM, $B=512$, $\gamma=0$ | train loss > 1, test acc < 60% | Failure with large batch + no momentum |
| SAM, decreasing $B$ | train loss $\downarrow$, test acc $\uparrow$ | Consistent with $\Delta_{SAM} \propto 1/B$ |
| SAM, increasing $\gamma$ | train loss $\downarrow$, test acc $\uparrow$ | Consistent with $(1-\gamma)^{-1}$ dependence |
| SGD vs SAM, $\gamma=0 \to 0.9$ | SGD +5%, SAM +20%+ | Much higher marginal gain for SAM |
| SAM, $\rho=0.1, \gamma=0.95$ (Full setup) | 95.08% (Best) | $\rho=0.5, \gamma=0$ only 86.20%, gap 5.79 |

### Key Findings
- SAM saddle trapping is not a niche case; it is verified across various functions and toy NNs, with the occurrence condition $\rho > -1/\lambda_1$.
- The escape gap $\Delta_{SGD}-\Delta_{SAM} \propto \rho|\lambda_j|^3/B$ explains why the combination of large $\rho$ and large $B$ fails.
- Momentum is practically a necessary condition for SAM; $\gamma=0.9$ increases SAM accuracy by 20%+, providing the first theoretical explanation for this empirical necessity.
- Larger $\rho$ requires larger $\gamma$ to maintain diffusion levels, guiding hyperparameter search.
- Oscillation of $\cos(\nabla\ell(\bm{w}_t), \nabla\ell(\bm{w}_t^p))$ between $-1$ and $1$ provides direct evidence for Case-III.
- The 5.79% Max-Min gap at $\rho=0.5$ confirms that aggressive SAM settings are highly sensitive to momentum selection.

## Highlights & Insights
- Decomposing "why SAM traps" into Case-III via the Lambda Lemma provides geometric rigor, supported by visualizations of cosine oscillations.
- Theorem 1's $\rho > -1/\lambda_1$ critical condition provides a direct design target for adaptive or normalized $\rho$ methods.
- Incorporating momentum into the diffusion formula and observing $(1-\gamma)^{-3}$ dependence provides the first theoretical explanation for the standard 0.9 momentum default.
- The closed-form difference $\Delta_{SGD}-\Delta_{SAM}$ is transferable and can be applied to analyze other SAM variants.

## Limitations & Future Work
- Theory relies on second-order Taylor expansion near points; global dynamics are extrapolated.
- Assumes first-order approximation for perturbations; does not account for ASAM/GSAM normalization.
- Scale is limited to CIFAR/ResNet-18; lacks ImageNet or LLM verification.
- Proves slower escape but does not identify settings where saddle point proximity might benefit generalization.
- Critical condition $\rho > -1/\lambda_1$ depends on difficult-to-estimate local Hessian information.
- No automated scheduling algorithm for $\rho/\gamma/B$ is proposed.

## Related Work & Insights
- **vs Compagnoni et al. 2023**: This work uses a dynamical systems analysis for trajectory driving forces, complementing noise smoothing perspectives.
- **vs Andriushchenko & Flammarion 2022**: Quantifies the empirical "small batch benefit" via $\Delta_{SAM} \propto 1/B$.
- **vs Kaddour et al. 2022**: Provides the first systematic theoretical characterization of previously reported anomalies.
- **vs Bartlett et al. 2023 / Chen et al. 2023**: Complements "bouncing" and "transient attraction" theories by identifying permanent trapping as a failure mode solved by momentum.
- **vs Long & Bartlett 2024**: The critical condition $\rho > -1/\lambda_1$ acts as a SAM-specific counterpart to Edge-of-Stability analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](cost-aware_stopping_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
