---
title: >-
  [Paper Note] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions
description: >-
  [ICML 2026][Interpretability][Structural Identifiability] This paper provides the first structural identifiability theorem for identifying the parameters $(\gamma_1…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Structural Identifiability"
  - "Second-Order ODE"
  - "Decoder-Free"
  - "Level-Set Slope Coverage"
  - "Variance Lower Bound Regularization"
date: 2026-05-08
content_hash: 2b4775fb7204f3b0
---

# Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions

**Conference**: ICML 2026  
**arXiv**: [2606.00115](https://arxiv.org/abs/2606.00115)  
**Code**: https://github.com/wenjiewang3/PhysicsFromVideo (Yes)  
**Area**: Video Physical Parameter Identification / Causal Representation Learning / Scientific Machine Learning  
**Keywords**: Structural Identifiability, Second-Order ODE, Decoder-Free, Level-Set Slope Coverage, Variance Lower Bound Regularization

## TL;DR
This paper provides the first structural identifiability theorem for identifying the parameters $(\gamma_1, \gamma_0)$ of second-order linear ODEs from raw video using only an encoder (without a decoder or pixel reconstruction). It characterizes the boundary between "single trajectory sufficiency" and "three trajectories necessity" using a geometric condition called **level-set slope coverage**. It proves that underdamped systems can be identified from a single video, while other damping regimes require three distinct trajectories. The authors further propose a finite-sample estimator combining "variance lower bound regularization" and "central difference."

## Background & Motivation
**Background**: Current video world models (e.g., Sora-like models, Physics IQ benchmarks) strive for pixel-level realism. However, evaluations like Physics-IQ demonstrate that they can generate videos that "look right" but are "physically wrong," exposing the decoupling between visual realism and physical correctness. Using a camera as a low-cost non-contact physical sensor to invert physical parameters (e.g., spring constants, damping ratios, pendulum length) from video has become a complementary research path.

**Limitations of Prior Work**: Mainstream methods fall into two categories: (1) autoencoders combined with differentiable simulation, relying on pixel reconstruction loss to learn dynamics; (2) fully decoder-free methods that impose physical constraints in the latent space (e.g., LPFV, Garcia 2025). The issue with the first category is that a sufficiently strong decoder can fit low pixel loss using incorrect physical parameters combined with compensatory appearance textures, leading to **parameter non-uniqueness**. The second category eliminates the decoder but leaves a deeper problem: the latent coordinate system is only defined up to an **arbitrary $C^2$ reparameterization** $f$. Even if the ODE residual drops to 0, the recovered $(\hat\gamma_1, \hat\gamma_0)$ may not equal the true values.

**Key Challenge**: The encoder-only setup lacks a theoretical guarantee—under what conditions does the data itself "lock" the latent coordinate system as an affine function of the true physical state? Without this, all decoder-free methods lack an identifiability foundation.

**Goal**: To answer three questions for the cleanest non-trivial model—homogeneous second-order linear time-invariant ODEs $z''(t)+\gamma_1 z'(t)+\gamma_0 z(t)=0$: (i) when a single video segment is sufficient to uniquely recover $(\gamma_1, \gamma_0)$; (ii) when multiple trajectories are required; and (iii) the non-asymptotic upper bound of estimation error in discrete noisy scenarios.

**Key Insight**: The authors observe that structural identifiability ultimately boils down to whether the latent reparameterization $f$ is forced to be an affine function. Forcing $f$ to be affine requires the trajectory to **repeatedly pass through the same physical state value but with different instantaneous velocities**—a purely geometric/dynamical condition that can be verified from the raw trajectory.

**Core Idea**: The concept of "locking the latent space" is translated into the "level-set slope coverage" condition. If every state level $u$ is traversed by at least 3 different instantaneous velocities $z'$, any $C^2$ reparameterization $f$ that simultaneously satisfies two sets of second-order ODEs must be affine. Therefore, $(\hat\gamma_1, \hat\gamma_0)$ must equal $(\gamma_1, \gamma_0)$.

## Method

### Overall Architecture
The pipeline is entirely encoder-only: each frame $\boldsymbol{x}_k$ passes through a shared CNN encoder $E_\phi$ to obtain a scalar latent $\hat{z}_k$. First and second-order derivatives $\hat{z}'_k, \hat{z}''_k$ are calculated using a **central difference** stencil. These are substituted into the ODE residual $r_k = \hat{z}''_k + \gamma_1 \hat{z}'_k + \gamma_0 \hat{z}_k$, and the sum of squares is used as the data-fitting loss $\mathcal{L}_{\mathrm{ODE}}$. Simultaneously, a variance lower bound regularization $\mathcal{L}_{\mathrm{var}}$ is applied to $\{\hat{z}_k\}$ to prevent latent collapse. Finally, a "UNIQUENESS CHECK" diagnostic box online determines whether the current clip satisfies the coverage condition, providing a CERTIFIED label. Input: $T+1$ video frames; Output: $(\hat\gamma_1, \hat\gamma_0)$ and whether it is certified as identifiable.

### Key Designs

1. **Level-Set Slope Coverage Geometric Condition (Core Theory)**:
    - **Function**: Translates the abstract notion of "latent reparameterization $f$ being forced to affine" into a geometric condition verifiable on the raw trajectory.
    - **Mechanism**: Coverage is defined on an open interval $U \subset \mathcal{R}_z$ if for every level $u \in U$, there exist at least three time points $t_1, t_2, t_3$ such that $z(t_i)=u$ and $z'(t_1), z'(t_2), z'(t_3)$ are **mutually distinct**. Theorem 4.2 proves that under the state consistency assumption $\hat{z}(t)=f(z(t))$, if the same $f$ allows both $z$ and $\hat{z}$ to satisfy second-order ODEs, coverage forces $f$ to be affine on $U$, thus $(\eta_1, \eta_0) = (\gamma_1, \gamma_0)$.
    - **Design Motivation**: Existing decoder-free works fail to address the "uniqueness of the latent coordinate system." This paper provides an answer with a **purely geometric** condition that does not depend on specific network architectures but only on the trajectory itself. Accompanying Theorem 4.3 gives a sufficient condition for "underdamped + window length $L \geq 2P=4\pi/\sqrt{\gamma_0-\gamma_1^2/4}$" for single trajectory identification, Theorem 4.5 shows single trajectories necessarily fail for critical/over/undamped cases, and Theorem 4.6 shows "three trajectories with different velocities" are sufficient. This set of results characterizes the **minimum data requirements** for different damping regimes for the first time.

2. **Central Difference Residual + Variance Lower Bound Regularization**:
    - **Function**: Maps continuous-time ODE residuals to discrete video frames while preventing the trivial latent collapse solution $\hat{z}_k \equiv 0$.
    - **Mechanism**: Uses central differences $\hat{z}'_k = (\hat{z}_{k+1}-\hat{z}_{k-1})/(2\Delta t)$ and $\hat{z}''_k = (\hat{z}_{k+1}-2\hat{z}_k+\hat{z}_{k-1})/\Delta t^2$ to replace the Euler/one-sided differences in LPFV, reducing the $\Delta t$ bias of the residual from first-order to second-order. The variance regularization is defined as $\mathcal{L}_{\mathrm{var}}=(\max\{0, \tau-\sqrt{\widehat{\mathrm{Var}}(\hat{z})+\varepsilon}\})^2$, which penalizes the latent only when the standard deviation is below the threshold $\tau$. It does not force a specific distribution shape.
    - **Design Motivation**: Pure $\mathcal{L}_{\mathrm{ODE}}$ has a degenerate optimal solution at $\hat{z}\equiv 0$. LPFV uses KL-to-$\mathcal{N}(0,1)$, but trajectories in non-oscillatory regions are strongly non-Gaussian and non-stationary, causing KL to distort the representation. The variance lower bound only manages scale, matching the "minimum eigenvalue of the design matrix $\psi_{\min}>0$" condition in finite-sample error bounds. Lemma D.1 proves the variance lower bound directly provides a checkable lower bound for $\psi_{\min}$.

3. **Non-asymptotic Finite-Sample Error Bound**:
    - **Function**: Provides a non-asymptotic upper bound for $\|\hat\eta-\gamma\|_2$ in discrete, noisy scenarios where the encoder is not strictly affine, decomposing three error sources.
    - **Mechanism**: Theorem 4.8 provides $\|\hat\eta-\gamma\|_2 \leq \frac{C_1\sigma}{\psi_{\min}}\sqrt{\log(3/\delta)/(T-1)} + \frac{C_2\sigma^2}{\psi_{\min}} + \frac{C_3\Delta t^2}{\psi_{\min}} + E_{\mathrm{enc}}$ with $1-\delta$ confidence. The terms represent: statistical error of sub-Gaussian noise $O(\sqrt{\log/T})$, noise second-order terms, central difference discretization bias $O(\Delta t^2)$, and deterministic mismatch $E_{\mathrm{enc}}$ from the encoder deviating from affine.
    - **Design Motivation**: Allows users to determine how long a video, how small a $\Delta t$, or how accurate an encoder is needed to suppress error to a certain level. A multi-clip pooling version (Appendix B.2) replaces $T-1$ with $N=\sum_m (T^{(m)}-1)$ and improves $\psi_{\min}$ through cross-trajectory velocity diversity.

### Loss & Training
The total objective is $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{ODE}} + \lambda_{\mathrm{var}} \mathcal{L}_{\mathrm{var}}$, with default values $\lambda_{\mathrm{var}}=1.0$, $\tau=1$, and $(\gamma_1, \gamma_0)$ initialized from $(1, 1)$. The encoder is a shared per-frame CNN, with results averaged over 5 random seeds. The multi-clip version calculates and averages variance regularization for each clip separately.

## Key Experimental Results

### Main Results
Identifiability theory was verified on synthetic pendulum videos across 4 damping regimes:

| System / Damping Zone | Ground Truth $(\gamma_0, \gamma_1)$ | Estimated $\hat\gamma_0$ | Estimated $\hat\gamma_1$ | Coverage |
|---|---|---|---|---|
| Pendulum Underdamped | (4.0016, 0.08) | 4.0037±0.0003 | 0.0800±0.0003 | Single clip satisfied (L≥2P) |
| Pendulum Undamped | (4, 0) | 4.0032±0.0010 | 0.0002±0.0004 | Identifiable under discrete sampling |
| Pendulum Critical (3 clips coverage+) | (4, 4) | 4.0218±0.0040 | 4.0352±0.0017 | 3 velocity-diverse trajectories |
| Pendulum Overdamped (3 clips coverage+) | (4, 5) | 3.9733±0.0033 | 4.9723±0.0010 | 3 velocity-diverse trajectories |
| Pendulum Critical (3 clips coverage–) | (4, 4) | 2.3462±0.0437 | 2.7642±0.0787 | Fail, verifies Thm 4.6 necessity |
| Intensity Sys. Underdamped | (4.0016, 0.08) | 4.007±0.007 | 0.088±0.004 | Holds for non-motion scenarios |
| Real Pendulum $L^\star=0.9$m | 0.90m | — | RMSE: Ours ≪ PAIG (0.116) | Successfully recovered length |

### Ablation Study

| Reg. Method / Damping Zone | $\hat\gamma_0$ (Truth 4) | $\hat\gamma_1$ (Varies) | Conclusion |
|---|---|---|---|
| No Reg / Underdamped | -0.0004±0.0009 | 0.0425±0.1194 | Collapses to $\hat{z}\equiv 0$ |
| KL-$\mathcal{N}(0,1)$ / Underdamped | 4.0037±0.0009 | 0.0798±0.0003 | OK for oscillatory region |
| **Var Lower Bound / Underdamped** | **4.0037±0.0003** | **0.0800±0.0003** | Comparable to KL |
| KL-$\mathcal{N}(0,1)$ / Critical | 9.2659±1.0532 | 6.2857±0.4848 | **Total failure** |
| **Var Lower Bound / Critical** | **4.0218±0.0040** | **4.0352±0.0017** | Only successful method for non-osc. |
| KL-$\mathcal{N}(0,1)$ / Overdamped | 6.8248±0.1909 | 7.3831±0.0906 | Failure |
| **Var Lower Bound / Overdamped** | **3.9733±0.0033** | **4.9723±0.0010** | Still accurate |

### Key Findings
- **Alignment of Theory and Empirics**: In the underdamped case, the first coverage-positive window appears at $1.1\pi$ (smaller than the sufficient condition $2\pi$), perfectly coinciding with the inflection point where parameter recovery becomes accurate, verifying that "coverage is the true cause."
- **Discrete Estimators are More Lenient**: While undamped systems are structurally unidentifiable in continuous time (Thm 4.5), central difference regression under discrete sampling has a unique zero-loss target at $(0, \gamma_0 + O(\Delta t^2))$ (Prop C.1). Long clips can thus still approximate recovery—a "benign surprise" of finite samples.
- **Variance Lower Bound is Mandatory for Non-oscillatory Regions**: KL regularization estimates $\hat\gamma_0$ between 6-9 in critical/overdamped zones because non-oscillatory trajectories are inherently non-Gaussian. Forcing a match to $\mathcal{N}(0,1)$ distorts the design matrix. The variance lower bound only manages scale, precisely matching the $\psi_{\min}$ condition in the error bound.
- **Generality Beyond Motion**: Synthetic videos using state to control grayscale intensity or circle radius also achieved accurate identification (4.007±0.007) underdamped, showing identifiability comes from trajectory geometry rather than motion cues.
- **Real-World Pendulums**: In real phone-captured videos of three cord lengths (0.45/0.90/1.50m), the proposed method's RMSE is significantly lower than PAIG (PAIG biased to ~1.01m across all lengths).

## Highlights & Insights
- **Turning Identifiability into a Geometric Condition**: Level-set slope coverage is a geometric condition that can be **visualized** and verified on raw trajectories (Fig. 3 in the paper uses phase portrait markers). It does not depend on network architecture or gradient computation and serves as an online diagnostic—decoupling "reliability" from "goodness of fit."
- **Physical Intuition for "Why 3 Velocities"**: A second-order ODE is a 2D submanifold in the $(z, z')$ plane. To infer an affine mapping from the latent space, a third independent dimension is required; velocity diversity provides this. This insight likely generalizes: an $n$-th order ODE probably requires $n+1$ independent velocities/accelerations to lock the latent space.
- **The Variance Lower Bound is an Underrated Trick**: Compared to KL, it only constrains a single scalar (std) but corresponds to the minimum eigenvalue of the design matrix—a precise design that unifies "anti-collapse" with "statistical efficiency."
- **Elegant Handling of Sufficient vs. Necessary Conditions**: The authors point out that $L\geq 2P$ is only sufficient; the empirical result $1.1\pi < 2\pi$ being coverage-positive helps the reader understand the gap between theoretical guarantees and practical experience.

## Limitations & Future Work
- **Limited to Homogeneous Second-Order Linear ODEs**: Appendix E extends this to non-homogeneous constant terms $g$, but there is no theory for non-linear, higher-order, or PDEs. Generalizing coverage to vector-valued states is an obvious next step.
- **Strong State Consistency Requirement**: It requires a $C^2$ function $f$ such that $\hat{z}(t)=f(z(t))$ holds strictly, which essentially excludes real videos with strongly time-varying lighting or backgrounds. Stress tests show it handles moderate perturbations, but strong ones break identifiability.
- **Per-frame Encoder**: Not utilizing temporal structure (e.g., optical flow, 3D-conv) makes it sensitive to noise. Maintaining identifiability while using temporal encoders is an open problem.
- **Computability of Coverage Diagnosis**: Continuous-time coverage can be computed precisely in synthetic experiments, but in real videos, the "true $z(t)$" itself must be estimated, requiring more clarity on the diagnostic box's practical utility.

## Related Work & Insights
- **vs LPFV (Garcia et al. 2025)**: Architecturally identical (encoder-only), but LPFV lacks identifiability theory. This paper provides that foundation while replacing Euler differences ($O(\Delta t)$) with central differences ($O(\Delta t^2)$) and KL with variance lower bounds.
- **vs reconstruction-driven (PAIG, etc.)**: Their identifiability depends on the inductive bias of decoders/renderers; parameters can be compensated by appearance nuisances. Ours is decoupled from appearance. PAIG's failure on varying real cord lengths is evidence of such "shortcuts."
- **vs Classical System Identification**: They usually assume direct observation of state $z$ or a fixed observation model. This paper must infer the latent from pixels—latent reparameterization ambiguity is unique to physics-from-video, and the coverage condition is the first clean answer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The **first** structural identifiability theorem for encoder-only video physical parameter identification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive verification across 4 damping zones and multiple scenarios, though real-world data is limited to pendulums.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure bridging theory, discrete estimators, and finite-sample bounds.
- Value: ⭐⭐⭐⭐⭐ Provides the missing theoretical foundation for decoder-free physics-from-video research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)
- [\[ICML 2026\] Universal 1/3 Time Scaling in Learning Peaked Distributions](universal_one-third_time_scaling_in_learning_peaked_distributions.md)
- [\[ICML 2026\] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](position_zeroth-order_optimization_in_deep_learning_is_underexplored_not_underpo.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)

</div>

<!-- RELATED:END -->
