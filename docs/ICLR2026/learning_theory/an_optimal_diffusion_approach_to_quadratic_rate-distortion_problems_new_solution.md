---
title: >-
  [Paper Note] An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods
description: >-
  [ICLR2026][Learning Theory][Rate-distortion function] This paper reformulates the calculation of the Rate-Distortion (RD) function for continuous sources under MSE distortion as a "Terminal-Entropy Regulated Stochastic Control" problem. It proves that the rate-distortion tradeoff is equivalent to the control energy-terminal entropy tradeoff and identifies that under regularity conditions, the optimal control is precisely the Stein score of the solution to the backward heat eq…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Information Theory"
  - "Rate-Distortion"
  - "Stochastic Control"
  - "Diffusion Processes"
  - "Rate-distortion function"
  - "Entropy-regularized optimal transport"
  - "Schrödinger bridge"
  - "Backward heat equation"
  - "Diffusion estimation"
date: 2026-05-08
content_hash: fde2e7fd0cabc902
---

# An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=upReXsENIl](https://openreview.net/forum?id=upReXsENIl)  
**Code**: https://github.com/ML-group-il/r2d2  
**Area**: Learning Theory / Information Theory / Rate-Distortion / Stochastic Control / Diffusion Processes  
**Keywords**: Rate-distortion function, Entropy-regularized optimal transport, Schrödinger bridge, Backward heat equation, Diffusion estimation

## TL;DR
This paper reformulates the calculation of the Rate-Distortion (RD) function for continuous sources under MSE distortion as a "Terminal-Entropy Regulated Stochastic Control" problem. It proves that the rate-distortion tradeoff is equivalent to the control energy-terminal entropy tradeoff and identifies that under regularity conditions, the optimal control is precisely the Stein score of the solution to the backward heat equation. This yields new closed-form solutions for sources like Gaussian mixtures and introduces R2D2, a diffusion neural estimator free from rate upper bound constraints.

## Background & Motivation
**Background**: Rate-distortion theory characterizes the minimum bitrate required to encode a continuous source given an allowable average distortion $D$: $R(D)=\min_{p_{\hat X|X}:\,D(\hat X,X)\le D} I(X;\hat X)$. While the classical Blahut–Arimoto (BA) algorithm iteratively solves this for discrete alphabets, efficiently computing $R(D)$ for **continuous sources** remains an open challenge.

**Limitations of Prior Work**: Closed-form RD solutions are known only for a few standard cases (e.g., Gaussian + MSE). Recent works like NERD and WGD leveraged the connection between BA and **entropy-regularized optimal transport (EOT)** but modeled reconstruction distributions as discrete atoms or finite particle sets. Consequently, the **bitrate is limited by batch/support size**: $R < \log M$. In **low-distortion (high-bitrate) regimes**—critical for modern high-bandwidth communications—these methods hit a ceiling at approximately 13 nats.

**Key Challenge**: To accurately compute RD and obtain the reconstruction distribution in continuous space, discrete approximations are inherently rate-limited, while direct optimization of the test channel lacks a computable analytical structure.

**Key Insight**: The authors observe that EOT is equivalent to a classic stochastic control problem—the **Schrödinger Bridge (SB)**. SB uses a minimum-energy drift $u$ to drive an initial distribution $P_0$ to a target $P_1$ under a noisy SDE $dX_t=u\,dt+\sqrt{\epsilon}\,dW_t$. Since modern diffusion generative models are essentially such finite-energy processes, the "diffusion" of a source to its distortion-optimal reconstruction should be expressible in the same framework.

**Core Idea**: Rewrite the RD problem as a **target-distribution-free** stochastic control problem—instead of fixing the terminal distribution, a penalty is placed on the "uncertainty of the terminal state (differential entropy)." This is termed **Terminal-Entropy Control (TEC)**. The authors prove it is strictly equivalent to the RD problem under MSE, translating "rate vs. distortion" into a "control energy vs. terminal entropy" tradeoff that is amenable to both analytical and numerical optimization.

## Method

### Overall Architecture
The paper revolves around an equivalence chain: **RD Lagrangian $\rightarrow$ Entropy-regularized OT $\rightarrow$ Schrödinger Bridge Control $\rightarrow$ Terminal-Entropy Control (TEC)**.

Starting from the entropy decomposition of the RD Lagrangian (where mutual information is expanded as $I = H(P_1) + H(P_0) - H(\pi)$), using quadratic cost $d(\hat x,x)=\tfrac12\|\hat x-x\|^2$ and average distortion $D=\tfrac12\mathbb{E}\|X-\hat X\|^2$, one obtains:

$$L_{RD}(P_0, \epsilon) = \min_{P_1} \min_{\pi \in \Pi(P_0, P_1)} \Big( \tfrac{1}{2\epsilon} \!\int\! \|\hat x - x\|^2 d\pi - H(\pi) + H(P_1) \Big) + H(P_0).$$

The inner coupling $\pi$ is realized via a **finite-energy diffusion trajectory** $dX_t = u(X_t, t)dt + \sqrt{\epsilon} dW_t$. Based on classical results by Léonard and Pavon–Wakolbinger, the KL cost of the optimal trajectory equals the control energy $\tfrac{1}{2\epsilon}\mathbb{E}\!\int_0^1\|u\|^2dt$. Substituting this back replaces RD with a surrogate objective for drift $u$ and terminal distribution $P_1$. Relaxing the fixed terminal distribution constraint in favor of a terminal entropy penalty yields TEC—a **target-free stochastic control problem with terminal entropy regularization**. This transforms the combinatorial optimization of RD into a control problem with PDE structures (Fokker–Planck / Backward Heat Equation).

### Key Designs

**1. Terminal-Entropy Control (TEC): Reformulating RD as an Energy-Entropy Tradeoff**

To address the lack of computable structures in continuous RD, the authors propose TEC:

$$\inf_{u \in \mathcal U} \ \tfrac12 \mathbb{E} \!\int_0^1 \|u(X_t, t)\|^2 dt + \epsilon H(X_1) \quad \text{s.t. } X_0 \sim P_0, \ P_{X_1} \text{ free}, \ dX_t = u \, dt + \sqrt\epsilon dW_t.$$

Intuitively, $u$ is a controller that **reduces terminal uncertainty while minimizing energy**. The energy term corresponds to the rate, and the terminal entropy corresponds to the "spread" of the reconstruction. $\epsilon$ is the Lagrange multiplier. Theorem 3.1 proves that under assumption A1, $L_{RD} = \tilde L_{RD}$, and the optimal terminal distribution $P_{X_1^*}$ from TEC is precisely the optimal reconstruction distribution.

**2. Optimal Control = Stein Score of the Backward Heat Equation (BHE) Solution**

With the variational form (var-TEC) and the Fokker–Planck equation $\partial_t p_t = -\nabla \cdot (p_t u) + \tfrac{1}{2\epsilon} \Delta p_t$, Theorem 3.2 provides an elegant result: if $p_t^*$ satisfies the **Backward Heat Equation (BHE)**

$$\partial_t p_t^*(x) = -\tfrac{1}{2\epsilon} \Delta_{xx} p_t^*(x), \qquad p_0^* \sim P_0,$$

then the optimal control is $u^* = \epsilon \nabla \log p_t^*(x)$, i.e., the (scaled) **Stein score function**. This reduces the control problem to "solving a PDE + taking the log-gradient," mirroring "score-based learning" in diffusion models.

**3. New Closed-form Solutions via Fourier Analysis**

Theorem 3.2 allows the derivation of previously unknown closed-form reconstruction distributions. Since BHE is linear, for a **Gaussian Mixture Model (GMM)** source $p_0 = \sum_i \tfrac{p_i}{\sqrt{2\pi\sigma_i^2}} e^{-(x-\mu_i)^2/2\sigma_i^2}$, the solution is a superposition:

$$p_{t}(x)=\sum_{i=1}^N \frac{p_i}{\sqrt{2\pi(\sigma^2_i-\epsilon t)}}\,e^{-\frac{(x-\mu_i)^2}{2(\sigma^2_i-\epsilon t)}}, \qquad \epsilon \in (0, \min_i \sigma_i^2).$$

The optimal control $u = \epsilon \nabla \log p_t$ is then determined. For **non-Gaussian mixtures** (e.g., band-limited $\mathrm{sinc}^4$ mixtures), frequency domain analysis yields $p_t(x) = \tfrac{1}{2\pi} \int e^{i\omega x + \frac{1}{2}\epsilon\omega^2 t} \hat p(\omega) d\omega$.

**4. R2D2: Diffusion Neural Estimator without Rate Upper Bound**

In practice, source distributions are represented by samples. The authors propose **R2D2 (Revealing RD functions with Diffusion)**: a DNN $u_\theta(x, t, \epsilon)$ models the controller. Since $\epsilon$ is an input, a **single model covers multiple points on the RD curve**. Training uses Euler–Maruyama simulation and optimizes the TEC surrogate loss $L_\theta^\epsilon = \tfrac{1}{2M} \sum_m \sum_{t_i} \|u_\theta\|^2 \Delta t + \epsilon \hat H(X_1)$, where terminal entropy $\hat H(X_1)$ is estimated via negentropy approximations or kernel methods. Unlike NERD/WGD, R2D2 has **no theoretical upper bound on bitrate**, making it ideal for the low-distortion regime.

### Loss & Training
The core objective is the TEC surrogate loss: the energy term $\tfrac{1}{2M} \sum_m \sum_{t_i} \|u_\theta(X_{t_i}^m, t_i, \epsilon)\|^2 \Delta t$ plus terminal entropy $\epsilon \hat H(X_1)$. During training, $\epsilon$ is sampled uniformly, trajectories are generated via discrete SDE, and $\theta$ is updated via backpropagation.

## Key Experimental Results

### Main Results
R2D2 was compared against EOT-based estimators NERD and WGD on synthetic and real sources.

| Source | Dim | Regime of Interest | R2D2 Performance | Baseline Performance |
| :--- | :--- | :--- | :--- | :--- |
| 1-D Gaussian | 1 | High + Low Rate | Lowest estimation error | Larger errors in NERD/WGD |
| GMM ($N=3$) | 1 | $\epsilon \in [4\times10^{-4}, 1.64\times 10^{-2}]$ | Matches closed-form Eq.(22) | — |
| CIFAR10 $4\times4$ patches | 16 | Full Range | Provides full RD curve | Benchmark comparison |
| Free Spoken Digit | 33 | Low Distortion | Reaches $>20$ nats | Capped at $\sim 13$ nats |

On the 33-dimensional speech feature set, NERD and WGD were constrained by $R < \log M$, capping their bitrate. R2D2 successfully estimated bitrates exceeding 20 nats.

### Key Findings
- **R2D2 Dominates the Low-Distortion Regime**: Discrete/atomic baselines hit a structural ceiling at high bitrates, whereas R2D2's continuous modeling excels.
- **Theoretical-Empirical Consistency**: For GMM sources, the empirical reconstruction distribution matches the closed-form BHE solution.
- **Multi-point Single Model**: Inputting $\epsilon$ into the network allows one controller to cover the entire RD curve.

## Highlights & Insights
- **Information Theory as PDE**: Using the backward heat equation to characterize optimal control and Stein scores creates a seamless link with the diffusion model paradigm.
- **Energy-Entropy Perspective**: While RL often maximizes policy entropy for exploration, this work penalizes **terminal entropy**, a novel contribution to control theory.
- **Closed-form Breakthroughs**: First-ever closed-form or semi-analytical RD results for GMMs and band-limited mixtures.
- **Removing the Rate Ceiling**: Diffusion drift optimization removes the $R < \log M$ bottleneck inherent in atomic modeling.

## Limitations & Future Work
- **Constraint to MSE/Quadratic Cost**: The equivalence chain relies on MSE to map to quadratic OT and SB.
- **Breakdown of A1 at Low Rates**: At very high $\epsilon$ (low rates), the optimal reconstruction distribution may become singular, violating continuity assumptions.
- **Ill-posedness of BHE**: The backward heat equation is generally unstable, limiting analytical results to specific well-behaved sources.
- **Terminal Entropy Estimation**: In high dimensions, the bias and variance of entropy estimators (kernel methods, etc.) remain a challenge.

## Related Work & Insights
- **vs. NERD / WGD**: These use EOT but are limited by discrete support size. R2D2 optimizes continuous drift, allowing unbounded bitrates.
- **vs. Schrödinger Bridge Solvers**: While existing SB solvers fix both distributions, TEC relaxes the target distribution into an entropy penalty.
- **vs. Diffusion Lossy Compression**: Unlike works using reverse diffusion for actual compression, R2D2 uses fixed-time forward diffusion to estimate the theoretical RD limit.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](a_new_approach_to_controlling_linear_dynamical_systems.md)
- [\[ICLR 2026\] Bi-Criteria Metric Distortion](bi-criteria_metric_distortion.md)
- [\[ICLR 2026\] Diffusion Language Models are Provably Optimal Parallel Samplers](diffusion_language_models_are_provably_optimal_parallel_samplers.md)
- [\[ICLR 2026\] The Effect of Attention Head Count on Transformer Approximation](the_effect_of_attention_head_count_on_transformer_approximation.md)
- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)

</div>

<!-- RELATED:END -->
