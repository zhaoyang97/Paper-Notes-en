---
title: >-
  [Paper Note] Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems
description: >-
  [ICML2026][Image Generation][Zeroth-Order Sampling] This paper proposes a zeroth-order Langevin sampling method with variance reduction, utilizing intermittent large-batch estimation and recursive small-batch updates to…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Zeroth-Order Sampling"
  - "Non-Log-Concave Distribution"
  - "Variance Reduction"
  - "Langevin Monte Carlo"
  - "Inverse Problems"
date: 2026-05-08
content_hash: 03b5ba8460fb6517
---

# Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems

**Conference**: ICML2026  
**arXiv**: [2605.30573](https://arxiv.org/abs/2605.30573)  
**Code**: None  
**Area**: Optimization / Sampling Algorithms / Black-box Inverse Problems  
**Keywords**: Zeroth-Order Sampling, Non-Log-Concave Distribution, Variance Reduction, Langevin Monte Carlo, Inverse Problems  

## TL;DR
This paper proposes a zeroth-order Langevin sampling method with variance reduction, utilizing intermittent large-batch estimation and recursive small-batch updates to replace the $O(d)$ function queries per step. It further extends this to ZO-APMC, utilizing a pre-trained score-based prior to perform posterior sampling with convergence guarantees for black-box inverse problems requiring only forward model queries.

## Background & Motivation
**Background**: Sampling from an unnormalized density $\pi \propto \exp(-f)$ is a fundamental tool in machine learning, Bayesian inference, and inverse problems. If $\nabla f$ is accessible, Langevin Monte Carlo can iterate along the score of the target distribution; if only function values can be queried, the common practice is to construct zeroth-order gradient estimators using finite differences or random directions.

**Limitations of Prior Work**: Zeroth-order estimators suffer from high variance in high-dimensional spaces. Naive batched ZO-LMC typically requires a batch size that grows linearly with dimension $d$ to ensure gradient accuracy, which translates to massive forward model calls and memory overhead in high-dimensional problems like MRI, black-hole imaging, or PDE inversion. Furthermore, existing theories primarily cover strongly log-concave targets, whereas real posteriors corresponding to score-based generative priors are often non-log-concave and multimodal.

**Key Challenge**: Black-box inverse problems urgently require posterior sampling to express uncertainty, yet black-box forward operators often lack available gradients, pseudo-inverses, or differentiable implementations. Direct zeroth-order estimation is too costly, and heuristic black-box posterior samplers lack non-asymptotic convergence guarantees. This paper aims to simultaneously satisfy the requirements of "computational feasibility" and "theoretical convergence."

**Goal**: First, establish a non-asymptotic theory for zeroth-order sampling under non-log-concave targets. Second, design a variance-reduced estimator requiring only constant-level function queries per step. Third, embed this estimator into an annealed posterior sampler with SGM priors to enable sampling for black-box inverse problems such as MRI, black-hole imaging, and Navier-Stokes using only forward evaluations.

**Key Insight**: The authors borrow the concept of "stationary point analysis" from non-convex optimization, treating the relative Fisher information in sampling as the gradient norm of the KL divergence in Wasserstein space. That is, rather than enforcing log-concavity, they prove that the time-averaged distribution approaches the target distribution in terms of FI or TV distance.

**Core Idea**: Instead of recomputing zeroth-order gradients with large batches at every step, a large-batch estimate is refreshed occasionally. At other times, the fact that adjacent Langevin iterations are highly correlated is leveraged to estimate gradient changes using small batches.

## Method
The methodology consists of two layers: the base layer is variance-reduced ZO-LMC for sampling from non-log-concave targets, and the application layer is ZO-APMC, which combines this estimator with an annealed score-based prior to solve posterior sampling in black-box inverse problems.

### Overall Architecture
The underlying problem is that only the potential $f(x)$ can be queried, while $\nabla f(x)$ is inaccessible. A naive zeroth-order estimator computes $\tilde{\nabla} f_\mu(x,u)=((f(x+\mu u)-f(x))/\mu)u$ in a random direction $u \sim \mathcal{N}(0,I)$ for LMC updates. This paper replaces the gradient estimator rather than the LMC structure: a recursive $g_k$ estimates the gradient of the smoothed potential $\nabla f_\mu(x_k)$, followed by the update $x_{k+1}=x_k-\gamma g_k+\sqrt{2}(B_{(k+1)\gamma}-B_{k\gamma})$.

In inverse problems, observations satisfy $y=A(x)+\xi$, where $A$ is a black-box query. The posterior is the product of likelihood and prior; the likelihood score is estimated via ZO, while the prior score is approximated by a pre-trained SGM $S_\theta(x,\sigma)$. ZO-APMC uses both $g_k$ and a weighted score prior at each annealing step: $x_{k+1}=x_k-\gamma(g_k-\alpha_k S_\theta(x_k,\sigma_k))+$ Langevin noise.

### Key Designs
1. **Variance-Reduced Zeroth-Order Gradient Estimator**:
	- **Function**: Reduces function queries and memory costs per step of ZO-LMC while maintaining analyzable estimation error.
	- **Mechanism**: With probability $p$, a full zeroth-order estimate with batch size $b$ is computed; with probability $1-p$, the previous estimate $g_{k-1}$ is used alongside a very small batch $b'$ to estimate $\tilde{\nabla}f_\mu(x_k,u)-\tilde{\nabla}f_\mu(x_{k-1},u)$. Since adjacent Langevin iterates are usually close, the variance of this difference is smaller than direct gradient estimation.
	- **Design Motivation**: Naive ZO estimation suppresses variance by increasing batch size, which scales with dimension. Recursive differencing shifts from "re-estimating absolute gradients every step" to "tracking gradient changes," trading correlation for computational efficiency.

2. **Non-Log-Concave Sampling Analysis via Relative Fisher Information**:
	- **Function**: Characterizes convergence without assuming strong log-concavity.
	- **Mechanism**: $FI(\nu\|\pi)$ is viewed as the gradient norm of KL in Wasserstein space, analogous to using gradient norms to define stationary points in non-convex optimization. Theorem 1 proves that the time-averaged distribution reaches an $\varepsilon$-relative FI error in $O(d^7 L_m^4/\varepsilon^4)$ iterations, yet function queries per step remain $O(1)$. If the target satisfies a Poincare inequality, this can be converted into a squared TV distance guarantee.
	- **Design Motivation**: Non-log-concave distributions can be multimodal, rendering traditional strongly convex/log-concave analysis inapplicable. FI provides an intermediate metric that expresses "score alignment" and leads to TV convergence.

3. **ZO-APMC Black-Box Posterior Sampling**:
	- **Function**: Generates posterior samples using SGM priors in inverse problems with only forward model evaluations.
	- **Mechanism**: The likelihood is defined by the black-box forward operator and observations, with its gradient estimated by the variance-reduced ZO estimator. The prior score is provided by a pre-trained SGM at noise scale $\sigma_k$. An annealing schedule simultaneously reduces the prior smoothing level and the weight $\alpha_k$, allowing sampling to escape low-probability plateaus under a smooth prior before gradually emphasizing the true likelihood.
	- **Design Motivation**: SGM priors are powerful, but existing posterior samplers often assume access to forward gradients. ZO-APMC allows non-differentiable, closed-source, PDE simulators, or rule-based systems to be integrated into the same Bayesian reconstruction framework.

### Loss & Training
This work does not train new generative models; experiments use existing or custom-trained SGM priors. Key hyperparameters for the algorithm include step size $\gamma$, ZO smoothing $\mu$, refresh probability $p$, large/small batch sizes $b, b'$, and annealing schedules $\alpha_k=\max\{\alpha_0\rho_1^k,1\}$, $\sigma_k=\max\{\sigma_0\rho_2^k,\sigma_{min}\}$. In theory, these parameters are set to scale with the number of iterations and dimension to balance ZO estimation bias, estimation variance, and Langevin discretization error.

## Key Experimental Results

### Main Results
The authors first validate FI convergence with toy experiments, then compare black-box posterior samplers on MRI, black-hole imaging, and Navier-Stokes inversion. MRI and black-hole imaging provide complete quantitative tables. In Navier-Stokes, ZO-APMC is visually stable, with NRMSE close to DPG but not comprehensively surpassing EnKG/DPG.

| Task | Metric | ZO-APMC | Strongest Black-box Baseline | Gradient-available Reference | Conclusion |
|------|------|---------|-------------------|--------------|------|
| Toy FI convergence | FI | Approaches 0 when p=1/0.75/0.5; unstable at p=0.3 | N/A | APMC | Below 0.01 FI within 2000 iterations for multiple settings with fixed per-step cost $pb=10$ |
| MRI reconstruction | PSNR / SSIM / NRMSE | 35.29 / 0.966 / 2.28e-2 | DPG: 32.17 / 0.953 / 5.4e-2 | APMC: 36.55 / 0.973 / 1.99e-2 | Best overall among black-box methods, close to gradient-based APMC |
| Black-hole imaging | PSNR / blurred PSNR | 26.71 / 32.86 | Forward-GSG: 26.21 / 31.47 | APMC: 26.23 / 31.32 | ZO-APMC leads black-box baselines in PSNR and measurement consistency |
| Navier-Stokes inverse problem | NRMSE / qualitative flow | Close to DPG | EnKG / DPG | No gradient baseline | Does not optimize all metrics but provides clearer convergence explanation |

### Ablation Study
The core analysis focuses on the performance/cost trade-off of $p, b, b'$ rather than traditional module ablation. The table below summarizes quantitative comparisons in MRI and black-hole imaging, reflecting the effectiveness of VR-ZO estimation in high-dimensional settings.

| Scenario | Method | PSNR | SSIM / blurred PSNR | Error Metric | Description |
|------|------|------|---------------------|----------|------|
| MRI | PnPDM | 30.81 | SSIM 0.946 | MSE 8.46e-4 | Gradient-available baseline, but lower quality than ZO-APMC |
| MRI | DPS | 34.38 | SSIM 0.965 | MSE 4.07e-4 | Close to ZO-APMC, but still 0.91 dB lower |
| MRI | APMC | 36.55 | SSIM 0.973 | MSE 2.55e-4 | Gradient-based upper bound reference |
| MRI | ZO-APMC | 35.29 | SSIM 0.966 | MSE 3.29e-4 | Uses function queries only, close to APMC |
| Black-hole | Forward-GSG | 26.21 | blurred PSNR 31.47 | $\chi^2_{cph}=6.77$ | Strong black-box baseline |
| Black-hole | Central-GSG | 21.63 | blurred PSNR 23.73 | $\chi^2_{cph}=80.31$ | Central difference failed to provide stable improvement |
| Black-hole | ZO-APMC | 26.71 | blurred PSNR 32.86 | $\chi^2_{cph}=5.42$ | Best PSNR and measurement fitting |

### Key Findings
- There is a clear trade-off in the refresh probability of variance reduction: a $p$ that is too small reduces total function evaluations but exacerbates error propagation, with instability appearing at $p=0.3$ in toy experiments; $p=0.5$ is well-balanced.
- In MRI, ZO-APMC with $p=0.2, b=10^4, b'=10^3$ approaches the gradient-based APMC on 256x256 high-dimensional images, suggesting that while the theoretical $d^7$ complexity is conservative, it is practical.
- Nonlinear forward models in black-hole imaging better demonstrate the advantage of black-box methods: ZO-APMC does not rely on differentiable forward operators yet outperforms heuristic methods like GSG, DPG, and EnKG in closure phase/amplitude error.

## Highlights & Insights
- This paper ports variance reduction ideas from zeroth-order optimization to sampling but avoids simple replication; it specifically handles the coupling of Langevin discretization error and ZO smoothing bias, which is more complex in sampling than in optimization.
- Using relative Fisher information to bridge non-convex optimization and non-log-concave sampling is insightful. It allows the analysis of whether a sampling algorithm approaches the target distribution to mirror "approaching a stationary point," while retaining probabilistic meaning.
- The value of ZO-APMC lies in extending SGM priors from "inverse problem solvers requiring forward gradients" to truly black-box simulators/proprietary systems. This can be transferred to medical devices, climate/fluid simulations, industrial modeling, and closed-source physics engines.

## Limitations & Future Work
- Theoretical complexity has a $d^7$ dependence, which is very conservative; although empirical performance is better, a gap remains between theory and practice.
- Assumption 1 requires the potential function to be globally Lipschitz, which is unnatural for many common distributions. The authors acknowledge this condition is better suited for compact domains, normalization, or gradient clipping in practice.
- ZO-APMC does not consistently outperform EnKG/DPG on Navier-Stokes, indicating that convergence guarantees do not equate to empirical optimality in all tasks. Future work should systematically study performance under different forward operators, noise levels, and prior mismatches.
- High-dimensional image experiments still require large batches (e.g., $b=10^4, b'=10^3$ in MRI), which may become a bottleneck when deploying to expensive simulators.

## Related Work & Insights
- **vs Roy et al.'s ZO-LMC**: Early ZO-LMC primarily analyzed strongly log-concave targets and required batch sizes growing with dimension; this work shifts to non-log-concave targets and reduces average function queries per step to a constant level using variance reduction.
- **vs APMC / DPS / PnPDM**: These posterior samplers rely on forward model gradients or differentiable structures in many settings; ZO-APMC sacrifices some efficiency for applicability and convergence guarantees under truly black-box forward operators.
- **vs GSG / DPG / EnKG**: These black-box methods are more heuristic; while competitive in some tasks, ZO-APMC integrates hyperparameters, annealing, and sampling error into a unified theoretical framework.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines VR-ZO estimation, non-log-concave sampling theory, and SGM inverse problems distinctively.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers toy, MRI, black-hole imaging, and PDE inversion, though ablation focuses more on parameter sensitivity than large-scale simulator cost analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Clear theoretical line and sufficient motivation; dense formulas present a barrier for non-sampling experts.
- Value: ⭐⭐⭐⭐☆ Highly valuable for black-box Bayesian inverse problems, especially where forward models are non-differentiable or proprietary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](../../NeurIPS2025/image_generation/npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[AAAI 2026\] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes](../../AAAI2026/image_generation/constrained_particle_seeking_solving_diffusion_inverse_problems_with_just_forwar.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)

</div>

<!-- RELATED:END -->
