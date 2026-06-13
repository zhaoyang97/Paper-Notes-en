---
title: >-
  [Paper Note] Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces
description: >-
  [ICML 2026][Text Generation][MCMC] SRMC uses a $d$-dimensional running score average (rather than an $|\mathcal{X}|$-dimensional empirical measure) to record history. This history is transformed via exponential score-til…
tags:
  - "ICML 2026"
  - "Text Generation"
  - "MCMC"
  - "Non-Markovian sampling"
  - "score-tilt"
  - "self-repulsion"
  - "stochastic approximation CLT"
date: 2026-05-08
content_hash: 107c85bbfac1e759
---

# Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2604.22948](https://arxiv.org/abs/2604.22948)  
**Code**: TBD  
**Area**: Scientific Computing / MCMC / Probabilistic Inference  
**Keywords**: MCMC, Non-Markovian sampling, score-tilt, self-repulsion, stochastic approximation CLT  

## TL;DR
SRMC uses a $d$-dimensional running score average (rather than an $|\mathcal{X}|$-dimensional empirical measure) to record history. This history is transformed via exponential score-tilt into a surrogate target $\pi_\theta$ that "repels already visited regions." By wrapping this around any base MCMC kernel, SRMC achieves a non-Markovian, low-variance, normalization-free sampler with constant memory in general state spaces.

## Background & Motivation
**Background**: MCMC is a cornerstone for everything from Bayesian inference to EBM sampling. However, for complex targets (multimodal posteriors, rugged energy landscapes, large discrete configuration spaces), chains often fall into the trap of being "ergodic in theory but stuck in practice," oscillating repeatedly in the same region, leading to highly correlated samples and unreliable estimates. Recent work on improving sampling efficiency mostly refines the Markov kernel itself—using locally informed proposal/balancing in discrete domains (Zanella, GWG) or Langevin/HMC and non-reversible samplers in continuous domains.

**Limitations of Prior Work**: These methods are memoryless—the kernel does not "remember" that it has already visited a location 100 times. Non-Markovian approaches have clean theory in finite state spaces: SRRW and HDT feed the empirical measure $\hat{\delta}_n = \frac{1}{n+1}\sum_i \delta_{X_i}$ back into the kernel, achieving near-zero variance as the repulsion strength $\alpha$ increases while remaining normalization-free. However, $\hat{\delta}_n$ is an $|\mathcal{X}|$-dimensional object: exponentially large in $\{0,1\}^d$ and an infinite-dimensional measure in continuous domains, making it impossible to store.

**Key Challenge**: To trade "remembering history" for variance reduction, one must store information; for it to be storable, it must be $O(\text{const})$ in size. Since history is essentially a distribution over the entire state space, how can it be compressed into constant dimensions while maintaining theoretical properties like asymptotic unbiasedness and variance decay? Existing compromises either require large buffers (Stein self-repulsive stores historical samples) or importance reweighting (adaptive biasing potentials), losing the simplicity of being normalization-free.

**Goal**: Construct a generic wrapper that (i) maintains constant $O(d)$ memory; (ii) is compatible with any base MCMC (MH, Langevin, HMC, GWG); (iii) remains normalization-free; and (iv) provides CLT and $\alpha$-scaling theoretical guarantees, extending the near-zero-variance properties of SRRW/HDT to general state spaces.

**Key Insight**: The Stein identity tells us that $\mathbb{E}_{X\sim\pi}[s(X)] = 0$ (where $s = \nabla\log\pi$), implying that for a well-exploring chain, the time average of the score should be near zero. If a chain lingers in a certain region, the score directions in that region will accumulate bias. The running score average $\theta_n$ thus becomes a "detector of the chain's deviation from the true distribution." This immediately reduces an $|\mathcal{X}|$-dimensional statistic to $d$ dimensions.

**Core Idea**: Use $\theta_n \in \mathbb{R}^d$ (score time average) as a history summary, then use an exponential tilt $\pi_\theta(x) \propto \pi(x)\exp\{-\alpha \theta^\top s(x)\}$ to penalize "directions that were over-visited in the past" to obtain a surrogate target. The base kernel performs one step on $\pi_{\theta_n}$, and then $\theta_{n+1}$ is updated iteratively.

## Method

### Overall Architecture
SRMC is a thin wrapper. Given a target $\pi(x)\propto e^{-U(x)}$, score $s(x) = -\nabla U(x)$, repulsion strength $\alpha \geq 0$, step size sequence $\gamma_n = (n+1)^{-\rho}$ ($\rho \in (1/2, 1]$, Robbins-Monro conditions), and any base kernel $P_q$ (MH/MALA/HMC/GWG). At iteration $n$: (1) use the current history $\theta_n$ to construct the surrogate $\pi_{\theta_n}(x) \propto \pi(x)\exp\{-\alpha\theta_n^\top s(x)\}$; (2) sample $X_{n+1}$ using $P_{\pi_{\theta_n}}$; (3) update history via a first-order recursion $\theta_{n+1} = \theta_n + \gamma_{n+1}(s(X_{n+1}) - \theta_n)$. This mechanism only adds $d$-dimensional memory ($\theta_n$) and one additional score evaluation.

### Key Designs

1.  **Running Score History with Constant Memory $\theta_n$**:
    *   **Function**: Replaces the $|\mathcal{X}|$-dimensional empirical measure with a $d$-dimensional vector as the "history summary."
    *   **Mechanism**: $\theta_n \in \mathbb{R}^d$ is a weighted moving average of past scores $\{s(X_i)\}_{i\leq n}$. When $\rho=1$, it reduces to a simple time average; when $\rho<1$, it favors recent samples, making it more sensitive to temporary trapping. Its physical meaning can be expressed as $\theta_n - \mathbb{E}_\pi[s(X)] = \int_\mathcal{X}[\frac{1}{n+1}\sum_i\delta_{X_i}(x) - \pi(x)]s(x)dx$—it is precisely the "deviation between the chain's empirical distribution and $\pi$ under the score projection," essentially acting as a low-dimensional imbalance detector.
    *   **Design Motivation**: The limitation of SRRW/HDT to finite spaces stems from the need to store $\hat\delta_n$. By projecting "distributional differences" into $d$ dimensions via the score, storage and computational complexity collapse to constants. Furthermore, the Stein identity $\mathbb{E}_\pi[s] = 0$ automatically ensures that $\theta^\star = 0$ is the equilibrium point, naturally calibrating the estimation.

2.  **Exponential Score-Tilt Surrogate Target $\pi_\theta$**:
    *   **Function**: Converts "historical bias" into a modified target that "penalizes biased directions" for the base kernel.
    *   **Mechanism**: $\pi_\theta(x)\propto \pi(x)\exp\{-\alpha \theta^\top s(x)\}$. When the chain stays in a metastable basin, $\theta$ points towards the concentrated "cone direction" of that basin's scores. Thus, $x$ values where $\theta^\top s(x) > 0$ (still inside the basin) are down-weighted by $\exp\{-\alpha\theta^\top s(x)\} < 1$. For MH, this only requires multiplying the acceptance rate by an $e^{-\alpha\theta^\top[s(y)-s(x)]}$ factor; the normalization $Z_\theta$ cancels out in the ratio. For Langevin/HMC, the score is replaced by the surrogate score $s_\theta(x) = s(x) - \alpha\nabla_x s(x)\theta = -\nabla U(x) + \alpha \nabla^2 U(x)\theta$, where Hessian-vector products are computed cheaply via autodiff or finite differences $(\nabla U(x+\epsilon\theta) - \nabla U(x))/\epsilon$.
    *   **Design Motivation**: (i) Exponential tilt is the most natural form for an "exponential family of linear score perturbations" and preserves non-negativity; (ii) it makes the mechanism normalization-free ($Z_\theta$ always cancels or is unnecessary), preserving the usability of classical MCMC; (iii) it is plug-and-play: no internal logic of the base kernel is modified, and any kernel effective for $\pi$ can be substituted with $\pi_{\theta_n}$. The "arrow flipping" visualization in Figure 1 shows how it dynamically lowers the "effective energy barrier" around metastable basins.

3.  **Coupled SA + CLT with $O(1/\alpha)$ Variance Decay**:
    *   **Function**: Proves almost sure convergence and joint CLT in general $\mathcal{X} = \mathbb{R}^d$ spaces and quantifies variance scaling as $\alpha$ increases.
    *   **Mechanism**: Define $\vartheta_n = (\theta_n, \mu_n)$ (where $\mu_n$ estimates $\mathbb{E}_\pi[f(X)]$) as a Stochastic Approximation (SA) process $\vartheta_{n+1} = \vartheta_n + \gamma_{n+1} H(\vartheta_n, X_{n+1})$. Under Assumption 1 ($L$-Lipschitz score, super-linear tail growth of $U$, asymptotically normal Hessian) and Assumption 2 (uniform drift of the kernel, Lipschitz in $\theta$), Theorem 3.3 shows $\vartheta_n \to (0, \mu)$ a.s. and $\gamma_n^{-1/2}(\vartheta_n - \vartheta^\star) \xrightarrow{d} \mathcal{N}(0, \Sigma_\vartheta)$, where $\Sigma_\vartheta$ satisfies a Lyapunov equation. The key Jacobian $A^\star$ shows $\alpha$ appearing in the covariance blocks. Proposition 3.4 proves the $\theta$-block $\Sigma_{\theta\theta}(\alpha) = O(1/\alpha)$ and is monotonically non-increasing in $\alpha$. For Gaussian targets, this scaling passes directly to $\Sigma_X(\alpha) = V\Sigma_{\theta\theta}(\alpha)V^\top$, reaching near-zero variance.
    *   **Design Motivation**: Extending near-zero variance to general spaces requires translating generic SA conditions into verifiable drift and kernel-Lipschitz conditions for MCMC. The technical contribution includes providing verifiable conditions for MH/MALA and upgrading "stability" from an assumption to a provable conclusion.

### Loss & Training
There is no training loss—SRMC is a sampling algorithm. Practical hyperparameters include: $\rho \in \{0.6, 0.8\}$ (to prevent $\gamma_n$ from dropping too fast), $\epsilon \approx \alpha$ (scale for finite differences to avoid numerical instability), and $\alpha$ chosen such that $\alpha|\theta_n^\top s(X_n)|$ is moderate to avoid over-tilting. For complex targets, an adaptive-$\alpha$ heuristic is used: small early on and larger later. In discrete domains, the discrete Stein operator $s_i(x) = \pi(x^{(i,K-x_i)})/\pi(x) - 1$ is used to maintain $\mathbb{E}_\pi[s] = 0$. For high-dimensional EBMs (e.g., Static MNIST), a relaxed gradient is used as a score proxy; while less theoretically rigorous, it remains effective in practice.

## Key Experimental Results

### Main Results
Comparison of MSE for SR-MALA / SR-HMC vs. baselines on a 10D continuous target (sample mean estimation, 100 independent trials, selecting $\alpha \in \{0, 0.01, 0.1, 1, 2, 5\}$):

| Target | Sampler | Best $\alpha$ | MSE Gain vs Baseline | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Correlated Gaussian (10D, ill-conditioned) | MALA → SR-MALA | $\alpha=2{-}5$ | Moderate reduction | Consistent across steps and CPU views |
| Correlated Gaussian | HMC → SR-HMC | $\alpha=2{-}5$ | Significant reduction, lowest MSE | $\alpha=5$ is optimal |
| Bayesian Logistic Regression (10D, 100 obs) | MALA → SR-MALA | $\alpha=1{-}2$ | ~5× MSE reduction | $\alpha=5$ too aggressive, high rejection |
| Bayesian Logistic Regression | HMC → SR-HMC | $\alpha \approx 1{-}2$ | Fastest MSE decay | $\alpha=5$ over-tilted and performed worse |

Discrete EBM (Static MNIST, $\{0,1\}^{784}$, 100 parallel chains initialized at digit '7', 10,000 steps):

| Metric | Baseline GWG | SR-GWG ($\alpha=10^{-4}$) | Relative Change |
| :--- | :--- | :--- | :--- |
| Cumulative KL (↓) | 4.16 | 0.68 | **−84%** |
| Batch Vendi Score (↑) | 2.6 | 6.4 | **+146%** |
| Escape Step from Initial Mode | Never escaped | ~2500 steps | — |
| Chain Diversity (at 10k steps) | Mostly '7' | Covers multiple digit classes | Significant mode exploration |

CIFAR-10 Gaussian mixture mode-coverage (Appendix D.2): SR-ULA achieved 100% mode coverage by 1035 steps, while ULA only reached 2.8%. With 10 parallel chains, SR-ULA covered 7/10 classes compared to 5/10 for ULA.

### Ablation Study

| Config | Key Observation | Explanation |
| :--- | :--- | :--- |
| $\alpha = 0$ | Reduces to base sampler | Confirms SRMC is a true wrapper layer |
| Moderate $\alpha \uparrow$ | Monotonic MSE / KL decrease | Consistent with the $O(1/\alpha)$ theory in Prop 3.4 |
| Very large $\alpha$ (e.g., 5) | Rejection rate spikes or over-tilt | Optimal $\alpha$ is usually moderate, not maximal for nonlinear targets |
| $\rho = 1$ vs. $\rho \in \{0.6, 0.8\}$ | Latter is more stable during transients | $\rho=1$ drops step size too fast for $\theta_n$ to adapt |
| $\epsilon$ too small | Poor Hessian-vec approximation | $\epsilon \sim \alpha$ provides the best stability |
| Adaptive $\alpha$ | Robust fallback | Useful when the optimal fixed $\alpha$ is unknown |
| Discrete: relaxed proxy vs. exact score | Proxy is necessary in high dimensions | Theory requires exact $s$ for $\mathbb{E}_\pi[s]=0$, yet proxy works on MNIST |

### Key Findings
*   The theoretical $\Sigma_{\theta\theta}(\alpha) = O(1/\alpha)$ scaling matches the MSE behavior in correlated Gaussian experiments.
*   For nonlinear targets (e.g., logistic regression), there is a "sweet spot" for $\alpha$; excessive tilting leads to extreme rejection rates.
*   The mode-mixing improvement in discrete EBMs is significant (84% KL reduction), demonstrating that SRMC's primary strength is "systematic forced exploration."
*   SR-ULA's advantage is more pronounced as the number of parallel chains decreases, suggesting SRMC as an alternative to increasing chain count under compute constraints.

## Highlights & Insights
*   **The idea of "using score time average as a proxy for visit counts" is elegant**: The insight that the score's expected value is zero allows a $d$-dimensional vector to act as a "deviation detector" for distribution. This paradigm of "summarizing distributions via low-dimensional statistics" could be applied beyond MCMC to RL exploration or adaptive optimization.
*   **Engineering value of the plug-and-play wrapper**: MH, MALA, HMC, ULA, and GWG are all easily adapted while maintaining normalization-free properties. SRMC serves as a general-purpose layer in the MCMC toolbox.
*   **Efficient Hessian-vector products**: By using $\nabla^2 U(x)\theta \approx (\nabla U(x+\epsilon\theta) - \nabla U(x))/\epsilon$, SRMC adds only one additional score evaluation, resulting in an overhead of roughly 2×.
*   **Solid theoretical contribution**: Translating Borkar’s SA theorems into verifiable MCMC conditions and proving stability without assuming bounded iterates is a substantial upgrade over SRRW/HDT analyses.

## Limitations & Future Work
*   For general non-Gaussian targets or arbitrary kernels, $\Sigma_{\mu\mu}(\alpha)$ lacks a closed-form $\alpha$ dependence, requiring empirical tuning of $\alpha$.
*   Optimal $\alpha$ values for nonlinear targets are often moderate; there is currently no fully automated $\alpha$ scheduler.
*   The discrete theory requires the discrete Stein score $\mathbb{E}_\pi[s] = 0$, but high-dimensional tasks like MNIST require relaxed gradients, leaving a gap between theory and practice.
*   Additional score evaluations: For samplers like HMC with many leapfrog steps, the overhead of finite-difference scores may offset variance gains in terms of raw CPU time.
*   Current framework focuses on single-chain; combinations with Parallel Tempering or Replica Exchange have not been systematically studied.

## Related Work & Insights
*   **vs. SRRW (Doshi 2023) / HDT-MCMC (Hu 2025)**: Both target non-Markovian self-repulsion, but SRRW/HDT are restricted to finite state spaces. SRMC uses $d$-dimensional score averages to extend these near-zero-variance properties to general spaces.
*   **vs. Stein self-repulsive dynamics (Ye 2020)**: The latter uses a sample buffer for pairwise repulsion; SRMC achieved unbiasedness with constant memory and finite $\alpha$.
*   **vs. Adaptive Biasing Potentials / Metadynamics**: These use density accumulation and reweighting; SRMC remains normalization-free.
*   **vs. Wang-Landau / Contour SGLD (Deng 2020)**: Those methods flatten the energy landscape but are tied to Langevin frameworks and stratification; SRMC is a general wrapper.
*   **vs. Adam-style SGLD (Kim 2022)**: While similar in modifying drift with running averages, SRMC modifies the target itself, making it universal for any sampler with a CLT.
*   **Insight**: Score-based history compression could be valuable for Variational Inference, Normalizing Flows, Diffusion guidance, and RL exploration—any scenario requiring a low-dimensional summary of history without breaking normalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](../../ACL2026/nlp_generation/adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ICML 2026\] Characterizing the Effect of Noise in Language Generation in the Limit](characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](../../ACL2026/nlp_generation/threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)

</div>

<!-- RELATED:END -->
