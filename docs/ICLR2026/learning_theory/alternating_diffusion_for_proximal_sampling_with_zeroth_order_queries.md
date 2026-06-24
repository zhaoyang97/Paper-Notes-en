---
title: >-
  [Paper Note] Alternating Diffusion for Proximal Sampling with Zeroth Order Queries
description: >-
  [ICLR2026][Sampling Theory][Proximal Sampling] This paper replaces the "reverse denoising" step in proximal sampling—traditionally implemented via rejection sampling—with direct SDE simulation. By using a Gaussian mixture model (GMM) formed by the current particles as a surrogate distribution, it performs Monte Carlo score estimation using only function values (zeroth-order) of $f$. The result is a sampler that requires no gradients, no rejection sampling…
tags:
  - "ICLR2026"
  - "Sampling Theory"
  - "Probabilistic Methods"
  - "MCMC"
  - "Proximal Sampling"
  - "Zeroth-Order Queries"
  - "Alternating Diffusion"
  - "Gaussian Mixture"
  - "Langevin Sampling"
date: 2026-05-08
content_hash: 61ca312cdfd9da91
---

# Alternating Diffusion for Proximal Sampling with Zeroth Order Queries

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=NjjRuJuMTd](https://openreview.net/forum?id=NjjRuJuMTd)  
**Code**: TBD  
**Area**: Sampling Theory / Probabilistic Methods / MCMC  
**Keywords**: Proximal Sampling, Zeroth-Order Queries, Alternating Diffusion, Gaussian Mixture, Langevin Sampling

## TL;DR
This paper replaces the "reverse denoising" step in proximal sampling—traditionally implemented via rejection sampling—with direct SDE simulation. By using a Gaussian mixture model (GMM) formed by the current particles as a surrogate distribution, it performs Monte Carlo score estimation using only function values (zeroth-order) of $f$. The result is a sampler that requires no gradients, no rejection sampling, and no model training with a fixed runtime. Theoretically, it inherits the exponential convergence of proximal sampling under isoperimetric conditions, and experimentally, it is nearly an order of magnitude faster than RGO-based proximal sampling.

## Background & Motivation
**Background**: Sampling from $\pi(x)\propto e^{-f(x)}$ is a fundamental task in statistics and machine learning (essential for Bayesian posteriors and score-based generative models). Mainstream approaches are Langevin-type methods (ULA, MALA), which have sharp non-asymptotic convergence guarantees under strong convexity or functional inequality conditions. Recently, **proximal sampling** (Lee et al. 2021; Liang & Chen 2023) emerged, introducing an auxiliary distribution close to the target—usually the Gaussian convolution $\pi_Y=\pi_X*\mathcal N(0,hI_d)$—and alternating between conditional updates of the target $x$ and auxiliary $y$. Chen et al. (2022) further interpreted these steps as **forward heat flow** (adding Gaussian noise) and **reverse denoising SDE**, proving exponential convergence when the target satisfies the Log-Sobolev Inequality (LSI).

**Limitations of Prior Work**: Despite its theoretical elegance, **scalable implementation of proximal sampling has been difficult**. The reverse step relies on a "Restricted Gaussian Oracle" (RGO), typically implemented via "local optimization of $f$ + rejection sampling." To maintain a reasonable acceptance rate, the step size $h$ must be taken very small (weak convolution), leading to a high number of outer iterations and fluctuating computational costs due to stochasticity.

**Key Challenge**: A larger $h$ yields faster convergence (theoretically requiring $\tilde O(C_{\mathrm{LSI}}/h)$ outer iterations), but rejection sampling imposes an upper bound on $h$—$h$ can only reflect the **local smoothness** of $f$, not the **global structure** of $\pi$ (e.g., distance between modes). Conversely, diffusion-based MCMC can simulate the denoising SDE directly to avoid rejection sampling but requires either training a score model, starting from a Gaussian distribution, or nesting an auxiliary sampler. Furthermore, proximal sampling was originally a **single-particle** iterative framework, which is disconnected from the practical need to sample many particles in parallel.

**Goal**: Can proximal sampling be implemented in its **theoretical form** (directly via Gaussian convolution + diffusion processes) in a way that is scalable, avoids rejection sampling, and supports multi-particle parallelism?

**Key Insight**: Notice that the drift term of the reverse SDE is the score $\nabla\log(\pi_X P_{h-t})$, and the bottleneck is this unknown score. The authors observe that the current batch of auxiliary particles $\{y_j\}$ itself constitutes an empirical **Gaussian mixture** approximation, which can serve as a surrogate for $\pi_X$, turning score estimation into "Monte Carlo sampling from a directly-sampleable GMM."

**Core Idea**: Use a GMM surrogate distribution formed by particles to replace the true target, converting the score estimation of reverse denoising into a Monte Carlo estimate using only $f$ function values. Thus, the proximal sampler becomes a zeroth-order diffusion algorithm alternating between "forward noising + reverse denoising."

## Method

### Overall Architecture
The method is called **Zeroth-Order Diffusive Proximal Sampler** (Algorithm 1). It maintains $N$ particles $\{x^{(i)}_k\}$. Each outer iteration $k$ performs two steps, corresponding precisely to the forward/backward steps of proximal sampling:

- **Step 1: Forward Heat Flow**. Gaussian noise is added to each particle: $y^{(j)}_{k+1/2}=x^{(j)}_k+\sqrt h\,\xi^{(j)}$. This is an exact Gaussian convolution $\pi_Y=\pi_X*\mathcal N(0,hI_d)$, requiring no approximation or oracles.
- **Step 2: Reverse Denoising (Surrogate Version)**. Starting from $z_T=x^{(i)}_k+\sqrt h\,\xi'$, a reverse denoising SDE is integrated across $T$ sub-steps to pull particles from the "noised distribution" back to the target $\pi_X$. While this originally requires the true score $\nabla\log(\pi_X P_{h-t})$, the authors substitute it with the score of the GMM surrogate distribution constructed from current particles, discretized using Euler–Maruyama.

In ideal proximal sampling, the forward path is $\pi_X\to\pi_Y$ and the backward path is $\pi_Y\to\pi_X$. Although a single step doesn't reach the target, repeated alternation contracts toward it (Theorem 1 gives exponential convergence). This paper preserves this "alternation" but replaces "rejection sampling" with "direct SDE simulation + GMM score estimation." A key property: since it **repeatedly** cycles through finite-time noise/denoise loops (Variance-Expanding VE diffusion), it **does not** require initialization from a Gaussian equilibrium as standard diffusion models do.

### Key Designs

**1. Particle GMM Surrogate: Score Estimation as Zeroth-Order Monte Carlo**

The drift term of the reverse SDE (5) is $\nabla\log(\pi_X P_{h-t})$. Since $\pi_X$ is unknown, this was the root cause for needing rejection sampling. The authors replace $\pi_X$ with a surrogate distribution constructed from **current particles**:
$$\hat q_{k+1}(x\mid Y_{k+1/2},X_k)\propto\frac1N\sum_{j=1}^N\frac{\pi_{X\mid Y=y_j}(x)}{\pi_Y(y_j)}\,\hat q_{k+1/2}(y_j\mid X_k),\quad \hat q_{k+1/2}(y\mid X_k)=\frac1N\sum_i\mathcal N(y;x_i,hI_d).$$
Substituting the explicit forms of $\pi_{X\mid Y=y_j}\propto\exp(-f(x)-\tfrac1{2h}\|x-y_j\|^2)$ and $\pi_Y$, the surrogate simplifies to $\hat q_{k+1}(x)\propto g^{k+1/2}_N(x)\,e^{-f(x)}$, where $g^{k+1/2}_N$ is an **unnormalized weighted Gaussian mixture** ($N$ components). Using Bayes' rule, the surrogate score can be written as an expectation over the GMM posterior:
$$\hat s_t(z)=\mathbb E_{g^{k+1/2}_N(x_0\mid z)}\!\Big[\frac{x_0-z}{\sigma_t^2}\,e^{-f(x_0)}\big/C_t\Big],\quad \sigma_t^2=h-t,$$
where the posterior $g^{k+1/2}_N(x_0\mid z)$ is itself a GMM (components with parameters $\bar\sigma^2=(h^{-1}+\sigma_t^{-2})^{-1}$, $m_j(z)=\bar\sigma^2(h^{-1}y_j+\sigma_t^{-2}z)$, and weights $w_j(z)$ determined by $\mathcal N(z;y_j,(h+\sigma_t^2)I)$). This implies score estimation **only requires**: drawing $M$ samples from a directly-sampleable GMM and weighting them by $e^{-f}$. It only queries the function value of $f$, requiring no gradients, no model training, and no rejection. This is the source of "zeroth-order."

**2. Inverse Reweighting: Suppressing Particle Collapse and Encouraging Exploration**

The surrogate distribution (9) includes an **inverse reweighting** term for $\hat q_{k+1/2}(y_j\mid X_k)$. Its function is to reduce the weight of $y_j$ if it falls in an over-clustered region of $X_k$ and increase its importance if it falls in a sparse region. Intuitively, this is an "anti-crowding" mechanism based on the empirical particle system that prevents particles from collapsing into a single mode and pushes them to explore uncovered regions. This introduces **interaction** between particles (they are not sampled independently). Experiments show this is crucial for accelerated mixing—removing interaction ($N=1$ independent parallel chains) significantly slows convergence.

**3. VE Alternating Diffusion: Cyclic Loops and Convergence Inheritance**

Unlike "one-way pushforward" diffusion MCMC, this method **repeatedly** applies the same SDE dynamics between two fixed distributions $\pi_X$ and $\pi_Y$. It uses **variance-expanding** (VE) diffusion: each round only adds noise for a finite time before denoising, never requiring the process to reach Gaussian equilibrium. This matches the inherent nature of proximal sampling, allowing the theoretical analysis to directly apply proximal sampling contraction lemmas (see convergence bound below). Unlike Variance-Preserving (VP) diffusion, where sampling error partly stems from the gap between the "finite-time mixed distribution" and "Gaussian equilibrium," this VE alternating framework bypasses that issue.

**4. Fixed Runtime + Multi-particle Parallelism**

All particle operations (for $i,j,l$) are designed for parallelism. A naive implementation requires $KTMN$ evaluations of $f$ ($K$ outer iterations, $T$ denoising sub-steps, $M$ MC samples, $N$ particles). With a **parallel oracle** that evaluates $f$ on multiple samples simultaneously, this reduces to $KT$. Unlike rejection sampling where costs fluctuate with acceptance rates, this method has a **deterministic** runtime (fixed steps), naturally suiting modern parallel computing environments.

### Loss & Training
Ours **requires no training**: there is no learnable score model and no loss function. The only "hyperparameters" are step size $h$, outer iterations $K$, denoising sub-steps $T$, MC samples $M$, particle count $N$, and noise schedule $\{\sigma_t^2\}$ (where $\sigma_T^2=h$ and $\sigma_0^2=\sigma_{\min}^2$). A key constraint is $T=O(h)$ to control time discretization error.

## Key Experimental Results

Theoretical Guarantee (Core Convergence Bound): Given that $\pi_X$ satisfies LSI (constant $C_{\mathrm{LSI}}$), Proposition 1 provides a single-step bound including discretization error $\Lambda_1^{(k)}$ and score estimation error $\Lambda_2^{(k)}$:
$$H_{\pi_X}(\rho^X_k)\le \frac{1}{r^k}H_{\pi_X}(\rho^X_0)+\frac{\alpha_u r}{r-1}\Lambda,\qquad r=(1+h/C_{\mathrm{LSI}})^{\,2-1/(2u^2)}>1,$$
where $\Lambda_1=O(h/T)$ (controllable via $T$) and $\Lambda_2=O(1/N+1/M)$ (Monte Carlo error from finite $N, M$). To achieve $H_{\pi_X}(\rho_k)\le\varepsilon$, the iteration complexity is $k=O\!\big(\log(H_{\pi_X}(\rho_0)/\varepsilon)/\log r\big)$, inheriting the exponential convergence rate of ideal proximal sampling. When $C_{\mathrm{LSI}}\gg h$, the factor is approximately $\tilde O(C_{\mathrm{LSI}}/h)$, implying larger $h$ is better—and this method allows large $h$ (provided $T=O(h)$), a key advantage over RGO.

### Main Results: Gaussian Lasso Mixture ($d=5$)

Replicating the setup from Liang & Chen (2023), the target is a mixture of Gaussian and Lasso distributions. The RGO baseline uses 100 chains with step size $h=1/135$. Ours uses $h=1/10$ (**13.5x larger**), with two settings: $N=100$ interacting particles, or $N=1$ running 100 chains. Convergence is measured by KL divergence.

| Method | Step Size $h$ | Iterations for Comparable KL | Remarks |
| :--- | :--- | :--- | :--- |
| RGO (Proximal Baseline) | $1/135$ | ~950 iterations (≈9500 RGO updates) | Limited by rejection sampling; step size stuck small |
| **Ours** (100 Interacting Particles) | $1/10$ | **~100 iterations** | ~10x fewer iterations; ~100x faster considering thinning |
| Ours w/o interaction ($N=1$ independent) | $1/10$ | Significantly slower | Ablation: Removed particle interaction |

On marginal distributions, Ours reaches the target in ~100 iterations (RGO requires ~950), and by ~250 iterations, it highly matches the long-run reference distribution.

### Ablation Study: Uniform Sampling on Bounded Domains

A second experiment samples uniformly from a **non-convex, disconnected** domain $K$ (two disjoint solid tori $T_1\cup T_2$) in $\mathbb R^3$. The potential function $f(x)=0$ ($x\in K$) or $100$ (otherwise), requiring only a **membership oracle**. Comparison with In-and-Out (Kook et al. 2024, proximal sampling for convex bodies).

| Method | $k=3$ | $k=200$ | Conclusion |
| :--- | :--- | :--- | :--- |
| In-and-Out | Finds proximal torus $T_1$ | Stays at $T_1$, **cannot reach** $T_2$ | Relies on convexity assumption |
| **Ours** | Fills $T_1$ first | Gradually drives particles to $T_2$ | Can cross disconnected modes |

### Key Findings
- **Particle interaction is key to accelerated mixing**: Removing interaction (independent parallel chains) significantly slows convergence, validating the "anti-crowding/exploration" role of inverse reweighting.
- **Large step sizes provide nearly an order of magnitude speedup**: Relaxing the strict step size limitations of RGO inherently improves convergence speed by ~10x; combined with multi-particle interaction, the effect is magnified.
- **Zeroth-order + Noisy scores aid mode jumping**: In the two-tori experiment, the noisy score approximation actually helps particles jump from the proximal mode to the distal mode (similar to phenomena in diffusion-based black-box optimization), whereas an exact proximal step would get stuck.

## Highlights & Insights
- **"Using particles as a surrogate distribution" is a clever perspective**: Previously, the proximal reverse step was restricted to rejection sampling because the true score was unknown. The authors realized that current particles naturally form a GMM, so score estimation reduces to "sampling from a GMM + $e^{-f}$ weighting," effectively eliminating gradients, training, and rejection.
- **Inverse reweighting serves two purposes**: It is mathematically necessary to recover $\pi_X$ as $N\to\infty$, and it acts as an "anti-crowding" agent for finite particles to prevent collapse, aligning theory with practice.
- **The choice of VE over VP is deliberate**: By using Variance-Expanding diffusion and repeated finite-time cycles, the method avoids the need for a Gaussian start and can apply proximal contraction lemmas for provable convergence—unifying "diffusion sampling" and "proximal sampling" theories.
- **Transferability**: The idea of "zeroth-order score estimation via empirical GMMs" can be transferred to any sampling/optimization scenario where only function values are available (e.g., black-box, membership oracles).

## Limitations & Future Work
- **Strong Theoretical Assumptions**: Convergence guarantees rely on LSI, Lipschitz bounds for scores in time/space, and consistent entropy bounds. Furthermore, the analysis handles $N,M\to\infty$ and finite cases separately; the authors admit a full theoretical characterization of the **interaction term** is not yet provided.
- **Cost Scaling with Dimension/Particles**: The naive implementation is $KTMN$ evaluations of $f$, heavily reliant on a "parallel oracle" to reach $KT$. Since the GMM component count equals $N$, the per-step score estimation cost in high dimensions or for large $N$ is non-trivial.
- **Limited Experimental Scale**: Main experiments are at $d=5$, and the second is in $\mathbb R^3$, lacking validation on high-dimensional real-world tasks (e.g., Bayesian Neural Network posteriors). KL estimation depends on long-run RGO, involving a circular evaluation dependency.
- **Future Directions**: Rigorous convergence characterization for the multi-particle interaction term; researching adaptive step sizes/noise schedules; decoupling component count from particle count to reduce costs.

## Related Work & Insights
- **vs RGO-based Proximal Sampling (Liang & Chen 2023, etc.)**: They use "local optimization + rejection sampling" for the reverse step, where step size is choked by acceptance rates and cost fluctuates. Ours simulates SDEs directly with GMM scores, allowing large steps and fixed runtime, roughly 10x faster.
- **vs ZOD-MC (He et al. 2024)**: Both are zeroth-order diffusion, but ZOD-MC uses a one-way pushforward outer loop with a nested proximal sampler for score estimation, requiring potential minima (gradient-located in practice). Ours **inverts** the structure—simulating proximal SDEs directly with purely zeroth-order information throughout.
- **vs SLIPS (Grenioux et al. 2024)**: Both alternate between denoising and updating auxiliary variables, but SLIPS is a one-way dynamic that converges as time span increases; Ours is a repeated cycle of fixed-length alternations.
- **vs DiGS (Chen et al. 2024)**: Both use "noise-denoise" for Gibbs-like updates, but DiGS uses VP diffusion with auxiliary samplers like MALA and **lacks** convergence guarantees. Ours uses VE diffusion consistent with proximal sampling, has provable contraction bounds, and leverages parallelism for lightweight score estimation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Particle GMM + Zeroth-order MC score" liberates proximal sampling from rejection sampling, unifying two theories with a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐ Two well-designed experiments with clear ablations, but low dimensionality ($d\le 5$) and lack of real-world high-dimensional tasks.
- Writing Quality: ⭐⭐⭐⭐ Complete theoretical derivation, clear motivation, and well-explained mapping between proximal steps and diffusion.
- Value: ⭐⭐⭐⭐ Provides a provable, parallelizable practical algorithm for sampling scenarios where only function values (no gradients) are accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Poly-attention: a general scheme for higher-order self-attention](poly-attention_a_general_scheme_for_higher-order_self-attention.md)
- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)
- [\[ICLR 2026\] Towards Sampling Data Structures for Tensor Products in Turnstile Streams](towards_sampling_data_structures_for_tensor_products_in_turnstile_streams.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] Quotient-Space Diffusion Models](quotient-space_diffusion_models.md)

</div>

<!-- RELATED:END -->
