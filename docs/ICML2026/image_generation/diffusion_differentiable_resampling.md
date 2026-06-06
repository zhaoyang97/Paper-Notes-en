---
title: >-
  [Paper Note] Diffusion Differentiable Resampling
description: >-
  [ICML 2026][Image Generation][Diffusion models] This paper proposes **diffusion resampling**: a **training-free** diffusion process is used to provide a naturally differentiable reparametrisation alternative for the resa…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion models"
  - "Particle filtering"
  - "SMC"
  - "Differentiable resampling"
  - "State-space models"
date: 2026-05-08
content_hash: cb355a6185f2f775
---

# Diffusion Differentiable Resampling

**Conference**: ICML 2026  
**arXiv**: [2512.10401](https://arxiv.org/abs/2512.10401)  
**Code**: https://github.com/zgbkdlm/diffres (Available)  
**Area**: Scientific Computing / Sequential Monte Carlo / Particle Filtering / Differentiable Sampling  
**Keywords**: Diffusion models, Particle filtering, SMC, Differentiable resampling, State-space models

## TL;DR
This paper proposes **diffusion resampling**: a **training-free** diffusion process is used to provide a naturally differentiable reparametrisation alternative for the resampling step in Sequential Monte Carlo (SMC). It is proven that this method achieves consistent convergence relative to the number of samples $N$ under the Wasserstein distance and outperforms existing differentiable resampling methods such as OT / Gumbel-Softmax / Soft on several particle filtering and parameter estimation benchmarks.

## Background & Motivation

**Background**: Particle filtering / SMC are primary tools for inference in State-Space Models (SSM), and **resampling** is a crucial step to mitigate particle degeneracy. The most commonly used multinomial resampling re-selects particles via categorical sampling $I_i \sim \mathrm{Categorical}(w_1,\dots,w_N)$.

**Limitations of Prior Work**: Multinomial resampling is a **discrete** operation, meaning path derivatives $\partial X_i^{\theta,*}/\partial\theta$ are undefined. When downstream tasks require learning SSM parameters (or even neural dynamics/decoders) based on gradients, automatic differentiation libraries silently drop these gradient components, leading to **incorrect gradient estimation**.

**Key Challenge**: Existing differentiable resampling methods face a trade-off between "unbiasedness / consistency" and "differentiability / computational cost":
- **REINFORCE types** (Score-based / Ścibior–Wood stop-gradient) suffer from high variance;
- **Soft / Gumbel-Softmax** act as biased interpolations between multinomial and uninformative sampling, requiring manual coefficient tuning to balance bias and statistical performance;
- **OT-based** (Corenflos et al., 2021) methods are consistent and differentiable but require solving Sinkhorn, with a computational cost of $O(N^2)$ and an **exponential** dependence on the entropy parameter $1/\varepsilon$; linear transport maps also struggle with complex distribution manifolds;
- **Neuralized** / **deterministic** resampling both introduce irreducible bias.

**Goal**: Construct a resampling method that is (i) naturally differentiable, (ii) does not disrupt existing SMC / SSM structures, (iii) consistently converges, (iv) has manageable computational costs, and (v) can adaptively inject prior information using the sequential structure of SMC.

**Key Insight**: The core of OT resampling is "finding a transport map $X_i^* = N\sum_j P_{i,j}^\varepsilon X_j$." The key insight of the authors is that **this map does not need to be "solved"; it can be "specified."** If a Langevin SDE is used to smoothly push the target $\pi$ toward a user-selected reference $\pi_{\mathrm{ref}}$ (forward) and then inverted (reverse SDE) to sample from $\pi_{\mathrm{ref}}$ back to $\pi$, the only source of randomness in the entire sampling chain is Gaussian noise, resulting in a **natural reparametrisation**.

**Core Idea**: Replace the Transport Matrix solved by Sinkhorn with a **training-free diffusion model** + **weighted sample-driven ensemble score approximation**, expressing SMC resampling as a differentiable SDE simulation.

## Method

### Overall Architecture
The input is a set of weighted samples $\{(w_i, X_i)\}_{i=1}^N \sim \pi$. The method consists of three steps:

1. **Specify a Langevin forward SDE** such that $\pi$ converges to a reference $\pi_{\mathrm{ref}}$ at time $t\to\infty$;
2. **Construct the corresponding reverse SDE** to invert $\pi_{\mathrm{ref}}$ back to $\pi$, where the required score $\nabla\log p_t$ is replaced by an **ensemble score (a non-parametric estimator in importance-sampling form)**—this step **requires zero training**;
3. Integrate this reverse SDE discretely to time $T$. The resulting $\{U_{i,T}\}_{i=1}^N$ are $\{(\frac{1}{N}, X_i^*)\}_{i=1}^N$, and the entire chain is differentiable with respect to the parameter $\theta$.

Substituting this into the Feynman–Kac / SMC main loop (Algorithm 2) yields a differentiable SMC where gradients can propagate end-to-end.

### Key Designs

1. **Training-free ensemble score (Core)**:

    - **Function**: Instantly estimates the score of the diffusion process using available weighted samples $\{(w_i, X_i)\}$, **eliminating all training**.
    - **Mechanism**: Rewrites $\nabla\log p_t(x)$ via importance sampling as $s_N(x,t) \coloneqq \sum_i \alpha_i(x,t)\nabla\log p_{t|0}(x|X_i)$, where $\alpha_i = w_i p_{t|0}(x|X_i) / \sum_j w_j p_{t|0}(x|X_j)$. This is equivalent to a "self-normalized IS with $\pi$ as the proposal and $p_{t|0}(\cdot|x_0)$ as the likelihood." Remark 1 provides a Doob $h$-function interpretation: $s_N = \nabla\log\sum_i h_i$, viewing the process as a **continuously differentiable reparametrisation** of multinomial resampling.
    - **Design Motivation**: Avoid the two-tier structure of "training a diffusion model → then using it for resampling"; naive evaluation is $O(N)$, but parallel implementation can reduce it to logarithmic complexity; reference $\pi_{\mathrm{ref}}$ implicitly encodes transport cost and Rao–Blackwellisation conditions, resulting in lower variance than multinomial.

2. **Mean-reverting Gaussian reference + adaptive injection of SMC information**:

    - **Function**: Ensures the reference is no longer a fixed $\mathrm{N}(0, I_d)$ but **dynamically approaches** the posterior of the current SMC step.
    - **Mechanism**: Uses weighted moment estimates $\mu_N, \Sigma_N$ of the particles to set the reference as $\nabla\log\pi_{\mathrm{ref}}(x) = -\Sigma_N^{-1}(x-\mu_N)$, resulting in an OU-type forward SDE $dX = -b^2\Sigma_N^{-1}(X-\mu_N)dt + \sqrt{2}b\,dW$. Its forward transition $p_{t|0}(x_t|x_0) = \mathrm{N}(x_t; m_t(x_0), V_t)$ has an analytical form (containing $e^{-b^2\Sigma_N^{-1}t}$ terms), allowing the $\nabla\log p_{t|0}$ required in the ensemble score to be computed in closed form.
    - **Design Motivation**: When $\pi$ is geometrically far from $\mathrm{N}(0, I_d)$, a naive reference requires a very large $T$ to converge; a Gaussian moment-matched reference minimizes the "diffusion distance," making it **more informative than the predictive samples used as references in OT** (OT uses $\{(w_{j-1,i}, Z_{j,i})\}$, whereas Ours can use the more accurate posterior $\{(w_{j,i}, Z_{j,i})\}$).

3. **Semi-linear exponential integrator to accelerate reverse SDE**:

    - **Function**: Integrates the reverse SDE to $T$ using fewer discrete steps $K$.
    - **Mechanism**: The reverse SDE under a Gaussian reference has a semi-linear structure $dU = (AU + f(U,t))dt + \sqrt{2}b\,dW$, where $A = b^2\Sigma_N^{-1}$ handles the linear rigid part separately. Using a Jentzen–Kloeden integrator: $U_{t_k} = e^{A\Delta_k}U_{t_{k-1}} + A^{-1}(e^{A\Delta_k}-I_d)f(U_{t_{k-1}}) + B_k$, where the Wiener integral $B_k\sim \mathrm{N}(0, \Sigma_N(e^{2A\Delta_k}-I_d))$ is sampled in closed form; if $A$ is non-invertible, it degrades to a lower-order Lord–Rougemont integrator.
    - **Design Motivation**: The Lipschitz constant of the ensemble score explodes near $t\to 0$, necessitating very small steps for a standard Euler–Maruyama; the exponential integrator "exactly integrates" the rigid main term, maintaining stability even with large step sizes.

### Loss & Training
This method **does not introduce new losses or training objectives**—it is a plug-and-play module within the SMC loop. For downstream learning of $L(\theta) = \prod_j L_j(\theta)$ (Feynman–Kac marginal likelihood estimation), $-\log L(\theta)$ is minimized directly, with gradients automatically backpropagated via (i) Gaussian noise reparametrisation + (ii) adjoint / discretise-then-differentiate methods from SDE solvers (using existing differentiable SDE tools like Bartosh / Li / Kidger).

Convergence analysis (Section 3) provides the main conclusion in Proposition 1:

$$\mathsf{W}_2^2(\widetilde{q}_t, q_t) \le \mathsf{W}_2^2(p_T, \pi_{\mathrm{ref}})\, e^{b^2(C_{\mathrm{ref}}-2C_p)t} + 2b^2 N^{-r} \overline{C}_e(t, T)$$

The error consists of score approximation (decaying with $N$ at the IS rate $r=1/2$) and the finite-time bias of $p_T \approx \pi_{\mathrm{ref}}$; Corollary 1 further proves the existence of a linear $t \mapsto T(t)$ such that $\mathsf{W}_2(\widetilde{q}_t, q_t) \to 0$. Remark 2 states that under a Gaussian reference, $N$ only requires **polynomial $T$**, which is significantly superior to the **exponential** dependency of $N$ on $1/\varepsilon$ in OT.

## Key Experimental Results

### Main Results (Gaussian mixture importance resampling, $N{=}10{,}000$, 100 independent runs)

| Method | SWD ($\times 10^{-1}$) ↓ | Resampling Variance ($\times 10^{-2}$) ↓ |
|------|--------------------------|------------------------------------|
| **Diffusion ($T{=}3, K{=}128$)** | **0.80 ± 0.21** | 3.74 ± 2.99 |
| OT ($\varepsilon{=}0.3$) | 0.84 ± 0.22 | 3.42 ± 3.26 |
| OT ($\varepsilon{=}0.6$) | 0.97 ± 0.20 | **3.41 ± 3.29** |
| Multinomial | 0.82 ± 0.25 | 3.78 ± 4.43 |
| Soft (0.9) | 0.83 ± 0.24 | 3.75 ± 3.77 |
| Gumbel-Softmax (0.1) | 1.40 ± 0.24 | 3.92 ± 3.74 |

Linear Gaussian SSM particle filtering ($N{=}32$, 128 steps, 100-run average):

| Method | $\|L-\hat L\|_2$ | Filtering KL ($\times 10^{-1}$) | $\|\theta-\hat\theta\|_2$ ($\times 10^{-1}$) |
|------|------------------|---------------------------------|------------------------------------------------|
| **Diffusion ($T{=}3, K{=}8$)** | **2.55 ± 1.89** | **4.26 ± 4.49** | 1.58 ± 0.75 |
| Diffusion ($T{=}1, K{=}4$) | 2.61 ± 2.08 | 4.94 ± 6.92 | **1.28 ± 0.70** |
| OT ($\varepsilon{=}0.4$) | 2.64 ± 2.13 | 5.07 ± 6.21 | 1.53 ± 1.16 |
| Multinomial | 2.80 ± 1.84 | 5.49 ± 6.87 | NaN (Diverged) |
| Soft (0.9) | 2.85 ± 1.80 | 4.66 ± 5.68 | NaN |
| Gumbel-Softmax (0.1) | 2.79 ± 2.14 | 4.83 ± 5.76 | NaN |

### Ablation Study
| Configuration / Phenomenon | Observation | Description |
|-------------|------|------|
| Diffusion w/ $K{=}8$ vs $K{=}128$ | SWD: 1.64 → 0.80 | Discrete steps directly determine accuracy; fine integration is needed. |
| Computational cost (increasing $N$) | Diffusion vs OT crossover moves left | At large sample sizes, diffusion resampling is **cheaper than OT**. |
| Computational cost ($K$ vs $1/\varepsilon$) | Crossover at $K \approx 6/\varepsilon$ for $N{=}8192$ | Both methods are of the same order; Diffusion avoids OT's exponential entropy dependency. |
| Lokta–Voltera neural dynamics learning ($N{=}64$) | Diffusion achieves lowest RMSE and stablest loss | Superior to OT / Soft / Gumbel / REINFORCE (Ścibior–Wood). |
| 32×32 visual pendulum dynamics learning | SSIM / PSNR comparable to or better than strongest baseline | Validates stable embedding in complex SMC pipelines with **high-dimensional image observations**. |

### Key Findings
- **Even without considering differentiability, diffusion resampling itself is a superior resampler**—on LGSSM, it outperforms multinomial / OT / Soft collectively, primarily because "using posterior particles for reference" is more informative than the predictive reference used by OT.
- **Gradient stability is the key factor for downstream optimization convergence**: Multinomial / Soft / Gumbel all provided "noisy" gradients to L-BFGS-B, causing NaN errors. Diffusion and OT were the only two that could successfully run with second-order optimizers.
- **Diffusion resampling is sensitive to $K$**: On Gaussian mixtures, $K{=}8$ is inferior to OT, while $K{=}128$ achieves SOTA. The advantage is that $K$ is a linear cost, which is more controllable than OT's exponential $1/\varepsilon$.
- The "mean-reverting" design of the Gaussian reference is the most cost-effective component: it prevents required $T$ from exploding and allows the exponential integrator to function effectively.

## Highlights & Insights
- **"Do not compute the transport map, specify it"** is the fundamental shift of this paper. The computational power Corenflos et al. spent on Sinkhorn is bypassed using a closed-form SDE, pulling differentiable resampling from $O(N^2/\varepsilon)$ down to nearly $O(N\log N \cdot K)$.
- Interpreting the ensemble score as a Doob $h$-function and thus a **continuously differentiable reparametrisation of multinomial** translates the "gradient problem of discrete categorical sampling" into the "reparametrisation of Gaussian noise in SDEs"—an elegant perspective shift that could inspire differentiable research for other discrete structures (e.g., categorical tokens, tree structures).
- The "using current SMC step posterior as reference" trick is valuable for all amortised inference / latent SDE learning: **information sources should adapt over time** in sequential structures rather than using static priors.
- The convergence proof **explicitly decouples** the two types of errors ($N$ vs $T$) and demonstrates that $N$ only needs to grow polynomially to match any $T$, providing direct practical guidance for designing SMC + differentiable sampling.

## Limitations & Future Work
- The authors acknowledge: backpropagating gradients through diffusion resampling is **sensitive to the choice of SDE solver**; the exponential integrator can still be unstable when the score explodes near $t\to 0$.
- The ensemble score is $O(N)$ (requires iterating over all particles per step), which remains a bottleneck at very large particle counts; parallel / tree-reduction is needed to compress this to $O(\log N)$.
- Self-assessment: The reference assumption relies on Gaussian / moment matching, which fails for **heavily multi-modal** targets. The authors mention "using Gaussian mixture references" as a future direction, but the corresponding semigroup approximation is non-trivial.
- Self-assessment: The vision-pendulum experiment uses 32×32 grayscale images; **whether variance remains stable under real image observations (higher res / RGB / deeper decoders)** is still an open question.
- Improvement directions: Replacing the finite-$T$ bias with chain correlation using forward-backward Gibbs chains (Corenflos et al., 2025 / Zhao et al. 2025) might eliminate the bias.

## Related Work & Insights
- **vs OT resampling (Corenflos et al., 2021)**: Key difference is transport map "computed vs specified." Ours bypasses Sinkhorn, replacing $1/\varepsilon$ exponential dependency with $T$ polynomial dependency; the reference can use more informative posteriors.
- **vs Soft / Gumbel-Softmax (Karkus 2018 / Jang 2017)**: Those are biased interpolations between multinomial and uninformative resampling; Ours is a **consistent** reparametrisation, theoretically cleaner and experimentally superior (especially avoiding NaN with L-BFGS-B).
- **vs Score-based / REINFORCE (Poyiadjis 2011 / Ścibior–Wood 2021)**: Those follow the expected gradient path, suffering from high variance and needing large $N$; Ours follows the pathwise route, leveraging the low-variance advantage of reparametrisation.
- **vs Wan & Zhao (2025)**: They also use diffusion for differentiable resampling but **train** a conditional diffusion, introducing bias, breaking consistency, and requiring gradient flow back to diffusion training. The training-free nature of Ours is the key differentiator.
- **vs Gourevitch et al. (2026, concurrent)**: They use stochastic interpolants for differentiable reparametrisation of **discrete one-hot** categories; Ours targets $\mathbb{R}^d$ continuous samples and focuses on the limit properties as $N\to\infty$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Specified SDE as transport map" is a truly clean paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers GMM / LGSSM / Lokta–Voltera / vision-pendulum, but image resolution is limited to 32×32.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation, theorems, and pseudo-code are well-structured; the Doob $h$-function explanation in Remark is exceptional.
- Value: ⭐⭐⭐⭐⭐ Provides a plug-and-play differentiable resampling module for probabilistic programming / latent SDE / neural SSM learning with high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](conditional_diffusion_sampling.md)
- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)

</div>

<!-- RELATED:END -->
