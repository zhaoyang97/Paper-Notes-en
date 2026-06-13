---
title: >-
  [Paper Note] Efficient Learning of Deep State Space Models via Importance Smoothing
description: >-
  [ICML 2026][Image Generation][Deep State Space Models] This paper proposes Parallel Variational Monte Carlo (PVMC), which uses prefix/suffix associative scans to compute the importance-weighted marginal smoothing distrib…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Deep State Space Models"
  - "Sequential Monte Carlo"
  - "Importance Smoothing"
  - "Parallel prefix scan"
  - "Variational Inference"
date: 2026-05-08
content_hash: 3c115a20b615d76a
---

# Efficient Learning of Deep State Space Models via Importance Smoothing

**Conference**: ICML 2026  
**arXiv**: [2605.21108](https://arxiv.org/abs/2605.21108)  
**Code**: https://github.com/John-JoB/parallel-variational-sequential-monte-carlo (Yes)  
**Area**: Time Series / Probabilistic Deep Learning / State Space Models  
**Keywords**: Deep State Space Models, Sequential Monte Carlo, Importance Smoothing, Parallel prefix scan, Variational Inference

## TL;DR
This paper proposes Parallel Variational Monte Carlo (PVMC), which uses prefix/suffix associative scans to compute the importance-weighted marginal smoothing distributions of deep state space models within an $\mathcal{O}(\log N \times \log T)$ span. It supports both supervised state estimation and generative modeling, achieving approximately 10× speedup over the fastest differentiable SMC baselines with higher accuracy.

## Background & Motivation

**Background**: Deep State Space Models (DSSMs) parameterize transition kernels $M_t$ and observation kernels $H_t$ using neural networks, serving as primary tools for time-series modeling in fields such as finance, ecology, target tracking, and neuroscience. Their training typically follows two separate paths: (a) treating the entire trajectory as a latent variable $\tilde{x}=x_{0:T}$ of a VAE and training with an IWAE-style ELBO (auto-encoding DSSM); (b) implementing sequential Monte Carlo/particle filtering (SMC) as differentiable operators and training via backpropagation through particle importance weights (differentiable SMC, DSMC).

**Limitations of Prior Work**: Both paths have significant drawbacks. While the VAE path is fully parallelizable, (i) it does not support supervised losses—the encoder only observes $y_{0:T}$ and cannot output a particle distribution at each time step to be compared with ground-truth states; (ii) its ELBO is a relaxed upper bound based on "importance weighting for a single trajectory," failing to exploit the exponential trajectory space formed by combining particles across different time steps. While the DSMC path provides reasonable marginal filtering posteriors for supervised losses (MSE / KNLL), its core resampling operator introduces global dependencies across particles, forcing sequential forward passes; it must either use biased gradient estimators like REINFORCE, sacrifice unbiasedness for low variance, or introduce differentiable relaxations that incur high computational overhead (e.g., Diffusion DPF training time is ~150× that of PVMC in Table 2).

**Key Challenge**: To simultaneously achieve "parallelism + supervision + tight variational bounds + unbiased gradients." The VAE path sacrifices supervision and tight bounds, while the DSMC path sacrifices parallelism and (in some methods) unbiasedness. This paper aims to satisfy all four requirements at once.

**Goal**: Construct an end-to-end differentiable estimator that is as hardware-parallelizable as a VAE, outputs marginal smoothing posteriors $Q_t(x_t \mid y_{0:T})$ for each time step like DSMC, and provides a tighter ELBO than IWAE.

**Key Insight**: The authors observe that if the "sampling" and "weighting" steps are completely decoupled—by making the proposal fully factorizable in the time dimension $V_{0:T}(x_{0:T}\mid y_{0:T})=\prod_t V_t(x_t\mid y_{0:T})$—sampling becomes naturally parallel. The remaining marginal weights $w_t^n$ take the form of a summation over all other time-step particle indices. This summation structure is a "forward × backward" chain tensor product, which can be solved using **associative prefix/suffix scans**. In other words, the sequential "resampling dependency" in SMC is replaced by a "re-summation dependency," the latter of which satisfies the associative property and allows for log-depth parallelism.

**Core Idea**: Replace particle filtering resampling with a **decomposable proposal + importance smoothing over temporal associative scans**, resulting in a DSSM training algorithm with $\mathcal{O}(\log N \times \log T)$ span complexity, unbiased gradients, and a strictly tighter ELBO than IWAE.

## Method

### Overall Architecture
Given a parameterized SSM $x_0 \sim P$, $x_t \sim M_t(\cdot\mid x_{t-1})$, $y_t \sim H_t(\cdot \mid x_t)$ and a neural network proposal $V_t(\cdot \mid y_{0:T})$, a single forward pass of PVMC proceeds as follows:

1. **Parallel Sampling**: Sample $N$ independent particles $X_t^{1:N}$ from $V_t$ in parallel across all $(t, n)$ pairs (Algorithm 1, lines 3-9).
2. **Parallel Kernel Computation**: Compute local importance kernels $K_t(X_t^m, X_{t-1}^n) = M_t(X_t^m\mid X_{t-1}^n) H_t(y_t\mid X_t^m) / V_t(X_t^m\mid y_{0:T})$ in parallel across all $(t, n, m)$ triplets (lines 11-19).
3. **Associative Scan**: Pack the $N\times N$ kernel matrices of adjacent time steps into semigroup elements $a_s$. Perform a prefix scan $b_s$ and a suffix scan $\hat b_s$, then combine them using a four-way product to obtain the marginal weight $w_t^n$ for each particle (Algorithm 2).
4. **Likelihood/Loss**: The normalization constant is $\hat L^N = \frac{1}{N^{T+1}}\sum_n W_t^n$ (valid for any $t$, as different marginalizations of the same joint). The final loss is either the negative PVMC ELBO $-\mathbb{E}[\log \hat L^N]$ (generative) or combined with $\sum_t \mathrm{MSE}(\sum_n w_t^n X_t^n, x_t^\star)$ (supervised state estimation).

The input is the observation sequence $y_{0:T}$ and model parameters $\theta$; the output is the set of weighted particles $\{(X_t^n, w_t^n)\}$ for each time step along with the likelihood estimate $\hat L^N$. The entire pipeline maintains $\mathcal{O}(\log N \times \log T)$ span during backpropagation.

### Key Designs

1. **Fully Decomposable Proposal + Joint Importance Measure**:
    - **Function**: Defines an importance measure $Q_{0:T}^N$ that weights all $N^{T+1}$ possible trajectories (picking one particle at each time step) simultaneously, such that its temporal marginal $Q_t^N$ is an unbiased estimator of the marginal smoothing posterior.
    - **Mechanism**: The proposal uses a "transverse" decomposition $V_{0:T}=\prod_t V_t(x_t\mid y_{0:T})$, and the kernel is defined as $K_t(X_t^{n_t}, X_{t-1}^{n_{t-1}}) = M_t H_t / V_t$. By multiplying all $K_t$ across $T+1$ time steps along a "trajectory index" $(n_0,\dots,n_T)$ and summing over all index combinations, one obtains the likelihood estimate $\hat L^N = \frac{1}{N^{T+1}}\sum_{n_0,\dots,n_T}\prod_t K_t$ (Eq. 19). Summing over all indices except $n_t$ yields the marginal weight $w_t^{n_t}$ at time $t$ (Eq. 18). It is proven that $\hat L^N$ is unbiased for $p(y_{0:T})$ (Prop 3.1) and converges at $\mathcal{O}_P(N^{-1/2})$ (Prop 3.2-3.3).
    - **Design Motivation**: DSMC cannot be parallelized because resampling makes the proposal at $t$ dependent on all particles at $t-1$. VAE bounds are loose because they only consider $N$ trajectories instead of $N^{T+1}$. Transverse decomposition + joint importance measures bypass both issues—proposals can be sampled independently, and the bound utilizes the exponential trajectory space.

2. **Prefix/Suffix Associative Scan for Marginal Weights**:
    - **Function**: Computes marginal weights $w_t^{1:N}$ in parallel across all time steps $t$ with $\mathcal{O}(\log N \times \log T)$ span complexity, avoiding sequential forward-backward passes.
    - **Mechanism**: Kernel matrices from adjacent steps are packed into $a_s=(\{K_{2s}\}, \{K_{2s+1}\})\in \mathbb{R}^{N\times N}\times \mathbb{R}^{N\times N}$ with the associative operator $(C_1, C_2)\oplus(D_1, D_2):=(C_1, C_2 D_1 D_2)$ (Eq. 20). After running prefix scan $b_s$ and suffix scan $\hat b_s$, Theorem 3.1 provides a closed-form expression for $w_t^i$ extracted from $\{b_s, \hat b_s\}$ based on temporal parity (Eq. 22). The span for two $N\times N$ matrix multiplications is $\mathcal{O}(\log N)$, and the scan adds $\mathcal{O}(\log T)$ in the time dimension, resulting in a total span of $\mathcal{O}(\log N \times \log T)$. Backpropagation proceeds through the same scan tree with unchanged depth.
    - **Design Motivation**: The summation over $n_{-t}$ indices appears to be a brute-force $N^T$ enumeration; however, the chain structure of $\prod_t K_t$ makes the index summation equivalent to matrix chain multiplication. The associativity of matrix multiplication allows for Blelloch-style scan log-depth parallelism. This is a critical engineering step to translate "probabilistic forward-backward inference" into "hardware parallel scans."

3. **PVMC ELBO: A Tighter Bound than IWAE**:
    - **Function**: The training objective $\mathcal{L}^N_{\text{PVMC}} = \mathbb{E}[\log \hat L^N]$ lower bounds $\log p(y_{0:T})$ via Jensen's inequality and is strictly tighter than the IWAE bound.
    - **Mechanism**: Theorem 3.2 establishes the hierarchy $\log p \geq \mathcal{L}^N_{\text{PVMC}} \geq \mathcal{L}^N_{\text{IWAE}} \geq \mathcal{L}^{\tilde N}_{\text{IWAE}} \geq \mathcal{L}^N_{\text{P-VAE}}=\mathcal{L}^N_{\text{VAE}}$ (Eq. 29). PVMC also tightens monotonically with $N$ (Eq. 30). Intuitively, IWAE sums over $\frac{1}{N}\sum_n \prod_t K_t(X_t^n, X_{t-1}^n)$—weighting only $N$ "diagonal" trajectories. In contrast, the PVMC sum $\hat L^N = \frac{1}{N^{T+1}}\sum_{n_0,\dots,n_T}\prod_t K_t$ weights all $N^{T+1}$ potential trajectory combinations, leading to a smaller Jensen gap.
    - **Design Motivation**: Tighter bounds directly translate to better likelihoods in generative tasks (e.g., financial time-series modeling) and provide more stable gradient signals for supervised tasks. The P-VAE ablation (using the PVMC sampler but training with a VAE-style objective) shows a performance drop in filtering MSE from 0.40 to 1.21 and 2-SWD from 2.96 to 20.9, highlighting the importance of tight bounds for making the trained DSSM reusable by classic particle filters.

### Loss & Training
- **Generative**: Directly maximize $\mathcal{L}^N_{\text{PVMC}} = \mathbb{E}[\log \hat L^N]$.
- **Supervised**: Minimize $-\mathcal{L}^N_{\text{PVMC}} + \beta \sum_t \|\sum_n w_t^n X_t^n - x_t^\star\|^2$ (a linear combination of ELBO and state estimation MSE).
- Since the proposal is fully factorizable and $V_t$ uses reparameterization sampling, gradients for the entire pipeline are unbiased (unlike most DSMC methods).
- **Implementation**: Implemented based on PyDPF (Brady et al., 2025) and tested on a single NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

**Linear Gaussian System** (5D state, compared with analytical RTS smoother):

| Method | $e_x$ (vs. RTS mean) | Time (s) | KSD |
| :--- | :--- | :--- | :--- |
| Kalman Filter | 0.132 | 0.13 | — |
| TFS (Classic Two-Filter Smoother) | 0.501 | 25.9 | 0.410 |
| d-SMC | 0.44 | 4.00 | 2.21 |
| **PVMC (Kalman proposal)** | **0.054** | **1.88** | **0.200** |
| **PVMC (learned proposal)** | **0.052** | **1.50** | **0.199** |

The learned neural proposal almost matches the Kalman analytical proposal, proving that the PVMC ELBO can learn effective proposals.

**Prey-Predator Supervised State Estimation** (256-step stochastic Lotka-Volterra + Poisson observations, 20 independent runs):

| Method | MSE | Filtering MSE | 2-SWD | Time (m:s) | Failures (/20) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Stop-gradient DPF | 0.83±0.50 | 0.72±0.46 | 14.8±9.4 | 16:27 | 2 |
| Soft DPF | 0.62±0.42 | 0.58±0.42 | 6.70±4.30 | 15:32 | 7 |
| Diffusion DPF | 0.52±0.22 | 0.56±0.16 | 10.2±4.28 | **267:10** | 0 |
| MDPS | 1.20±0.55 | 1.32±0.64 | 13.5±10.0 | 26:23 | 14 |
| P-VAE Ablation | 0.43±0.06 | 1.21±0.11 | 20.9±2.6 | 1:49 | 0 |
| **PVMC** | **0.32±0.04** | **0.40±0.03** | **2.96±0.74** | **1:49** | **0** |

Ours achieved 100% convergence across 20 runs and outperformed others on all metrics. Training time was ~10× faster than Soft DPF and ~150× faster than Diffusion DPF.

**Financial Time-Series Generation** (SPX daily returns, 120-day window, 2014-2024): PVMC best captured the short-term autocorrelation structure of |return| and squared return across six non-overlapping 360-day trajectories. DMM and Soft-DPF failed to learn volatility clustering, while P-VAE and TC-VAE underestimated the magnitude and spread of skewness/kurtosis.

### Ablation Study

| Config | MSE | Filtering MSE | 2-SWD | Note |
| :--- | :--- | :--- | :--- | :--- |
| PVMC (Full) | 0.32 | 0.40 | 2.96 | ELBO + scan |
| P-VAE (VAE objective) | 0.43 | 1.21 | 20.9 | Same sampler/architecture, different loss |
| PVMC (Kalman proposal) | 0.054 ($e_x$) | — | 0.200 (KSD) | Analytical instead of learned proposal |
| PVMC (learned proposal) | 0.052 ($e_x$) | — | 0.199 (KSD) | Default |

### Key Findings
- **Role of Tight Bounds**: P-VAE performs reasonably on MSE (0.43) but fails on filtering MSE (1.21) and 2-SWD (20.9), indicating that DSSMs trained with loose bounds collapse when reused with classic particle filters. The PVMC ELBO learns truly self-consistent DSSMs.
- **Parallel vs. Sequential**: Soft / Stop-grad / Diffusion / MDPS require temporal resampling, with single-epoch times ranging from 15-267 minutes. PVMC requires only 1:49. While Diffusion DPF has the lowest variance, the 150× compute gap makes it impractical for large-scale training.
- **Training Stability**: The DPF family had 2/7/0/14 failures, while PVMC had 0. This is attributed to unbiased gradients and the absence of REINFORCE through discrete resampling variables.
- **Learned vs. Analytical Proposals**: In the linear Gaussian setting, PVMC with a learned proposal matched the Kalman analytic proposal, indicating that the ELBO signal is sufficient to replace structural prior knowledge.

## Highlights & Insights
- **Associative scan as a replacement for forward-backward is a elegant engineering abstraction**: The core of classic smoothing is a chain tensor product. If the proposal is factorizable in time, this product becomes a scan over an associative semigroup, which is hardware-friendly. This maps Bayesian operators (traditionally considered sequential) to GPU prefix scans, applicable to HMMs, CRFs, CTC, etc.
- **Transverse proposals are an undervalued choice**: DSMC often assumes proposals must depend on the previous state ("propagate-and-weight"). Removing this assumption enables parallelism and unlocks a tighter ELBO because the joint proposal now spans all $N^{T+1}$ potential trajectories.
- **The bound hierarchy provides a clear theoretical narrative**: The transition from VAE → IWAE → PVMC represents a step-by-step tightening of relaxation by utilizing particle combinations vs. single trajectories. This allows PVMC to gain a strictly tighter bound than IWAE without adding sampling overhead.
- **Reusability by classic filters as an evaluation metric**: Reporting both learning-time MSE and "filtering MSE using a bootstrap PF on the learned SSM" exposes that VAE-style methods, while looking good on internal metrics, fail when a different inference engine is used. This is a valuable robustness measure.

## Limitations & Future Work
- The authors acknowledge that more complex proposals (e.g., structured inference models, PF proposals) "require more refined importance sampler derivations." The current restriction to fully factorizable proposals may limit fitting quality for sequences with very strong long-range dependencies.
- Regarding spatial complexity, the $N\times N$ kernel matrices must be stored for all $T$ steps, leading to memory growth of $N^2 T$. VRAM limitations may restrict the use of very large $N$.
- The SPX task only evaluates distributional moments (autocorrelation, etc.); it lacks downstream financial backtesting metrics.
- Future work could extend the method to non-factorizable proposals, integrate auxiliary PFs, or link the scan framework to deterministic SSMs like S4/Mamba for probabilistic extensions.

## Related Work & Insights
- **vs. Differentiable SMC (Soft/Stop-grad/Diffusion DPF)**: DPFs maintain differentiability by smoothing resampling, but the forward pass remains sequential. PVMC skips resampling in favor of proposal-only sampling + scans, achieving 1-2 orders of magnitude speedup with unbiased gradients.
- **vs. MDPS (Mixture Density Particle Smoother)**: MDPS fuses two particle filters and inherits biased gradients; PVMC provides consistent estimates by performimg importance weighting directly on the joint smoothing measure.
- **vs. IWAE / DMM / TC-VAE**: VAE approaches lack particle interaction, have loose bounds, and lack per-step supervision. PVMC allows particles to interact "via weights" in the scan (without resampling), enabling both supervision and tighter bounds.
- **vs. Särkkä-García-Fernández (2021) and Zhao-Linderman (2023)**: These methods rely on linear Gaussian structures for scans. PVMC generalizes scans to non-linear, non-Gaussian scenarios via general SSMs and importance weighting.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first end-to-end differentiable, unbiased, log-depth parallel particle smoother that bridges the gap between VAE and DSMC. The combination of associative scans and transverse proposals is a clean, theoretically grounded abstraction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers linear-Gaussian, non-linear supervised estimation, and real-world financial generation. Supervised tasks included 20 repetitions with failure rates. Scalability curves for N and T and systematic memory reports are missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theorems and algorithms are clear and reproducible. Figure 1 effectively distinguishes the sampling-weighting structures of PVMC, IWAE, and DSMC.
- **Value**: ⭐⭐⭐⭐⭐ Simultaneously providing 10× speedup, zero failures, and tighter bounds offers immediate benefits to time-series modeling, state estimation, and generative modeling researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](../../CVPR2026/image_generation/smoothing_the_score_function_for_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](../../ICLR2026/image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](../../CVPR2026/image_generation/dip_taming_diffusion_models_in_pixel_space.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
