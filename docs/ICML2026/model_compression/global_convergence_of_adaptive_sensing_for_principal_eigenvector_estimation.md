---
title: >-
  [Paper Note] Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation
description: >-
  [ICML 2026][Model Compression][Paper Note] This paper establishes the optimal convergence rate for compressed streaming PCA. The upper bound for the Oja algorithm using two **adaptive** measurements per step under noisy observations matches the information-theoretic lower bound ($\Theta(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$). It reveals for the first time th
tags:
  - ICML 2026
  - Model Compression
date: 2026-05-08
content_hash: 3bd755193a02a88e
---
# Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation

**Conference**: ICML 2026  
**arXiv**: [2505.10882](https://arxiv.org/abs/2505.10882)  
**Code**: To be confirmed  
**Area**: Optimization Theory / Streaming PCA  
**Keywords**: Principal Component Analysis, Adaptive Sensing, Streaming Learning, Compressed Measurements, Convergence Analysis

## TL;DR
This paper establishes the optimal convergence rate for compressed streaming PCA. The upper bound for the Oja algorithm using two **adaptive** measurements per step under noisy observations matches the information-theoretic lower bound ($\Theta(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$). It reveals for the first time that the fundamental cost of compression relative to full observation is an extra factor of $d$, while adaptivity saves a factor of $d$ compared to non-adaptive sensing.

## Background & Motivation

**Background**: Classic PCA requires full $d$-dimensional samples. However, in hardware-constrained scenarios such as mmWave communication, neural signals, and radar, only a few scalar measurements can be taken per sample. The Oja streaming algorithm is the benchmark for handling such constraints, but existing analyses are predominantly based on full observations.

**Limitations of Prior Work**: Designing and analyzing PCA under compressed observations is difficult. Adaptive GROUSE only analyzes the noise-free case ($\lambda_2 = 0$) and cannot handle real data with tail eigenvalues. More importantly, there is a lack of information-theoretic lower bounds for compressed PCA—leaving the fundamental limit of "two measurements per step" unknown.

**Key Challenge**: Adaptive sensing (along the direction of the current estimate) can improve convergence speed but introduces signal-noise coupling that complicates analysis. At the same time, it is necessary to balance "exploitation" (along the current estimate) and "exploration" (along orthogonal directions).

**Goal**: (1) Prove global convergence guarantees for compressed Oja under noisy observations; (2) Establish the optimal rate for adaptive compressed PCA compared to full observation and non-adaptive sensing; (3) Provide the first information-theoretic lower bound for compressed eigenvector estimation.

**Key Insight**: The problem is formalized as taking two adaptive linear measurements per step, with a measurement matrix designed to balance exploitation and exploration. The origin of the $d^2$ factor is revealed through Assouad’s lemma combined with a measurement energy budget argument.

**Core Idea**: By tracking the stochastic recurrence of the expected cosine alignment and utilizing a measurement budget argument, it is shown that the additional cost of adaptive compressed PCA relative to full observation is **exactly** a factor of $d$, whereas it saves a factor of $d$ relative to non-adaptive sensing.

## Method

### Overall Architecture
This paper investigates streaming PCA under hardware constraints: samples $v_t \sim \mathcal{N}(0, \Sigma)$ arrive one by one, but the environment can only obtain **2 scalar measurements** per step instead of the full $d$-dimensional vector. The goal is to estimate the principal eigenvector $\bar{u}$ online from these compressed observations. The core mechanism of the algorithm (Compressed Oja, Algorithm 1) is: using the current estimate $u_t$ to select a $2 \times d$ measurement matrix $A_t$, compressing the sample into $x_t = A_t v_t \in \mathbb{R}^2$, reconstructing a projection from these 2 values, and feeding it into a standard Oja update. The entire analysis revolves around a single scalar—the cosine alignment $c_t = \bar{u}^\top u_t$—tracking the stochastic recurrence of $\mathbb{E}[c_t^2]$ and attributing the costs of "compression + adaptivity + noise" to the constants within this recurrence.

### Key Designs

**1. Adaptive Sensing: Using the Current Estimate as a Searchlight to Balance Exploitation and Exploration**

With a budget of only 2 scalars per step, how to allocate them is the starting point. Selecting measurement directions completely at random results in an expected overlap with the true principal direction $\bar{u}$ of only $O(1/d)$, effectively wasting the budget on directions nearly orthogonal to the signal. Ours approach makes the measurement matrix follow the estimate: $A_t = [u_t^\top;\, b_t^\top]$, where the first row is along the current estimate $u_t$ (**Exploitation**, to reinforce existing alignment) and the second row $b_t$ is a unit vector sampled uniformly from the orthogonal complement of $u_t$ (**Exploration**, to obtain corrective information). The observations are thus two projections $x_t = [u_t^\top v_t;\, b_t^\top v_t]$. During the update, the projection $\tilde{v}_t = (u_t u_t^\top + b_t b_t^\top) v_t$ is reconstructed first, followed by the Oja step $u_{t+1} = \text{Norm}(u_t + \eta_t \tilde{v}_t v_t^\top u_t)$. Crucially, this is a positive feedback mechanism: even if the initial alignment is poor, the measurement along $u_t$ amplifies the "small existing alignment" into an effective gradient, gradually turning the searchlight toward $\bar{u}$. By introducing auxiliary scalars $z = \bar{u}^\top b$, $g = v^\top u$, and $h = v^\top b$, the impact of each update on $c_t^2$ can be written as a closed-form recurrence.

**2. Two-stage Step-size: Monotonic Ascent followed by Optimal Decay**

A single global step-size cannot handle the entire trajectory—the early stage needs stability due to low alignment, while the later stage needs speed. This paper splits training into two phases. The warm-up phase uses a fixed step-size $\eta_0 = (d-1)/(2 S \Delta)$ to ensure $c_t^2$ rises monotonically without oscillation, where the constant $S = 3 \lambda_1 \lambda_2 d^2 / \Delta^2 + 15 \lambda_1 d / \Delta$ absorbs the effects of tail noise $\lambda_2$ and dimension $d$. Once alignment exceeds $c_t^2 \geq 0.5$ and enters the local convergence region, the algorithm switches to a decaying step-size $\eta_t = 2(d-1) / [\Delta (4S + t - t_0)]$. This allows the residual $1 - c_t^2$ to anneal at $O(1/t)$, avoiding oscillations while leveraging $1/t$ "inertia" to reach the optimal rate. The final upper bound (Theorem 4.1) states that after the warm-up period $t_0 = (4S+1)\log(d/2)$:

$$\mathbb{E}[1 - (\bar{u}^\top u_t)^2] \leq \frac{C_1}{4S + (t - t_0)} + \frac{C_2}{[4S + (t-t_0)]^2},$$

which is asymptotically $\mathcal{O}(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$—this is the source of the $d^2$ factor in $S$.

**3. Signal-Noise Decomposition + Energy Budget Lower Bound: Translating "2D Compression" into Energy Allocation**

This point supports both the analyzability of the upper bound and the tightness of the lower bound. The difficulty in the upper bound is dealing with tail eigenvalues $\lambda_2 > 0$. The GROUSE family is easy to analyze because it assumes $\lambda_2 = 0$, causing cross-terms to cancel; however, real data is noisy and these terms cannot be ignored. Ours uses the Isserlis theorem (Gaussian fourth moments) and Cauchy-Schwarz to bound the coupling term $\mathbb{E}[gh \mid c, z] = u^\top \Sigma b$ by $\sqrt{(u^\top \Sigma u)(b^\top \Sigma b)}$, where squared terms contribute $2 a^2 b^2$ (with $a^2 = \Delta c^2 + \lambda_2$ and $b^2 = \Delta z^2 + \lambda_2$). Jensen's inequality and monotonicity are then used to bypass adaptive coupling from matrix concentration, allowing the variance of the recurrence to close. On the lower bound side, Assouad’s lemma is paired with a measurement energy budget argument: 2 unit-norm measurements per step for $t$ steps yield a total energy of $2t$. When spread across $d-1$ coordinates, each coordinate receives only $O(t/d)$ energy, corresponding to a per-coordinate error of $O(d/t)$. Summing these gives $\Theta(d^2/t)$. Thus, the lower bound (Theorem 4.2) $\inf_{\hat{u}} \sup_P \mathbb{E}[1 - (\bar{u}^\top \hat{u}_t)^2] \geq \Omega(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$ **tightly matches** the upper bound. Comparing the three schemes makes the "cost of compression" clear: full observation is $\Theta(d/t)$, adaptive compression is $\Theta(d^2/t)$, and non-adaptive compression is $\Omega(d^3/t)$. Compression costs an extra factor of $d$ compared to full observation, but adaptivity recovers a factor of $d$ over non-adaptive sensing.

## Key Experimental Results

### Main Results

| Dimension $d$ | Adaptive Iterations | Non-adaptive Iterations | Speedup |
|---------|------------|--------------|---------|
| 16 | $3.8 \times 10^4$ | $1.6 \times 10^5$ | 4.2× |
| 32 | $1.9 \times 10^5$ | $1.3 \times 10^6$ | 7.1× |
| 64 | $8.4 \times 10^5$ | $1.2 \times 10^7$ | 14× |

Median iterations required to reach a target error of $10^{-2}$ (20 trials, $\eta = 0.01/d$).

### Dimension Scaling

| Dimension $d$ | Iterations | $t_d / t_{d/2}$ |
|---------|-------|----------------|
| 16 | 35,500 | — |
| 32 | 172,750 | 4.87 |
| 64 | 879,190 | 5.09 |
| 128 | 4,091,830 | 4.65 |
| 256 | 17,950,000 | 4.39 |
| 512 | 68,500,000 | 3.82 |
| 1024 | 284,650,000 | 4.15 |

The ratios between adjacent dimensions are between 3.8–5.1, consistent with the theoretical $d^2$ (expected ratio of 4). The fitted exponent is 2.16, slightly exceeding 2 due to the $O(d)$ term in $S$ and the $\log d$ warm-up contributions.

### Key Findings
- The adaptive speedup factor grows with dimension (4× → 14×), quantifying the advantage of adaptive directions having much higher overlap than fixed directions.
- The upper and lower bounds differ only by a constant factor (~10⁴), showing a tight match.
- In a non-stationary setting where the optimal point moves at speed $V$, the optimal step-size $\eta^* = \sqrt{V/S}$ yields a steady-state error of $V + \sqrt{V S}$, verifying generalization to non-stationary data.

## Highlights & Insights
- **First Information-Theoretic Lower Bound for Compressed PCA**: The $\Omega(d^2)$ lower bound is provided using a measurement energy budget argument, and the $d$-fold extra cost for non-adaptive sensing intuitively reveals the "value of adaptivity."
- **Breaking Noise-Free Limits of the GROUSE Family**: The combination of signal-noise decomposition, Isserlis fourth-moment correction, and explicit integration of exploration directions allows the noisy setting to be analyzed for the first time.
- **Clear Separation of Three Schemes**: The complexity comparison of full observation ($d^1$), adaptive compression ($d^2$), and non-adaptive compression ($d^3$) provides theoretical guidance for choosing sampling strategies.
- **New Application of Assouad’s Lemma**: A byproduct is a new proof of the classic full-observation PCA lower bound $\Omega(\lambda_1 \lambda_2 d / (\Delta^2 t))$, replacing existing Fano arguments with a more general technique.

## Limitations & Future Work
- The quadratic dependence on dimension limits practical utility in very high dimensions ($d > 10^4$), which is a fundamental bottleneck of "two measurements."
- Limited to rank-1 estimation; extending to rank $k > 1$ requires handling $k$ coupled recurrences and orthogonalization. The authors speculate that taking $m = 2k$ measurements incurs a $(d/m)^2$ penalty.
- The Gaussian assumption could be generalized via sub-Gaussian moment bounds and Le Cam’s $\chi^2$ divergence, though details are not expanded.
- The compressed version of sparse PCA remains an open problem.

## Related Work & Insights
- **vs. Full Observation Oja**: $\Theta(\lambda_1 \lambda_2 d / \Delta^2 t)$ vs. Ours $\Theta(d^2 / \Delta^2 t)$; both reveal that compression costs a factor of $d$.
- **vs. Adaptive GROUSE (Ongie et al. 2017)**: Only noise-free analysis is provided; ours handles $\lambda_2 > 0$ for the first time, borrowing recurrence ideas but adding signal-noise decomposition.
- **vs. Randomized SVD (Halko et al. 2011)**: Batch vs. streaming; both aim to bypass the curse of dimensionality. Ours shows that adaptivity can compensate for part of the cost.
- **Insight**: The energy budget argument is a general tool for lower bounds in constrained observation problems (radar, MRI, covariance estimation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First tight bound for noisy compressed PCA; original lower bound technique using energy budget + Assouad.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified adaptive vs. non-adaptive, dimension scaling, and tracking stability; lacks experiments on real-world radar/MRI data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, technical details well-balanced with intuitive explanations, and precise theorem statements.
- Value: ⭐⭐⭐⭐⭐ Fills a gap at the intersection of compressed sensing and streaming learning; provides theoretical guidance for hardware-limited applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ECCV 2024\] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](../../ECCV2024/model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [\[ECCV 2024\] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](../../ECCV2024/model_compression/adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)
- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICML 2026\] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)

</div>

<!-- RELATED:END -->
