---
title: >-
  [Paper Note] Improved High-Dimensional Estimation with Langevin Dynamics and Stochastic Weight Averaging
description: >-
  [ICLR 2026][Learning Theory][Langevin Dynamics] This paper proves that combining spherical Langevin dynamics with **iterative time averaging** can recover the hidden direction $\theta^\star$ of single-index models / Tensor PCA using only $n \gtrsim d^{\lceil k^\star/2 \rceil}$ samples. Noise injection plus averaging spontaneously simulates the effect of "landscape smoothing" without requiring explicit smoothing operations.
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "High-Dimensional Statistical Estimation"
  - "Langevin Dynamics"
  - "Weight Averaging"
  - "Information Exponent"
  - "Sample Complexity"
  - "Single-Index Models"
  - "Tensor PCA"
date: 2026-05-08
content_hash: 4e6335b08e426c8b
---

# Improved High-Dimensional Estimation with Langevin Dynamics and Stochastic Weight Averaging

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=GYXUD3hGTh](https://openreview.net/forum?id=GYXUD3hGTh)  
**Code**: TBD  
**Area**: Learning Theory / High-Dimensional Statistical Estimation  
**Keywords**: Langevin Dynamics, Weight Averaging, Information Exponent, Sample Complexity, Single-Index Models, Tensor PCA  

## TL;DR
This paper proves that combining spherical Langevin dynamics with **iterative time averaging** can recover the hidden direction $\theta^\star$ of single-index models / Tensor PCA using only $n \gtrsim d^{\lceil k^\star/2 \rceil}$ samples. Noise injection plus averaging spontaneously simulates the effect of "landscape smoothing" without requiring explicit smoothing operations.

## Background & Motivation

**Background**: In high-dimensional problems such as Tensor PCA and single-index models $\sigma(\theta^\star\cdot x)$, the feasibility of recovering the hidden direction $\theta^\star \in S^{d-1}$ using gradient descent is primarily determined by the **information exponent** $k^\star$ of the link function (the order of the first non-zero Hermite coefficient of $\sigma$, which corresponds to the order of the saddle point at the initialization of the population landscape). Ben Arous et al. (2021) proved that online SGD requires and only requires $n \gtrsim d^{\max(1,k^\star-1)}$ samples, while Ben Arous et al. (2020) provided a lower bound of the same order for Langevin dynamics.

**Limitations of Prior Work**: Damian et al. (2023) found a way to **bypass** this lower bound by running SGD on a **smoothed landscape**, reducing the sample complexity to the optimal $n \gtrsim d^{\max(1,k^\star/2)}$. The principle is that smoothing enhances the signal-to-noise ratio in the equatorial region ($|\theta \cdot \theta^\star| \lesssim d^{-1/2}$). however, this relies on **explicit landscape smoothing** operations.

**Key Challenge**: There are two ends to the signal-to-noise ratio—smoothing "actively raises the SNR." Conversely, the negative conclusions of Ben Arous asserted that "low SNR, purely noise-driven" Langevin dynamics **cannot escape the equator** and are destined to fail. Is it possible to achieve the same optimal rate using native Langevin noise without any explicit smoothing?

**Goal**: To prove that Langevin dynamics itself is sufficient, provided a neglected readout method is used: focusing on the **time average of iterates** rather than the **final iterate**.

**Core Idea**: **Noise Injection + Iterative Averaging $\approx$ Implicit Landscape Smoothing**. The algorithmic trajectory $\theta(t)$ remains close to the equator throughout, with extremely low correlation to $\theta^\star$ (confirming Ben Arous's "cannot escape the equator" observation). However, after performing **time averaging** on the entire trajectory, the average quantity converges to $\theta^\star$. This is established through an **ergodic concentration** argument for spherical Brownian motion.

## Method

### Overall Architecture
The algorithm runs a spherical Langevin SDE with a temperature parameter $\epsilon$ (spherical Brownian motion + a weak drift term) until time $T$. It then computes the time average of the trajectory to obtain a first-order estimator $\hat\theta = \frac{1}{T}\int_0^T \theta_t \, dt$ and a second-order estimator $\hat M = \frac{1}{T}\int_0^T \theta_t \theta_t^\top \, dt$. When the information index $k^\star$ is odd, it returns the normalized $\hat\theta$; when $k^\star$ is even, it returns the top eigenvector of $\hat M$. The core of the analysis involves decomposing $\theta_t$ into "pure Brownian motion $\beta_t$ + error $E_t$ of magnitude $O(\epsilon)$" and performing concentration term-by-term using spherical ergodicity.

```mermaid
flowchart LR
    A[Uniform Init θ₀∼μ on S^{d-1}] --> B[Run spherical Langevin SDE to T<br/>dθ = (-(d-1)/2·θ + ε·b θ)dt + P⊥dW]
    B --> C[Time Averaging]
    C --> D1[1st Order: θ̂ = 1/T ∫θ_t dt]
    C --> D2[2nd Order: M̂ = 1/T ∫θ_t θ_tᵀ dt]
    D1 -->|k* odd| E1[Return θ̂/‖θ̂‖]
    D2 -->|k* even| E2[Return top eigenvector of M̂]
    E1 --> F[Optional: Warm-start online SGD<br/>Saves additional √d → d^{k*/2}]
    E2 --> F
```

### Key Designs

**1. Spherical Langevin Dynamics: Driven by pure noise, not smoothing.** The SDE executed by the algorithm (Algorithm 1) is:

$$d\theta = \Big(-\frac{d-1}{2}\theta + \epsilon\, b(\theta)\Big)dt + P^\perp_\theta\, dW_t,\qquad b(\theta):=-\nabla_\theta L_n(\theta)$$

where $P^\perp_\theta = I - \theta\theta^\top$ projects the gradient back onto the tangent space, ensuring $\theta_t$ stays on the sphere. This is the natural analogue of standard Langevin on $S^{d-1}$. The inverse temperature $\epsilon$ controls the relative strength of signal and noise. As $\epsilon \to 0$, the SDE degenerates into **pure Brownian motion** on the sphere $d\beta = -\frac{d-1}{2}\beta \, dt + P^\perp_\beta dW_t$, whose stationary distribution is the uniform measure $\mu$. Contrary to smoothing methods that deliberately raise equatorial SNR, this work **actively remains in the low SNR regime**: letting noise dominate and "integrating out" the weak signal hidden in the noise through subsequent averaging. Another distinction from previous work is the use of **ERM (Empirical Risk Minimization) loss** on full batches rather than online sample-by-sample updates.

**2. Ergodic Concentration: Averaging Brownian components to zero to precipitate the signal.** Substituting $\theta_t = \beta_t + E_t$ (coupled to the same noise $W_t$, with $\theta_0 = \beta_0$ and $E_0 = 0$) into the time average:

$$\frac{1}{T}\int_0^T \theta_t \, dt = \underbrace{\frac{1}{T}\int_0^T \beta_t \, dt}_{\text{Brownian component} \to 0} + \underbrace{\frac{1}{T}\int_0^T E_t \, dt}_{\text{Signal component}}$$

The first term concentrates to $0$ via the ergodicity of Brownian motion. Specifically, for any zero-mean $f \in L^2(\mu)$, the authors prove (Lemma 1–2) that $\frac{1}{T}\int_0^T f(\beta_t) dt$ can be written as $\frac{\phi(\beta_0) - \phi(\beta_T)}{T} + \frac{M_T}{T}$ (where $M_T$ is a martingale). Its magnitude is controlled by the Ricci curvature $\rho = d-2$ of $S^{d-1}$, so the convergence rate is determined by terms such as $\frac{\sup\|\nabla f\|}{(d-2)T}$—this is the engine of the entire proof. The second term can be approximated as $\frac{1}{T}\int_0^T E_t \, dt \approx \frac{\epsilon}{d} \cdot \frac{1}{T}\int_0^T b(\theta_t) \, dt$, which again converges via ergodicity to the direction of $\mathbb{E}_{z\sim\mu}[\nabla L_n(z)]$. In both settings, this happens to be exactly the partial-trace estimator direction required to recover $\theta^\star$. A matching high-probability uniform bound (Lemma 3) ensures $\sup_t \|E_t\| = O(\epsilon \sup \|b\|/d)$ holds throughout (using a mapping from OU processes to sub-Gaussian processes + chaining), which is the key component for both odd and even cases.

**3. Case Analysis for Parity: 1st Order vs. 2nd Order.** For **odd** $k^\star$ (Theorem 2), since $\sigma'$ is an odd function, the first-order time average $\hat\theta$ converges toward the direction of $\bar b = \mathbb{E}_{z\sim\mu}[b(z)]$ (with magnitude $\tilde O(\epsilon)$), thereby recovering $\theta^\star$ with $n \gtrsim d^{\lceil k^\star/2 \rceil}$. For **even** $k^\star$ (Theorem 3), due to the symmetry of the uniform distribution/Brownian motion, $\mathbb{E}_{z\sim\mu}[\nabla L_n(z)] \approx 0$, rendering first-order quantities useless. In this case, one must examine the second-order information from the time average of $\theta\theta^\top$:

$$\hat M = \frac{1}{T}\int_0^T \theta_t \theta_t^\top \, dt \ \longrightarrow \ \frac{I}{d} + \frac{\epsilon}{d} \, \mathbb{E}_{z\sim\mu}[zb(z)^\top + b(z)z^\top]$$

The main term $I/d$ is an isotropic background. When $n \gtrsim d^{k^\star/2}$, a rank-one spike for $\theta^\star \theta^{\star \top}$ emerges in the middle term. Thus, taking the **top eigenvector** of $\hat M$ recovers $\theta^\star$.

**4. Warm-start saving $\sqrt{d}$ + Bridge to minibatch SGD.** The estimators obtained above can serve as a **warm-start** for online SGD. When $n = \Omega(d^{k^\star/2})$ (reducing the complexity by a factor of $\sqrt{d}$ compared to Theorem 2), the average estimator already achieves a correlation of $\Theta(d^{-1/4})$ with $\theta^\star$. Transitioning to online SGD (Ben Arous 2021) from this point allows the total sample complexity to be compressed to the optimal $d^{k^\star/2}$ (Corollary 1). Furthermore, the authors perform an SDE approximation of normalized minibatch SGD with batch size 1, finding that when the learning rate $\eta \ll d^{-1/2}$, $d\theta \approx \big(-\frac{d-1}{2}\theta + \frac{1}{\eta} b(\theta)\big)dt + P^\perp_\theta dW_t$. This is exactly equivalent to the Langevin setting with $\epsilon := 1/\eta$, leading to the **conjecture that pure minibatch SGD can achieve the same rate without any additional noise injection**.

## Key Experimental Results

This is a theoretical work; "experiments" are numerical simulations for sanity-checking the theory, using $d=100$, $\sigma=h_{k^\star}$ (link function with information exponent $k^\star$), and minibatch updates with batch size 1.

### Main Results (Numerical Verification)

| Setting | $k^\star$ | Sample Size | 1st Order Estimator $\hat\theta$ | 2nd Order Top Eigenvector | Actual Iterate $\theta_t$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Odd (Fig.1) | 3, 5 | $n=10\,d^{\lceil k^\star/2\rceil}$ | ✅ Converges to $\theta^\star$ | — | Stays at equator |
| Even (Fig.2) | 4 | $n=10\,d^2$ | ❌ Does not converge | ✅ Converges to $\theta^\star$ | Stays at equator |

### Key Findings
- **Average iterates converge, instantaneous iterates do not**: Dark curves (time averages) steadily climb toward a correlation of $1$, while light curves (actual $\theta_t$) stay near the equator—visibly confirming that "one can estimate $\theta^\star$ without escaping the equator."
- **Bipolar behavior of learning rates**: Smaller learning rates behave more like gradient flow (explicit optimization), while larger ones act more like Brownian motion and stay closer to the equator, consistent with Langevin predictions.
- **Higher-order necessity for even cases**: At $k^\star=4$, the 1st order estimator fails as expected, while the principal eigenvector of the 2nd order estimator succeeds, validating the necessity of the parity-based approach.

## Highlights & Insights
- **Counter-intuitive Positive Result**: In the setting where Ben Arous et al. (2020) asserted "Langevin fails on Tensor PCA due to the divergence of the calculation-statistical gap," this paper reverses the verdict by changing the readout—it is the final iterate that fails, not the algorithm itself.
- **Noise as a Tool rather than an Enemy**: Instead of pushing SNR higher (smoothing), this work uses low SNR + averaging, allowing noise to spontaneously perform what smoothing is intended to do. It reveals the equivalence of "Implicit Smoothing = Noise + Averaging."
- **Geometry-driven Proof**: The sample complexity is reduced to the ergodic concentration of spherical Brownian motion, with the convergence rate directly linked to the Ricci curvature $\rho = d-2$ of $S^{d-1}$, yielding a clean theoretical structure.
- **Bridging Langevin and SGD**: The SDE approximation maps SGD with batch size 1 to Langevin with $\epsilon = 1/\eta$, providing a credible conjecture that "pure SGD should have the same guarantees," connecting two parallel lines of research.

## Limitations & Future Work
- **Still $\sqrt{d}$ from Optimal (without warm-start)**: Running Algorithm 1 alone yields $d^{\lceil k^\star/2 \rceil}$; matching the optimal $d^{k^\star/2}$ still requires a subsequent phase of online SGD.
- **Minibatch SGD remains a conjecture**: The core challenge is not just discretization error, but also that the stationary distribution of the pure noise process is **no longer isotropic and depends on the data**, introducing additional coupling into the analysis that this paper did not prove.
- **Strong Assumptions**: Relies on standard Gaussian inputs and bounded link functions $\sigma$ and their first two derivatives (relaxing to polynomial tails is possible but complicates the proof). It also assumes the information exponent equals the generative index (assuming label transformations are pre-applied).
- **Temperature/Time Tuning**: The ranges for $\epsilon$ and $T$ (e.g., $T \gtrsim d^{k^\star}/\epsilon^2$) depend on $k^\star$; how to select these in practice when $k^\star$ is unknown remains unclear.

## Related Work & Insights
- **Information Exponent Taxonomy**: Ben Arous et al. (2021) defined the information exponent and gave $d^{k^\star-1}$ for SGD; Damian et al. (2023) improved this to $d^{k^\star/2}$ using smoothing, matching CSQ lower bounds; Damian et al. (2024) used the "generative index" via label transformation to further reduce the rate. This paper follows this line, providing a path to the same rate **without explicit smoothing**.
- **Tensor PCA Tools**: Methods such as tensor unfolding (Montanari & Richard 2014), partial-trace estimators (Hopkins et al. 2016), and homotopy/smoothing (Anandkumar et al. 2017; Biroli et al. 2020) all reach $d^{k/2}$. The time-averaging estimator in this paper essentially recovers the direction of partial-trace estimators but uses dynamics + ergodicity instead of explicit construction.
- **Inspiration**: For any high-dimensional non-convex optimization where "the final iterate is drowned in noise," trajectory-based time/weight averaging may be a free path toward implicit smoothing or variance reduction—elevating stochastic weight averaging from an empirical trick to a theoretically grounded estimation principle.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Reverses the verdict in a setting where "Langevin is destined to fail" by using average iterates and ergodic concentration, revealing "Noise + Averaging = Implicit Smoothing."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — As a purely theoretical work, numerical simulations cover odd/even $k^\star$, clearly validating "equatorial residence but average convergence" and parity-based logic. Scale is limited but sufficient to support the conclusions.
- **Writing Quality**: ⭐⭐⭐⭐ — The main argument (noise + averaging ≈ smoothing) is clear, and the proof structure is well-organized. Theoretical density is high, making it a steep climb for non-specialists.
- **Value**: ⭐⭐⭐⭐ — Nearly matches the optimal sample complexity, unifies Langevin and smoothing approaches, and indicates a credible direction for minibatch SGD guarantees. A substantial advancement in high-dimensional estimation theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] High-dimensional Analysis of Synthetic Data Selection](high-dimensional_analysis_of_synthetic_data_selection.md)
- [\[ICML 2026\] Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy](../../ICML2026/learning_theory/asymptotic_optimality_of_the_high-dimensional_gaussian_mechanism_and_improved_lo.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](../../ICML2026/learning_theory/on_the_robustness_of_langevin_dynamics_to_score_function_error.md)
- [\[ICLR 2026\] High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification](high-dimensional_analysis_of_single-layer_attention_for_sparse-token_classificat.md)

</div>

<!-- RELATED:END -->
