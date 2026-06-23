---
title: >-
  [Paper Note] Diffusion Bridge Variational Inference for Deep Gaussian Processes
description: >-
  [ICLR 2026][learning_theory][Paper Note] Aiming at posterior inference for inducing variables in Deep Gaussian Processes (DGP), this paper transforms the "reverse diffusion from a fixed Gaussian prior" in DDVI (Denoising Diffusion Variational Inference) into a "diffusion bridge starting from a learnable, data-dependent initial distribution." By leveraging the
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 5eab8849602a95e7
---
# Diffusion Bridge Variational Inference for Deep Gaussian Processes

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=zyRmy0Ch9a](https://openreview.net/forum?id=zyRmy0Ch9a)  
**Code**: To be confirmed  
**Area**: Probabilistic Methods / Variational Inference / Deep Gaussian Processes  
**Keywords**: Deep Gaussian Processes, Variational Inference, Diffusion Bridge, Doob h-transform, Amortized Inference

## TL;DR
Aiming at posterior inference for inducing variables in Deep Gaussian Processes (DGP), this paper transforms the "reverse diffusion from a fixed Gaussian prior" in DDVI (Denoising Diffusion Variational Inference) into a "diffusion bridge starting from a learnable, data-dependent initial distribution." By leveraging the Doob h-transform while maintaining the Girsanov-ELBO mathematical framework, it shortens the inference trajectory, achieving faster convergence and more accurate posteriors than DDVI in regression, classification, and image reconstruction tasks.

## Background & Motivation

**Background**: Deep Gaussian Processes (DGP) cascade multiple layers of Gaussian Processes to achieve hierarchical Bayesian expressivity far superior to single-layer GPs. However, posterior inference is notoriously difficult due to non-conjugate likelihoods and strong cross-layer coupling. For scalability, many inducing variables $u^{(l)}$ (located at inducing inputs $Z^{(l)}$) are introduced per layer. Standard approaches use Stochastic Variational Inference (SVI) to approximate the posterior as a factorized Gaussian, but such simple distributions often fail to capture the complex, multimodal true posteriors of deep models.

**Limitations of Prior Work**: The recently proposed DDVI (Denoising Diffusion Variational Inference) models the variational posterior of inducing variables as the **marginal distribution of a reverse-time diffusion SDE at $t=1$**. It uses a neural network to parameterize the score function in the reverse drift, enabling the representation of complex posteriors while inheriting a tractable ELBO via Girsanov's theorem. However, DDVI's reverse diffusion **always starts from a fixed, unconditional Gaussian $U_0 \sim \mathcal{N}(0, \sigma^2 I)$**.

**Key Challenge**: The true posterior of inducing variables is often far from this fixed Gaussian starting point. Consequently, the reverse-time SDE must traverse a long and winding trajectory to reach the target distribution. Long trajectories lead to inefficient inference, high variance, and slow convergence. Furthermore, since the starting point ignores observed data, the sampling is "input-independent," hindering amortization and scalability.

**Goal**: To make the **starting point of the reverse diffusion closer to the posterior** and **data-dependent** to support amortized inference, all while preserving the elegant reverse-time SDE and Girsanov-ELBO mechanism of DDVI.

**Key Insight**: Since the problem lies in the "fixed and distant starting point," the starting point can be treated as a **learnable, data-dependent distribution** that the ELBO gradient naturally pushes toward the posterior. Once the starting point is modified, the entire diffusion process transitions from a standard reverse diffusion to a **bridge process**, which can be rigorously characterized by the Doob h-transform.

**Core Idea**: Use an amortized network $\mu_\theta(x)$ to provide a **data-dependent starting point** for the reverse diffusion, and reinterpret it using the **Doob h-transform** as a diffusion bridge with constrained endpoints. This allows for the derivation of a trainable Bridge-ELBO within the DDVI framework, resulting in Diffusion Bridge Variational Inference (DBVI).

## Method

### Overall Architecture
DBVI addresses the "poor starting point in DDVI" by modifying three components: the starting point, the dynamics, and the training objective. The pipeline works as follows: Given a mini-batch, an **amortized network** computes a data-dependent initial distribution $p_0^\theta(U_0 \mid x) = \mathcal{N}(\mu_\theta(x), \sigma^2 I)$ based on inducing inputs. From this start, an **observation-conditioned reverse bridge SDE** is run, with its drift using the conditional score $s_{\text{cond}} = s_\phi + h$ (where $h$ is derived from the Doob h-transform). The endpoint of the reverse bridge at $t=1$, $U_1$, serves as the posterior sample for inducing variables, which are fed into the DGP forward pass to compute the likelihood $f^{(L)}$. Finally, a **Bridge-ELBO** (expressing KL in a score-matching form) is used to jointly train the amortized network $\mu_\theta$, the score network $s_\phi$, and the DGP hyperparameters $\gamma$.

Notably, DBVI only modifies the "start point + bridge correction term" while keeping the reverse SDE and Girsanov-ELBO intact. When $\mu_\theta(x) \equiv 0$, the DBVI loss **reduces exactly to the DDVI loss**, making it a strict generalization of DDVI.

### Key Designs

**1. Amortized Data-Dependent Initial Distribution: Turning Fixed Start into Learnable Start**

The root of DDVI's inefficiency is that $\mathcal{N}(0, \sigma^2 I)$ is far from the posterior. DBVI parameterizes the mean of the starting point as an amortized output:
$$p_0^\theta(U_0 \mid x) = \mathcal{N}\big(U_0;\ \mu_\theta(x),\ \sigma^2 I\big),$$
where only $\mu_\theta(x)$ is output by the network and $\sigma^2$ is fixed. This places the start closer to the posterior, shortening the SDE path. Since the start point is a learnable parameter in the ELBO, **gradients naturally push the initial distribution toward the posterior**, reducing the inference gap.

**2. Diffusion Bridge via Doob h-transform: Start Point Dynamics**

By making the starting point data-dependent, the process becomes a bridge process constrained by initial distributions. Using the **Doob h-transform** (Proposition 1), let
$$h(U_t, t, U_0) = \nabla_{U_t} \log p(U_0 \mid U_t).$$
The forward bridge drift adds $g(t)^2 h$ to the original drift $f(U_t, t)$, "steering" the path toward the target. The reverse-time bridge SDE is:
$$dU_t = \big[f(U_t, t) - g(t)^2 s_{\text{cond}}(U_t, t, U_0)\big]dt + g(t)\,dW_t,$$
where $s_{\text{cond}} = s(U_t, t, U_0) + h(U_t, t, U_0)$.

**3. Closed-form Gaussian Marginals and Score-Matching ELBO**

Proposition 2 proves that under linear drift and Doob correction, the bridge marginal at each $t$ remains Gaussian $p_t(U_{\text{Bri}} \mid x) = \mathcal{N}(U_{\text{Bri}}; m_t, \kappa_t I)$, with $m_t$ and $\kappa_t$ determined by coupled ODEs. Proposition 3 expresses the path KL between the variational reverse bridge $Q_\phi$ and the reference bridge in a solvable **score-matching** form:
$$\ell_{\text{DBVI}} = \mathbb{E}_{Q_\phi}\Big[-\log p_0^\theta(U_1) + \tfrac{N}{B}\log p(y_I \mid f^{(L)}) - \tfrac{1}{2}\!\int_0^1\! g(t)^2\big\|\tfrac{1}{\kappa_t}(U_t - m_t) + s_{\text{cond}}\big\|^2 dt + \log p_{\text{prior}}(U_1) - \mathrm{KL}\big(\mathcal{N}(\mu_\theta, \sigma^2 I) \,\|\, \mathcal{N}(m_1, \kappa_1 I)\big)\Big].$$

**4. Inducing Inputs $Z^{(l)}$ as Amortization Inputs: Solving Mismatch and Scalability**

Amortizing over the entire dataset $x$ is computationally infeasible, and mini-batches may introduce bias or dimension mismatch (input $[B, d_{\text{in}}]$ vs. $l$-th layer inducing variables $[M_l, d_{\text{out}}]$). DBVI uses **inducing inputs $Z^{(l)} \in \mathbb{R}^{M_l \times d_{\text{in}}}$ as inputs to the amortizer**. Inducing points are inherently representative of the data and their shape naturally aligns with $u^{(l)}$. Each layer's amortizer $\mu_\theta^{(l)}$ maps $Z^{(l)}$ point-wise to $\mathbb{R}^{M_l \times d_{\text{out}}}$.

### Loss & Training
The objective is the mini-batch ELBO $\ell_{\text{DBVI}}(\theta, \phi, \gamma)$.
1. Precompute marginal parameters $(m_t, \kappa_t)$ via ODE integration.
2. Sample mini-batch and amortized start $U_0$ from $p_0^\theta(\cdot \mid X_I)$.
3. Use Euler-Maruyama to solve the reverse bridge SDE for $K$ steps, accumulating score-matching terms.
4. Use the endpoint $U_1$ as inducing variables for the DGP forward pass.
5. Update $\theta, \phi, \gamma$ jointly using Adam.

## Key Experimental Results

### Main Results
Evaluation spans UCI regression (10 datasets), image classification (MNIST, Fashion-MNIST, CIFAR-10), large-scale physics (SUSY, HIGGS), and Frey Faces reconstruction. Baselines include DSVI, IPVI, SGHMC, and DDVI.

**Image Classification Test Accuracy (%) (L=3/4):**

| Dataset | Method | Acc(L=3) | Acc(L=4) |
| :--- | :--- | :--- | :--- |
| MNIST | DDVI | 98.84 | 99.01 |
| MNIST | **DBVI** | **99.02** | **99.10** |
| Fashion | DDVI | 90.36 | 90.85 |
| Fashion | **DBVI** | **90.53** | **91.07** |
| CIFAR-10 | DDVI | 95.23 | 95.56 |
| CIFAR-10 | **DBVI** | **95.42** | **95.68** |

**Large-scale Physics AUC (M=128):**

| Dataset | Method | L=2 | L=4 | L=5 |
| :--- | :--- | :--- | :--- | :--- |
| SUSY | DDVI | 0.883 | 0.887 | 0.886 |
| SUSY | **DBVI** | **0.885** | **0.889** | **0.889** |

### Ablation Study
When $\mu_\theta(x) \equiv 0$, DBVI reduces to DDVI. The performance gain of DBVI over DDVI is entirely attributable to the "learnable, data-dependent start + Doob bridge correction $h$."

| Configuration | Equivalent To | Effect |
| :--- | :--- | :--- |
| Full DBVI | —— | Consistently outperforms DDVI |
| w/o Amortized Start | DDVI | Distant start $\to$ long trajectory, slower convergence |

### Key Findings
- **Gains from Start Point, not Model Capacity**: DBVI's per-iteration time is only slightly higher than DDVI (e.g., 0.74s vs 0.69s on CIFAR-10), yet it converges faster and more accurately, indicating improvements come from shortening the diffusion path.
- **Superior on Large Data**: Amortized bridge initialization provides more significant gains on large datasets like YearMSD where unconditional DDVI struggles with slow convergence.
- **Consistent Improvements**: While absolute accuracy/AUC gains are often in the second decimal place, they are consistent across all datasets, layers, and tasks.

## Highlights & Insights
- **"Right Start, Shorter Path"**: While most research focuses on score network expressivity, this paper addresses the "fixed start point" limitation.
- **"No-Loss" Extension**: Since DBVI includes DDVI as a special case ($\mu_\theta \equiv 0$), it theoretically shouldn't perform worse than DDVI.
- **Doob h-transform in VI**: Reusing h-transforms for DGP posterior inference maintains Girsanov-ELBO solvability while grounding data-dependency in stochastic process theory.
- **$Z^{(l)}$ as Amortization Input**: A clever engineering trick that solves dimension mismatch and saves memory by using representative data summaries.

## Limitations & Future Work
- **Small Absolute Gains**: The performance increment over DDVI is small; its cost-benefit ratio in production needs more discussion.
- **Incomplete Theoretical Guarantees**: Convergence and bias bounds for diffusion bridges in VI remain for future work.
- **Linear Drift Assumption**: Closed-form marginals rely on linear forward SDEs; nonlinear cases are not addressed.
- **Inducing Input Dependency**: Reliance on $Z^{(l)}$ might propagate errors if inducing points are poorly learned or in very deep layers.

## Related Work & Insights
- **vs DDVI**: DBVI generalizes DDVI by replacing fixed Gaussian starts with learnable ones, shortening trajectories and adding amortization.
- **vs IPVI**: IPVI uses GAN-style training which is unstable; DBVI uses a stable, solvable ELBO.
- **vs SGHMC**: SGHMC is high-accuracy but computationally heavy (8s/iter vs 0.7s/iter for DBVI on CIFAR-10).
- **vs Diffusion/Schrödinger Bridges**: While usually used for generative modeling, DBVI adapts these for DGP posterior inference.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Variational Inference for Cyclic Learning](variational_inference_for_cyclic_learning.md)
- [\[ICLR 2026\] Variational Deep Learning via Implicit Regularization](variational_deep_learning_via_implicit_regularization.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [\[ICLR 2026\] Alternating Diffusion for Proximal Sampling with Zeroth Order Queries](alternating_diffusion_for_proximal_sampling_with_zeroth_order_queries.md)

</div>

<!-- RELATED:END -->
